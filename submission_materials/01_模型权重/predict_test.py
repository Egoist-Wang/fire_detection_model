"""Resumable image scoring and strict filename -> 0/1 competition export."""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RELEASE = ROOT.parent / 'release_three_model'
MODELS = ('yolo', 'siglip2', 'dinov3')


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def atomic_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.partial')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def read_manifest(path, images, expected_count):
    with path.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    names = [r['image_name'] for r in rows]
    if len(rows) != expected_count or len({n.casefold() for n in names}) != len(names):
        raise ValueError('Manifest count/uniqueness mismatch')
    for row in rows:
        name = row['image_name']
        if name != Path(name).name or '/' in name or '\\' in name:
            raise ValueError(f'Invalid image key: {name}')
        path = images / name
        if not path.is_file() or digest(path) != row['sha256']:
            raise ValueError(f'Missing or changed image: {name}')
    return names


def read_cache(path, names):
    scores = {}
    if not path.exists():
        return scores
    with path.open(encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ['image_name', 'score']:
            raise ValueError(f'Bad cache header: {path}')
        for row in reader:
            name, value = row['image_name'], float(row['score'])
            if name not in names or name in scores or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f'Invalid/duplicate cache row: {path}: {name}')
            scores[name] = value
    return scores


def append_scores(path, batch, values):
    if len(batch) != len(values) or any(not math.isfinite(v) or not 0 <= v <= 1 for v in values):
        raise ValueError('Model returned incomplete or invalid scores')
    exists = path.exists()
    with path.open('a', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(['image_name', 'score'])
        writer.writerows(zip(batch, values))
        f.flush()
        os.fsync(f.fileno())


def fuse(names, scores, config):
    if config['rule'] != 'majority2of3':
        raise ValueError('Only the frozen majority2of3 rule is supported')
    thresholds = {m: float(config[m + '_threshold']) for m in MODELS}
    if any(not math.isfinite(v) or not 0.01 <= v <= 1 for v in thresholds.values()):
        raise ValueError('Threshold must be in [0.01, 1]; YOLO scores below 0.01 were not collected')
    for model in MODELS:
        if set(scores[model]) != set(names):
            raise ValueError(f'Incomplete {model} predictions')
    rows, predictions = [], {}
    for name in names:
        values = {m: float(scores[m][name]) for m in MODELS}
        if any(not math.isfinite(v) or not 0 <= v <= 1 for v in values.values()):
            raise ValueError(f'Invalid score: {name}')
        votes = sum(values[m] >= thresholds[m] for m in MODELS)
        predictions[name] = int(votes >= 2)
        rows.append({'image_name': name, **{m + '_score': values[m] for m in MODELS},
                     'positive_models': votes, 'prediction': predictions[name]})
    return rows, predictions


def infer_stage(model_key, missing, images, weights, config, cache_path, device_name, batch_size, hf_cache):
    import torch
    from PIL import Image
    device = torch.device(device_name)
    use_half = device.type == 'cuda'
    torch.set_float32_matmul_precision('high')

    class LinearHead(torch.nn.Module):
        def __init__(self, dimension, dropout):
            super().__init__()
            self.layers = torch.nn.Sequential(torch.nn.LayerNorm(dimension), torch.nn.Dropout(dropout), torch.nn.Linear(dimension, 2))

        def forward(self, x):
            return self.layers(x)

    head = processor = transform = None
    if model_key == 'yolo':
        from ultralytics import YOLO
        model = YOLO(str(weights))
        size = 4  # Explicit slicing; a list source otherwise bypasses batch=4.
    else:
        checkpoint = torch.load(weights, map_location='cpu', weights_only=False)
        if checkpoint['model'] != config[model_key + '_model']:
            raise ValueError(f'Checkpoint/backbone mismatch: {model_key}')
        head = LinearHead(int(checkpoint['feature_dimension']), float(checkpoint['dropout']))
        head.load_state_dict(checkpoint['head_state_dict'])
        head = head.float().eval().to(device)
        size = batch_size
        if model_key == 'siglip2':
            from transformers import AutoModel, AutoProcessor
            processor = AutoProcessor.from_pretrained(config['siglip2_model'], cache_dir=str(hf_cache), local_files_only=True)
            model = AutoModel.from_pretrained(config['siglip2_model'], cache_dir=str(hf_cache), local_files_only=True).eval().to(device)
        else:
            import timm
            from timm.data import create_transform, resolve_model_data_config
            model = timm.create_model(config['dinov3_model'], pretrained=True, num_classes=0).eval().to(device)
            transform = create_transform(**resolve_model_data_config(model), is_training=False)
    print(f'{model_key}: loaded; {len(missing)} images remaining; batch={size}', flush=True)
    for start in range(0, len(missing), size):
        batch = missing[start:start + size]
        if model_key == 'yolo':
            # RGB arrays avoid Unicode path decoding issues and have deterministic square padding.
            import numpy as np
            arrays = []
            for name in batch:
                with Image.open(images / name) as im:
                    arrays.append(np.asarray(im.convert('RGB'))[:, :, ::-1].copy())
            results = list(model.predict(source=arrays, imgsz=960, batch=4, rect=False,
                                         conf=0.01, iou=0.7, device=str(device), quantize=16 if use_half else 32,
                                         augment=False, stream=True, verbose=False))
            if len(results) != len(batch):
                raise ValueError('YOLO result count mismatch')
            values = []
            for result in results:
                boxes = result.boxes
                values.append(max((float(boxes.conf[i].item()) for i in range(len(boxes))
                                   if int(boxes.cls[i].item()) == 0), default=0.0) if boxes is not None else 0.0)
            del arrays, results
        else:
            rgb = []
            for name in batch:
                with Image.open(images / name) as im:
                    rgb.append(im.convert('RGB'))
            try:
                if model_key == 'siglip2':
                    inputs = processor(images=rgb, return_tensors='pt')
                    inputs = {k: v.to(device) for k, v in inputs.items() if k in {'pixel_values', 'pixel_attention_mask', 'spatial_shapes'}}
                else:
                    inputs = torch.stack([transform(im) for im in rgb]).to(device)
                with torch.inference_mode():
                    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_half):
                        features = model.get_image_features(**inputs) if model_key == 'siglip2' else model(inputs)
                    if hasattr(features, 'pooler_output'):
                        features = features.pooler_output
                    # Match training-score extraction: normalize and run head in FP32, outside autocast.
                    features = torch.nn.functional.normalize(features.float(), dim=1)
                    values = torch.softmax(head(features), dim=1)[:, 1].cpu().tolist()
                del inputs, features
            finally:
                for im in rgb:
                    im.close()
        append_scores(cache_path, batch, values)
        done = min(start + size, len(missing))
        if start == 0 or done % (size * 10) == 0 or done == len(missing):
            print(f'{model_key}: {done}/{len(missing)} remaining images scored', flush=True)
    del model, head, processor, transform
    gc.collect()
    if use_half:
        torch.cuda.empty_cache()


def runtime_info(executable):
    code = """import importlib.metadata as m,json,sys
versions={}
for name in ('torch','ultralytics','transformers','timm','Pillow','numpy'):
 try: versions[name]=m.version(name)
 except m.PackageNotFoundError: versions[name]=None
print(json.dumps({'python':sys.version,'executable':sys.executable,'packages':versions}))
"""
    return json.loads(subprocess.check_output([str(executable), '-X', 'utf8', '-c', code], text=True, encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', type=Path, default=ROOT / 'images')
    parser.add_argument('--manifest', type=Path, default=ROOT / 'test_manifest.csv')
    parser.add_argument('--expected-count', type=int, default=5313)
    parser.add_argument('--config', type=Path, default=RELEASE / 'configs/final_candidate.json')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'baseline')
    parser.add_argument('--score-cache', type=Path, default=ROOT / 'score_cache')
    parser.add_argument('--cache-dir', type=Path, default=Path('C:/AI/hf_cache/hub'))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=8, help='Classifier batch only; YOLO is always 4')
    parser.add_argument('--export-only', action='store_true', help='Reuse complete scores; never load a model')
    parser.add_argument('--yolo-python', type=Path, default=Path('C:/ANACONDA2/envs/torch_gpu/python.exe'))
    parser.add_argument('--dino-python', type=Path, default=Path('C:/ANACONDA2/envs/torch_gpu/python.exe'))
    parser.add_argument('--worker-plan', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_plan:
        plan = json.loads(args.worker_plan.read_text(encoding='utf-8'))
        infer_stage(plan['model_key'], plan['missing'], Path(plan['images']), Path(plan['weights']),
                    plan['config'], Path(plan['cache_path']), plan['device'], plan['batch_size'], Path(plan['hf_cache']))
        return
    if args.batch_size < 1:
        raise ValueError('Invalid batch size')
    os.environ['HF_HOME'] = str(args.cache_dir.parent)
    os.environ['HF_HUB_CACHE'] = str(args.cache_dir)
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    config = json.loads(args.config.read_text(encoding='utf-8-sig'))
    weights = {m: Path(config[m + '_weights']) for m in MODELS}
    weights = {m: p if p.is_absolute() else RELEASE / p for m, p in weights.items()}
    print('Verifying input image hashes and checkpoint hashes...', flush=True)
    names = read_manifest(args.manifest, args.images, args.expected_count)
    runtimes = {'yolo': args.yolo_python, 'siglip2': Path(sys.executable), 'dinov3': args.dino_python}
    runtime_details = {str(p): runtime_info(p) for p in set(runtimes.values())}
    identity = {'manifest_sha256': digest(args.manifest), 'script_sha256': digest(Path(__file__)),
                'weights_sha256': {m: digest(p) for m, p in weights.items()},
                'models': {m: config[m + '_model'] for m in ('siglip2', 'dinov3')},
                'runtimes': {m: runtime_details[str(p)] for m, p in runtimes.items()},
                'device': args.device, 'classifier_batch': args.batch_size,
                'yolo_batch': 4, 'imgsz': 960, 'iou': 0.7, 'predict_conf': 0.01, 'rect': False,
                'head_precision': 'float32', 'backbone_cache': str(args.cache_dir.resolve())}
    args.score_cache.mkdir(parents=True, exist_ok=True)
    identity_path = args.score_cache / 'identity.json'
    if identity_path.exists():
        if json.loads(identity_path.read_text(encoding='utf-8')) != identity:
            raise ValueError('Cache belongs to different images/model/code/environment. Use a new --score-cache folder.')
    else:
        if any(args.score_cache.glob('*.csv')):
            raise ValueError('Score cache lacks provenance; refusing to use it')
        atomic_json(identity_path, identity)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    request = {'identity': identity, 'config': config}
    request_path = args.output_dir / 'request.json'
    if request_path.exists() and json.loads(request_path.read_text(encoding='utf-8')) != request:
        raise ValueError('Output belongs to another configuration; use a new --output-dir')
    atomic_json(request_path, request)
    scores = {}
    for model_key in MODELS:
        cache_path = args.score_cache / (model_key + '.csv')
        scores[model_key] = read_cache(cache_path, set(names))
        missing = [n for n in names if n not in scores[model_key]]
        print(f'{model_key}: cached {len(scores[model_key])}/{len(names)}', flush=True)
        if missing:
            if args.export_only:
                raise ValueError(f'Incomplete cache: {model_key}; export-only cannot infer')
            plan_path = args.score_cache / (model_key + '_worker.json')
            atomic_json(plan_path, {'model_key': model_key, 'missing': missing, 'images': str(args.images.resolve()),
                                   'weights': str(weights[model_key].resolve()), 'config': config,
                                   'cache_path': str(cache_path.resolve()), 'device': args.device,
                                   'batch_size': args.batch_size, 'hf_cache': str(args.cache_dir.resolve())})
            subprocess.run([str(runtimes[model_key]), '-X', 'utf8', '-u', str(Path(__file__).resolve()),
                            '--worker-plan', str(plan_path.resolve())], check=True)
            scores[model_key] = read_cache(cache_path, set(names))
    rows, predictions = fuse(names, scores, config)
    csv_path = args.output_dir / 'image_scores.csv'
    with csv_path.with_suffix('.csv.partial').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    csv_path.with_suffix('.csv.partial').replace(csv_path)
    output = args.output_dir / 'submission.json'
    atomic_json(output, predictions)
    check = json.loads(output.read_text(encoding='utf-8'))
    if set(check) != set(names) or any(type(v) is not int or v not in (0, 1) for v in check.values()):
        raise RuntimeError('Submission verification failed')
    summary = {'images': len(check), 'predicted_fire': sum(check.values()),
               'predicted_no_fire': len(check) - sum(check.values()), 'submission_sha256': digest(output),
               'scores_sha256': digest(csv_path), 'config': config, 'f1': None,
               'note': 'No official ground truth. Predicted class counts are not performance metrics.'}
    atomic_json(args.output_dir / 'submission_check.json', summary)
    print(f'PASS: {len(check)} unique image keys, integer 0/1 values.\nSubmission: {output}', flush=True)


if __name__ == '__main__':
    main()
