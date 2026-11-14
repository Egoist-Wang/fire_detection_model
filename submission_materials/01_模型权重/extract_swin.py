from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import timm
from safetensors.torch import load_file

BASE=Path(r'C:\fire_detection_clean')
IMG=BASE/'competition_20261014'/'images'
MAN=BASE/'competition_20261014'/'test_manifest.csv'
WEIGHT=Path(r'C:\Users\薇尔莉特\.cache\huggingface\hub\models--timm--swin_tiny_patch4_window7_224.ms_in1k\snapshots\4cc3a7275b50b53a7bec45f32c236ebe64227cff\model.safetensors')
OUT=Path(__file__).resolve().parent/'swin_features.npz'

def main():
 import csv
 names=[r['image_name'] for r in csv.DictReader(MAN.open(encoding='utf-8-sig'))]
 ap=argparse.ArgumentParser(); ap.add_argument('--batch',type=int,default=64); ap.add_argument('--max-items',type=int); args=ap.parse_args()
 if args.max_items: names=names[:args.max_items]
 if OUT.exists() and len(np.load(OUT,allow_pickle=False)['names'])>=len(names): print('cache exists',OUT); return
 torch.backends.cudnn.benchmark=True
 model=timm.create_model('swin_tiny_patch4_window7_224.ms_in1k',pretrained=False,num_classes=1000)
 model.load_state_dict(load_file(str(WEIGHT)),strict=False)
 model=model.cuda().eval()
 tf=transforms.Compose([transforms.Resize((224,224),interpolation=transforms.InterpolationMode.BICUBIC),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
 feats=[]; start=time.time()
 with torch.inference_mode():
  for i in range(0,len(names),args.batch):
   batch=[]; keep=[]
   for n in names[i:i+args.batch]:
    try:
     with Image.open(IMG/n) as im: batch.append(tf(im.convert('RGB'))); keep.append(n)
    except Exception as e: print('skip',n,e)
   x=torch.stack(batch).cuda(non_blocking=True)
   y=model.forward_head(model.forward_features(x),pre_logits=True).float().cpu().numpy(); feats.append(y)
   done=i+len(keep); 
   if done%256<args.batch or done==len(names): print(json.dumps({'done':done,'total':len(names),'sec_per':(time.time()-start)/done}),flush=True)
 arr=np.concatenate(feats,0); np.savez_compressed(OUT,names=np.array(names),features=arr); print('saved',OUT,arr.shape)
if __name__=='__main__': main()
