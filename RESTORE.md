# Restore Guide

本仓库与 Release 一起构成识火模型的完整可恢复备份。

## 1. 仓库内容

`main` 分支包含：

- 推理代码：`inference/predict_fusion.py`
- 训练/评测脚本：`src/`
- 配置：`configs/`
- 数据协议与第三方说明：`docs/`
- 提交材料与日志：`submission_materials/`、`submissions/`
- 小型分类头：`weights/dinov3_head.pt`、`weights/siglip2_head.pt`
- 模型卡：`MODEL_CARD.md`
- 本恢复说明：`RESTORE.md`
- 备份/删除清单：`BACKUP_MANIFEST.md`
- Qwen 基座模型元数据清单：`base_models_and_upload_manifest.json`

## 2. Release 资产

Release: https://github.com/Egoist-Wang/fire_detection_model/releases/tag/v1.0.0

| 资产 | 用途 | 大小 | SHA256 |
|---|---|---:|---|
| `yolo26m_epoch77.pt` | 最终 YOLO26m 识火检测器 | 166.97 MB | `f38afc89548218c47b74da8e713b6e4efde560fb197b9f9a4a80ab1af85ff270` |
| `best.pt` | 主训练 run 的 best checkpoint | 42.00 MB | `1d88d793b1617967e9daf8848d31fb46e50284a4f1391a1d88fb6bba5e1fdf4a` |
| `last.pt` | 主训练 run 的 last checkpoint | 42.00 MB | `7a840c459353258873310b7340e93734415d9f8e3258e185812cc56228c625fa` |
| `yolo26m_final_train810_seed4_ckpt-2_logs.zip` | 训练曲线、混淆矩阵、batch 可视化、`results.csv`、`args.yaml` | 4.22 MB | `f3500c2b0b37190d7427aff38fe63e32bcd0939afe8550230544d671ffa7089e` |
| `base_yolo_weights.zip` | `yolo26m.pt`、`yolo26n.pt`、`yolo26s.pt` 基座权重 | 61.65 MB | `1329e1a79b2fcd7364fa24fd509c8641b4f860ebb7e3fb0b0685d5ea058bc2b6` |
| `qwen3_vl_metadata.zip` | Qwen3-VL 4B/8B 的 config/tokenizer/index 等小文件 | 7.28 MB | `7d0074b0a2f686f0c2b4c77c1612e67ec198371d2bebdb9ad5100100b27cac64` |
| `fire_detection_clean_source.zip` | `fire_detection_clean` 的源码/配置/日志文本（不含大模型、数据、图片） | 14.51 MB | `efca1c5a8b41e84212588668f7f926c882c5891aa31d55b9236ca45de0b83d61` |

## 3. 恢复最终推理模型

```powershell
git clone https://github.com/Egoist-Wang/fire_detection_model.git
cd fire_detection_model
# 从 Release 下载 yolo26m_epoch77.pt 到 weights/
# 仓库已包含 weights/dinov3_head.pt 和 weights/siglip2_head.pt
python inference/predict_fusion.py --images <图片目录> --output outputs\predictions.csv --cache-dir <HF缓存目录>
```

## 4. 恢复训练

1. 下载 `best.pt` / `last.pt` 和 `yolo26m_final_train810_seed4_ckpt-2_logs.zip`。
2. 下载 `base_yolo_weights.zip` 获取 `yolo26m.pt` 等基座权重。
3. 训练数据不在 GitHub 仓库中，需要从官方数据源重新准备；数据协议见 `docs/data_protocol.md`。
4. 用 `last.pt` 可恢复训练状态；用 `yolo26m_epoch77.pt` 可复现最终推理模型。

## 5. Qwen3-VL 基座模型

Qwen3-VL 4B/8B 是通用视觉语言基座模型，不是训练后的识火权重，未上传到 GitHub。恢复时从 Hugging Face 下载：

- `Qwen/Qwen3-VL-8B-Instruct`
- `Qwen/Qwen3-VL-4B-Instruct`

本地文件的精确大小和 SHA256 记录在 `base_models_and_upload_manifest.json`。`qwen3_vl_metadata.zip` 保存了 config、tokenizer、index 等小文件，便于核对版本。

注意：本地 8B 目录中的 `model-00004-of-00004.safetensors.part` 是未完成的下载分片；重新下载时应以 Hugging Face 官方文件为准。

## 6. 数据与许可证

- 官方训练图像、标注、测试图像和测试标签不进入 GitHub。
- DINOv3、SigLIP2 骨干权重不进入 GitHub，运行时按官方模型标识加载。
- 第三方组件许可证见 `docs/third_party.md`。