# Backup Manifest and Deletion Log

本文件记录本次 GitHub 备份、可重新下载内容和本地删除范围。

## 1. 已备份到 GitHub

### 仓库 `main`

- 代码、配置、文档、提交材料、训练/推理日志
- `weights/dinov3_head.pt`
- `weights/siglip2_head.pt`
- 模型卡、恢复说明、本清单、Qwen 元数据清单

### Release `v1.0.0`

- `yolo26m_epoch77.pt`
- `best.pt`
- `last.pt`
- `yolo26m_final_train810_seed4_ckpt-2_logs.zip`
- `base_yolo_weights.zip`
- `qwen3_vl_metadata.zip`
- `fire_detection_clean_source.zip`

## 2. 可从官方源重新下载

- `Qwen/Qwen3-VL-8B-Instruct`
- `Qwen/Qwen3-VL-4B-Instruct`
- DINOv3：`hf-hub:timm/vit_base_patch16_dinov3.lvd1689m`
- SigLIP2：`google/siglip2-base-patch16-384`
- 官方训练/测试数据：按数据协议从官方渠道重新准备

## 3. 本次删除范围

- `C:\fire_detection_clean\runs\pure_yolo\yolo26m_final_train810_seed4_ckpt-2\weights\*.pt`
  - 102 个文件，约 16.39 GB
  - `best.pt`、`last.pt` 已上传 Release；`epoch77.pt` 与 Release 的 `yolo26m_epoch77.pt` 哈希一致
  - 其余 `epoch*.pt` 为中间训练检查点，不再保留
- `C:\fire_detection_clean\models\Qwen3-VL-8B-Instruct`
  - 约 16.21 GB；基座模型，元数据已上传，权重可从 Hugging Face 重新下载
- `C:\fire_detection_clean\models\Qwen3-VL-4B-Instruct`
  - 约 8.28 GB；同上
- `C:\Users\薇尔莉特\Desktop\AI`
  - 约 7.00 GB；经相对路径/大小比对，是 `C:\AI` 的子集，无独有文件
- 已验证重复的 YOLO 权重副本（如有）
  - `submission_materials/01_模型权重/yolo26m_epoch77.pt`
  - `release_three_model/weights/yolo26m_epoch77.pt`

合计约 48 GB。

## 4. 保留内容

- `C:\AI`：当前保留的模型/缓存工作副本，约 7.00 GB
- `C:\tem\AI`：包含旧版训练产物和独有 `crop_data_v1`，约 8.88 GB；未直接删除
- `C:\fire_detection_clean\data`：训练数据，约 1.27 GB
- `C:\fire_detection_clean\official_raw`：官方原始图像，约 0.46 GB
- `C:\fire_detection_clean\archive_candidate_20261015`：旧候选，约 0.90 GB
- `C:\Users\薇尔莉特\Desktop\fire_detection`：旧项目副本，约 3.09 GB
- `C:\fire_detection_final_handoff`：交接副本，约 2.13 GB
- 当前 `ChatGPT\fire_detection` 项目中的一份最终推理模型和代码

## 5. 校验

- Release 资产的 SHA256 以 GitHub API 返回的 `digest` 为准。
- 本地删除前已核对关键文件哈希：
  - `epoch77.pt` = `F38AFC89548218C47B74DA8E713B6E4EFDE560FB197B9F9A4A80AB1AF85FF270`
  - `best.pt` = `1D88D793B1617967E9DAF8848D31FB46E50284A4F1391A1D88FB6BBA5E1FDF4A`
  - `last.pt` = `7A840C459353258873310B7340E93734415D9F8E3258E185812CC56228C625FA`
- Qwen 权重的本地 SHA256 记录在 `base_models_and_upload_manifest.json`。