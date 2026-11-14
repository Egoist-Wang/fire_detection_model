# Fire Detection Model Card

本仓库是独立的识火模型发布仓库，包含 YOLO26m 检测器、DINOv3/SigLIP2 分类头、推理代码、配置和训练/评测记录。模型文件的分析基于实际加载 checkpoint 得到的元数据，而不是仅根据文件名推断。

## 1. 权重清单

| 文件 | 类型 | 大小 | SHA256 | 分发方式 |
|---|---|---:|---|---|
| `yolo26m_epoch77.pt` | Ultralytics YOLO26m 检测器 | 166.97 MB | `F38AFC89548218C47B74DA8E713B6E4EFDE560FB197B9F9A4A80AB1AF85FF270` | GitHub Release 资产 |
| `dinov3_head.pt` | DINOv3 图像分类头 | 约 15 KB | `77358896F6F2F7D4EFE844DFEE856F17C78DE5A7A93740819DF5024C69E5A6B1` | 仓库内 `weights/` |
| `siglip2_head.pt` | SigLIP2 图像分类头 | 约 15 KB | `62BAE91732A5E165DF2CCDF5AAE0C8EAAAEA6E8369B28AE08432A0922DCC0F5D` | 仓库内 `weights/` |

Release: https://github.com/Egoist-Wang/fire_detection_model/releases/tag/v1.0.0

## 2. YOLO26m 检测器

加载 `yolo26m_epoch77.pt` 后可以确认：

- 模型类型：`ultralytics` `DetectionModel`
- 任务：目标检测，单类别 `fire`
- 参数量：21,774,430
- 模型规模：`m`
- 训练轮次：第 77 轮（训练配置为最多 100 轮）
- 最佳适应度：0.30981
- 输入尺寸：960
- batch size：4
- seed：20261004
- 初始学习率：0.01
- 预训练：是
- 训练数据配置：`C:\fire_detection_clean\data\grouped_official_v1\data_final_train810_valcalib.yaml`
- Ultralytics 版本：8.4.87
- checkpoint 日期：2026-10-02T16:11:49
- checkpoint 顶层包含：`epoch`、`best_fitness`、`model`、`ema`、`updates`、`optimizer`、`scaler`、`train_args`、`train_metrics`、`train_results`、`date`、`version`、`git`、`license`、`docs`

该 checkpoint 是完整训练状态，不是只含推理权重的精简文件，因此体积为 166.97 MB。

## 3. DINOv3 分类头

`dinov3_head.pt` 的实际结构：

- 输入特征维度：768
- 分类头参数：`layers.0.weight`、`layers.0.bias`、`layers.2.weight`、`layers.2.bias`
- 输出类别数：2
- dropout：0.2
- 训练轮次：14
- seed：20261001
- 骨干模型标识：`hf-hub:timm/vit_base_patch16_dinov3.lvd1689m`
- 骨干权重不随本仓库分发，运行时由 timm/Hugging Face Hub 按标识加载
- 二模型规则中的阈值：0.62
- 三模型规则中的阈值：0.17

阈值来自 `configs/`，不在分类头 checkpoint 内。

## 4. SigLIP2 分类头

`siglip2_head.pt` 的实际结构：

- 输入特征维度：768
- 分类头参数：`layers.0.weight`、`layers.0.bias`、`layers.2.weight`、`layers.2.bias`
- 输出类别数：2
- 类别名：`{0: 'no_fire', 1: 'fire'}`
- dropout：0.2
- 训练轮次：19
- 验证损失：0.4351003170013428
- seed：20261005
- 骨干模型标识：`google/siglip2-base-patch16-384`
- 骨干权重不随本仓库分发，运行时由 Hugging Face Hub 加载
- 三模型规则中的阈值：0.39

## 5. 未上传的模型和数据

以下内容没有放入本仓库：

- `Qwen3-VL-8B-Instruct`（16.21 GB）和 `Qwen3-VL-4B-Instruct`（8.28 GB）：通用视觉语言基座模型，主要用于标注/诊断，不是训练后的识火专用权重；体积也远超 GitHub 普通文件限制。
- DINOv3、SigLIP2 的骨干权重：通过官方模型标识在运行时加载；是否允许再分发取决于各自许可证。
- 官方训练图像、标注、测试图像和测试标签：按数据协议不进入仓库。
- 同一份 166.97 MB YOLO checkpoint 在 `submission_materials/01_模型权重/` 下的重复副本：为避免重复存储而省略。

## 6. 推理方式

```powershell
python inference/predict_fusion.py `
  --images C:\path\to\test_images `
  --output outputs\predictions.csv `
  --cache-dir C:\AI\hf_cache\hub
```

配置：

- `configs/final_candidate.json`：YOLO26m + DINOv3，规则为 `YOLO >= 0.02 OR DINOv3 >= 0.62`
- `configs/final_candidate_three_model.json`：YOLO26m + DINOv3 + SigLIP2，规则为三模型多数投票

使用前请把 `yolo26m_epoch77.pt` 从 Release 下载到 `weights/`。`weights/siglip2_head.pt` 已随仓库提供，修复了原配置引用但 `weights/` 中缺少该文件的问题。

## 7. 验证记录

`configs/` 中记录的 137 张 calibration split 结果：

| 配置 | precision | recall | F1 |
|---|---:|---:|---:|
| YOLO + DINOv3 | 0.9113 | 1.0000 | 0.9536 |
| YOLO + DINOv3 + SigLIP2 | 0.9496 | 1.0000 | 0.9741 |

这些数字是 calibration split 上的规则选择结果，不是竞赛测试集成绩。`holdout_used_for_rule_selection` 为 `false`。

加载验证：使用 PyTorch 2.8.0 + Ultralytics 8.4.87 成功加载 YOLO checkpoint，`model.names` 为 `{0: 'fire'}`；在 `official_raw/images` 的一张样例图上以 960 输入、conf=0.02 完成一次推理，返回 0 个检测框。该结果仅说明模型可加载和可推理，不作为整体精度证据。

## 8. 第三方组件与许可

本仓库不分发第三方骨干权重，也不对第三方组件重新授权。使用前请核对：

- Ultralytics YOLO：AGPL-3.0
- PyTorch：BSD 风格许可
- timm：Apache-2.0
- DINOv3：以官方模型卡/许可证为准
- SigLIP2：以官方模型卡/许可证为准

更多说明见 `docs/third_party.md`。