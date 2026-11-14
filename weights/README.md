# Weights

- `dinov3_head.pt`：DINOv3 二分类头，768 维输入。
- `siglip2_head.pt`：SigLIP2 二分类头，768 维输入，类别为 `no_fire` / `fire`。
- `yolo26m_epoch77.pt`：166.97 MB YOLO26m 单类别 `fire` 检测器，因超过 GitHub 100 MB 普通文件限制，放在 GitHub Release 资产中。

下载地址：https://github.com/Egoist-Wang/fire_detection_model/releases/tag/v1.0.0

完整 checkpoint 分析、SHA256 和使用说明见仓库根目录 `MODEL_CARD.md`。