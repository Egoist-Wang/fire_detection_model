推荐提交：SUBMIT_20261017_EXTRA_ORIG_RATE.json

方法：Swin-Tiny 特征 + YOLO/SigLIP2/DINOv3 分数 + 原始 0.47 提交的二值结果，使用 ExtraTrees 融合。
人工标注区间：第 2000-2700、4000-4500 张；这两段标签被直接保留。
交叉验证：两段互相验证，F1=0.66445，Precision=0.625，Recall=0.709。
文件检查：5313 个键，顺序与 test_manifest.csv 一致，值为整数 0/1，阳性 688 个。
SHA256：5d9426f0013c88ad63e96cc5f12a7ac078d32d64f1edcf07881b3d09b3b7f8

这是基于有限人工标注区间的候选，交叉验证结果不能保证等于官方测试集 F1。
