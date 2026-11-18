# 项目一：用 NumPy 复现 OpenCLIP ViT-B-32 图像编码器

本仓库为 Python 程序设计课程项目一的提交。

## 文件说明

| 文件 | 说明 |
| --- | --- |
| `image_encoder.py` | **本人完成**。实现 TODO-01 至 TODO-06，纯 NumPy 前向计算。 |
| `check.py` | 课程提供，逐阶段数值校验。 |
| `helpers.py` | 课程提供。LayerNorm、精确 GELU、注意力系数、向量单位化。 |
| `preprocess_image.py` | 课程提供。缩放、中心裁剪、CLIP 归一化。 |
| `run.py` / `runtime.py` | 课程提供。批量推理与权重加载。 |
| `config.json` | 课程提供。模型结构参数。 |

## 数据准备

本仓库**不包含**课程提供的大文件，因此 clone 后需要先从课程附件取回下列内容，放到仓库根目录的相同位置：

- `weights/image_encoder.npz`（约 326 MB，152 个视觉张量）
- `assets/`（10 张测试图 `images/` 与参考输出 `reference_outputs.npz`）

按作业要求"不要重复提交大型参数文件与参考数据"，故未上传。放置完成后的目录结构应为：

```
project1-numpy-student/
├── image_encoder.py
├── check.py
├── helpers.py
├── preprocess_image.py
├── run.py
├── runtime.py
├── config.json
├── requirements.txt
├── weights/
│   └── image_encoder.npz
└── assets/
    ├── images/
    └── reference_outputs.npz
```

## 运行方法

Python 3.9 及以上，仅依赖 NumPy 与 Pillow：

```bash
python -m pip install -r requirements.txt
```

全阶段校验：

```bash
python check.py --stage all
```

也可单独跑某一阶段，按实现顺序为：

```bash
python check.py --stage patches
python check.py --stage tokens
python check.py --stage block0
python check.py --stage last_tokens
python check.py --stage output
python check.py --stage batch
```

对自带图片生成输出向量：

```bash
python run.py --output output_tokens.npz
```

对任意图片生成输出向量：

```bash
python run.py --images path/to/a.jpg path/to/b.png --output my_tokens.npz
```

## 校验结果

`python check.py --stage all` 输出全部 PASS：

| 阶段 | 形状 | max abs error | mean abs error |
| --- | --- | --- | --- |
| preprocess | (10, 3, 224, 224) | 0.000e+00 | 0.000e+00 |
| patches | (2, 49, 3072) | 0.000e+00 | 0.000e+00 |
| tokens | (2, 50, 768) | 4.005e-05 | 1.289e-06 |
| block0 | (2, 50, 768) | 7.063e-06 | 3.250e-07 |
| last_tokens | (2, 50, 768) | 4.768e-05 | 2.696e-06 |
| output | (10, 512) | 5.439e-07 | 8.311e-08 |
| batch | (3, 512) | 4.619e-07 | 7.632e-08 |

输出 token 容差为 atol=2e-5、rtol=2e-4，实测最大绝对误差 5.4e-7，低于容差两个数量级。

## 实现要点

- 输入 `[B, 3, 224, 224]`，patch 大小 32，故 224/32 = 7，得到 7×7 = 49 个 patch。
- patch 展平顺序为 C、H、W，单 patch 维度为 3×32×32 = 3072。
- patch 投影无 bias，使用 `patch_weight.T`。
- 线性层权重按 `[out, in]` 存储，故计算时取 `weight.T`。
- `visual.proj` 已是 `[in, out] = [768, 512]`，**不转置**。
- 全流程强制 float32，避免 NumPy 与 float64 标量运算时的类型提升。
- 图像分支每个 block 有两次残差相加，无 LayerScale 参数。

## 已知限制

实现针对本 checkpoint 的固定配置（224×224 输入、12 层、12 头），不做通用化处理。
