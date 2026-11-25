# 小项目 1 NumPy 图像编码器

使用课程提供的 OpenCLIP ViT-B-32 权重完成图像编码器的 NumPy 前向计算。输入是预处理后的 `float32[B,3,224,224]`，输出是每张图一个 512 维单位向量，形状为 `float32[B,512]`。

## 准备课程数据

从畅课小项目 1 的附件 `project1-numpy-student` 中取出完整的 `assets/` 和 `weights/`，放在仓库根目录。这两个目录包含课程的 10 张样图、清单、参考输出及图像模型权重。课程数据由原附件提供，仓库的 `.gitignore` 会排除这两个目录。

```text
project1-numpy/
├── assets/            课程样图、参考输出和清单
├── weights/           课程图像权重和清单
├── image_encoder.py   本项目实现
├── helpers.py         课程数值函数
├── preprocess_image.py  课程预处理
├── config.json        固定模型配置
├── runtime.py         权重和图片加载
├── check.py           课程分阶段检查
├── run.py             图像编码命令行入口
└── requirements.txt   依赖清单
```

本机工作副本已经连接到原课程附件。新克隆的仓库按上面的目录结构放置数据即可。

## 安装与运行

在仓库根目录执行。推荐使用 Python 3.12，本次验证使用 CPU。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

python check.py --stage all
python run.py --output output_tokens.npz
python run.py --images assets/images/image_03.jpg assets/images/image_01.jpg --output my_tokens.npz
```

也可把 `--images` 后的参数替换成自己的图片路径。输出 `.npz` 包含与输入顺序一致的 `ids` 和 `output_tokens`，用 NumPy 读取。

```python
import numpy as np

with np.load("output_tokens.npz", allow_pickle=False) as result:
    print(result["ids"])
    print(result["output_tokens"].shape)  # 样图全量运行得到 (10, 512)
```

## 实现与验证

`image_encoder.py` 将每张图拆成 49 个图块，投影后拼接 CLS 并加入位置向量。12 个 Transformer 块分别执行多头注意力和两次残差更新，最后用 CLS 生成归一化的输出向量。

课程检查覆盖权重和数据完整性、预处理、图块、输入 token、首块、末块、最终向量及重排和重复输入的批次。最终向量的逐元素校验容差为 `atol=2e-5`、`rtol=2e-4`。全部阶段通过，10 张图的输出最大绝对误差为 `6.780e-07`。环境版本、各阶段误差和额外验证见 [实测记录](results/validation.txt)。
