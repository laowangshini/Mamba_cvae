# Mamba-CVAE

基于 Mamba / SS2D 的图像条件 CVAE 实验代码（Encoder–Decoder + CelebA 属性条件等）。

## 环境

- Python 3.10+
- PyTorch（CUDA 与 `mamba-ssm`、`causal-conv1d` 等需与本地 CUDA 版本匹配）
- 训练/评估常用：`pyyaml`、`torchvision`、`tqdm`、`matplotlib`、`torchmetrics[image]`、`ptflops`、`pandas` 等

安装示例（版本请按机器调整）：

```bash
pip install torch torchvision pyyaml tqdm matplotlib pandas
pip install mamba-ssm  # 需已配置好 CUDA 与 PyTorch
pip install torchmetrics[image] ptflops
```

## 数据准备

本仓库**不包含** CelebA 人脸图像与大体积特征文件（见 `.gitignore`）。

1. 获取 [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) 对齐人脸集，将 `img_align_celeba/` 解压到：

   `data/celeba/img_align_celeba/`

2. 仓库内已跟踪小的列表文件（如 `data/celeba/list_attr_celeba.csv` 等），用于属性与划分；若你本地路径不同，请相应修改 `train.py` / 配置中的 `DATA_ROOT`。

3. 若使用 CLIP 预处理生成 `celeba_clip_features.pkl`，请在本地生成，勿提交该文件。

## 训练

```bash
cd Mamba-CVAE
python train.py --config configs/exp_b_mamba_ss2d.yaml
```

其他配置见 `configs/`。

## 评估

```bash
python evaluate.py --auto-latest
# 或指定实验子目录
python evaluate.py --exp-dirs <experiments下的目录名>
```

## 目录说明

| 路径 | 说明 |
|------|------|
| `models/` | 模型结构（encoder / decoder / mamba_blocks / cvae） |
| `configs/` | 实验 YAML |
| `train.py` / `evaluate.py` | 训练与定量评估 |
| `scripts/` | 预处理等辅助脚本 |
| `data/celeba/` | 小 CSV 可提交；**图像目录与 pkl 不提交** |

## License

请根据需要自行添加 LICENSE。
