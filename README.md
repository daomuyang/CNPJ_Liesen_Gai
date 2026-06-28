# CNPJ — UK Biobank T1 MRI 去噪项目
## 盖烈森23307130013
计算神经学期末作业，UK Biobank T1 脑 MRI 图像去噪，采用"三层递进评估 + 一层全量验证"实验框架。

## 项目结构

> 标签说明：  
> `[GitHub]` 已上传至 GitHub  
> `[网盘]` 需从百度网盘下载（链接见"数据准备"）  
> `[本地]` 需用户自行提供或脚本生成  
> `[—]` 垃圾文件，未上传

```
CNPJ/                                   [GitHub]
├── README.md                           [GitHub]
├── requirements.txt                    [GitHub]
├── references.bib                      [GitHub]
├── report.tex                          [GitHub]
├── report.pdf                          [GitHub]
│
├── data/
│   ├── __init__.py                     [GitHub]
│   ├── manifest.csv                    [GitHub]
│   ├── raw/                            [本地] 每例子文件夹含 T1_noisy.nii.gz + T1_clean.nii.gz
│   └── preprocessed/                   [本地] preprocess.py 生成，{caseid}/{noisy.npy, clean.npy}
│
├── experiments/                        [网盘] 百度网盘下载，解压至此处
│   ├── preliminary_summary.json        # 预实验汇总
│   ├── server_summary.json             # 全量训练汇总
│   ├── stageA_* / stageB_* / stageC_*  # 各阶段独立实验目录
│   ├── server_UNet_* / server_SE_*     # Stage D 全量训练目录
│   └── visualization/                  # 可视化输出
│
├── src/                                [GitHub]
│   ├── config.py
│   ├── dataset.py
│   ├── train.py
│   ├── models/
│   │   ├── factory.py
│   │   ├── unet.py                     # U-Net (32ch / 64ch)
│   │   ├── red_cnn.py                  # RED-CNN
│   │   ├── attention_unet.py           # SE-UNet (32ch)
│   │   ├── unetpp_lite.py              # UNet++ Lite
│   │   └── dncnn.py                    # DnCNN（未参与最终实验）
│   └── utils/
│       ├── metrics.py
│       ├── seed.py
│       └── io.py
│
├── preprocess.py                       [GitHub]
├── run_preliminary.py                  [GitHub]
├── run_full_train.py                   [GitHub]
├── run_baseline.py                     [GitHub]
├── run_visualization.py                [GitHub]
├── generate_figures.py                 [GitHub]
│
└── report_figures/                     [GitHub]
    ├── loss_ablation.png
    ├── arch_screening.png
    ├── multiseed_stability.png
    ├── full_train.png
    ├── final_result.png
    ├── signal_bin_bar.png
    ├── param_efficiency.png
    ├── triplet_best.png
    ├── triplet_median.png
    └── triplet_worst.png
```

详细复现指导见报告附录（`report.pdf` 第 18--19 页）。

## 环境配置

| 参数 | 值 |
|------|-----|
| 操作系统 | Ubuntu 22.04 |
| CUDA | 12.8.1 |
| Python | 3.12.13 |
| PyTorch | 2.10.0+cu128 |
| GPU | NVIDIA A10 (24 GB) |

```bash
cd CNPJ/
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 数据准备

1. **原始数据**：将 UK Biobank T1 数据集复制至 `data/raw/` 目录，每例一个子文件夹，内含：
   - `T1_noisy.nii.gz` — 噪声 T1 图像
   - `T1_clean.nii.gz` — 干净 T1 图像
   
   目录结构：`data/raw/{caseid}/T1_noisy.nii.gz` 与 `data/raw/{caseid}/T1_clean.nii.gz`

2. **实验中间结果**：从百度网盘下载 `experiments/` 文件夹，解压后放置于 `CNPJ/experiments/`：
   - 链接：https://pan.baidu.com/s/1mEfSFJ5ys1GsNMqpV41tRQ?pwd=2xx2
   - 提取码：`2xx2`

## 完整复现流程

```bash
# 1. 预处理（从 raw NIfTI → .npy 缓存）
python preprocess.py

# 2. 预实验 (Stage A+B+C, 约 83 min)
python run_preliminary.py

# 3. 全量训练 (Stage D, 约 54 min)
python run_full_train.py

# 4. 传统基线 NLM（可选，约 30 min）
python run_baseline.py

# 5. 后处理可视化（约 3 min）
python run_visualization.py

# 6. 生成报告图表
python generate_figures.py

# 7. 编译报告
xelatex report.tex && xelatex report.tex
```

## 训练配置

| 参数 | 值 |
|------|-----|
| 数据划分 | 80/10/10 (train/val/test), seed=42 |
| 优化器 | AdamW (lr=1e-3, wd=1e-5, β=(0.9, 0.999)) |
| Batch Size | 8 |
| 训练步数 | Stage A/B: 1500, Stage C: 1000, Stage D: 5000 |
| 评估间隔 | 每 75 steps |
| 损失函数 | 纯 L1（tissue-only 口径，Stage A 消融后选定） |
| 数据增强 | 50% 概率水平翻转 |
| 预处理 | 三线性插值 128³ → 脑组织掩膜内独立 1-99 百分位归一化 |
| 评测口径 | tissue-only（脑组织掩膜强度 > 0 内计算所有指标） |

## 模型选择建议

### 精度–稳定性综合最优（推荐）
**U-Net 32ch** — test PSNR 48.14 dB (gain 2.68 dB), 跨种子 2σ=0.19 dB, 7.76M 参数。
单次训练即可获得稳定结果，适合科研复现与批量部署。

### 纯精度最优（种子稳定性略差）
**SE-UNet 32ch** — test PSNR 48.23 dB (gain 2.77 dB), 7.85M 参数。
仅 +1.1% 参数增量换取 +0.09 dB 提升。3 种子中 1 例训练失效 (2σ=1.77 dB)，
建议至少运行 5–10 次独立种子训练，选取验证集最优 checkpoint。

### 参数效率最优（小样本/短训练场景）
**UNet++ Lite** — 2.00M 参数, 120 例 1500 步下 gain 2.62 dB。
嵌套密集跳连使小数据短步长下收敛极快。但全量 480 例 5000 步下因容量不足崩溃，
仅适用于训练数据量 ≤ ~120 例的有限预算场景。

### 高容量方案（不推荐）
**U-Net 64ch** — 31.04M 参数（4× 于 32ch），test gain 仅 +0.05 dB 提升，跨种子 2σ=0.82 dB。
128² 分辨率下 32 基础通道已触及容量上限，翻倍通道无实质增益。

### 最低参数量方案（不推荐）
**RED-CNN** — 1.02M 参数, Stage B gain 2.47 dB (Δ=0.16 dB vs. 最优)。
感受野线性增长限制全局噪声建模，未进入后续筛选。

## 文件上传说明

- **GitHub**（https://github.com/daomuyang/CNPJ_Liesen_Gai）：所有标注 `[GitHub]` 的文件已上传
- **百度网盘**（链接见"数据准备"）：`experiments/` 下载后解压至 `CNPJ/experiments/`
- **数据集**：课程原始数据，请自行放置至 `data/raw/`（结构见上方项目结构）
- **预处理缓存**：`data/preprocessed/` 由 `preprocess.py` 自动生成
