# Targeted-Screened Hard-Carbon ML 仓库

[English](./README.md) | [简体中文](./README.zh-CN.md) | [日本語](./README.ja.md)

这个仓库整合了原始稿件代码与审稿后补充分析，用于硬碳储钠机器学习研究的统一公开发布。数据、可复现代码、预计算结果和审稿回复材料都保存在同一仓库中，便于审稿人和读者直接检索。

当前快照已经包含大部分结果文件。读者可以直接查看仓库中的 CSV、JSON 和 PNG 结果；只有在需要重新构建结果时才需要重新运行代码。

## 从哪里开始

| 目的 | 先看这里 | 主要位置 |
| --- | --- | --- |
| 查看论文主体对应的机器学习流程 | `README.md`、`src/run_fig_s1_benchmark.py`、`src/run_rf_final.py` | `results/`、`outputs/` |
| 查看审稿回复相关分析 | `docs/reviewer_runbook.md` | `results/reviewer/`、`results/reviewer_10fold/` |
| 查看审稿后扩展数据集分析 | `python -m src.run_all_extended_capacity_analyses` | `data/`、`results/reviewer_extension_fixed/`、`outputs/` |
| 查看保留的审稿回复原始文档 | `docs/reviewer_materials/` | `.docx` 与 `.xlsx` 文件 |
| 需要简明中文运行说明 | `docs/quickstart_zh.md` | 命令摘要 |

## 仓库结构

```text
.
├── data/                     # 基础数据集与扩展 Excel 派生数据
├── docs/                     # 审稿运行说明、快速开始、保留的审稿文档
├── outputs/                  # 导出的图件与图表辅助表格
├── results/                  # 数值结果；根目录为论文主体结果，子目录为审稿相关结果
├── scripts/                  # 独立辅助脚本
├── src/                      # 可复现的 Python 模块与入口脚本
├── requirements.txt
└── .gitignore
```

## 安装

建议使用 Python 3.10-3.12。

```bash
pip install -r requirements.txt
```

说明：贝叶斯搜索相关脚本依赖 `scikit-optimize`。在部分 Python 3.13 环境中该依赖可能无法正常安装，因此如果要完整重跑结果，Python 3.10-3.12 更稳妥。

## 快速开始

### 1. 复现论文主体中的 benchmark 与最终随机森林模型

ICE benchmark：

```bash
python -m src.run_fig_s1_benchmark \
  --task ice \
  --data data/hc_dataset_ice.csv \
  --n-iter 20 \
  --cv-folds 5 \
  --random-state 42 \
  --outdir results
```

Plateau benchmark：

```bash
python -m src.run_fig_s1_benchmark \
  --task plateau \
  --data data/hc_dataset_plateau.csv \
  --n-iter 20 \
  --cv-folds 5 \
  --random-state 42 \
  --outdir results
```

最终随机森林模型：

```bash
python -m src.run_rf_final --task ice --data data/hc_dataset_ice.csv --n-iter 20 --cv-folds 5 --random-state 42 --outdir results/reviewer
python -m src.run_rf_final --task plateau --data data/hc_dataset_plateau.csv --n-iter 20 --cv-folds 5 --random-state 42 --outdir results/reviewer
```

### 2. 一条命令复现审稿后扩展分析

```bash
python -m src.run_all_extended_capacity_analyses
```

这条命令会重建扩展 CSV、可逆容量基线、渐进式特征加入分析，以及缺失特征统计分析。

### 3. 不重跑代码，直接使用仓库内已给出的结果

当前仓库已经包含以下关键输出：

- 论文主体参考表：`results/cv_r2_scores.csv`、`results/reference_fig_s1_r2.csv`、`results/rf_predictions_ice_logit.csv`、`results/rf_predictions_plateau.csv`
- 论文/审稿图件：`outputs/fig_S1_cv_r2_corrected.png`、`outputs/fig_S2_rf_performance.png`、`outputs/fig_S2_rf_best.png`、`outputs/fig_S3_shap_rf_best.png`
- 审稿核心结果：`results/reviewer/`、`results/reviewer_10fold/`
- 审稿扩展结果：`results/reviewer_extension_fixed/`

## 数据文件

| 文件 | 作用 |
| --- | --- |
| `data/hc_dataset_ice.csv` | ICE 建模的 565 行基础数据集；机器学习目标为 `lce = log(ice / (1 - ice))`。 |
| `data/hc_dataset_plateau.csv` | plateau capacity 建模的 565 行基础数据集。 |
| `data/hard_carbon_database_20260323_revised.xlsx` | 审稿扩展分析使用的扩展版 Excel 数据库。 |
| `data/hc_dataset_extended_preprocessed.csv` | 由扩展 Excel 数据库导出的清洗后 CSV。 |
| `data/hc_dataset_original_master_with_missing_features.csv` | 将原始 565 行数据集与扩展数据库中的缺失特征映射合并后的主表。 |
| `data/hc_dataset_original_master_mapping_report.csv` | 主表映射覆盖情况汇总。 |

更详细说明见 `data/README.md`。

## 主要代码入口

核心脚本如下：

- `src/run_fig_s1_benchmark.py` — 7 模型 benchmark 与贝叶斯超参数搜索
- `src/run_rf_final.py` — 最终随机森林调参与训练/测试评估
- `src/run_model_comparison.py` — 审稿回复中的多模型比较表
- `src/run_rf_posthoc_from_best_params.py` — 固定参数 RF 的 SHAP 与稳健性导出
- `src/run_all_extended_capacity_analyses.py` — 一条命令执行扩展数据集分析

更完整的脚本索引，包括辅助脚本与为保留溯源而保留的工具，见 `src/README.md`。

## 可复现性说明

- 贝叶斯优化（`BayesSearchCV`）本身带有随机性。仓库已固定随机种子，但不同 Python、BLAS 或操作系统环境下仍可能出现极小的数值偏差。
- ICE 机器学习任务使用 logit 变换后的目标 `lce`；审稿扩展中的部分描述性或统计分析使用原始 `ice`。这是有意保留的设计，相关脚本中已有说明。
- `results/` 和 `outputs/` 根目录下的文件主要对应论文主体参考结果；审稿专用重跑结果位于 `results/` 下的子目录。

## 文档索引

- `docs/reviewer_runbook.md` — 审稿回复分析的逐命令说明
- `docs/quickstart_zh.md` — 中文快速开始说明
- `docs/reviewer_materials/README.md` — 保留的审稿文档与规范化文件名说明
- `results/README.md`、`outputs/README.md`、`data/README.md` — 各子目录说明
