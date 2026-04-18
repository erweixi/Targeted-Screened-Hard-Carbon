# Targeted-Screened Hard-Carbon ML リポジトリ

[English](./README.md) | [简体中文](./README.zh-CN.md) | [日本語](./README.ja.md)

このリポジトリは、ハードカーボン系ナトリウム貯蔵の機械学習研究について、元の論文コードベースと査読対応後の拡張解析を統合したものです。データ、再現可能なコード、事前計算済みの出力、査読対応資料を一つの公開用リポジトリにまとめています。

このスナップショットには主要な結果ファイルがすでに含まれています。読者は CSV、JSON、PNG の成果物をそのまま確認できます。コードの再実行が必要なのは、結果を再構築したい場合のみです。

## まず見る場所

| 目的 | 最初に見るもの | 主な場所 |
| --- | --- | --- |
| 論文本体に対応する ML ワークフローを確認する | `README.md`、`src/run_fig_s1_benchmark.py`、`src/run_rf_final.py` | `results/`、`outputs/` |
| 査読対応解析を確認する | `docs/reviewer_runbook.md` | `results/reviewer/`、`results/reviewer_10fold/` |
| 査読後の拡張データ解析を確認する | `python -m src.run_all_extended_capacity_analyses` | `data/`、`results/reviewer_extension_fixed/`、`outputs/` |
| 保存された査読返信資料を確認する | `docs/reviewer_materials/` | `.docx` と `.xlsx` ファイル |
| 簡潔な中国語クイックスタートが必要 | `docs/quickstart_zh.md` | コマンド要約 |

## リポジトリ構成

```text
.
├── data/                     # 基本データセットと拡張 Excel 由来データ
├── docs/                     # 査読用ランブック、クイックスタート、保存済み査読資料
├── outputs/                  # 出力図と図作成補助テーブル
├── results/                  # 数値結果。ルートは論文本体、サブフォルダは査読関連
├── scripts/                  # 単独実行用の補助スクリプト
├── src/                      # 再現可能な Python モジュールとエントリーポイント
├── requirements.txt
└── .gitignore
```

## インストール

Python 3.10-3.12 を推奨します。

```bash
pip install -r requirements.txt
```

注記：ベイズ探索スクリプトには `scikit-optimize` が必要です。一部の Python 3.13 環境ではこの依存関係が正しくインストールできない可能性があるため、完全再実行には Python 3.10-3.12 の方が安全です。

## クイックスタート

### 1. 論文本体の benchmark と最終 Random Forest モデルを再現する

ICE benchmark:

```bash
python -m src.run_fig_s1_benchmark \
  --task ice \
  --data data/hc_dataset_ice.csv \
  --n-iter 20 \
  --cv-folds 5 \
  --random-state 42 \
  --outdir results
```

Plateau benchmark:

```bash
python -m src.run_fig_s1_benchmark \
  --task plateau \
  --data data/hc_dataset_plateau.csv \
  --n-iter 20 \
  --cv-folds 5 \
  --random-state 42 \
  --outdir results
```

最終 Random Forest モデル:

```bash
python -m src.run_rf_final --task ice --data data/hc_dataset_ice.csv --n-iter 20 --cv-folds 5 --random-state 42 --outdir results/reviewer
python -m src.run_rf_final --task plateau --data data/hc_dataset_plateau.csv --n-iter 20 --cv-folds 5 --random-state 42 --outdir results/reviewer
```

### 2. 査読後の拡張解析を一括で再現する

```bash
python -m src.run_all_extended_capacity_analyses
```

このコマンドは、派生拡張 CSV、可逆容量ベースライン、段階的特徴追加解析、欠損特徴の統計解析を再構築します。

### 3. 再実行せずに同梱済み出力を使う

主要な出力はすでに含まれています。

- 論文本体の参照テーブル：`results/cv_r2_scores.csv`、`results/reference_fig_s1_r2.csv`、`results/rf_predictions_ice_logit.csv`、`results/rf_predictions_plateau.csv`
- 論文・査読用図：`outputs/fig_S1_cv_r2_corrected.png`、`outputs/fig_S2_rf_performance.png`、`outputs/fig_S2_rf_best.png`、`outputs/fig_S3_shap_rf_best.png`
- 査読コア出力：`results/reviewer/`、`results/reviewer_10fold/`
- 査読拡張出力：`results/reviewer_extension_fixed/`

## データファイル

| ファイル | 役割 |
| --- | --- |
| `data/hc_dataset_ice.csv` | ICE モデリング用の 565 行基本データセット。ML ターゲットは `lce = log(ice / (1 - ice))`。 |
| `data/hc_dataset_plateau.csv` | plateau capacity モデリング用の 565 行基本データセット。 |
| `data/hard_carbon_database_20260323_revised.xlsx` | 査読対応拡張解析で用いた拡張 Excel データベース。 |
| `data/hc_dataset_extended_preprocessed.csv` | 拡張 Excel データベースから書き出した前処理済み CSV。 |
| `data/hc_dataset_original_master_with_missing_features.csv` | 元の 565 行データセットと拡張データベース内の追加欠損特徴をマージしたマスターテーブル。 |
| `data/hc_dataset_original_master_mapping_report.csv` | マージの対応状況を要約したレポート。 |

詳細は `data/README.md` を参照してください。

## 主なコードのエントリーポイント

主要スクリプトは次のとおりです。

- `src/run_fig_s1_benchmark.py` — 7 モデル benchmark とベイズハイパーパラメータ探索
- `src/run_rf_final.py` — 最終 Random Forest のチューニングと train/test 評価
- `src/run_model_comparison.py` — 査読対応用のモデル比較表
- `src/run_rf_posthoc_from_best_params.py` — 固定パラメータ RF の SHAP と頑健性出力
- `src/run_all_extended_capacity_analyses.py` — 拡張データ解析の一括実行ワークフロー

補助スクリプトや由来保持のため残してあるユーティリティを含む詳細な一覧は `src/README.md` にあります。

## 再現性に関する注記

- ベイズ最適化（`BayesSearchCV`）には確率的要素があります。乱数シードは固定していますが、Python、BLAS、OS の違いによりごく小さな数値差が出る可能性があります。
- ICE の ML タスクでは logit 変換したターゲット `lce` を使います。一方、査読拡張の一部の記述的・統計的解析では生の `ice` を使います。これは意図的であり、対応スクリプトに明記されています。
- `results/` と `outputs/` のルートファイルは主に論文本体用の参照成果物です。査読向け再実行結果は `results/` 配下のサブディレクトリにあります。

## ドキュメント対応表

- `docs/reviewer_runbook.md` — 査読対応解析のコマンド別ガイド
- `docs/quickstart_zh.md` — 中国語の簡潔なクイックスタート
- `docs/reviewer_materials/README.md` — 保存済み査読資料と正規化ファイル名の対応表
- `results/README.md`、`outputs/README.md`、`data/README.md` — 各フォルダの説明
