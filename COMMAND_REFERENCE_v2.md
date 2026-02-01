# ESDE Phase 9: コマンドリファレンス

**Version**: v2.0 (Phase 9 完了時点)  
**Updated**: 2026-02-02

---

## ワークフロー

```
Step 1: Harvest（1回だけ）
  python -m harvester.cli harvest --dataset mixed
  → Wikipedia取得 → ローカルキャッシュ保存

Step 2: Analyze（何度でも、オプション組み合わせ自由）
  python -m statistics.pipeline.run_full_pipeline --dataset mixed --lens hybrid --edge-filter mutual_knn --knn-k auto --threshold-mode quantile
  → キャッシュ読み込み → パイプライン実行（ネットワーク不要）
```

---

## Harvester コマンド

### データ取得（harvest）
```bash
# 混合データセット（軍事指導者5 + 学者5 + 都市5）
python -m harvester.cli harvest --dataset mixed

# 戦国武将10人
python -m harvester.cli harvest --dataset warlords

# 強制再取得（キャッシュ上書き）
python -m harvester.cli harvest --dataset mixed --force
```

### キャッシュ一覧（list）
```bash
python -m harvester.cli list
```

### データセット詳細（info）
```bash
python -m harvester.cli info --dataset mixed
```

---

## Pipeline コマンド

### 基本実行（キャッシュから）
```bash
# デフォルト（Structure lens 相当: section軸, token mode, 固定閾値 0.9）
python -m statistics.pipeline.run_full_pipeline --dataset mixed

# Lens 指定（推奨：--lens が --axis と feature mode を自動設定）
python -m statistics.pipeline.run_full_pipeline --dataset mixed --lens hybrid
python -m statistics.pipeline.run_full_pipeline --dataset mixed --lens semantic
python -m statistics.pipeline.run_full_pipeline --dataset mixed --lens structure
```

### 閾値制御
```bash
# 固定閾値（旧互換）
python -m statistics.pipeline.run_full_pipeline --dataset mixed --threshold 0.85

# 動的閾値（quantile mode: t_abs + t_rel → t_resolved）
python -m statistics.pipeline.run_full_pipeline --dataset mixed --threshold-mode quantile --threshold-q 0.98

# quantile + lens の組み合わせ（推奨構成）
python -m statistics.pipeline.run_full_pipeline --dataset mixed --lens hybrid --threshold-mode quantile
```

### エッジフィルタ + k-sweep
```bash
# Mutual-kNN（固定 k=3）
python -m statistics.pipeline.run_full_pipeline --dataset mixed --lens hybrid --edge-filter mutual_knn --knn-k 3

# Mutual-kNN（自動 k 選定: k-sweep → 最小適格 k）★推奨
python -m statistics.pipeline.run_full_pipeline --dataset mixed --lens hybrid --edge-filter mutual_knn --knn-k auto --threshold-mode quantile

# エッジフィルタなし（単連結、連鎖が発生しうる）
python -m statistics.pipeline.run_full_pipeline --dataset mixed --lens hybrid --edge-filter none
```

### フル構成（全機能）
```bash
python -m statistics.pipeline.run_full_pipeline \
  --dataset mixed \
  --lens hybrid \
  --threshold-mode quantile \
  --threshold-q 0.98 \
  --edge-filter mutual_knn \
  --knn-k auto \
  --min-island 2 \
  --output ./output
```

### ライブ取得（非推奨：毎回 Wikipedia 接続）
```bash
python -m statistics.pipeline.run_full_pipeline --fetch --dataset mixed --lens hybrid
```

### モックデータ（オフライン、Harvester 不要）
```bash
python -m statistics.pipeline.run_full_pipeline
```

---

## CLI オプション一覧

### データ選択

| オプション | デフォルト | 説明 |
|-----------|-----------|------|
| `--dataset` | `warlords` | データセット名。組込: `warlords`, `mixed`。Harvester でキャッシュ済みなら任意の名前 |
| `--fetch` | off | Wikipedia API から直接取得（非推奨。先に harvester を使う） |

### Lens（v1.7+）

| オプション | デフォルト | 説明 |
|-----------|-----------|------|
| `--lens` | なし | Lens プリセット。`structure` / `semantic` / `hybrid`。指定すると `--axis` と feature mode を自動設定 |
| `--axis` | `section` | 条件軸（`--lens` 未指定時に使用）。下表参照 |

`--lens` を指定すると `--axis` は無視される。Lens を使うのが推奨。

### 閾値（v1.8+）

| オプション | デフォルト | 説明 |
|-----------|-----------|------|
| `--threshold` | `0.9` | 固定閾値（`--threshold-mode fixed` 時に使用） |
| `--threshold-mode` | `fixed` | `fixed`（固定値）/ `quantile`（動的3層） |
| `--threshold-q` | `0.98` | quantile mode の分位値（0.0〜1.0）。0.98 = 上位2% |
| `--threshold-resolve` | `safety_first` | 合成方式。現在は `safety_first` のみ（max(t_abs, t_rel, floor)） |

### エッジフィルタ（v1.8+）

| オプション | デフォルト | 説明 |
|-----------|-----------|------|
| `--edge-filter` | `none` | `none`（単連結）/ `mutual_knn`（双方向 k 近傍フィルタ） |
| `--knn-k` | なし | k 値。`auto`（k-sweep で自動選定）/ 整数（固定）/ 未指定（ceil(log2(N))） |

`--knn-k auto` は k-sweep を実行し、最小適格 k を自動選定。k-sweep テーブルを CSV と Markdown に出力。

### 出力

| オプション | デフォルト | 説明 |
|-----------|-----------|------|
| `--output` | `./output` | 出力ディレクトリ |
| `--min-island` | `2` | 島の最小メンバー数 |

---

## 条件軸（--axis）

`--lens` 未指定時に使用。`--lens` 指定時は自動設定される。

| 値 | Provider | 条件数 | 説明 |
|----|----------|--------|------|
| `section` | SectionConditionProvider | 可変 | セクション名による分割 |
| `document` | DocumentConditionProvider | 可変 | ドキュメント名（記事単位） |
| `passive` | PassiveConditionProvider | 2 | 受動態か否か |
| `paren` | ParenthesesConditionProvider | 2 | 括弧内か否か |
| `quote` | QuoteConditionProvider | 2 | 引用内か否か |
| `propn` | ProperNounConditionProvider | 2 | 固有名詞を含む文か |
| `section_passive` | SectionPassiveConditionProvider | 可変 | section × passive の組み合わせ |

---

## Lens プリセット

| Lens | --axis 相当 | Feature Mode | 何が見えるか |
|------|------------|-------------|-------------|
| `structure` | section | Token (S-Score) | Wikipedia テンプレート構造 |
| `semantic` | document | Vector (20-dim) | 記事間の主題的類似性 |
| `hybrid` | section | Vector (20-dim) | セクション内の意味的偏り |

---

## データセット

| 名前 | 記事数 | 内容 |
|------|--------|------|
| `warlords` | 10 | 戦国武将（信長、秀吉、家康...） |
| `mixed` | 15 | 軍事指導者(5) + 学者(5) + 都市(5) |

### mixed データセット

| グループ | Prefix | 記事 |
|---------|--------|------|
| 軍事指導者 | `mil_` | Oda Nobunaga, Cao Cao, Napoleon, Hannibal, Alexander the Great |
| 学者 | `sch_` | Einstein, Maslow, Descartes, Fabre, Paracelsus |
| 都市 | `city_` | Tokyo, London, New York City, Paris, Berlin |

---

## ストレージ構造

```
data/
├── artifacts/              # Layer A: 生データ（「捨てるな」）
│   ├── mil_oda_nobunaga.json
│   ├── sch_albert_einstein.json
│   └── city_tokyo.json
├── datasets/               # Layer B: 蒸留テキスト（「記述せよ」）
│   ├── mixed/
│   │   ├── manifest.json   # データセットメタ情報 + Substrate traces
│   │   ├── mil_oda_nobunaga.txt
│   │   ├── sch_albert_einstein.txt
│   │   └── city_tokyo.txt
│   └── warlords/
│       ├── manifest.json
│       └── ...
└── threshold/              # GlobalThresholdModel 累積データ
    ├── structure_token.json
    ├── semantic_vector.json
    └── hybrid_vector.json
```

---

## 出力ファイル

```
./output/
├── report.md             # 人間向けレポート（threshold trace, island一覧）
├── analysis.json         # 機械向けデータ（全構造 + traces）
├── structure_stats.json  # 構造統計（文長、段落数等）
└── k_sweep.csv           # k-sweep 結果テーブル（--knn-k auto 時）
```

### k_sweep.csv の内容

| 列 | 説明 |
|----|------|
| k | Mutual-kNN の k 値 |
| edges | エッジ数 |
| islands | 島数 |
| noise | noise セクション数 |
| largest | 最大島のサイズ |
| gcr | Giant Component Ratio |
| mean_intra | 島内平均類似度 |
| ok | Policy 制約を満たすか（✓ / 空白） |

---

## よく使うレシピ

### 初回セットアップ
```bash
# 1. データ取得
python -m harvester.cli harvest --dataset mixed

# 2. 全 Lens で実行して比較
python -m statistics.pipeline.run_full_pipeline --dataset mixed --lens structure
python -m statistics.pipeline.run_full_pipeline --dataset mixed --lens semantic
python -m statistics.pipeline.run_full_pipeline --dataset mixed --lens hybrid --edge-filter mutual_knn --knn-k auto --threshold-mode quantile
```

### 相転移の観察
```bash
# k-sweep で臨界点を見つける
python -m statistics.pipeline.run_full_pipeline --dataset mixed --lens hybrid --edge-filter mutual_knn --knn-k auto --threshold-mode quantile

# k_sweep.csv を確認 → gcr が急増する k が相転移点
```

### 特定の k で詳細分析
```bash
# k=2（望遠: 微細クラスタ）
python -m statistics.pipeline.run_full_pipeline --dataset mixed --lens hybrid --edge-filter mutual_knn --knn-k 2 --threshold-mode quantile

# k=3（標準: 中粒度クラスタ）
python -m statistics.pipeline.run_full_pipeline --dataset mixed --lens hybrid --edge-filter mutual_knn --knn-k 3 --threshold-mode quantile
```

---

*Phase 9 v2.0 (完了) + Harvester v0.1.0*
