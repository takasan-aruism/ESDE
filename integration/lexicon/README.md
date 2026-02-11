# ESDE Lexicon v2 — WordNet Pipeline

## Overview

326 atom 全件を WordNet 経由で展開 → 全体統計レポートを生成するパイプライン。

**設計原則**: まず最大値を出す → 統計で判断 → 引き算で最適化

## Pipeline (3 steps)

```
esde_dictionary.json
        │
        ▼
┌─────────────────┐
│ wn_auto_seed.py  │  Step 1: 各 atom の WordNet seed synset を自動生成
└────────┬────────┘
         │ seeds.json
         ▼
┌─────────────────┐
│ wn_batch_expand  │  Step 2: 全 atom を WordNet 展開 (12 step, 全リレーション)
└────────┬────────┘
         │ expanded/*.json (326 files)
         ▼
┌─────────────────┐
│ wn_cross_stats   │  Step 3: 全体統計レポート (GPT 10-column)
└────────┬────────┘
         │ report.csv
         ▼
    GPT 監査 / 引き算判断
```

## Usage

```bash
# Prerequisites
pip install nltk
python3 -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"

# Step 1: Generate seeds for all 326 atoms
python3 wn_auto_seed.py --dictionary esde_dictionary.json --out seeds.json

# Step 2: Expand all atoms (全ステップ、制限なし)
python3 wn_batch_expand.py --seeds seeds.json --outdir expanded/

# Step 2b: (Optional) 特定ステップだけで展開
python3 wn_batch_expand.py --seeds seeds.json --outdir expanded_trimmed/ --steps 0,2,3,4,6,7,9

# Step 2c: (Optional) 特定 atom だけテスト
python3 wn_batch_expand.py --seeds seeds.json --outdir test/ --atoms EMO.like,EMO.anger

# Step 3: Cross-atom statistics report
python3 wn_cross_stats.py --dir expanded/ --dictionary esde_dictionary.json --out report.csv
```

## Report Columns (GPT/Gemini consolidated)

| # | Column | Description |
|---|--------|-------------|
| 1 | atom_id / category | Atom ID and category |
| 2 | seed_count / seed_lemma_count | Seed 起因の量 |
| 3 | total_keys | unique lemma::pos 数 |
| 4 | pos_*_pct | 品詞分布 (n/v/adj/adv) |
| 5 | sibling_ratio_pct | Sibling ステップの比率 (source_counts 内) |
| 6 | unique_keys / unique_ratio_pct | 他 atom と非重複の語数 |
| 7 | mean_atoms_per_word / p95_atoms_per_word | 直交性指標 |
| 8 | generic_at_*pct | K% 以上の atom に出現する語の数 |
| 9 | top5_overlap_json (+ jaccard) | 最も重なる 5 atom |
| 10 | sym_overlap_keys / from_antonym / from_sibling | 対称ペア漏洩 |

Plus what-if columns:
- remaining_after_5pct / 10pct / 20pct → generic 除外後の残り語数

## Single-Atom Tools (prototyping)

```bash
# Detailed single-atom expansion with per-step word listing
python3 wn_max_expand.py EMO.like

# Original v1 pipeline (seed + expand + clean + mapper prompt)
python3 wn_lexicon.py EMO.like --seed
python3 wn_lexicon.py EMO.like --dry   # LLM prompt output
```

## Files

| File | Purpose |
|------|---------|
| `wn_auto_seed.py` | Step 1: Auto seed generation from esde_dictionary.json |
| `wn_batch_expand.py` | Step 2: Batch WordNet expansion (all atoms) |
| `wn_cross_stats.py` | Step 3: Cross-atom statistics (GPT 10-column report) |
| `wn_max_expand.py` | Single-atom max expansion with per-step word listing |
| `wn_lexicon.py` | Single-atom pipeline (seed → expand → clean → mapper prompt) |
