# ESDE Module Reference — Lexicon v2 Pipeline (v5.7.0 追加セクション)

## lexicon_wn/（Lexicon v2: WordNet-Based Vocabulary Supply）

**Phase**: Lexicon v2  
**役割**: 326 Atom の語彙を WordNet から自動供給し、Core/Deviation に分離して統計監査する

### Pipeline 概要

```
esde_dictionary.json (326 atoms 定義)
        │
        ▼
┌─────────────────┐
│ wn_auto_seed.py  │  Step 1: 各 atom の WordNet seed synset を自動生成
└────────┬────────┘
         │ seeds.json
         ▼
┌─────────────────────┐
│ wn_batch_expand.py   │  Step 2: 全 atom を WordNet 展開 (12 relations)
└────────┬────────────┘
         │ expanded/*.json (326 files)
         ▼
┌─────────────────────┐
│ wn_lexicon_entry.py  │  Step 3: Core/Deviation 分離 → Lexicon Entry 生成
└────────┬────────────┘
         │ lexicon/*.json (326 files) + _summary.json
         ▼
┌────────────────────────────┐
│ wn_cross_stats.py          │  Step 4a: 全体統計 (full expansion)
│ wn_core_stats.py           │  Step 4b: Core-only 統計 (Mapper's world)
└────────┬───────────────────┘
         │ report.csv / core_report.csv
         ▼
┌─────────────────────┐
│ wn_proposal_gen.py   │  Step 5: Constitution v1.0 に基づく Proposal 自動生成
└────────┬────────────┘
         │ proposals.json
         ▼
    Taka 審査 → 承認/棄却
```

### ファイル一覧

| ファイル | 役割 | 入力 → 出力 |
|----------|------|-------------|
| `wn_auto_seed.py` | Seed 自動生成 | esde_dictionary.json → seeds.json |
| `wn_batch_expand.py` | 326 atom 一括 WordNet 展開 | seeds.json → expanded/*.json |
| `wn_lexicon_entry.py` | Core/Deviation 分離 | expanded/*.json → lexicon/*.json |
| `wn_cross_stats.py` | 全体統計（10カラム GPT レポート） | expanded/*.json → report.csv |
| `wn_core_stats.py` | Core-only 統計 | lexicon/*.json → core_report.csv |
| `wn_proposal_gen.py` | Proposal 自動生成（Constitution v1.0） | core_report.csv → proposals.json |
| `wn_max_expand.py` | 単一 atom 詳細展開（デバッグ用） | atom_id → 詳細 JSON |
| `wn_lexicon.py` | 単一 atom パイプライン（レガシー） | — |

### 展開ステップ（12 relations）

| Step | WordNet Relation | Pool | 説明 |
|------|-----------------|------|------|
| 0_seed | Seed lemmas | Core | 定義の核 |
| 2_hypernym_d1 | 上位語 depth=1 | Deviation | 汎用的すぎる |
| 3_hyponym_d1 | 下位語 depth=1 | Core | 直接の具体化 |
| 4_hyponym_d2 | 下位語 depth=2 | Deviation | 深すぎる |
| 5_hyponym_d3 | 下位語 depth=3 | Deviation | さらに深い |
| 6_derivational | 派生形 | Core | 品詞違い同概念 |
| 7_similar_to | 類語（adj） | Core | 同義語圏 |
| 8_also_see | 関連語 | Deviation | 弱いリンク |
| 9_antonym | 対義語（seed のみ） | Core | 対称ペア境界 |
| 10_sibling | 同親語 | Deviation | **主要汚染源＆情報源** |
| 11_pertainym | 関連形 | Deviation | 散発的 |
| 12_verb_group | 動詞群 | Deviation | 散発的 |

### 統計レポート カラム定義（GPT 設計 10 カラム）

| カラム | 意味 | 健全条件 |
|--------|------|----------|
| total_keys / core_count | 語数 | > 0 |
| unique_ratio_pct | その atom 固有の語の割合 | > 5% |
| mean_atoms_per_word (APW) | 1語が平均何 atom に出現 | < 8 |
| pos_n/v/adj_pct | 品詞分布 | バランス |
| generic_at_5/10/20pct | N% 以上の atom に出る語数 | 少ないほど良い |
| top1_jaccard | 最も重なる atom との Jaccard | < 0.4 |
| sym_overlap_keys | 対称ペアとの共有語数 | 少ないほど良い |

### Constitution v1.0 処理ルール

| Pattern | 条件 | 処置 | 該当数 |
|---------|------|------|--------|
| 🔴 A_MERGE | J≥0.75, 同カテゴリ, サイズ近似 | alias 化 + 多核 Core | 3 |
| 🟠 D_SUBSUME | J≥0.60, 同カテゴリ, 非対称 | parent/child 階層 | 1 |
| 🔵 B_COUPLE | J≥0.50, 異カテゴリ | Phase 9 バイパス | 6 |
| ⚪ MONITOR | J 0.40-0.50 | ログのみ | 7 |

全 proposal は `auto_status: flagged`。Taka 承認必須（「記述せよ、決定するな」）。

### 3AI 役割分担

| AI | 担当 | 成果物 |
|----|------|--------|
| **Claude** | アーキテクチャ・実装 | Pipeline スクリプト群、Core/Dev 分離ロジック、Proposal 生成 |
| **Gemini** | 統計・運用リアリティ | 10 カラムレポート設計、Constitution 閾値設定、カテゴリ別分析 |
| **GPT** | 監査・ガバナンス | Constitution v1.0 最終稿、双方向 Jaccard ルール、処理優先順位 |

---

*Lexicon v2: "Sibling はノイズではなく偏り（deviation）。座標決定からは分離し、生成過程として Phase 7/9 へ流す。"*
