# ESDE Module Reference — Lexicon v2 Pipeline (v5.7.0 追加セクション)

## 概要

Lexicon v2 は **2段パイプライン** で構成される:

1. **語彙供給パイプライン** (`lexicon_wn/`) — WordNet から 326 Atom の語彙を自動展開し、Core/Deviation に分離
2. **A1 観測パイプライン** (`integration/lexicon/`) — Core Pool の各語に 48 次元共鳴度プロファイルを QwQ-32B で観測・監査

```
lexicon_wn/                          integration/lexicon/
  ┌──────────────────┐                 ┌──────────────────┐
  │ 語彙供給          │                 │ A1 観測          │
  │                  │                 │                  │
  │ wn_auto_seed     │                 │ mapper_a1        │
  │ wn_batch_expand  │    lexicon/     │ auditor_a1       │
  │ wn_lexicon_entry │──→ *.json ──→   │ batch_report     │
  │ wn_core_stats    │   (326 files)   │ run_all_atoms.sh │
  │ wn_proposal_gen  │                 │ test_mapper_a1   │
  └──────────────────┘                 └──────────────────┘
        Phase: 語彙基盤                    Phase: 意味観測
```

---

## Part 1: lexicon_wn/（語彙供給パイプライン）

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

---

## Part 2: integration/lexicon/（A1 観測パイプライン）

**Phase**: Lexicon v2 — A1 Semantic Observation  
**役割**: Core Pool の各語を QwQ-32B で 48 次元スロットに共鳴度観測し、構造的品質監査を実施する

### 設計原則（3AI 合意）

- **"Describe, do not decide"** — winner=null を維持、スロット間の勝者は決めない
- **共鳴度スコアリング**: 0-10 の連続値（バイナリ所属ではない）
- **Writer / Auditor 分離**: 同一 LLM でも生成タスク vs 検証タスクでバイアス方向が異なる
- **検出のみ、修正しない**: Auditor は異常を検出し、Writer が制約付きで再観測する

### Pipeline 概要

```
lexicon/*.json (Part 1 の出力: Core Pool)
        │
        ▼
┌────────────────────┐
│ mapper_a1.py        │  Phase 1: QwQ → 48次元共鳴度観測
│                    │  Word → prompt → QwQ-32B → parse JSON
│                    │  → softmax normalize → entropy → focus_rate
└────────┬───────────┘
         │ mapper_output/{atom}_a1.jsonl
         ▼
┌────────────────────┐
│ auditor_a1.py       │  Phase 2: 構造的品質監査 (5 checks: C1-C5)
│                    │  → code pre-screen → LLM audit (flagged only)
│                    │  → PASS / REVISE 判定
│                    │  → REVISE: 制約付き re-observation
└────────┬───────────┘
         │ audit_output/{atom}_a1_final.jsonl
         │ audit_output/{atom}_audit.jsonl
         ▼
┌────────────────────┐
│ batch_report.py     │  Phase 3: 全 atom 横断統計レポート
└────────┬───────────┘
         │ batch_report.md
         ▼
    品質評価・次フェーズ判断
```

### ファイル一覧

| ファイル | 役割 | 入力 → 出力 |
|----------|------|-------------|
| `mapper_a1.py` | A1 Mapper (QwQ 観測) | lexicon/*.json → mapper_output/*.jsonl |
| `mapper_a1.py.bak` | Mapper バックアップ（閾値変更前） | — |
| `auditor_a1.py` | A1 Auditor (構造的監査) | mapper_output/*.jsonl → audit_output/*_final.jsonl |
| `auditor_a1.py.bak` | Auditor バックアップ（INFLATION_NONZERO_THRESHOLD=40 版） | — |
| `batch_report.py` | Batch Report Generator | audit_output/ → batch_report.md |
| `run_all_atoms.sh` | 326 atom 一括実行スクリプト | lexicon/ → mapper_output/ → audit_output/ |
| `test_mapper_a1.py` | Mapper テストスイート | — |

### mapper_a1.py — A1 Mapper

QwQ-32B を観測者として、各語の 48 次元共鳴度プロファイルを生成する。

**構成**:
| 関数 | 役割 |
|------|------|
| `build_a1_prompt()` | 観測プロンプト生成（atom定義 + sym_pair + 48 slot 定義） |
| `call_qwq()` | LLM API 呼出し（リトライ付き） |
| `parse_qwq_response()` | JSON 抽出（think block / markdown fence 除去） |
| `softmax()` | raw scores → 確率分布正規化 (τ=1.0) |
| `shannon_entropy()` | 正規化エントロピー (0=確定, 1=均一) |
| `focus_rate()` | `1.0 - entropy_norm` (高い=焦点明確) |
| `classify_status()` | F < 0.30 → Diffuse_Observation |
| `process_raw_scores()` | full post-processing pipeline |
| `map_single_word()` | 1語の完全パイプライン |
| `run_atom()` | 1 atom の全語バッチ処理（並列対応） |

**LLM 設定**:
| パラメータ | 値 | 備考 |
|-----------|-----|------|
| Model | QwQ-32B (`qwq32b_tp1_short8k`) | Tensor Parallel on dual RTX 5090 |
| Endpoint | `http://100.107.6.119:8001/v1` | TP2 (2-GPU) |
| Temperature | 0.3 | greedy-ish |
| Max tokens | 6000 | |
| Timeout | 180s | |
| Retries | 2 | JSON parse failure 時 |

**出力レコード形式** (JSONL):
```json
{
  "word": "bask",
  "pos": "v",
  "atom": "EMO.like",
  "raw_scores": {"temporal.emergence": 0, ...},
  "normalized_scores": {"temporal.emergence": 0.001, ...},
  "entropy_norm": 0.4277,
  "focus_rate": 0.5723,
  "status": "OK",
  "top5": [{"slot": "epistemological.experience", "p": 0.463}],
  "evidence": "Bask strongly resonates with ...",
  "llm_elapsed_sec": 14.9,
  "timestamp": "2026-02-11T16:25:51Z"
}
```

### auditor_a1.py — A1 Auditor

Mapper 出力に対する構造的品質監査。意味的判断は行わず、数値的異常のみを検出する。

**アーキテクチャ**: 2 フェーズ

| フェーズ | 実行者 | 説明 |
|---------|--------|------|
| Phase 1: Pre-screen | コード（高速・決定的） | 5 構造チェック → flag 生成 |
| Phase 1b: LLM Audit | QwQ-32B | flag 付きレコードのみ LLM 検証 → PASS/REVISE |
| Phase 2: Re-observe | QwQ-32B (Writer) | REVISE レコードを制約付きで再観測 |

**5 構造チェック (C1-C5)**:

| Check | 名称 | 検出対象 | 閾値 |
|-------|------|---------|------|
| C1 | Distribution Anomaly | all-zero, score inflation, spread過多 | sum≥150, NZ≥25, high≥15 |
| C2 | Symmetric Pair Leak | 対義語が非 destructive スロットで高スコア | non-destructive > 3 |
| C3 | Evidence-Score Mismatch | evidence テキストと top スロットの不整合 | keyword 不在 |
| C4 | Axis-Generic Inflation | 軸全体が均一に高い（非弁別的） | axis mean ≥ 4.0, spread ≤ 2 |
| C5 | POS Coherence | 品詞と ontological/experience スロットの不整合 | adj/adv + material ≥ 6 |

**key insight**: 検出は有効だが修正は無効。同一モデルは同じ意味的バイアスを再生産するため、検出のみ行い、制約付き re-observation で対応する。

**Auditor 設定（Before/After 対照）**:
| パラメータ | After Fix (現行) | Before Fix (.bak) | 効果 |
|-----------|-----------------|-------------------|------|
| INFLATION_NONZERO_THRESHOLD | **25** | 40 | nz_mean 38.7→13.6 |
| Diffuse→REVISE 連動 | **有効** | 無効 | Diffuse 捕捉 0/13→12/12 |
| INFLATION_SUM_THRESHOLD | 150 | 150 | — |
| HIGH_SCORE_CEILING | 8 | 8 | — |
| MAX_HIGH_SLOTS | 15 | 15 | — |
| AXIS_MEAN_THRESHOLD | 4.0 | 4.0 | — |
| LLM_AUDIT_TEMPERATURE | 0.2 | 0.2 | Writer より低い（決定的チェック目的） |

**出力レコード形式** (audit JSONL):
```json
{
  "word": "enjoyer",
  "pos": "n",
  "atom": "EMO.like",
  "pre_screen_flags": ["C1_inflation_sum", "C1_inflation_spread"],
  "pre_screen_details": {"C1_inflation_sum": {"detail": "Total sum=168 exceeds 150", "severity": "warn"}},
  "llm_audit": {"overall": "REVISE", "revision_note": "..."},
  "final_status": "REVISE",
  "audit_elapsed_sec": 5.7
}
```

### batch_report.py — Batch Report Generator

全 atom の監査結果を横断集計し、Markdown レポートを生成する。

**レポート内容**:
- Overview: atom 数、総語数、re-observe 数、diffuse 数、mass fail 数
- By Category: カテゴリ別集計（revise率、avg focus、diffuse、mass fail）
- C4/C5 Flag Rates by Category: カテゴリ固有の構造フラグ傾向
- Rare/Thin Words: Seed 改善候補リスト（NZ 少、sum 低）

### run_all_atoms.sh — 全 326 Atom バッチ実行

**機能**:
- lexicon/*.json を自動探索し、mapper → auditor+re-observe を順次実行
- 完了済み atom を自動スキップ（resume-safe）
- SUBSET=0/1 でデュアル GPU 独立実行に分割可能
- GPU 温度監視 + クールダウン待機（GPU1 > 85°C → 60s間隔で冷却待ち）
- ETA 自動計算、per-atom タイミングログ
- 失敗 atom のリスト出力

**設定**:
| 変数 | デフォルト | 説明 |
|------|----------|------|
| PARALLEL | 8 | mapper 並列数 |
| PARALLEL_REOBS | 4 | re-observe 並列数 |
| LLM_HOST | http://100.107.6.119:8001/v1 | TP2 デュアル GPU |
| GPU_COOL_TARGET | 85 | 冷却目標温度 (°C) |
| GPU_COOL_CHECK | 1 | 監視対象 GPU ID |

### test_mapper_a1.py — テストスイート

| テスト | 検証内容 |
|--------|---------|
| test_softmax_basic | softmax 正規化の合計=1 |
| test_softmax_all_zero | 全ゼロ入力 → 均一分布 |
| test_entropy_uniform | 均一分布 → entropy=1.0 |
| test_entropy_single_peak | 単峰分布 → entropy≈0 |
| test_focus_rate_range | focus_rate ∈ [0, 1] |
| test_process_raw_scores_ok | 正常入力 → OK ステータス |
| test_process_raw_scores_diffuse | 全ゼロ → Diffuse_Observation |
| test_process_raw_scores_missing_slots | 欠損スロット → 0 補完 |
| test_parse_json_direct | 直接 JSON → 正常パース |
| test_parse_json_with_think_block | `<think>` 付き → 正常パース |
| test_parse_json_with_preamble | プリアンブル付き → 正常パース |
| test_parse_no_json_raises | JSON 不在 → ValueError |

---

## Score Inflation 問題と対策

A1 パイロット（EMO.like）で発見された品質問題。QwQ-32B が共鳴度スコアを過度に広く付与する傾向。

**症状**: 48 スロット中 39 以上に nonzero を付与（期待値: 8-15）

**原因**: LLM の「関連性があればスコアを付ける」傾向。zero-default（関係なければ 0）の徹底不足。

**対策 2 点**:
1. **プロンプトエンジニアリング**: zero-default スコアリングの明示的指示強化（mapper_a1.py 更新）
2. **Auditor 自動フラグ**: INFLATION_NONZERO_THRESHOLD を 40 → 25 に引き下げ、Diffuse 検出→REVISE 連動（auditor_a1.py 更新）

### Before/After 実測結果

mapper_a1.py.bak / auditor_a1.py.bak が修正前、現行ファイルが修正後:

| 指標 | Before Fix | After Fix | 目標 | 判定 |
|------|-----------|-----------|------|------|
| nz_mean（平均 nonzero スロット数） | 38.7 | **13.6** | ≤25 | ✅ 大幅改善 |
| Diffuse 残存率 | 18.0% | **0.0%** | ≤5% | ✅ 完全消滅 |
| Diffuse → REVISE 捕捉 | 0/13 | **12/12** | 全捕捉 | ✅ 完全捕捉 |
| OK 率 | 78.4% | **97.3%** | — | ✅ 大幅改善 |

**nz_mean 38.7→13.6**: 修正前は 48 スロット中 39 近くに nonzero を付与していたものが、13.6 まで絞り込まれた。各語が本当に共鳴するスロットだけにスコアが集中している状態。

**Diffuse 残存 18%→0%**: 修正前は OK 判定された語の 18% が実際には Diffuse（焦点不明確）だったが、修正後は全て捕捉されるようになった。

**Diffuse→REVISE 0/13→12/12**: 修正前は Diffuse 観測が REVISE に連動せず素通りしていたが、修正後は全件が REVISE として再観測対象になる。

---

## 3AI 役割分担

| AI | 担当 | 成果物 |
|----|------|--------|
| **Claude** | アーキテクチャ・実装 | Pipeline スクリプト群、Core/Dev 分離ロジック、Proposal 生成、A1 Mapper/Auditor 実装 |
| **Gemini** | 統計・運用リアリティ | 10 カラムレポート設計、Constitution 閾値設定、カテゴリ別分析 |
| **GPT** | 監査・ガバナンス | Constitution v1.0 最終稿、双方向 Jaccard ルール、処理優先順位、Auditor スキーマ設計 |

---

*Lexicon v2: "Sibling はノイズではなく偏り（deviation）。座標決定からは分離し、生成過程として Phase 7/9 へ流す。"*
