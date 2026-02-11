# ESDE Glossary

**Version**: 5.7.0  
**Updated**: 2026-02-11  
**Spec**: Existence Symmetry Dynamic Equilibrium  
**Status**: Synapse Expansion Phase 1-3 完了 + 実走 v3.2 まで完了 + **Lexicon v2 Pipeline 完成 + Constitution v1.0 確定**

---

## Change Log

| Version | Date | Changes |
|---------|------|---------|
| 5.4.8-MIG.2 | 2026-01-25 | Migration Phase 2, Substrate Layer |
| 5.5.0 | 2026-02-02 | Phase 9 完了。W層再定義、Lens/Threshold/Edge Policy/Mutual-kNN 追加。旧W0-W6定義を廃止し実装準拠に更新。File Locations・Phase History・Key Metrics を全面改訂 |
| 5.5.2 | 2026-02-05 | Observation C: Relation Pipeline 用語追加。Synapse 動詞接地限界の発見を記録 |
| 5.6.0 | 2026-02-06 | Synapse Expansion Phase 1-3 完了。SynapseStore/Overlay/Tombstone、SynapseEdgeProposer/4-Pack Rewrite、CLI（propose-synapse / evaluate-synapse-patch）、DiagnosticResult/Audit Gate 用語追加。Phase History 更新 |
| 5.6.1 | 2026-02-08 | 実走 v3.1/v3.2 反映。CLI に --synapse-patches 追加（propose/evaluate 両対応）。evaluate の Baseline Patch Auto-Inherit（GPT §5）実装。Synapse Version History に v3.1/v3.2 追加。Coverage Gap/Misground の実測値更新。Phase History に SYN-EXP2/3, SYN-RUN1/2 追加。Key Thresholds に Synapse Expansion 閾値追加。GPT Audit Amendments 一覧新設 |
| 5.7.0 | 2026-02-11 | Lexicon v2: WordNet 供給パイプライン完成。Core/Deviation 二層化、Lexicon Constitution v1.0 確定。17 proposals（3 merge, 1 subsume, 6 couple, 7 monitor）Taka 承認済み。Pipeline: wn_auto_seed → wn_batch_expand → wn_lexicon_entry → wn_core_stats → wn_proposal_gen。用語追加: Lexicon Entry / Core Pool / Deviation Pool / Lexicon Constitution / Proposal Pattern A-D |

---

## Core Philosophy

### Aruism (アリズム)
The philosophical foundation of ESDE, based on the primordial recognition: "There is" (Aru wa, Aru). All understanding derives from this fundamental acknowledgment of existence.

### "Describe, but do not decide" (記述せよ、しかし決定するな)
Core principle for observation layers. Systems observe and record without making semantic judgments or classifications. Uncertainty is a valid outcome, not a failure state.

---

## Semantic Structure (Phase 8: Strong Meaning)

### Atom
The indivisible unit of meaning in ESDE. The Foundation Layer defines 326 canonical atoms across 16 categories (ACT, EMO, REL, etc.). Atoms are the strong meaning system — stable reference points for observation. 163 symmetric pairs.

### Molecule
A structured composition of atoms that represents observed meaning in context. Format:
```
{
  "active_atoms": [{"atom": "EMO.love", "axis": "ethical", "level": 3}],
  "formula": "EMO.love"
}
```

### Axis (Phase 8)
One of 8 canonical axes that provide dimensional context for atom activation: *cognitive*, *ethical*, *social*, *creative*, *ontological*, *temporal*, *spatial*, *physical*.

### Level
A 5-point scale (1-5) indicating intensity or degree along an axis.

### Synapse
The bridge between natural language and semantic atoms. Maps WordNet synsets to ESDE atoms with trigger words. v3.0: 11,557 synsets, 22,285 edges。v3.1/v3.2 パッチにより動詞カバレッジを段階的に拡張中（Overlay 方式、Base JSON は不変）。

### SynapseStore (v5.6.0 新設)
Synapse データの単一ソース（`synapse/store.py`）。Base JSON + Overlay patch を統合して提供。Phase 8 Sensor / Observation C / Phase 7 Engine が同一インスタンスを共有する（GO 条件）。

### SynapsePatchEntry
Synapse パッチ1件のデータモデル（`synapse/schema.py`）。`op`（add_edge / disable_edge）と `edge_key`（`{synset_id}::{atom_id}`）で一意識別。

### Overlay
Base Synapse JSON の上にパッチを重ねて適用する仕組み。衝突解決ルール: disable_edge が常に勝つ（tombstone）、add_edge 重複は後勝ち。Design Spec v2.1 §2 で定義。

### Tombstone
`disable_edge` で無効化された edge。SynapseStore 内でメモリ保持されるが `get_edges()` には出現しない。一度 tombstone 化された edge は re-add 不可（永久除外）。

### edge_key
Synapse edge の一意識別子。形式: `{synset_id}::{atom_id}`（例: `kill.v.01::ACT.destroy`）。Overlay の衝突解決・監査追跡に使用。

### SynapseEdgeProposer (v5.6.0 新設)
カバレッジギャップ動詞から候補 Synapse edge を生成するモジュール（`synapse/proposer.py`）。入力は lemma のみ（上流は synset を知る必要がない）。内部で WordNet synset 展開 → 4-Pack Rewrite → embedding 比較 → スコア付き候補出力を行う。

### 4-Pack Rewrite
SynapseEdgeProposer の候補生成戦略。各 synset に対し4段階の変換を行う: ① lemma → synsets（WordNet 展開）→ ② synset → definition（定義文取得）→ ③ definition → embedding（ベクトル化）→ ④ embedding → scored candidates（326 Atom との cosine 類似度）。RewritePack データクラスがこの中間状態を保持する。

### RewritePack
4-Pack Rewrite の中間状態を保持するデータクラス。synset_id, definition, embedding, scored_atoms の4要素。全候補の rewrite trace はファイルに書き出され、人間レビューの監査証跡となる。

### DiagnosticResult (v5.6.0 新設)
`run_relations.py` の診断レポートの型付きラッパー（`synapse/diagnostic.py`）。SYNAPSE_COVERAGE_GAP / CONSISTENT_MISGROUND / CATEGORY_MISMATCH への構造化アクセスと、Before/After 比較の diff ロジック・Audit Gate 判定を提供する。環境メタデータ（GPT 監査 §2）を注入する `with_env_meta()` ファクトリを持つ。

### Audit Gate (監査ゲート)
`evaluate-synapse-patch` の機械判定ロジック（Design Spec v3.1 §3.2）。DiagnosticResult.diff() が Before/After を比較し、以下の判定を返す:

- **PASS** (exit 0): ギャップ解消あり、かつ回帰なし
- **WARN** (exit 1): 改善なし（resolved_gaps == 0 かつ delta_rate ≤ 0.001）
- **FAIL** (exit 2): CATEGORY_MISMATCH > 0 または新規 CONSISTENT_MISGROUND ≥ 1

### propose-synapse (CLI Command A)
診断 → 提案 → ベースライン保存の一連フロー。Coverage gap 動詞を抽出し、SynapseEdgeProposer で候補 edge を生成し、Run Directory に全成果物を隔離保存する。`--synapse-patches` で既採用パッチを overlay した状態で診断可能（v5.6.0）。2巡目以降の propose で「解消済み gap を再検出しない」ために必須。

### evaluate-synapse-patch (CLI Command B)
パッチ評価フロー。SynapseStore にパッチを一時的に overlay → 同じコーパスで再診断 → Before/After diff → Audit Gate 判定。GPT 監査 §4 により patches/ ディレクトリには一切書き込まない。v5.6.1 で `--synapse-patches` と Baseline Patch Auto-Inherit（GPT §5）を追加。

### Baseline Patch Auto-Inherit (GPT §5, v5.6.1 新設)
`evaluate-synapse-patch` において、`--synapse-patches` 未指定時に `diagnostic_before.json` の `env_meta.patches_loaded` からベースラインパッチを自動継承する仕組み。Before/After の「比較世界」を一致させ、diff の信頼性を保証する。明示指定時は before 側の値と一致チェックを行い、不一致は WARN を出力する。

### Run ID
実行の一意識別子。形式: `run_{YYYYMMDD_HHMMSS}_{dataset}_{rand4}`。timestamp + PID + nanosecond の SHA-256 先頭4文字で衝突耐性を確保（GPT 監査 §1）。

### Run Directory
提案・評価の全成果物を格納する隔離ディレクトリ（`proposals/{run_id}/`）。含まれるファイル: diagnostic_before.json, patch_candidate.json, proposal_report.md, （evaluate 後）diagnostic_after.json, diagnostic_diff.json, diagnostic_diff.md。

### Environment Metadata (環境メタデータ)
DiagnosticResult に注入される8フィールド（GPT 監査 §2）。diff の信頼性を保証するため、Before/After が同一環境で実行されたことを記録する: synapse_base_path, patches_loaded, dictionary_version, min_score, min_freq, dataset, run_id, code_version。

### PipelineRunnerFn
`run_relations.py` との統合プロトコル（依存性注入点）。`Callable[[str, SynapseStore, float, Path], Dict]` 型。テスト時は mock runner を注入し、本番では `default_pipeline_runner` が実際の Relation Pipeline を呼び出す。

---

## Lexicon v2 (WordNet-Based Vocabulary Supply)

### Background: Synapse の構造的限界

Synapse v3.0 は Glossary 定義（名詞的記述）と WordNet 定義（動詞的記述）の embedding 類似度に依存する。これにより概念的に明白な接続（kill → EXS.death）が閾値以下となる構造的バイアスがあった。Lexicon v2 はこの問題を根本から解決するため、**WordNet を候補供給源として使い、LLM を分類器として使う** 役割分離アーキテクチャを採用した。

### Design Principle: 三役分離

| Role |担当 | What it does |
|------|------|--------------|
| **Supplier** | WordNet (機械的) | 各 Atom に対して候補語を自動生成 |
| **Mapper** | LLM (判断のみ) | Core Pool の語を 48 slot に配置 |
| **Constitution** | ESDE 326 Atoms | 不変の座標系。Lexicon が変わっても Atom は変わらない |

### Lexicon Entry (語彙エントリ)

1 Atom に対する語彙データの構造化単位。Core Pool と Deviation Pool を持ち、Status で管理される。

```json
{
  "atom": "EMO.like",
  "status": "proposed",
  "core_pool": { "count": 44, "words": [...] },
  "deviation_pool": { "count": 70, "words": [...] },
  "deviation_stats": { "dev_ratio": 0.614, ... }
}
```

### Core Pool (座標決定用プール)

Mapper が参照する語群。以下の WordNet 関係から収集された語のみを含む:

| Step | Relation | Rationale |
|------|----------|-----------|
| 0_seed | Seed synset の lemma | 定義の核 |
| 3_hyponym_d1 | 直接下位語 | 直接の具体化 |
| 6_derivational | 派生形 | 品詞違い同概念 |
| 7_similar_to | 類語（形容詞） | 同義語圏 |
| 9_antonym | 対義語（seed のみ） | 対称ペア境界 |

分類ルール: 1 つでも Core step で発見された語は Core（sibling 経由でも見つかっていても Core）。

### Deviation Pool (偏り観測用プール)

座標決定には使わないが、観測データとして永続保持される語群:

| Step | Relation | Rationale |
|------|----------|-----------|
| 2_hypernym_d1 | 上位語 | 汎用的すぎる |
| 4-5_hyponym_d2+ | 深い下位語 | 特殊的すぎる |
| 8_also_see | 緩い関連 | 弱いリンク |
| 10_sibling | 同親語 | **主要汚染源 かつ 主要情報源** |
| 11_pertainym | 関連形 | 散発的 |
| 12_verb_group | 動詞群 | 散発的 |

Deviation Pool は消さない。Phase 7（Unknown fuel）、Phase 9（Relations fuel）、Atom 再編の根拠データとして活用される。

### Status (三状態)

| Status | 意味 | Mapper 参照 |
|--------|------|-------------|
| `proposed` | 機械生成の生データ | ✗ |
| `audited` | 監査 AI 確認済み（未確定） | △（実験用） |
| `core` | Taka 承認済み（憲法） | ✓ |

### Core/Deviation 分離の効果（実測値）

| 指標 | 分離前 (Full) | 分離後 (Core) | 改善 |
|------|------:|------:|------|
| mean_APW median | 4.7 | **2.0** | -57% |
| mean_APW > 8 (汚染 atom 数) | 66 | **0** | 完全消滅 |
| unique_ratio median | 21% | **50%** | +29pt |
| Symmetry leak > 10 keys | 82 pairs | **8 pairs** | -90% |
| Max symmetry leak | 649 | **33** | -95% |

Core 全体: 33,394 語 (avg 102/atom)。Deviation 全体: 97,456 語 (avg 299/atom)。

### Lexicon Constitution v1.0 (語彙憲法)

Core Pool の Jaccard 類似度に基づく Atom ペア処理ルール。3AI（Claude/Gemini/GPT）合意。

**優先順位**: Pattern A > Pattern D > Pattern B/C

#### Pattern A: Merge (相転移)
- 条件: `pair_jaccard >= 0.75`, 同一カテゴリ, `size_diff_ratio <= 0.25`
- 処置: 主 ID 維持、従 ID は `alias_of` として登録。多核 Core 化
- 該当: 3 件 (FND.temporality↔time, PRP.aged↔old, ACT.build↔make)

#### Pattern D: Subsume (包含)
- 条件: `pair_jaccard >= 0.60`, 同一カテゴリ, Pattern A 非該当
- 処置: 両 Atom ID 維持、`parent_of`/`child_of` 付与
- 該当: 1 件 (ACT.create → ACT.make)

#### Pattern B/C: Couple (共鳴)
- 条件: `pair_jaccard >= 0.50`, 異カテゴリ
- 処置: 独立維持、`couple_of` データを Phase 9 にバイパス
- 該当: 6 件 (COG↔FND, COM↔REL, SPC↔WLD, BOD↔COM, ABS↔REL)

#### Monitor (監視)
- 条件: `0.40 <= pair_jaccard < 0.50`
- 処置: ログのみ、行動なし
- 該当: 7 件

全 17 件 `auto_status: flagged` → Taka 承認済み (2026-02-11)。

### Seed Synset

各 Atom の WordNet 展開の出発点となる synset 群。`wn_auto_seed.py` が esde_dictionary.json の定義から自動選定するが、手動指定も可能。

### Atoms Per Word (APW)

1 つの単語が平均していくつの Atom の Pool に出現するかを示す指標。Core Pool での APW が低いほど座標として直交性が高い。APW > 8 は「汚染」と判定。

### pair_jaccard

2 つの Atom 間の Core Pool 語彙重複度。`max(A→B の top1_jaccard, B→A の top1_jaccard)` で双方向を考慮。Constitution の発火条件に使用。

---

## Statistical Structure (Phase 9: Weak Meaning)

### Island
A cluster of **sections** whose writing patterns are statistically similar. Formed by W5 through threshold filtering, edge selection, and connected component analysis. Each section belongs to at most one island (many-to-one), or is classified as **noise**.

Not to be confused with genre classification — islands reflect **editorial patterns** (how something is written), not topic categories (what it is about).

### Noise (Island context)
Sections that belong to no island. This is a valid observation result, not a failure. Noise sections have no mutual top-k neighbor satisfying the threshold, indicating unique or unstable writing profiles.

### Condition Factor (条件因子)
A classification axis extracted from **text internal structure** (not external metadata). Used by W2 to slice token statistics into groups for comparison.

v0.1 (obsolete): `source_type`, `language_profile`, `time_bucket` — external metadata  
v2.0 (current): Section name, document name, passive voice flag, etc. — internal structure

### ConditionProvider
Pluggable module that extracts one condition axis from token features. Available providers:

| Provider | Condition Axis | Output Example |
|----------|---------------|----------------|
| SectionConditionProvider | Section name | `cao_cao__early_life` |
| DocumentConditionProvider | Document name | `cao_cao` |
| PassiveConditionProvider | Passive voice (0/1) | `passive_1` |
| ParenthesesConditionProvider | Inside parentheses (0/1) | `paren_1` |
| QuoteConditionProvider | Inside quote (0/1) | `quote_1` |
| ProperNounConditionProvider | Contains PROPN (0/1) | `propn_1` |

### Lens (レンズ)
A (ConditionProvider, FeatureMode) pair. Determines what aspect of the text is observed.

| Lens | Condition | Feature Mode | What It Reveals |
|------|-----------|-------------|-----------------|
| **Structure** | Section | Token (S-Score) | Wikipedia template topology (Hub/Narrative/Institutional) |
| **Semantic** | Document | Vector (20-dim) | Subject similarity across articles |
| **Hybrid** | Section | Vector (20-dim) | Semantic bias within structural sections |

The optical analogy: changing the lens shows different structures in the same specimen, just as changing a microscope objective reveals different features.

### Feature Mode
How token features are aggregated for comparison:
- **Token mode**: Token frequency → S-Score → Resonance Vector (dimension = number of axis candidates)
- **Vector mode**: 20-dimensional feature vector mean → f-score profile → Cosine similarity

### z-score Profile
A 20-dimensional vector representing a condition's deviation from the global baseline, standardized by standard deviation. Computed by W3 (vector mode).

Key insight: comparing **deviations** (z-scores) rather than **raw means** is essential. Raw means converge to the global baseline by the law of large numbers, making all conditions appear identical.

### Threshold (3-Layer Dynamic)
The similarity threshold for island formation. Not a fixed value — dynamically resolved from three sources:

```
t_abs = Q_global(q)       — Quantile from all historical similarity data
t_rel = Q_run(q)          — Quantile from current run's similarity data  
t_resolved = max(t_abs, t_rel, floor)  — Final threshold (safety-first)
```

- First run: t_abs = fallback (no historical data)
- Subsequent runs: t_abs computed from accumulated global model
- All decisions recorded in **threshold_trace**

### GlobalThresholdModel
Accumulates similarity pair data across runs, organized by (lens, feature_mode). Stored in `data/threshold/{lens}_{feature_mode}.json`. Returns global quantile when n ≥ 30, otherwise fallback.

### Mutual-kNN (Mutual k-Nearest Neighbors)
Edge selection algorithm that prevents single-linkage chaining artifacts. An edge (i,j) is kept only if:
1. j is in the top-k neighbors of i
2. i is in the top-k neighbors of j (mutual requirement)
3. sim(i,j) ≥ threshold

One-sided affinity chains are broken. Without this, the Hybrid lens produces a giant component of 475/492 nodes.

### k (Focal Length)
The k parameter in Mutual-kNN. Not a free parameter — it is the **lens's focal length**:

- k large (wide angle) → global template patterns visible, few large islands
- k small (telephoto) → local thematic clusters visible, many small islands

### k-sweep (EdgePolicyResolver)
Automatic k selection by sweeping candidate values [2, 3, 4, 5, 7, 9, 12, 15] and observing clustering behavior at each. Selects the **smallest k satisfying all policy constraints** (max_giant_ratio ≤ 0.20, min_mean_intra ≥ 0.25). Does not "decide" — observes and recommends. Researcher can override.

### Percolation Threshold (相転移点)
The critical k value where the network transitions from "islands have meaning" to "chaining dominates." Observed in experiments as k=3→4 (gcr jumps from 0.13 to 0.66). This is a property of the data, not a parameter choice.

### Giant Component Ratio (gcr)
`largest_island_size / total_node_count`. Primary indicator of chaining:
- gcr < 0.20: healthy (no dominant island)
- gcr > 0.50: chaining detected (one island dominates)

### Chaining
An artifact where single-linkage clustering creates a chain A→B→C→...→Z through weak one-sided similarities, absorbing all nodes into a single giant component. Prevented by Mutual-kNN.

---

## Integration Layer (Observation C)

### Observation C: Relation Pipeline
Phase 8 と Phase 9 を橋渡しする関係抽出層。テキストから SVO（Subject-Verb-Object）トリプルを抽出し、動詞述語を Synapse 経由で Atom に接地する。LLM を使わない決定論的パイプライン。

### SVO Triple
Subject-Verb-Object の3項関係。spaCy の依存構造解析から抽出される構造的事実。受動態 (passive)、否定 (negated)、接続詞展開 (conjunction) を検出する。

### Grounding Status
Relation Pipeline における動詞の Atom 接地結果を示すタグ。
- **GROUNDED**: Atom が割り当てられた（候補がフィルタを通過）
- **UNGROUNDED**: 候補がフィルタ後に残らなかった（Coverage gap 候補）
- **UNGROUNDED_LIGHTVERB**: 軽動詞のため Atom 付与を抑制（Edge は保持）

### Light Verb (軽動詞)
意味が文脈に強く依存する機能語的動詞。have, make, do, get, take, give, go, come, be, become, include, feature, provide の13語。Phase 8 の強い意味としては扱わず、Phase 9 の文脈分析に委譲する。

### POS Guard (品詞整合性フィルタ)
動詞の Synapse 接地時に、名詞カテゴリ（NAT/MAT/PRP/SPA）の Atom 候補を除外するフィルタ。Synapse が名詞の概念空間に最適化されているために発生する品詞混同（例: include→PRP.dirty）を防ぐ。

### Score Threshold (最低スコア閾値)
Synapse の raw_score がこの閾値未満の候補を UNGROUNDED に倒すフィルタ。デフォルト 0.45（CLI --min-score で可変）。値は暫定であり、ドメイン別の感度分析で調整する。

### CONSISTENT_MISGROUND (一貫した誤接地)
同一動詞が同一 Atom に繰り返し接続される統計的異常パターン。「意味的に間違い」の証明ではなく「怪しい」パターンの検出。現行の grounding は lemma ベース（文脈を見ない）のため、全出現が同一 Atom に飛ぶ。修正には人間判断（disable_edge + 正しい edge 追加）が必要。Diagnostic Report の SYMPTOMS セクションに出力される。

### SYNAPSE_COVERAGE_GAP (接地不能動詞)
Synapse DB に当該動詞の edge 自体が存在せず、どの Atom にも到達できない状態。propose-synapse で候補 edge を自動生成 → 人間レビュー → evaluate-synapse-patch → パッチ承認で解消する。

### Entity Graph
Relation Pipeline の集約出力。ノード（エンティティ）とエッジ（Atom 付き関係）を持つグラフ構造。UI 表示に使用。

### Section Relation Profile
Relation Pipeline の集約出力。セクション別の predicate_atom ベクトルと構造統計（negated_ratio, passive_ratio, directionality）。Phase 9 Lens への入力として設計。

### Diagnostic Report
Relation Pipeline の品質診断レポート。CONSISTENT_MISGROUND（一貫した誤接地）、SYNAPSE_COVERAGE_GAP（接地不能な頻出動詞）、CATEGORY_MISMATCH（品詞混同）の3カテゴリの症状を検出する。

### Harvester
Wikipedia 記事のフェッチとローカルキャッシュを行うデータ収集モジュール。"Fetch once, analyze many times" の原則に従い、ネットワーク I/O を分析処理から分離する。

---

## Key Metrics

### Rigidity (R) — Phase 8
Measures pattern fixation for a concept:
```
R = N_mode / N_total
```
| Range | Status | Strategy |
|-------|--------|----------|
| R < 0.3 | Volatile | STABILIZING |
| 0.3 ≤ R ≤ 0.9 | Healthy | NEUTRAL |
| R > 0.9 | Rigid | DISRUPTIVE |

### S-Score — Phase 9 (token mode)
Condition specificity measure:
```
S(token, condition) = log(P_cond / P_global)
```
Positive = condition-specific, Negative = condition-avoided. Used in Structure lens (W3 token mode).

### Resonance Vector — Phase 9 (token mode)
Per-article projection onto W3 axis candidates, computed by W4Projector. Dimension = number of axis candidates.

### z-score Vector — Phase 9 (vector mode)
Per-condition deviation profile. 20-dimensional. Used in Semantic and Hybrid lenses. Input to W4 cosine similarity.

### Chaining Metrics — Phase 9 (W5)
Diagnostic indicators recorded after every island formation:

| Metric | Definition | Healthy Range |
|--------|-----------|---------------|
| Giant Component Ratio (gcr) | largest / total | < 0.20 |
| Mean Intra-Similarity | average similarity within islands | > 0.25 |
| Edge Sparsity | edges / max_possible_edges | depends on k |
| Chaining Detected | boolean flag | false |

### Grounding Rate — Observation C (v5.6.1 追記)
SVO トリプルのうち動詞が Atom に接続できた割合（軽動詞 1,032 件は分母から除外）。

| Synapse Version | Rate | Delta | Resolved Gaps |
|-----------------|------|-------|---------------|
| v3.0 (base) | 55.2% | — | — |
| v3.0 + v3.1 | 61.0% | +5.8pt | write, serve, join, hold, attack, kill, occupy |
| v3.0 + v3.1 + v3.2 | 63.0% | +2.0pt | cross, found, introduce, represent |

Total triples: 5,342 (excl. lightverb)。回帰ゼロ（全ラウンド）。

---

## Layer Architecture

### Foundation Layer
Contains Glossary (326 atoms) and Synapse (v3.0 + patches). Provides the semantic grounding for all other layers.

### Substrate Layer (Layer 0)
Cross-cutting foundational layer providing machine-observable trace storage. Follows the principle "Describe, but do not decide." No semantic interpretation, only raw observation data.

Key components:
- **ContextRecord**: Immutable observation unit with traces
- **SubstrateRegistry**: Append-only JSONL storage
- **Traces**: Key-value pairs in `namespace:name` format

### Phase 7: Unknown Resolution
Handles tokens outside established semantic space. The weak meaning system — concepts that have not yet acquired stable semantic grounding.

### Phase 8: Introspective Engine
Self-reflection system monitoring concept processing patterns. Implements Rigidity detection and feedback loops. The 326 atoms represent the strong meaning system.

### Phase 9: Weak Axis Statistics [COMPLETE]

Statistical analysis of text writing patterns. Discovers structure through observation, not labeling.

**W-Layer Pipeline (v2.0 — implementation-definitive):**

| W Layer | Name | Input | Output |
|---------|------|-------|--------|
| W1 | Feature Extraction | Raw text | 20-dimensional token features (spaCy) |
| W2 | Conditional Aggregation | W1 + ConditionProvider | Per-condition statistics |
| W3 | Profile Computation | W2 statistics | z-score profiles (vector) or S-Score candidates (token) |
| W4 | Similarity Computation | W3 profiles | Pairwise cosine similarity matrix |
| W5 | Island Formation | W4 + Threshold + Mutual-kNN | Island structure + chaining metrics |
| W6 | Export | W5 structure | JSON / Markdown / CSV / k-sweep table |

**Superseded W-layer definitions (v1.x — do not use):**
W0 (ContentGateway), W1 (Global Statistics), W2 (Conditional Statistics), W3 (Axis Candidates), W4 (Structural Projection), W5 (Structural Condensation), W6 (Structural Observation) — these names described the original single-pipeline design before Lens integration. The v2.0 definitions above reflect the actual implementation.

---

## Experimental Discoveries (Phase 9)

### Wikipedia Template Topology
Structure lens revealed three editorial pattern layers:
- **Hub**: Concentric section layout (city articles: demographics, economy, transport...)
- **Narrative**: Chronological section layout (biographical: early life, campaign, legacy...)
- **Institutional**: Institutional section layout (organizations, legal entities)

These are not genre categories but editorial structural types.

### Phase Transition at k=3→4
In the mixed dataset (15 articles, 492 sections, Hybrid lens), the percolation threshold occurs between k=3 and k=4. Largest island jumps from 62 to 323 (5×). This is the network's intrinsic property, not a parameter artifact.

### Synapse の動詞接地限界（Observation C 発見）

Synapse v3.0 は名詞の概念空間に最適化されており、動詞を同じ辞書で引くと3種の構造的問題が発生する：

1. **CATEGORY_MISMATCH**: 動詞が名詞カテゴリ (PRP/NAT/MAT/SPA) の Atom に接地される（例: include→PRP.dirty）→ POS Guard で完全解消済み
2. **CONSISTENT_MISGROUND**: 多義語の間違った語義が一貫して選ばれる（例: receive→EMO.like, use→ACT.give）→ 統計的検出のみ、修正には人間判断が必要
3. **SYNAPSE_COVERAGE_GAP**: 頻出動詞が Synapse のどの Atom にも到達しない → v3.1/v3.2 パッチで段階的に解消中

v0.2.0 の3フィルタ（Light Verb / POS Guard / Score Threshold）により CATEGORY_MISMATCH は完全解消。

### ドメイン別接地特性

| ドメイン | 典型的な残存 Gap | 傾向 |
|----------|-----------------|------|
| 武将 (mil) | defeat, host, launch | 軍事動詞の不足。kill, attack は v3.1 で解消 |
| 学者 (sch) | operate, publish(misground) | 知的動詞は比較的良好。publish は misground 側の問題 |
| 都市 (city) | contain, host, employ, comprise | 軽動詞比率が高い（19-35%）。rate の分母が小さい |

### Synapse Expansion 実走結果（v5.6.1 時点）

Synapse の動詞カバレッジギャップを解消するための段階的拡張機構。326 Atoms は不変（座標系）、Synapse edges のみ append-only で成長させる。

**実装完了（v5.6.0）:** Phase 1（SynapseStore）、Phase 2（SynapseEdgeProposer）、Phase 3（CLI + Audit Gate）  
**実走完了（v5.6.1）:** v3.1 パッチ（42 edges, PASS）、v3.2 パッチ（27 edges, PASS）

#### 逓減パターン

| ラウンド | 解消動詞 | Rate 上昇 |
|---------|---------|----------|
| v3.1（1巡目） | write, serve, join, hold, attack, kill, occupy | +5.8pt |
| v3.2（2巡目） | cross, found, introduce, represent | +2.0pt |

高頻度かつ WordNet 定義が Atom と嚙み合う動詞が先に解消し、残存動詞は embedding スコアが低い（< 0.55）ため自動提案の限界に近づいている。

#### 現在の症状分布（v3.0 + v3.1 + v3.2）

**Coverage Gaps（10件, 計 215 triples）:**
host(39), defeat(34), contain(25), operate(23), visit(19), marry(17), launch(17), employ(14), comprise(14), spend(13)

**Consistent Misgrounds（10件, 計 107 triples）:**
receive→EMO.like(21), use→ACT.give(15), win→EMO.pride(14), publish→COM.announce(10), carry→BOD.mouth(9), cover→COM.answer(9), offer→SOC.request(8), surround→SPC.outside(7), order→FND.temporality(7), lose→ECO.loss(7)

ガバナンス: Relation Pipeline 診断 → 候補 Edge 自動生成 → 人間レビュー → パッチ承認（Audit Gate PASS 必須）

---

## File Locations

### Foundation Layer
| Component | Path |
|-----------|------|
| Glossary Data | esde_dictionary.json |
| Synapse Data | esde_synapses_v3.json |

### Synapse Expansion (v5.6.0 新設, v5.6.1 更新)
| Component | Path |
|-----------|------|
| SynapseStore | synapse/store.py |
| PatchEntry Schema | synapse/schema.py |
| Edge Proposer | synapse/proposer.py |
| Diagnostic Wrapper | synapse/diagnostic.py |
| CLI (propose/evaluate) | synapse/cli.py |
| Patch Files | patches/ |
| Run Directories | proposals/ |
| Store Tests | tests/test_synapse_store.py |
| CLI Tests | tests/test_phase3_cli.py |
| Migration Guide | MIGRATION_SYNAPSE_STORE.md |

### Substrate Layer
| Component | Path |
|-----------|------|
| Context Registry | data/substrate/context_registry.jsonl |
| Schema | esde/substrate/schema.py |
| ID Generator | esde/substrate/id_generator.py |

### Lexicon v2 Pipeline
| Component | Path |
|-----------|------|
| Auto Seed Generator | lexicon_wn/wn_auto_seed.py |
| Batch WordNet Expander | lexicon_wn/wn_batch_expand.py |
| Core/Deviation Splitter | lexicon_wn/wn_lexicon_entry.py |
| Full Expansion Stats | lexicon_wn/wn_cross_stats.py |
| Core-Only Stats | lexicon_wn/wn_core_stats.py |
| Proposal Generator | lexicon_wn/wn_proposal_gen.py |
| Single-Atom Expander | lexicon_wn/wn_max_expand.py |
| Single-Atom Legacy | lexicon_wn/wn_lexicon.py |
| ESDE Dictionary | lexicon_wn/esde_dictionary.json |
| Seed Definitions | lexicon_wn/seeds.json |
| Expanded Atoms (326 files) | lexicon_wn/expanded/*.json |
| Lexicon Entries (326 files) | lexicon_wn/lexicon/*.json |
| Lexicon Summary | lexicon_wn/lexicon/_summary.json |
| Full Stats Report | lexicon_wn/report.csv |
| Core Stats Report | lexicon_wn/core_report.csv |
| Proposals | lexicon_wn/proposals.json |

### Phase 9 Pipeline (v2.0)
| Component | Path |
|-----------|------|
| Main Pipeline (CLI) | statistics/pipeline/run_full_pipeline.py |
| Lens Definitions | statistics/pipeline/lens.py |
| ConditionProviders | statistics/pipeline/condition_provider.py |
| W2 Aggregator | statistics/pipeline/w2_aggregator.py |
| W3 S-Score (token) | statistics/pipeline/w3_calculator.py |
| W3 z-score (vector) | statistics/pipeline/w3_vector.py |
| W4 Resonance (token) | statistics/pipeline/w4_projector.py |
| W4 Cosine (vector) | statistics/pipeline/w4_vector.py |
| W5 + W6 Adapter | statistics/pipeline/w5_w6_adapter.py |
| ThresholdResolver | statistics/pipeline/threshold.py |
| GlobalThresholdModel | statistics/pipeline/global_model.py |
| MutualKNNSelector | statistics/pipeline/edge_selector.py |
| EdgePolicyResolver | statistics/pipeline/edge_policy.py |
| Chaining Metrics | statistics/pipeline/chaining_metrics.py |

### Phase 9 Data
| Component | Path |
|-----------|------|
| Global Threshold Data | data/threshold/{lens}_{feature_mode}.json |
| k-sweep Results | output/k_sweep.csv |
| Analysis Output | output/analysis.json |
| Markdown Report | output/report.md |

---

## Key Thresholds and Parameters

### Phase 8 (config.py — unchanged)

| Parameter | Value | Purpose |
|-----------|-------|---------|
| COMPETE_TH | 0.15 | Minimum score for competing hypothesis |
| VOL_LOW_TH | 0.25 | Below = candidate status |
| VOL_HIGH_TH | 0.50 | Above = quarantine status |
| UNKNOWN_MARGIN_TH | 0.20 | Variance Gate margin threshold |
| UNKNOWN_ENTROPY_TH | 0.90 | Variance Gate entropy threshold |
| TYPO_MAX_EDIT_DISTANCE | 2 | Maximum edit distance for typo detection |

### Phase 9 (dynamic — no fixed config)

| Parameter | Source | Purpose |
|-----------|--------|---------|
| threshold_floor | lens.py per-lens | Minimum similarity (Structure: 0.85, Semantic/Hybrid: 0.0) |
| quantile_q | CLI (default 0.50) | Quantile for t_rel / t_abs |
| t_resolved | ThresholdResolver | Final threshold = max(t_abs, t_rel, floor) |
| k | EdgePolicyResolver or CLI | Mutual-kNN neighbor count (auto-sweep or fixed) |
| max_giant_ratio | edge_policy.py (0.20) | Policy constraint for k-sweep |
| min_mean_intra | edge_policy.py (0.25) | Policy constraint for k-sweep |
| k_candidates | edge_policy.py | Sweep values: [2, 3, 4, 5, 7, 9, 12, 15] |
| min_island_size | CLI (default 2) | Minimum members to form an island |

### Synapse Expansion (v5.6.0 新設)

| Parameter | Value | Purpose |
|-----------|-------|---------|
| MIN_SCORE (propose) | 0.28 | 候補生成の最低閾値（広めに取得、top-5 per synset） |
| Adoption threshold | 0.55 | パッチ採用時の品質閾値（人間判断で適用） |
| min_score (grounding) | 0.45 | Relation Pipeline でのランタイム grounding 閾値 |
| min_freq | 2 | Coverage gap 動詞の最低出現回数 |

### Lexicon v2 / Constitution v1.0

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Core APW contamination | > 8 | 座標崩壊の閾値 (Core では 0 に改善済み) |
| pair_jaccard (Merge) | >= 0.75 | Pattern A 発火条件 |
| pair_jaccard (Subsume) | >= 0.60 | Pattern D 発火条件 |
| pair_jaccard (Couple) | >= 0.50 | Pattern B/C 発火条件 |
| pair_jaccard (Monitor) | >= 0.40 | 監視対象 |
| size_diff_ratio (Merge) | <= 0.25 | 同一カテゴリ内サイズ近似条件 |
| core_count thin | < 15 | 座標不安定アラート |

---

## Synapse Version History

| Version | Date | Synsets | Edges | Notes |
|---------|------|---------|-------|-------|
| v2.1 | 2025-12-22 | 2,037 | 2,116 | Concept name search only |
| v3.0 | 2026-01-19 | 11,557 | 22,285 | triggers_en support, 100% concept coverage |
| v3.1 (patch) | 2026-02-07 | +16 | +42 | 7 gap 解消 (write, serve, join, hold, attack, kill, occupy). Rate 55.2%→61.0%. 回帰ゼロ |
| v3.2 (patch) | 2026-02-08 | +2 | +27 | 4 gap 解消 (cross, found, introduce, represent). Rate 61.0%→63.0%. 回帰ゼロ |

---

## Phase Version History

| Phase | Version | Date | Description |
|-------|---------|------|-------------|
| 7 | v5.3.2 | 2025-12 | Unknown Resolution with multi-hypothesis routing |
| 8 | v5.3.9 | 2026-01 | Introspection with Rigidity modulation |
| 9 (W0-W3) | v5.4.2 | 2026-01 | ContentGateway → Global Stats → Conditional Stats → S-Score |
| 9 (W4) | v5.4.4 | 2026-01 | Structural Projection (Resonance) |
| 9 (W5) | v5.4.5 | 2026-01 | Weak Structural Condensation (Islands) |
| 9 (W6) | v5.4.6 | 2026-01 | Weak Structural Observation (Evidence) |
| SUB | v0.1.0 | 2026-01 | Substrate Layer (Context Fabric) |
| MIG-2 | v0.2.1 | 2026-01-25 | Migration Phase 2 (Policy-Based Statistics) |
| 9 (v1.7) | v5.4.7 | 2026-01-29 | Lens integration (Structure/Semantic/Hybrid), ConditionProvider |
| 9 (v1.8) | v5.4.8 | 2026-01-31 | Mutual-kNN, Chaining Metrics, Threshold 3-layer |
| 9 (v1.9) | v5.5.0 | 2026-02-02 | EdgePolicyResolver, k-sweep, GlobalThresholdModel. **Phase 9 complete** |
| OBS-C | v5.5.1 | 2026-02-04 | Observation C: Relation Pipeline (SVO + Synapse Grounding) |
| OBS-C2 | v5.5.2 | 2026-02-05 | Grounding Logic Hardening (POS Guard / Stoplist / Threshold) |
| SYN-EXP1 | v5.6.0 | 2026-02-06 | Synapse Expansion Phase 1: SynapseStore + Overlay patch system |
| SYN-EXP2 | v5.6.0 | 2026-02-06 | Synapse Expansion Phase 2: SynapseEdgeProposer + 4-Pack Rewrite |
| SYN-EXP3 | v5.6.0 | 2026-02-06 | Synapse Expansion Phase 3: CLI + Audit Gate (propose/evaluate) |
| SYN-RUN1 | v5.6.0 | 2026-02-07 | v3.1 パッチ実走: 42 edges, +5.8pt, PASS. run_relations.py に --synapse-patches 追加 |
| SYN-RUN2 | v5.6.1 | 2026-02-08 | v3.2 パッチ実走: 27 edges, +2.0pt, PASS. evaluate に Baseline Patch Auto-Inherit 追加 |
| LEX-SEED | v5.7.0 | 2026-02-09 | Lexicon v2 Step 1: Auto seed generation (326 atoms) |
| LEX-EXPAND | v5.7.0 | 2026-02-09 | Lexicon v2 Step 2: Batch WordNet expansion (325/326 success, 12 relations) |
| LEX-STATS | v5.7.0 | 2026-02-10 | Lexicon v2 Step 4a: Full expansion cross-stats (10-column GPT report). Sibling 汚染の統計的発見 |
| LEX-SPLIT | v5.7.0 | 2026-02-10 | Lexicon v2 Step 3: Core/Deviation 分離. APW 4.7→2.0, 汚染 66→0 |
| LEX-CONST | v5.7.0 | 2026-02-11 | Lexicon Constitution v1.0 確定. 17 proposals (3 merge, 1 subsume, 6 couple, 7 monitor) Taka 承認 |

---

## GPT Audit Amendments (監査条項一覧)

| 条項 | 内容 | 導入 |
|------|------|------|
| §1 | Run ID 衝突耐性（timestamp + PID + nanosecond hash） | v5.6.0 |
| §2 | Environment Metadata（8フィールド、Before/After 環境同一性保証） | v5.6.0 |
| §3 | 機械判定 FAIL 条件（CATEGORY_MISMATCH / 新規 CONSISTENT_MISGROUND） | v5.6.0 |
| §4 | patches/ への書き込み禁止（evaluate は run-dir 内のみ） | v5.6.0 |
| §5 | Baseline Patch Auto-Inherit（evaluate 時の比較世界一致保証） | v5.6.1 |
| LEX-§0 | Constitution 優先順位: Pattern A > D > B/C | v5.7.0 |
| LEX-§1 | pair_jaccard は双方向 max(A→B, B→A) を採用 | v5.7.0 |
| LEX-§2 | Pattern A (Merge): alias 保持 + 多核 Core 化 + Deviation 合流 | v5.7.0 |
| LEX-§3 | Pattern D (Subsume): 機械的判定ヒント (size_ratio < 0.85 or child_low_unique) | v5.7.0 |
| LEX-§4 | 全 proposal は auto_status: flagged。Taka 承認必須 | v5.7.0 |

---

## Historical Note

Phase numbering begins at 7 due to the iterative nature of early development. Foundation Layer components (Glossary, Synapse) were developed before the current phase system was established. This numbering is preserved for file compatibility.

Phase 9 W-layer numbering changed at v1.7: the original W0-W6 names (ContentGateway through Structural Observation) described a single-pipeline architecture. The v2.0 names (Feature Extraction through Export) reflect the Lens-integrated multi-pipeline implementation. Both sets of names may appear in older documents and code comments.

---

*End of Glossary*  
*Philosophy: Aruism — "Describe, but do not decide"*
