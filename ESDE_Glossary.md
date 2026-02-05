# ESDE Glossary

**Version**: 5.5.2  
**Updated**: 2026-02-05  
**Spec**: Existence Symmetry Dynamic Equilibrium  
**Status**: Observation C 完了時点

---

## Change Log

| Version | Date | Changes |
|---------|------|---------|
| 5.4.8-MIG.2 | 2026-01-25 | Migration Phase 2, Substrate Layer |
| 5.5.0 | 2026-02-02 | Phase 9 完了。W層再定義、Lens/Threshold/Edge Policy/Mutual-kNN 追加。旧W0-W6定義を廃止し実装準拠に更新。File Locations・Phase History・Key Metrics を全面改訂 |
| 5.5.2 | 2026-02-05 | Observation C: Relation Pipeline 用語追加。Synapse 動詞接地限界の発見を記録 |

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
The bridge between natural language and semantic atoms. Maps WordNet synsets to ESDE atoms with trigger words. v3.0: 11,557 synsets, 22,285 edges.

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

---

## Layer Architecture

### Foundation Layer
Contains Glossary (326 atoms) and Synapse (v3.0). Provides the semantic grounding for all other layers.

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

1. **CATEGORY_MISMATCH**: 動詞が名詞カテゴリ (PRP/NAT/MAT/SPA) の Atom に接地される（例: include→PRP.dirty）
2. **CONSISTENT_MISGROUND**: 多義語の間違った語義が一貫して選ばれる（例: have→EMO.like）
3. **SYNAPSE_COVERAGE_GAP**: 頻出動詞が Synapse のどの Atom にも到達しない（例: kill, host, marry）

v0.2.0 の3フィルタ（Light Verb / POS Guard / Score Threshold）により CATEGORY_MISMATCH は完全解消。残った Coverage Gap 動詞（write, host, defeat, serve, join 等）が真の辞書拡張候補として浮上した。

### ドメイン別接地特性

| ドメイン | 典型的な Coverage Gap | 傾向 |
|----------|---------------------|------|
| 武将 (mil) | kill, defeat, invade, conquer | 軍事動詞の不足 |
| 学者 (sch) | write, publish, propose | 知的動詞は比較的良好 |
| 都市 (city) | host, serve, contain, occupy | 都市記事は軽動詞比率が高い（19-35%） |

---

## File Locations

### Foundation Layer
| Component | Path |
|-----------|------|
| Glossary Data | esde_dictionary.json |
| Synapse Data | esde_synapses_v3.json |

### Substrate Layer
| Component | Path |
|-----------|------|
| Context Registry | data/substrate/context_registry.jsonl |
| Schema | esde/substrate/schema.py |
| ID Generator | esde/substrate/id_generator.py |

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

---

## Synapse Version History

| Version | Date | Synsets | Edges | Notes |
|---------|------|---------|-------|-------|
| v2.1 | 2025-12-22 | 2,037 | 2,116 | Concept name search only |
| v3.0 | 2026-01-19 | 11,557 | 22,285 | triggers_en support, 100% concept coverage |

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

---

## Historical Note

Phase numbering begins at 7 due to the iterative nature of early development. Foundation Layer components (Glossary, Synapse) were developed before the current phase system was established. This numbering is preserved for file compatibility.

Phase 9 W-layer numbering changed at v1.7: the original W0-W6 names (ContentGateway through Structural Observation) described a single-pipeline architecture. The v2.0 names (Feature Extraction through Export) reflect the Lens-integrated multi-pipeline implementation. Both sets of names may appear in older documents and code comments.

---

*End of Glossary*  
*Philosophy: Aruism — "Describe, but do not decide"*
