# ESDE Glossary

Quick Reference for AI Systems  
v5.4.6-P9.6 (Synapse v3.0) | Read this first before any ESDE task

---

## Core Concepts

| Term | Definition | Context |
|------|------------|---------|
| ESDE | Existence Symmetry Dynamic Equilibrium - AI introspection engine | System name |
| Aruism | Philosophical foundation emphasizing existence symmetry and ternary emergence | Philosophy |
| 326 Atoms | Immutable semantic primitives (163 symmetric pairs) forming the meaning foundation | Foundation |
| Glossary | The dictionary of 326 atoms with definitions | Data: glossary_results.json |
| Synapse | Connection map from WordNet synsets to atoms (11,557 synsets, 22,285 edges) | Data: esde_synapses_v3.json |
| Molecule | Combination of atoms representing a semantic observation | Phase 8 output |

---

## Phase 7 Terms (Unknown Resolution)

| Term | Definition | Formula/Values |
|------|------------|----------------|
| Volatility | Uncertainty metric for route classification (higher = more uncertain) | V = 1 - (max - 2nd_max) |
| Route A | Typo hypothesis - token is misspelling of known word | Edit distance ≤ 2 |
| Route B | Entity hypothesis - proper noun or title requiring external lookup | Wikipedia, etc. |
| Route C | Novel hypothesis - new word requiring Molecule generation | Queue for Phase 8 |
| Route D | Noise hypothesis - stopword or random characters | Discard |
| Candidate | Low volatility status - proceed with classification | V < 0.3 |
| Deferred | Medium volatility - wait for more observations | 0.3 ≤ V ≤ 0.6 |
| Quarantine | High volatility - requires human review | V > 0.6 |
| Variance Gate | Mechanism that triggers abstain when hypotheses compete | margin < 0.2 OR entropy > 0.9 |

---

## Phase 8 Terms (Introspection)

| Term | Definition | Formula/Values |
|------|------------|----------------|
| Rigidity | How fixed a concept's processing pattern has become | R = N_mode / N_total |
| Crystallization | Pathological state where R approaches 1.0 | Alert: R ≥ 0.98, N ≥ 10 |
| Sensor | Component that extracts concept candidates from text | Input stage |
| Generator | Component that produces Molecules via LLM | LLM interface |
| Ledger (L1) | Immutable append-only history with Hash Chain | semantic_ledger.jsonl |
| Index (L2) | Reconstructable projection for Rigidity calculation | In-memory |
| Modulator | Component that selects strategy based on Rigidity | Decision stage |
| NEUTRAL | Normal generation mode | 0.3 ≤ R ≤ 0.9, temp=0.1 |
| DISRUPTIVE | Pattern-breaking mode for rigid concepts | R > 0.9, temp=0.7 |
| STABILIZING | Consistency-building mode for scattered concepts | R < 0.3, temp=0.0 |

---

## Phase 9 Terms (Weak Axis Statistics)

### W-Layer Architecture

| Term | Definition | Context |
|------|------------|---------|
| W0 (ContentGateway) | Integration layer that normalizes external data into ArticleRecords | Input normalization |
| W1 (Global Statistics) | Condition-blind token statistics across all documents | W1Aggregator |
| W2 (Conditional Statistics) | Token statistics sliced by condition factors | W2Aggregator |
| W3 (Axis Candidates) | Per-token specificity analysis using S-Score | W3Calculator |
| W4 (Structural Projection) | Per-article resonance vectors computed from W3 S-Scores | W4Projector |
| W5 (Weak Structural Condensation) | Resonance-based clustering of W4 vectors into islands | W5Condensator |
| W6 (Weak Structural Observation) | Evidence extraction and topology from W5 structures | W6Analyzer |

### W0-W3 Data Structures

| Term | Definition | Key Fields |
|------|------------|------------|
| ArticleRecord | Normalized input unit from ContentGateway | source_id, raw_text, source_meta |
| W1Record | Global statistics for a single token_norm | total_count, doc_count |
| W2Record | Conditional statistics for token × condition | count, condition_signature |
| W3Record | Axis candidate analysis result | positive_candidates, negative_candidates |
| CandidateToken | Token with specificity score | token_norm, s_score, p_cond, p_global |
| ConditionEntry | Registry mapping signature to condition factors | signature, factors, total_token_count |

### W4 Data Structures (Phase 9-4)

| Term | Definition | Key Fields |
|------|------------|------------|
| W4Record | Resonance vector for a single article | article_id, resonance_vector, used_w3 |
| resonance_vector | Per-condition specificity scores | Dict[condition_signature, float] |
| w4_analysis_id | Deterministic ID for reproducibility | SHA256(article_id + w3_ids + versions) |
| used_w3 | Traceability mapping | Dict[condition_signature, w3_analysis_id] |
| token_count | Total valid tokens in article (for length bias awareness) | Integer ≥ 0 |

### W5 Data Structures (Phase 9-5)

| Term | Definition | Key Fields |
|------|------------|------------|
| W5Structure | Snapshot of structural condensation | structure_id, islands, noise_ids |
| W5Island | Condensation unit (connected component) | island_id, member_ids, representative_vector |

### W6 Data Structures (Phase 9-6)

| Term | Definition | Key Fields |
|------|------------|------------|
| W6Observatory | Complete observation of a W5 structure | observation_id, islands, topology_pairs |
| W6IslandDetail | Detailed observation of a single island | evidence_tokens, representative_articles |
| W6EvidenceToken | Token identified as evidence for an island | token_norm, evidence_score, source_w3_ids |
| W6TopologyPair | Distance between two islands | pair_id, island_a_id, island_b_id, distance |
| W6RepresentativeArticle | Sample article from island with snippet | article_id, resonance_magnitude, snippet |
| island_id | Deterministic ID from member IDs | SHA256({"members": sorted_member_ids}) |
| structure_id | Deterministic ID from inputs and parameters | SHA256(w4_analysis_ids + params) |
| representative_vector | Raw mean of member resonance vectors (rounded) | Dict[condition_signature, float] |
| cohesion_score | Edge average similarity within island | Float [0.0, 1.0] |
| noise_ids | Articles that didn't form islands | List[article_id] |

### Condition Factors

| Factor | Definition | Values |
|--------|------------|--------|
| source_type | Classification of content origin | news, dialog, paper, social, unknown |
| language_profile | Detected language of content | en, ja, mixed, unknown |
| time_bucket | Temporal grouping (monthly) | YYYY-MM format |
| Condition Signature | SHA256 hash of sorted condition factors | 32-char hex string |

### S-Score (Specificity Score)

| Term | Definition | Formula |
|------|------------|---------|
| S-Score | Per-token KL contribution measuring condition specificity | S = P(t\|C) × log((P(t\|C) + ε) / (P(t\|G) + ε)) |
| P(t\|C) | Probability of token under condition | count_cond / total_cond |
| P(t\|G) | Probability of token globally | count_global / total_global |
| ε (epsilon) | Fixed smoothing constant to prevent log(0) | 1e-12 |
| Positive Candidate | Token MORE specific to condition (S > 0) | Over-represented |
| Negative Candidate | Token LESS specific to condition (S < 0) | Suppressed |

### Resonance Score (Phase 9-4)

| Term | Definition | Formula |
|------|------------|---------|
| Resonance | Per-article dot product measuring condition affinity | R(A,C) = Σ count(t,A) × S(t,C) |
| count(t,A) | Token count in article A | Integer ≥ 0 |
| S(t,C) | S-Score from W3 for token t under condition C | Float (positive or negative) |
| Positive Resonance | Article aligns with condition (R > 0) | Over-represented tokens dominate |
| Negative Resonance | Article diverges from condition (R < 0) | Suppressed tokens dominate |
| projection_norm | Normalization method for resonance | "raw" (v9.4 = no normalization) |

### W5 Condensation (Phase 9-5)

| Term | Definition | Formula/Values |
|------|------------|----------------|
| Similarity | L2-normalized Cosine similarity between W4 vectors | cos(v1_norm, v2_norm), rounded to 12 decimals |
| Threshold | Minimum similarity for edge creation | Default: 0.70 |
| min_island_size | Minimum members for valid island | Default: 3 |
| Noise | Articles with < min_island_size connections | Excluded from islands |
| Cohesion | Average similarity of edges that formed the island | Σ edge_sim / edge_count |

### Key Thresholds (Phase 9)

| Parameter | Value | Purpose |
|-----------|-------|---------|
| EPSILON | 1e-12 | S-Score smoothing constant |
| DEFAULT_MIN_COUNT_FOR_W3 | 2 | Minimum count for W3 inclusion |
| DEFAULT_TOP_K | 100 | Number of candidates to extract |
| W4_PROJECTION_NORM | "raw" | v9.4: No length normalization applied |
| W4_ALGORITHM | "DotProduct-v1" | Resonance calculation method |
| W5_DEFAULT_THRESHOLD | 0.70 | Similarity threshold for W5 edges |
| W5_DEFAULT_MIN_ISLAND_SIZE | 3 | Minimum island size |
| W5_MAX_BATCH_SIZE | 2000 | P1-4: Prevent O(N²) explosion |
| W5_VECTOR_ROUNDING | 9 | Decimal places for vector values |
| W5_SIMILARITY_ROUNDING | 12 | Decimal places for similarity (P0-B) |

### W6 Observation (Phase 9-6)

| Parameter | Value | Purpose |
|-----------|-------|---------|
| W6_EVIDENCE_POLICY | mean_s_score_v1 | Evidence formula (P0-X1) |
| W6_SNIPPET_POLICY | head_chars_v1 | First 200 chars for snippets |
| W6_METRIC_POLICY | cosine_dist_v1 | Topology distance metric |
| W6_DIGEST_POLICY | abs_val_desc_v1 | Vector digest sorting |
| W6_EVIDENCE_K | 20 | Top-K evidence tokens per island |
| W6_TOPOLOGY_K | 10 | Top-K topology pairs to export |
| W6_DISTANCE_ROUNDING | 12 | Decimal places for distance |
| W6_EVIDENCE_ROUNDING | 8 | Decimal places for evidence score |

---

## Invariants (INV)

Design constraints that must never be violated.

### Phase 9 Invariants

| INV Code | Name | Description |
|----------|------|-------------|
| INV-W0-001 | Immutable Record | ArticleRecord is never modified after creation |
| INV-W1-001 | No Per-Token Surface Forms | W1 does not store individual surface forms |
| INV-W2-001 | Read-Only Condition | W2 reads condition factors, never writes/infers them |
| INV-W2-002 | Time Bucket Fixed | time_bucket is YYYY-MM (monthly) in v9.x |
| INV-W2-003 | Denominator Ownership | ConditionEntry denominators written by W2 only |
| INV-W3-001 | No Labeling | W3 output is factual only, no axis labels |
| INV-W3-002 | Immutable Input | W3 never modifies W1/W2 data |
| INV-W3-003 | Deterministic | Same input produces identical output (tie-break: token_norm asc) |
| INV-W4-001 | No Labeling | Output keys are condition_signature only, no natural language labels |
| INV-W4-002 | Deterministic | Same article + same W3 set produces identical scores |
| INV-W4-003 | Recomputable | W4 can always be regenerated from W0 + W3 |
| INV-W4-004 | Full S-Score Usage | W4 uses both positive AND negative candidates from W3 |
| INV-W4-005 | Immutable Input | W4Projector does NOT modify ArticleRecord or W3Record |
| INV-W4-006 | Tokenization Canon | W4 MUST use W1Tokenizer + normalize_token (no independent implementation) |
| INV-W5-001 | No Naming | Output must not contain any natural language labels |
| INV-W5-002 | Topological Identity | island_id is generated from member IDs only, no floating-point |
| INV-W5-003 | Fixed Metric | Similarity = L2-normalized Cosine, threshold operator is >= (fixed) |
| INV-W5-004 | Parameter Traceability | structure_id includes input w4_analysis_ids and observation parameters |
| INV-W5-005 | Canonical Vector | Vector values must be rounded before storage |
| INV-W5-006 | ID Collision Safety | ID generation uses Canonical JSON serialization. String join forbidden |
| INV-W5-007 | Structure Identity | structure_id uses w4_analysis_id (not article_id) as input identity |
| INV-W5-008 | Canonical Output | created_at and other non-deterministic fields excluded from identity check |
| INV-W6-001 | No Synthetic Labels | No natural language categories, no LLM summaries |
| INV-W6-002 | Deterministic Export | Same input produces bit-identical output |
| INV-W6-003 | Read Only | W1-W5 data is never modified |
| INV-W6-004 | No New Math | No new statistical models, only extraction/transformation |
| INV-W6-005 | Evidence Provenance | All evidence tokens traceable to W3 |
| INV-W6-006 | Stable Ordering | All list outputs have complete tie-break rules |
| INV-W6-007 | No Hypothesis | No "Axis Hypothesis", "Confidence" or judgment logic |
| INV-W6-008 | Strict Versioning | Tokenizer, Normalizer, W3 version compatibility tracked |
| INV-W6-009 | Scope Closure | W5 member set must match W4/ArticleRecord input sets exactly |

---

## Direction Symbols

| Symbol | Name | Meaning | Effect |
|--------|------|---------|--------|
| =>+ | Creative Emergence | New structure forming | Connectivity increases |
| -\|> | Destructive Emergence / Reboot | Structure dissolving | Connectivity decreases |
| => | Neutral | Observation only | No structural change |

---

## Aruism Terms

| Term | Definition |
|------|------------|
| Aru (ある) | Primordial fact: "There is" - foundation of all existence |
| Ternary Emergence | Manifestation requires three linked terms (A ↔ B ↔ C) |
| Dual Symmetry | World_A (physics-fixed) and World_B (mind-fixed) as dual faces |
| Dynamic Equilibrium | ε × L ≈ K_sys - flexibility and connectivity must balance |
| Weak Meaning System | Phase 7/9 - concepts seeking stability through statistics |
| Strong Meaning System | Phase 8 - stable concepts forming observation framework |
| Axis Reboot | Discontinuous transition that breaks rigid patterns (-\|>) |

---

## File Naming Conventions

| Pattern | Meaning | Example |
|---------|---------|---------|
| *_v5.4.x.py | Version-tagged implementation | esde-engine-v545.py |
| *_7bplus* | Phase 7B+ (multi-hypothesis routing) | evidence_ledger_7bplus.jsonl |
| *_live.py | Production version (uses real LLM) | esde_cli_live.py |
| *_mock.py | Test version (no LLM required) | Uses MockPipeline |
| 旧/ or legacy/ | Archived files - do not use | legacy_20251222/ |

### Phase 9 File Structure

| Path | Purpose |
|------|---------|
| esde/integration/ | W0 ContentGateway |
| esde/statistics/ | W1, W2, W3, W4, W5 modules |
| esde/discovery/ | W6 observation and export modules |
| data/stats/w1_global.json | W1 statistics storage |
| data/stats/w2_records.jsonl | W2 record storage |
| data/stats/w2_conditions.jsonl | Condition registry |
| data/stats/w3_candidates/ | W3 axis candidate outputs |
| data/stats/w4_projections/ | W4 per-article resonance vectors |
| data/stats/w5_structures/ | W5 structural condensation outputs |
| data/discovery/w6_observations/ | W6 observation exports (JSON/MD/CSV) |

---

## Key Thresholds (config.py)

| Parameter | Value | Purpose |
|-----------|-------|---------|
| COMPETE_TH | 0.15 | Minimum score for competing hypothesis |
| VOL_LOW_TH | 0.25 | Below = candidate status |
| VOL_HIGH_TH | 0.50 | Above = quarantine status |
| UNKNOWN_MARGIN_TH | 0.20 | Variance Gate margin threshold |
| UNKNOWN_ENTROPY_TH | 0.90 | Variance Gate entropy threshold |
| TYPO_MAX_EDIT_DISTANCE | 2 | Maximum edit distance for typo detection |

---

## Historical Note

Phase numbering begins at 7 due to the iterative nature of early development. Foundation Layer components (Glossary, Synapse) were developed before the current phase system was established. This numbering is preserved for file compatibility.

Phase 9 introduces the "Weak Axis Statistics" layer (W0-W6), providing statistical foundation for axis discovery without human labeling.

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
| 9-0 | v5.4.2 | 2026-01 | W0 ContentGateway |
| 9-1 | v5.4.2 | 2026-01 | W1 Global Statistics |
| 9-2 | v5.4.2 | 2026-01 | W2 Conditional Statistics |
| 9-3 | v5.4.2 | 2026-01 | W3 Axis Candidates (S-Score) |
| 9-4 | v5.4.4 | 2026-01 | W4 Structural Projection (Resonance) |
| 9-5 | v5.4.5 | 2026-01 | W5 Weak Structural Condensation (Islands) |
| 9-6 | v5.4.6 | 2026-01 | W6 Weak Structural Observation (Evidence) |

---

*End of Glossary*
