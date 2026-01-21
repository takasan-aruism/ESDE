# ESDE Glossary

Quick Reference for AI Systems  
v5.4.4-P9.4 (Synapse v3.0) | Read this first before any ESDE task

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
| W5 (Axis Confirmation) | Human review and axis labeling (future) | Planned |

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

### Key Thresholds (Phase 9)

| Parameter | Value | Purpose |
|-----------|-------|---------|
| EPSILON | 1e-12 | S-Score smoothing constant |
| DEFAULT_MIN_COUNT_FOR_W3 | 2 | Minimum count for W3 inclusion |
| DEFAULT_TOP_K | 100 | Number of candidates to extract |
| W4_PROJECTION_NORM | "raw" | v9.4: No length normalization applied |
| W4_ALGORITHM | "DotProduct-v1" | Resonance calculation method |

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
| *_v5.4.x.py | Version-tagged implementation | esde-engine-v544.py |
| *_7bplus* | Phase 7B+ (multi-hypothesis routing) | evidence_ledger_7bplus.jsonl |
| *_live.py | Production version (uses real LLM) | esde_cli_live.py |
| *_mock.py | Test version (no LLM required) | Uses MockPipeline |
| 旧/ or legacy/ | Archived files - do not use | legacy_20251222/ |

### Phase 9 File Structure

| Path | Purpose |
|------|---------|
| esde/integration/ | W0 ContentGateway |
| esde/statistics/ | W1, W2, W3, W4 modules |
| data/stats/w1_global.json | W1 statistics storage |
| data/stats/w2_records.jsonl | W2 record storage |
| data/stats/w2_conditions.jsonl | Condition registry |
| data/stats/w3_candidates/ | W3 axis candidate outputs |
| data/stats/w4_projections/ | W4 per-article resonance vectors |

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

Phase 9 introduces the "Weak Axis Statistics" layer (W0-W5), providing statistical foundation for axis discovery without human labeling.

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

---

*End of Glossary*
