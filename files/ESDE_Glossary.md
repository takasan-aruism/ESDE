# ESDE Glossary

Quick Reference for AI Systems  
v5.4.7-SUB.1 (Synapse v3.0) | Read this first before any ESDE task

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

## Substrate Layer (Layer 0)

The Substrate Layer (Context Fabric) is the lowest layer of ESDE, providing observation trace storage beneath all phases.

**Philosophy:** "Describe, but do not decide." (記述せよ、しかし決定するな)

| Term | Definition | Context |
|------|------------|---------|
| Substrate | Cross-cutting observation layer (Layer 0) beneath all phases | Architecture |
| Context Fabric | Alternative name for Substrate Layer | Architecture |
| ContextRecord | Immutable record of observation traces (frozen dataclass) | Schema |
| Traces | Schema-less observation values (namespace:name format) | Data format |
| context_id | Deterministic SHA256 hash of (retrieval_path, traces, capture_version) | Identity |
| capture_version | Version of trace extraction logic | Versioning |
| SubstrateRegistry | Append-only JSONL storage for ContextRecords | Storage |

### Substrate Namespaces

| Namespace | Purpose | Examples |
|-----------|---------|----------|
| `html:` | HTML structure | `html:tag_count`, `html:has_h1` |
| `text:` | Text statistics | `text:char_count`, `text:word_count` |
| `meta:` | Retrieval metadata | `meta:domain`, `meta:content_type` |
| `time:` | Temporal information | `time:year`, `time:month` |
| `struct:` | Structural patterns | `struct:reply_depth`, `struct:has_quoted_text` |
| `env:` | Environment info | `env:user_agent`, `env:platform` |
| `legacy:` | Migration only | `legacy:source_type`, `legacy:language_profile` |

### Forbidden Namespaces (INV-SUB-002)

These namespaces are permanently banned as they imply semantic interpretation:

| Namespace | Reason |
|-----------|--------|
| `meaning:` | Semantic interpretation |
| `category:` | Classification |
| `intent:` | Intent inference |
| `quality:` | Quality judgment |
| `importance:` | Importance ranking |
| `sentiment:` | Sentiment analysis |
| `topic:` | Topic classification |
| `type:` | Type classification |

---

## Phase 7 Terms (Unknown Resolution)

| Term | Definition | Formula/Values |
|------|------------|----------------|
| Volatility | Uncertainty metric for route classification (higher = more uncertain) | V = 1 - (max - 2nd_max) |
| Route A | Typo hypothesis - token is misspelling of known word | Edit distance ≤ 2 |
| Route B | Entity hypothesis - proper noun or title requiring external lookup | Wikipedia, etc. |
| Route C | Novel hypothesis - new concept needing molecular decomposition | WordNet exploration |
| Route D | Functional hypothesis - particles, conjunctions, filters | Stopword list |

---

## Phase 8 Terms (Introspection)

| Term | Definition |
|------|------------|
| Atom | Single semantic unit from the 326 Glossary entries |
| Molecule | Combination of atoms with operator connections |
| Formula | Text representation of molecule structure (e.g., "aa_1 × aa_2") |
| Operator | Connection type between atoms (×, ▷, ○, etc.) |
| Rigidity | Measurement of observation pattern stability |
| Canonical Schema | v8.3 flat structure (axis/level at top level, no nesting) |

---

## Phase 9 Terms (Weak Axis Statistics)

| Term | Definition |
|------|------------|
| W0 | ContentGateway - entry point for observation |
| W1 | Global Statistics - cross-sectional token counts |
| W2 | Conditional Statistics - per-condition token counts |
| W3 | Axis Candidates - S-Score based token selection |
| W4 | Structural Projection - article resonance vectors |
| W5 | Structural Condensation - island clustering |
| W6 | Structural Observation - evidence extraction |
| S-Score | Specificity score = KL divergence (P_cond vs P_global) |
| Condition Signature | Hash of condition factors (source_type + time_bucket) |
| Resonance Vector | Per-article vector of S-Scores across conditions |
| Island | Cluster of similar articles in resonance space |

---

## Invariants (INV Codes)

### Substrate Layer Invariants

| INV Code | Name | Description |
|----------|------|-------------|
| INV-SUB-001 | Upper Read-Only | Upper layers can only read Substrate (no update/delete, append-only) |
| INV-SUB-002 | No Semantic Transform | Record raw observation values only, no interpretation |
| INV-SUB-003 | Machine-Observable | Only machine-computable values allowed (no human judgment) |
| INV-SUB-004 | No Inference | No ML inference, no probabilistic judgment |
| INV-SUB-005 | Append-Only | Records can only be added, never updated or deleted |
| INV-SUB-006 | ID Determinism | context_id = SHA256(canonical(path, traces, version)) |
| INV-SUB-007 | Canonical Export | Output order and format must be deterministic |

### Phase 8 Invariants

| INV Code | Name | Description |
|----------|------|-------------|
| INV-MOL-001 | Flat Schema | CanonicalAtom uses flat structure (axis/level at top, no coordinates nesting) |

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
| INV-W4-001 | No Labeling | Output keys are condition_signature only |
| INV-W4-002 | Deterministic | Same article + W3 produces identical scores |
| INV-W4-003 | Recomputable | W4 can be regenerated from W0 + W3 |
| INV-W4-004 | Full S-Score Usage | Both positive AND negative candidates used |
| INV-W4-005 | Immutable Input | W4 does NOT modify ArticleRecord or W3 |
| INV-W4-006 | Tokenization Canon | W4 MUST use W1Tokenizer + normalize_token |
| INV-W5-001 | No Naming | No natural language labels in output |
| INV-W5-002 | Topological Identity | island_id from member IDs only |
| INV-W5-003 | Fixed Metric | Similarity = L2-normalized Cosine, >= operator |
| INV-W5-004 | Parameter Traceability | structure_id includes params |
| INV-W5-005 | Canonical Vector | Values rounded before storage |
| INV-W5-006 | ID Collision Safety | Canonical JSON hash, no string join |
| INV-W5-007 | Structure Identity | Uses w4_analysis_id, not article_id |
| INV-W5-008 | Canonical Output | created_at excluded from identity |
| INV-W6-001 | No Synthetic Labels | No LLM summaries |
| INV-W6-002 | Deterministic Export | Bit-identical output |
| INV-W6-003 | Read Only | W1-W5 never modified |
| INV-W6-004 | No New Math | Only extraction/transformation |
| INV-W6-005 | Evidence Provenance | Tokens traceable to W3 |
| INV-W6-006 | Stable Ordering | Complete tie-break rules |
| INV-W6-007 | No Hypothesis | No judgment logic |
| INV-W6-008 | Strict Versioning | Version compatibility tracked |
| INV-W6-009 | Scope Closure | W5 members match input sets |

---

## File Structure

### Substrate Layer Files

| Path | Purpose |
|------|---------|
| esde/substrate/ | Layer 0 Package |
| esde/substrate/schema.py | ContextRecord (frozen dataclass) |
| esde/substrate/registry.py | SubstrateRegistry (JSONL storage) |
| esde/substrate/id_generator.py | Deterministic ID generation |
| esde/substrate/traces.py | Trace validation and normalization |
| esde/substrate/NAMESPACES.md | Namespace definitions |
| data/substrate/context_registry.jsonl | Default registry storage |

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

## Substrate Constants

| Parameter | Value | Purpose |
|-----------|-------|---------|
| FLOAT_PRECISION | 9 | Decimal places for float normalization |
| CONTEXT_ID_LENGTH | 32 | Hex characters in context_id |
| STRING_MAX_LENGTH | 4096 | Max characters per trace string value |
| INT_MIN | -2³¹ | Minimum integer value |
| INT_MAX | 2³¹-1 | Maximum integer value |
| FILE_ENCODING | "utf-8" | Canonical file encoding |
| FILE_NEWLINE | "\n" | Canonical newline (LF only) |

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
| *_v5.4.x.py | Version-tagged implementation | esde-engine-v547.py |
| *_7bplus* | Phase 7B+ (multi-hypothesis routing) | evidence_ledger_7bplus.jsonl |
| *_live.py | Production version (uses real LLM) | esde_cli_live.py |
| *_mock.py | Test version (no LLM required) | Uses MockPipeline |
| 旧/ or legacy/ | Archived files - do not use | legacy_20251222/ |

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
| **SUB** | **v0.1.0** | **2026-01** | **Substrate Layer (Context Fabric)** |

---

## Historical Note

Phase numbering begins at 7 due to the iterative nature of early development. Foundation Layer components (Glossary, Synapse) were developed before the current phase system was established. This numbering is preserved for file compatibility.

Phase 9 introduces the "Weak Axis Statistics" layer (W0-W6), providing statistical foundation for axis discovery without human labeling.

**Substrate Layer** is a cross-cutting layer (Layer 0) that sits beneath all phases, providing machine-observable trace storage without semantic interpretation. It follows the philosophy "Describe, but do not decide."

---

*End of Glossary*
