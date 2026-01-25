# ESDE

## Existence Symmetry Dynamic Equilibrium

*The Introspection Engine for AI*

### Detailed Design Document

For Design Review and AI Context Transfer

**Version 5.4.8-MIG.2**

**2026-01-25**

*Based on Aruism Philosophy*

---

## Table of Contents

1. Executive Summary
2. Foundation Layer: Semantic Dictionary
3. **Substrate Layer: Context Fabric (Layer 0)**
4. **Migration Phase 2: Policy-Based Statistics**
5. Phase 7: Unknown Resolution (Weak Meaning System)
6. Phase 8: Introspective Engine (Strong Meaning System)
7. Phase 9: Weak Axis Statistics Layer (W0-W6) **[COMPLETE]**
8. System Architecture: Dual Symmetry
9. Philosophical Foundation: Aruism
10. Mathematical Formalization
11. Future Roadmap
12. Appendices

---

## 1. Executive Summary

### 1.1 What is ESDE?

ESDE (Existence Symmetry Dynamic Equilibrium) is an introspection engine that enables AI systems to observe their own cognitive patterns and autonomously adjust behavior when necessary.

Current Large Language Models (LLMs) lack three critical capabilities:

- **Self-awareness**: Cannot recognize their own response patterns
- **Rigidity detection**: Cannot identify when they are repeating the same answers
- **Autonomous adjustment**: Cannot modify behavior based on self-observation

ESDE addresses these limitations by providing an external observation structure that gives AI a form of pseudo-metacognition.

### 1.2 Core Architecture

ESDE operates through integrated layers:

| Layer | Function | Analogy |
|-------|----------|---------|
| **Substrate (Layer 0)** | Machine-observable trace storage | OS Kernel / File System |
| Foundation | 326 semantic atoms + connection dictionary | AI vocabulary and dictionary |
| Phase 7 | Unknown word detection, classification, resolution | Learning new words |
| Phase 8 | Thought pattern monitoring and adjustment | Self-reflection capability |
| Phase 9 | Statistical foundation for axis discovery (W0-W6) | Pattern emergence detection |
| **Migration** | Bridge between layers (Substrate ↔ Statistics) | Integration infrastructure |

### 1.3 Key Design Principles

ESDE is built on the following principles derived from Aruism philosophy:

- **Observation over judgment**: Detect uncertainty rather than assert confidence
- **Thresholds for computation, not truth**: Parameters control processing flow, not determine correctness
- **Ternary emergence**: Meaning manifests through three-term relationships, not binary oppositions
- **Symmetric duality**: Phase 7 (weak meaning) and Phase 8 (strong meaning) form a symmetric pair
- **Describe, but do not decide**: Substrate Layer records facts without interpretation

---

## 2. Foundation Layer: Semantic Dictionary

### 2.1 Glossary: 326 Semantic Atoms

The foundation of ESDE is a set of 326 semantic atoms representing the minimal units of human conceptual space. These atoms are organized as 163 symmetric pairs.

#### 2.1.1 Design Principle: Existential Symmetry

Following Aruism's principle of existential symmetry, each concept exists in relation to its symmetric counterpart. Love cannot be defined without hate; life cannot be understood without death.

---

## 3. Substrate Layer: Context Fabric (Layer 0)

### 3.1 Philosophy

The Substrate Layer is the foundational data layer for ESDE, providing machine-observable trace storage without semantic interpretation. It follows the core principle:

**"Describe, but do not decide."** (記述せよ、しかし決定するな)

### 3.2 Purpose

Substrate Layer stores raw observations that can be used by upper layers (W2, W3, etc.) for statistical analysis. It does NOT interpret meaning, classify content, or make judgments.

### 3.3 Architecture Position

```
┌─────────────────────────────────────────────┐
│  Phase 9: W0-W6 Statistics (Upper Layers)   │
│  ↓ reads from ↓                              │
├─────────────────────────────────────────────┤
│  Migration Phase 2: Policy Layer (Bridge)    │
│  ↓ connects ↓                                │
├─────────────────────────────────────────────┤
│  Substrate Layer (Layer 0)                   │
│  - ContextRecord                             │
│  - SubstrateRegistry                         │
│  - Trace Storage                             │
└─────────────────────────────────────────────┘
```

### 3.4 ContextRecord

The fundamental data unit of Substrate Layer:

```python
@dataclass(frozen=True)
class ContextRecord:
    # === Identity (Canonical) ===
    context_id: str                 # SHA256[:32] of canonical content
    retrieval_path: Optional[str]   # Source URL or path
    capture_version: str            # Trace extraction logic version
    traces: Dict[str, Any]          # Schema-less observations
    
    # === Metadata (Non-Canonical) ===
    observed_at: str                # ISO8601
    created_at: str                 # ISO8601
```

### 3.5 Invariants

| ID | Name | Definition |
|----|------|------------|
| INV-SUB-001 | Upper Read-Only | Upper layers cannot update/delete; append-only |
| INV-SUB-002 | No Semantic Transform | Raw observation values only |
| INV-SUB-003 | Machine-Observable | No human judgment required |
| INV-SUB-004 | No Inference | No ML/probabilistic values |
| INV-SUB-005 | Append-Only | No update, no delete |
| INV-SUB-006 | ID Determinism | context_id from inputs only (no timestamp) |
| INV-SUB-007 | Canonical Export | Deterministic output order |

---

## 4. Migration Phase 2: Policy-Based Statistics

### 4.1 Overview

Migration Phase 2 (v0.2.1) introduces Policy-based condition signature generation, bridging Substrate Layer with W2 statistics. This enables consistent statistical grouping across the observation pipeline.

### 4.2 Architecture

```
ArticleRecord
    │ substrate_ref (context_id)
    ▼
SubstrateRegistry.get(substrate_ref)
    │
    ▼
ContextRecord.traces
    │ legacy:source_type, legacy:language_profile, ...
    ▼
StandardConditionPolicy.compute_signature()
    │
    ▼
64-char SHA256 condition signature
    │
    ▼
W2Aggregator statistics
```

### 4.3 Components

#### 4.3.1 BaseConditionPolicy (Abstract)

```python
class BaseConditionPolicy(ABC):
    """Abstract base for condition signature policies."""
    policy_id: str      # Unique identifier
    version: str        # Policy version
    
    @abstractmethod
    def compute_signature(self, record: ContextRecord) -> str:
        """Return full SHA256 hex (64 chars)."""
        pass
    
    @abstractmethod
    def extract_factors(self, record: ContextRecord) -> Dict[str, Any]:
        """Return factors with types preserved (no str coercion)."""
        pass
```

#### 4.3.2 StandardConditionPolicy

```python
@dataclass
class StandardConditionPolicy(BaseConditionPolicy):
    policy_id: str
    target_keys: List[str]  # e.g., ["legacy:source_type", "legacy:language_profile"]
    version: str = "v1.0"
```

**Key Features:**
- Policy ID mixed into hash (collision prevention)
- Type preservation (bool, int, float not coerced to str)
- Canonical JSON (Substrate-unified: `separators=(',',':')`)
- Missing keys tracked separately

### 4.4 P0 Requirements

| ID | Requirement | Description |
|----|-------------|-------------|
| P0-MIG-1 | Policy ID Mixing | `policy_id` included in hash payload |
| P0-MIG-2 | Type Preservation | No `str()` coercion; original types preserved |
| P0-MIG-3 | Canonical JSON | Unified with Substrate spec (no spaces) |
| P0-MIG-4 | Missing Key Handling | Explicit `missing` list, not empty factors |

### 4.5 W2Aggregator Integration

W2Aggregator now accepts optional `registry` and `policy` parameters:

```python
aggregator = W2Aggregator(
    records_path="data/stats/w2_records.jsonl",
    conditions_path="data/stats/w2_conditions.jsonl",
    registry=substrate_registry,      # Optional
    policy=condition_policy,          # Optional
)
```

**Signature Resolution:**
1. If `substrate_ref` + `registry` + `policy` → **Policy path**
2. Otherwise → **Legacy fallback** (reads `source_meta` directly)

### 4.6 Legacy Fallback

For backward compatibility:
- `_compute_legacy_hash()` uses Substrate-unified canonical JSON
- Returns full 64-char SHA256 (not truncated)
- No exceptions on missing data; graceful degradation

### 4.7 Test Coverage

| Test Suite | Tests | Status |
|------------|-------|--------|
| test_migration_phase2.py | 12 | ✅ PASS |
| test_e2e_gateway_substrate_w2.py | 4 | ✅ PASS |
| Regression tests | 3 | ✅ PASS |

---

## 5. Phase 7: Unknown Resolution

### 5.1 Purpose: The Weak Meaning System

Phase 7 handles tokens that fall outside the established semantic space. It represents the weak meaning system - concepts that have not yet acquired stable semantic grounding.

[... Phase 7 content continues as before ...]

---

## 6. Phase 8: Introspective Engine

### 6.1 Purpose: The Strong Meaning System

Phase 8 implements self-reflection by monitoring how concepts are being processed and detecting when patterns become rigid.

The 326 atoms represent the strong meaning system - stable reference points that enable observation.

[... Phase 8 content continues as before ...]

---

## 7. Phase 9: Weak Axis Statistics Layer [COMPLETE]

### 7.1 Overview

Phase 9 implements the Weak Axis Statistics Layer (W0-W6), providing statistical foundation for axis discovery without human labeling.

**Status: COMPLETE** - Phase 9 ended at W6. Next is Phase 10.

### 7.2 W-Layer Architecture

```
W0 (ContentGateway) → External data normalization
    ↓
W1 (Global Statistics) → Unconditional token statistics
    ↓
W2 (Conditional Statistics) → Condition-sliced statistics
    ↓
W3 (Axis Candidates) → S-Score based candidate extraction
    ↓
W4 (Structural Projection) → Article → W3 resonance vectors
    ↓
W5 (Structural Condensation) → Clustering into islands
    ↓
W6 (Structural Observation) → Evidence extraction, topology
```

### 7.3 Implementation Status

| Layer | Component | Status |
|-------|-----------|--------|
| W0 | ContentGateway | ✅ Complete |
| W1 | Global Statistics | ✅ Complete |
| W2 | Conditional Statistics | ✅ Complete |
| W3 | Axis Candidates (S-Score) | ✅ Complete |
| W4 | Structural Projection (Resonance) | ✅ Complete |
| W5 | Weak Structural Condensation (Islands) | ✅ Complete |
| W6 | Weak Structural Observation (Evidence) | ✅ Complete |

**Test Results**: All phases passing (W0-W6)

### 7.4 Key Invariants

| ID | Layer | Definition |
|----|-------|------------|
| INV-W0-001 | W0 | Gateway must not interpret meaning |
| INV-W2-003 | W2 | No winner declaration |
| INV-W3-001 | W3 | Output sorted by |S-Score| DESC |
| INV-W3-003 | W3 | Deterministic output |
| INV-W4-001~006 | W4 | Determinism, traceability, no labeling |
| INV-W5-001~008 | W5 | Island identity, canonical output |
| INV-W6-001~009 | W6 | No synthetic labels, deterministic export |

---

## 8. System Architecture: Dual Symmetry

### 8.1 Phase 7 ↔ Phase 8 Symmetry

| Aspect | Phase 7 (Weak Meaning) | Phase 8 (Strong Meaning) |
|--------|------------------------|--------------------------|
| Direction | External → ESDE | ESDE → ESDE |
| Target | Unknown tokens | Known concepts (326 atoms) |
| Question | What is this? | Am I biased? |
| Output | Classification hypothesis | Strategy selection |
| Uncertainty metric | Volatility | Rigidity |

### 8.2 No Cross-Flow by Design

Phase 7 discoveries do NOT automatically flow into Phase 8. This is intentional:

- The 326 atoms constitute the immutable strong meaning system
- Phase 7 handles transient weak meanings that may never stabilize
- Mixing the two would compromise the observation axis

---

## 9. Philosophical Foundation: Aruism

### 9.1 Core Principle: Aru (There Is)

Aruism begins with the primordial recognition: There is. Before any definition, categorization, or judgment, existence simply is.

*Aru wa, aru. (There is, there is.)*

### 9.2 The Seven Axioms

| Axiom | Name | Statement |
|-------|------|-----------|
| 0 | Aru | The primordial fact of existence |
| E | Identification | Existence can be identified |
| L | Linkage | All existences are linked |
| Eq | Equality | All existences equal in status |
| C | Creativity | Creativity orthogonal to gradient |
| ε | Error | Error always exists |
| T | Ternary | Manifestation requires three terms |

---

## 10. Future Roadmap

### 10.1 Phase 9: Complete

Phase 9 (W0-W6) is complete. The Weak Axis Statistics Layer provides:
- Statistical foundation for axis discovery
- Evidence extraction pipeline
- Topology calculation
- Human-reviewable observation outputs

### 10.2 Substrate + Migration: Complete

- Substrate Layer (v0.1.0): Context Fabric for trace storage
- Migration Phase 2 (v0.2.1): Policy-based statistics bridge

### 10.3 Phase 10: Next

Phase 10 scope to be determined. Potential directions:
- Multi-instance operation
- Distributed ledger consensus
- Collective emergence detection
- Web UI visualization

### 10.4 Future Phases

| Phase | Theme | Status |
|-------|-------|--------|
| 7 | Unknown Resolution | ✅ Complete |
| 8 | Introspective Engine | ✅ Complete |
| 9 | Weak Axis Statistics (W0-W6) | ✅ Complete |
| SUB | Substrate Layer | ✅ Complete |
| MIG-2 | Migration Phase 2 | ✅ Complete |
| **10** | **TBD** | **Next** |

---

## Appendix D: Module Reference

### D.4 Phase 9: Statistics Modules

| File | Class | Role |
|------|-------|------|
| statistics/schema.py | W1Record, W1GlobalStats | W1 data structures |
| statistics/schema_w2.py | W2Record, ConditionEntry | W2 data structures |
| statistics/schema_w3.py | W3Record, CandidateToken | W3 data structures |
| statistics/schema_w4.py | W4Record | W4 data structures |
| statistics/schema_w5.py | W5Island, W5Structure | W5 data structures |
| **statistics/policies/base.py** | **BaseConditionPolicy** | **Migration Phase 2** |
| **statistics/policies/standard.py** | **StandardConditionPolicy** | **Migration Phase 2** |
| statistics/tokenizer.py | HybridTokenizer | Token extraction |
| statistics/normalizer.py | normalize_token | Token normalization |
| statistics/w1_aggregator.py | W1Aggregator | Global statistics |
| statistics/w2_aggregator.py | W2Aggregator | Conditional statistics |
| statistics/w3_calculator.py | W3Calculator | S-Score calculation |
| statistics/w4_projector.py | W4Projector | Resonance vector projection |
| statistics/w5_condensator.py | W5Condensator | Structural condensation |

### D.5 Discovery Modules

| File | Class | Role |
|------|-------|------|
| discovery/schema_w6.py | W6Observatory, W6IslandDetail | W6 data structures |
| discovery/w6_analyzer.py | W6Analyzer | Evidence extraction |
| discovery/w6_exporter.py | W6Exporter | MD/CSV/JSON export |

---

*--- End of Document ---*

**ESDE v5.4.8-MIG.2** | Existence Symmetry Dynamic Equilibrium

Philosophy: Aruism
