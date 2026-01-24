# ESDE Technical Specification

**Version 5.4.7-SUB.1**

**2026-01-24**

---

## 1. Overview

This document provides the technical specification for ESDE (Existence Symmetry Dynamic Equilibrium), covering all layers from Substrate (Layer 0) through Phase 9.

---

## 2. Substrate Layer (Layer 0) - NEW

### 2.1 Purpose

The Substrate Layer provides permanent, append-only storage for machine-observable traces without semantic interpretation.

**Philosophy:** "Describe, but do not decide."

### 2.2 Package Structure

```
esde/substrate/
├── __init__.py          # Package exports
├── schema.py            # ContextRecord (frozen dataclass)
├── registry.py          # SubstrateRegistry (JSONL storage)
├── id_generator.py      # Deterministic ID generation
├── traces.py            # Trace validation and normalization
└── NAMESPACES.md        # Namespace definitions
```

### 2.3 ContextRecord Schema

```python
@dataclass(frozen=True)
class ContextRecord:
    """Immutable observation record."""
    
    # === Identity (Deterministic) ===
    context_id: str              # SHA256(canonical)[:32]
    
    # === Observation Facts (Canonical) ===
    retrieval_path: Optional[str]   # URL, file path, etc.
    capture_version: str            # Trace extraction version
    traces: Dict[str, Any]          # Schema-less observations
    
    # === Metadata (Non-Canonical) ===
    observed_at: str                # ISO8601
    created_at: str                 # ISO8601
    schema_version: str             # "v0.1.0"
```

### 2.4 Context ID Generation

```python
def compute_context_id(
    retrieval_path: Optional[str],
    traces: Dict[str, Any],
    capture_version: str,
) -> str:
    # 1. Normalize traces (sort keys, round floats to 9 decimals)
    normalized = normalize_traces(traces)
    
    # 2. Build canonical payload
    payload = {
        "capture_version": capture_version,
        "retrieval_path": retrieval_path,
        "traces": normalized,
    }
    
    # 3. Canonical JSON (sorted keys, no spaces)
    canonical_str = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    
    # 4. SHA256 hash, truncate to 32 chars
    return hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()[:32]
```

### 2.5 Trace Validation Rules

#### 2.5.1 Key Format

```
namespace:name
```

- Namespace: `[a-z][a-z0-9_]*`
- Name: `[a-z][a-z0-9_]*`
- Example: `html:tag_count`, `meta:domain`

#### 2.5.2 Forbidden Namespaces

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

#### 2.5.3 Forbidden Key Names

Interpretation words banned regardless of namespace:

```python
FORBIDDEN_KEY_NAMES = {
    "source_type", "content_type", "document_type",
    "is_short", "is_long", "is_important", "is_relevant",
    "quality_score", "importance_score", "relevance_score",
    "label", "tag", "class", "classification",
    ...
}
```

**Exceptions:**
- `legacy:source_type` - Migration only
- `meta:content_type` - MIME type only

#### 2.5.4 Value Type Constraints

| Type | Constraints |
|------|-------------|
| `str` | Max 4096 characters |
| `int` | Range: [-2³¹, 2³¹-1] |
| `float` | 9 decimal precision, no NaN/Inf |
| `bool` | True/False |
| `None` | Null |
| `list` | **FORBIDDEN** |
| `dict` | **FORBIDDEN** |

### 2.6 SubstrateRegistry API

```python
class SubstrateRegistry:
    def __init__(self, storage_path: str = "data/substrate/context_registry.jsonl"):
        ...
    
    def register(self, record: ContextRecord) -> str:
        """
        Register a ContextRecord.
        Returns context_id.
        Deduplication: same context_id = no re-write.
        """
    
    def register_traces(
        self,
        traces: Dict[str, Any],
        retrieval_path: Optional[str] = None,
        capture_version: str = "v1.0",
    ) -> str:
        """Convenience method to register traces directly."""
    
    def get(self, context_id: str) -> Optional[ContextRecord]:
        """Retrieve by context_id."""
    
    def exists(self, context_id: str) -> bool:
        """Check existence."""
    
    def count(self) -> int:
        """Total record count."""
    
    def query_by_trace_key(self, key: str, value: Any) -> List[ContextRecord]:
        """Query by trace key-value."""
    
    def query_by_namespace(self, namespace: str) -> List[ContextRecord]:
        """Query by namespace prefix."""
    
    def export_canonical(self, output_path: str) -> int:
        """Export sorted by context_id (INV-SUB-007)."""
```

### 2.7 File I/O Constants

```python
FILE_ENCODING = "utf-8"     # Canonical encoding
FILE_NEWLINE = "\n"         # Unix LF only
```

### 2.8 Substrate Invariants

| ID | Name | Definition |
|----|------|------------|
| INV-SUB-001 | Upper Read-Only | Upper layers can read only; no update/delete (append-only) |
| INV-SUB-002 | No Semantic Transform | Raw observation values only, no interpretation |
| INV-SUB-003 | Machine-Observable | Only machine-computable values (no human judgment) |
| INV-SUB-004 | No Inference | No ML inference, no probabilistic values |
| INV-SUB-005 | Append-Only | Records can only be added, never updated/deleted |
| INV-SUB-006 | ID Determinism | context_id computed from inputs only (no timestamp/random) |
| INV-SUB-007 | Canonical Export | Output order and format must be deterministic |

### 2.9 Legacy Migration

```python
def convert_source_meta_to_traces(source_meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert legacy source_meta to Substrate traces.
    Prefix: "legacy:"
    """
    traces = {}
    for key, value in source_meta.items():
        if value is None:
            continue
        trace_key = f"legacy:{key.lower()}"
        traces[trace_key] = value if isinstance(value, (str, int, float, bool)) else str(value)
    return traces
```

---

## 3. Phase 7: Unknown Resolution

### 3.1 Volatility Score

```python
V = 1 - (max_score - second_max_score)
```

### 3.2 Route Classification

| Route | Condition | Resolution |
|-------|-----------|------------|
| A | Edit distance ≤ 2 | Spell correction |
| B | Proper noun detected | External lookup |
| C | Novel concept | Molecular decomposition |
| D | Functional word | Filter |

### 3.3 Thresholds

| Parameter | Value |
|-----------|-------|
| COMPETE_TH | 0.15 |
| VOL_LOW_TH | 0.25 |
| VOL_HIGH_TH | 0.50 |
| TYPO_MAX_EDIT_DISTANCE | 2 |

---

## 4. Phase 8: Introspection

### 4.1 Canonical Schema (v8.3)

```python
@dataclass
class CanonicalAtom:
    id: str                          # "aa_1"
    atom: str                        # "EMO.love"
    axis: Optional[str] = None       # Top-level
    level: Optional[str] = None      # Top-level
    text_ref: Optional[str] = None
    span: Optional[Tuple[int, int]] = None

@dataclass
class CanonicalMolecule:
    molecule_id: str
    source_observation_ids: List[str]
    formula: str                      # "aa_1 × aa_2"
    active_atoms: List[CanonicalAtom]
    meta: Dict[str, Any]
```

### 4.2 Invariants

| ID | Definition |
|----|------------|
| INV-MOL-001 | Flat schema: axis/level at top level, no coordinates nesting |

---

## 5. Phase 9: Weak Axis Statistics

### 5.1 W0: ContentGateway

```python
class ContentGateway:
    def ingest(
        self,
        text: str,
        source_url: Optional[str] = None,
        source_id: Optional[str] = None,
        source_meta: Optional[Dict[str, Any]] = None,
    ) -> ArticleRecord
```

### 5.2 W1: Global Statistics

```python
@dataclass
class W1Record:
    token_norm: str              # Primary key
    total_count: int
    document_frequency: int
    surface_forms: Dict[str, int]
    entropy: float
```

### 5.3 W2: Conditional Statistics

```python
@dataclass
class W2Record:
    token_norm: str
    condition_signature: str     # SHA256(condition_factors)
    count: int
    document_frequency: int
```

### 5.4 W3: Axis Candidates

```python
@dataclass
class W3Candidate:
    token_norm: str
    s_score: float              # KL divergence
    p_cond: float
    p_global: float

@dataclass
class W3Record:
    analysis_id: str
    condition_signature: str
    positive_candidates: List[W3Candidate]
    negative_candidates: List[W3Candidate]
```

**S-Score Formula:**
```
S(t, C) = log₂(P(t|C) / P(t|Global)) + ε
```

### 5.5 W4: Structural Projection

```python
@dataclass
class W4Record:
    article_id: str
    w4_analysis_id: str
    resonance_vector: Dict[str, float]  # condition_sig -> score
    used_w3: Dict[str, str]             # condition_sig -> w3_analysis_id
```

**Resonance Formula:**
```
R(A, C) = Σ count(t, A) × S(t, C)
```

### 5.6 W5: Structural Condensation

```python
@dataclass
class W5Island:
    island_id: str              # Hash of sorted member_ids
    member_ids: List[str]       # w4_analysis_ids
    size: int
    representative_vector: Dict[str, float]
    cohesion_score: float

@dataclass
class W5Structure:
    structure_id: str
    islands: List[W5Island]
    noise_ids: List[str]
    threshold: float = 0.70
    min_island_size: int = 3
```

### 5.7 W6: Structural Observation

```python
@dataclass
class W6EvidenceToken:
    token_norm: str
    mean_s_score: float
    member_count: int
    total_count: int
    source_w3_ids: List[str]

@dataclass
class W6IslandDetail:
    island_id: str
    member_count: int
    evidence_tokens: List[W6EvidenceToken]
    top_conditions: List[str]
```

**Evidence Formula:**
```
mean_s_score = Σ s_score / count(members)
```

---

## 6. Phase 9 Invariants

### 6.1 W0-W2 Invariants

| ID | Definition |
|----|------------|
| INV-W0-001 | ArticleRecord is immutable after creation |
| INV-W1-001 | W1 does not store individual surface forms |
| INV-W2-001 | W2 reads condition factors, never writes/infers |
| INV-W2-002 | time_bucket is YYYY-MM (monthly) |
| INV-W2-003 | Denominators written by W2 only |

### 6.2 W3 Invariants

| ID | Definition |
|----|------------|
| INV-W3-001 | No natural language labels |
| INV-W3-002 | Never modifies W1/W2 |
| INV-W3-003 | Deterministic (tie-break: token_norm asc) |

### 6.3 W4 Invariants

| ID | Definition |
|----|------------|
| INV-W4-001 | Keys are condition_signature only |
| INV-W4-002 | Deterministic given same inputs |
| INV-W4-003 | Recomputable from W0 + W3 |
| INV-W4-004 | Uses both positive and negative candidates |
| INV-W4-005 | Does NOT modify inputs |
| INV-W4-006 | MUST use W1Tokenizer + normalize_token |

### 6.4 W5 Invariants

| ID | Definition |
|----|------------|
| INV-W5-001 | No natural language labels |
| INV-W5-002 | island_id from member IDs only |
| INV-W5-003 | Similarity = L2-normalized Cosine, >= operator |
| INV-W5-004 | structure_id includes params |
| INV-W5-005 | Vector values rounded before storage |
| INV-W5-006 | Canonical JSON hash, no string join |
| INV-W5-007 | Uses w4_analysis_id, not article_id |
| INV-W5-008 | created_at excluded from identity |

### 6.5 W6 Invariants

| ID | Definition |
|----|------------|
| INV-W6-001 | No synthetic labels or LLM summaries |
| INV-W6-002 | Deterministic export |
| INV-W6-003 | W1-W5 data never modified |
| INV-W6-004 | No new statistics, only extraction |
| INV-W6-005 | All evidence traceable to W3 |
| INV-W6-006 | Complete tie-break rules for ordering |
| INV-W6-007 | No hypothesis or judgment logic |
| INV-W6-008 | Version compatibility tracked |
| INV-W6-009 | W5 members match input sets exactly |

---

## 7. Constants Summary

### 7.1 Substrate Constants

| Parameter | Value | Description |
|-----------|-------|-------------|
| FLOAT_PRECISION | 9 | Decimal places for float normalization |
| CONTEXT_ID_LENGTH | 32 | Hex characters in context_id |
| STRING_MAX_LENGTH | 4096 | Max trace string value length |
| INT_MIN | -2147483648 | Minimum integer |
| INT_MAX | 2147483647 | Maximum integer |
| FILE_ENCODING | "utf-8" | Canonical encoding |
| FILE_NEWLINE | "\n" | Canonical newline |

### 7.2 Phase 7 Constants

| Parameter | Value |
|-----------|-------|
| COMPETE_TH | 0.15 |
| VOL_LOW_TH | 0.25 |
| VOL_HIGH_TH | 0.50 |
| TYPO_MAX_EDIT_DISTANCE | 2 |

### 7.3 Phase 9 Constants

| Parameter | Value | Description |
|-----------|-------|-------------|
| EPSILON | 1e-12 | S-Score smoothing |
| DEFAULT_MIN_COUNT_FOR_W3 | 2 | Minimum count filter |
| DEFAULT_TOP_K | 100 | Default candidates |
| W3_ALGORITHM | "KL-PerToken-v1" | W3 algorithm |
| W4_ALGORITHM | "DotProduct-v1" | W4 algorithm |
| W4_PROJECTION_NORM | "raw" | No normalization |
| W5_ALGORITHM | "ResonanceCondensation-v1" | W5 algorithm |
| W5_VECTOR_POLICY | "mean_raw_v1" | Centroid method |
| W5_DEFAULT_THRESHOLD | 0.70 | Similarity threshold |
| W5_DEFAULT_MIN_ISLAND_SIZE | 3 | Minimum island size |
| W5_MAX_BATCH_SIZE | 2000 | Batch limit |
| W5_VECTOR_ROUNDING | 9 | Decimal places |
| W5_SIMILARITY_ROUNDING | 9 | Decimal places |

---

## 8. File Structure

```
esde/
├── substrate/                    # Layer 0
│   ├── __init__.py
│   ├── schema.py                 # ContextRecord
│   ├── registry.py               # SubstrateRegistry
│   ├── id_generator.py           # Canonical ID
│   ├── traces.py                 # Validation
│   └── NAMESPACES.md             # Namespace registry
│
├── integration/                  # W0
│   ├── __init__.py
│   ├── schema.py                 # ArticleRecord
│   ├── gateway.py                # ContentGateway
│   └── segmenter.py              # Text segmentation
│
├── statistics/                   # W1-W5
│   ├── __init__.py
│   ├── schema.py                 # W1Record
│   ├── schema_w2.py              # W2Record
│   ├── schema_w3.py              # W3Record
│   ├── schema_w4.py              # W4Record
│   ├── schema_w5.py              # W5Structure
│   ├── w1_aggregator.py
│   ├── w2_aggregator.py
│   ├── w3_analyzer.py
│   ├── w4_projector.py
│   └── w5_condensator.py
│
├── discovery/                    # W6
│   ├── __init__.py
│   ├── w6_analyzer.py
│   ├── w6_exporter.py
│   └── w6_observatory.py
│
└── sensor/                       # Phase 8
    ├── __init__.py
    ├── loader_synapse.py
    └── validator_v83.py
```

---

## 9. Data Storage

### 9.1 Substrate

| Path | Format | Purpose |
|------|--------|---------|
| data/substrate/context_registry.jsonl | JSONL | ContextRecord storage |

### 9.2 Phase 9

| Path | Format | Purpose |
|------|--------|---------|
| data/stats/w1_global.json | JSON | W1 statistics |
| data/stats/w2_records.jsonl | JSONL | W2 records |
| data/stats/w2_conditions.jsonl | JSONL | Condition registry |
| data/stats/w3_candidates/ | JSON | W3 outputs |
| data/stats/w4_projections/ | JSON | W4 outputs |
| data/stats/w5_structures/ | JSON | W5 outputs |
| data/discovery/w6_observations/ | JSON/MD/CSV | W6 exports |

---

## 10. Version History

| Version | Date | Changes |
|---------|------|---------|
| v5.4.2-P9.0 | 2026-01-18 | W0 ContentGateway |
| v5.4.2-P9.1 | 2026-01-18 | W1 Global Statistics |
| v5.4.2-P9.2 | 2026-01-18 | W2 Conditional Statistics |
| v5.4.2-P9.3 | 2026-01-19 | W3 Axis Candidates |
| v5.4.4-P9.4 | 2026-01-20 | W4 Structural Projection |
| v5.4.5-P9.5 | 2026-01-21 | W5 Structural Condensation |
| v5.4.6-P9.6 | 2026-01-22 | W6 Structural Observation |
| **v5.4.7-SUB.1** | **2026-01-24** | **Substrate Layer v0.1.0** |

---

## 11. References

- ESDE Core Specification v0.2.1
- ESDE v3.3: Ternary Emergence and Dual Symmetry
- ESDE v3.3.1: Emergence Directionality
- Semantic Language Integrated v1.1
- ESDE Operator Spec v0.3
- Aruism Philosophy
- Substrate Layer Specification v0.1.0 (Gemini Design)

---

*Document generated: 2026-01-24*  
*Engine Version: 5.4.7-SUB.1*  
*Framework: Existence Symmetry Dynamic Equilibrium*  
*Philosophy: Aruism*
