# ESDE

## Existence Symmetry Dynamic Equilibrium

*The Introspection Engine for AI*

### Detailed Design Document

For Design Review and AI Context Transfer

**Version 5.4.7-SUB.1**

**2026-01-24**

*Based on Aruism Philosophy*

---

## Table of Contents

1. Executive Summary
2. Foundation Layer: Semantic Dictionary
3. **Substrate Layer: Context Fabric (NEW)**
4. Phase 7: Unknown Resolution (Weak Meaning System)
5. Phase 8: Introspective Engine (Strong Meaning System)
6. Phase 9: Weak Axis Statistics Layer
7. System Architecture: Dual Symmetry
8. Philosophical Foundation: Aruism
9. Mathematical Formalization
10. Future Roadmap
11. Appendices

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
| Phase 9 | Statistical foundation for axis discovery | Pattern emergence detection |

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

Following Aruism's principle of existential symmetry, each concept exists in relation to its symmetric counterpart. Love cannot be defined without hate; life cannot be understood without death. This is not binary opposition but mutual definition.

```
EMO.love <-> EMO.hate     (Emotion pair)
EXS.life <-> EXS.death    (Existence pair)
ACT.create <-> ACT.destroy (Action pair)
VAL.truth <-> VAL.falsehood (Value pair)
```

#### 2.1.2 Atom Structure

Each semantic atom is defined by:

| Property | Description | Example |
|----------|-------------|---------|
| ID | Unique identifier | aa_1 |
| Atom | Category.concept format | EMO.love |
| Definition | Precise semantic definition | Deep affection and care for another |
| Symmetric | Paired concept ID | aa_2 (EMO.hate) |

#### 2.1.3 Immutability Principle

The 326 atoms constitute the strongest meaning system and are immutable by design. Once defined, they do not change. This constraint ensures:

- **Stable observation axis**: Without a fixed reference frame, measurement becomes meaningless
- **Accumulation integrity**: Historical records remain interpretable across time
- **Cross-instance compatibility**: Different ESDE instances share a common semantic foundation

*Note: Phase 7 discoveries do NOT flow into Phase 8 glossary. The weak meaning system and strong meaning system remain structurally separate.*

### 2.2 Axes and Levels (10 Axes × 48 Levels)

Raw semantic atoms are too abstract for practical use. The axis-level coordinate system provides contextual precision.

#### 2.2.1 Ten Semantic Axes

| Axis | Question | Range |
|------|----------|-------|
| temporal | When? How long? | momentary → eternal |
| scale | What scope? | individual → cosmic |
| epistemological | How known? | perception → creation |
| ontological | What is it? | material → semantic |
| symmetry | How does it change? | destructive → cyclical |
| lawfulness | How constrained? | chaotic → deterministic |
| experience | How felt? | surface → profound |
| value_generation | What worth? | instrumental → intrinsic |
| interconnection | How connected? | isolated → universal |
| emergence | How arising? | latent → manifest |

### 2.3 Synapse: Connection Dictionary

Synapse bridges everyday language (WordNet synsets) to ESDE semantic atoms.

| Version | Synsets | Edges | Coverage |
|---------|---------|-------|----------|
| v3.0 | 11,557 | 22,285 | 100% (326/326 concepts) |

---

## 3. Substrate Layer: Context Fabric (NEW)

### 3.1 Overview

The Substrate Layer (also called Context Fabric) is **Layer 0** of ESDE—the lowest foundational layer that sits beneath all phases. It provides permanent storage for machine-observable facts without any semantic interpretation.

**Philosophy:** "Describe, but do not decide." (記述せよ、しかし決定するな)

### 3.2 Design Principles

| Principle | Description |
|-----------|-------------|
| **Not a Phase** | Substrate has no completion state; it operates continuously |
| **No Semantics** | Records "URL is example.com" not "this is news" |
| **Deterministic** | Same inputs always produce same context_id |
| **Append-Only** | Records are never updated or deleted |
| **Machine-Observable** | Only values computable without human judgment |

### 3.3 Architecture

```
┌─────────────────────────────────────────────┐
│  Phase 10: Interpretation (Future)          │
├─────────────────────────────────────────────┤
│  Phase 9: Weak Axis Statistics (W0-W6)      │
│    └─ Reads substrate_ref for condition     │
│       signature generation via Policy       │
├─────────────────────────────────────────────┤
│  Phase 7-8: Unknown/Introspection           │
├═════════════════════════════════════════════┤
│  SUBSTRATE LAYER (Layer 0)                  │
│    - ContextRecord (immutable traces)       │
│    - SubstrateRegistry (append-only JSONL)  │
│    - No semantic interpretation             │
└─────────────────────────────────────────────┘
```

### 3.4 Core Data Structure: ContextRecord

```python
@dataclass(frozen=True)
class ContextRecord:
    # === Identity (Deterministic) ===
    context_id: str  # SHA256(canonical(retrieval_path, traces, capture_version))[:32]
    
    # === Observation Facts (Canonical) ===
    retrieval_path: Optional[str]   # URL, file path, etc.
    capture_version: str            # Trace extraction logic version
    traces: Dict[str, Any]          # Schema-less observations
    
    # === Metadata (Non-Canonical) ===
    observed_at: str                # ISO8601
    created_at: str                 # ISO8601
```

**Key Property:** `context_id` is computed ONLY from `retrieval_path`, `traces`, and `capture_version`. Timestamps do not affect identity.

### 3.5 Trace Format

Traces are key-value pairs following the `namespace:name` format:

```python
traces = {
    "html:tag_count": 42,
    "html:has_h1": True,
    "text:char_count": 1000,
    "meta:domain": "example.com",
    "time:year": 2026,
}
```

#### 3.5.1 Allowed Value Types

| Type | Constraints |
|------|-------------|
| `str` | Max 4096 characters |
| `int` | Range: -2³¹ to 2³¹-1 |
| `float` | Normalized to 9 decimal places, no NaN/Inf |
| `bool` | True/False |
| `None` | Null value |

**Forbidden:** List, Dict (prevents nesting)

#### 3.5.2 Forbidden Namespaces (INV-SUB-002)

| Namespace | Reason |
|-----------|--------|
| `meaning:` | Semantic interpretation |
| `category:` | Classification |
| `intent:` | Intent inference |
| `quality:` | Quality judgment |

#### 3.5.3 Forbidden Key Names (P1-SUB-002)

Interpretation words are banned regardless of namespace:

- `source_type`, `is_short`, `is_important`, `quality_score`, etc.

**Exception:** `legacy:source_type` allowed for migration only.

### 3.6 SubstrateRegistry

The registry provides append-only JSONL storage:

```python
class SubstrateRegistry:
    def register(self, record: ContextRecord) -> str:
        """Append record (deduplicate by context_id)"""
    
    def get(self, context_id: str) -> Optional[ContextRecord]:
        """Retrieve by ID"""
    
    def export_canonical(self, output_path: str) -> int:
        """Export sorted by context_id (INV-SUB-007)"""
```

### 3.7 Invariants

| ID | Name | Definition |
|----|------|------------|
| INV-SUB-001 | Upper Read-Only | Upper layers cannot update/delete; append-only |
| INV-SUB-002 | No Semantic Transform | Raw observation values only |
| INV-SUB-003 | Machine-Observable | No human judgment required |
| INV-SUB-004 | No Inference | No ML/probabilistic values |
| INV-SUB-005 | Append-Only | No update, no delete |
| INV-SUB-006 | ID Determinism | context_id from inputs only (no timestamp) |
| INV-SUB-007 | Canonical Export | Deterministic output order |

### 3.8 Integration with W2

**Current (v0.1.0):** ArticleRecord has `source_meta` (legacy).

**Future (v0.2.0+):** ArticleRecord will have `substrate_ref` pointing to ContextRecord. W2 Aggregator will use `ConditionSignaturePolicy` to extract condition factors from traces.

```python
# Future W2 Integration
class ConditionSignaturePolicy:
    policy_id: str
    target_keys: List[str]    # e.g., ["meta:domain", "time:month"]
    projection_rules: Dict    # e.g., "text:char_count" -> Bucket
```

---

## 4. Phase 7: Unknown Resolution (Weak Meaning System)

### 4.1 Purpose

Phase 7 handles tokens that cannot be mapped through Synapse. These "unknown" tokens represent the boundary between established meaning (Glossary) and emerging meaning (new concepts).

### 4.2 Volatility Score

```
V = 1 - (max_score - second_max_score)
```

When V is high, the token's classification is uncertain—multiple hypotheses compete.

### 4.3 Four Routes

| Route | Hypothesis | Resolution |
|-------|------------|------------|
| A | Typo | Spell correction → Synapse lookup |
| B | Entity | External knowledge lookup (Wikipedia) |
| C | Novel | Molecular decomposition |
| D | Functional | Filter/ignore (particles, conjunctions) |

### 4.4 Multi-Hypothesis Tracking

Phase 7B+ maintains all hypotheses in parallel until evidence resolves classification. Winner selection requires explicit human review.

---

## 5. Phase 8: Introspective Engine (Strong Meaning System)

### 5.1 Molecule Generation

Phase 8 converts observed text into molecules using the 326 atoms as building blocks.

```
Input: "I love you"
Output: Molecule(EMO.love × REL.connection)
```

### 5.2 Rigidity Monitoring

Rigidity measures how repetitive the observation patterns become. High rigidity triggers axis reboot consideration.

### 5.3 Schema: v8.3 Canonical

```python
@dataclass
class CanonicalAtom:
    id: str           # "aa_1"
    atom: str         # "EMO.love"
    axis: str         # Top-level (NOT nested)
    level: str        # Top-level (NOT nested)
```

**INV-MOL-001:** No `coordinates` nesting allowed.

---

## 6. Phase 9: Weak Axis Statistics Layer

### 6.1 Overview

Phase 9 provides statistical infrastructure for discovering semantic axes without human labeling. It consists of six sub-layers (W0-W6).

### 6.2 Pipeline

```
W0 (Gateway) → W1 (Global) → W2 (Conditional) → W3 (Candidates)
                                                      ↓
W6 (Observation) ← W5 (Condensation) ← W4 (Projection)
```

### 6.3 Key Concepts

| Layer | Input | Output | Key Metric |
|-------|-------|--------|------------|
| W0 | Text | ArticleRecord | - |
| W1 | Articles | Token counts | document_frequency |
| W2 | W1 + Conditions | Per-condition counts | - |
| W3 | W1 + W2 | Axis candidates | S-Score |
| W4 | Article + W3 | Resonance vector | Dot product |
| W5 | W4 vectors | Island clusters | Cosine similarity |
| W6 | W5 structures | Evidence export | mean_s_score |

---

## 7. System Architecture: Dual Symmetry

### 7.1 Weak/Strong Duality

| Aspect | Weak (Phase 7, 9) | Strong (Phase 8) |
|--------|-------------------|------------------|
| Purpose | Discovery | Observation |
| Stability | Evolving | Fixed |
| Data flow | → | ← |
| Truth | Emerging | Established |

### 7.2 Layer Stack

```
┌─────────────────────────────────────────┐
│ Application Layer (CLI, API)            │
├─────────────────────────────────────────┤
│ Phase 8: Strong Meaning (Introspection) │
├─────────────────────────────────────────┤
│ Phase 7: Weak Meaning (Unknown)         │
├─────────────────────────────────────────┤
│ Phase 9: Statistics (W0-W6)             │
├─────────────────────────────────────────┤
│ Foundation: Glossary + Synapse          │
├═════════════════════════════════════════┤
│ SUBSTRATE LAYER (Layer 0)               │
└─────────────────────────────────────────┘
```

---

## 8. Philosophical Foundation: Aruism

### 8.1 Core Tenets

1. **Aru (ある)**: The primordial fact "there is" precedes all categorization
2. **Existential Symmetry**: Every concept requires its counterpart for definition
3. **Ternary Emergence**: Meaning requires three-term relationships (A ↔ B ↔ C)
4. **Dynamic Equilibrium**: Systems maintain stability through balanced flexibility

### 8.2 Substrate Philosophy

The Substrate Layer embodies Aruism's observation-first principle:

> "Before categorizing what something IS, record that it EXISTS and HOW it was observed."

This is why Substrate records `meta:domain = "example.com"` but never `category:source_type = "news"`. The interpretation of domain-to-category is left to upper layers (W2 Policy, Phase 10).

---

## 9. Mathematical Formalization

### 9.1 S-Score (W3)

```
S(t, C) = log₂(P(t|C) / P(t|Global)) + ε
```

Where:
- P(t|C) = token probability under condition C
- P(t|Global) = token probability across all data
- ε = smoothing constant (1e-12)

### 9.2 Resonance (W4)

```
R(A, C) = Σ count(t, A) × S(t, C)
```

### 9.3 Context ID (Substrate)

```
context_id = SHA256(canonical_json({
    "retrieval_path": path,
    "traces": normalized_traces,
    "capture_version": version
}))[:32]
```

---

## 10. Future Roadmap

### 10.1 Substrate Evolution

| Phase | Version | Content |
|-------|---------|---------|
| Phase 1 | v0.1.0 | ContextRecord, SubstrateRegistry ✅ |
| Phase 2 | v0.2.0 | ArticleRecord.substrate_ref, ConditionSignaturePolicy |
| Phase 3 | v1.0.0 | Legacy source_meta removal |

### 10.2 Phase 10: Interpretation Layer

Future layer that will:
- Name axis candidates discovered by W3/W5/W6
- Create human-readable labels from statistical patterns
- Connect Weak Meaning discoveries to Strong Meaning framework

---

## 11. Appendices

### A. File Structure

```
esde/
├── substrate/               # Layer 0
│   ├── schema.py           # ContextRecord
│   ├── registry.py         # SubstrateRegistry
│   ├── id_generator.py     # Canonical ID
│   ├── traces.py           # Validation
│   └── NAMESPACES.md       # Namespace registry
│
├── integration/            # W0
│   ├── schema.py           # ArticleRecord
│   └── gateway.py          # ContentGateway
│
├── statistics/             # W1-W5
│   ├── schema.py           # W1Record
│   ├── schema_w2.py        # W2Record
│   ├── schema_w3.py        # W3Record
│   ├── schema_w4.py        # W4Record
│   └── schema_w5.py        # W5Structure
│
└── discovery/              # W6
    └── w6_observatory.py   # W6Observatory
```

### B. Version History

| Version | Date | Changes |
|---------|------|---------|
| v5.4.6-P9.6 | 2026-01-22 | W6 completion |
| v5.4.7-SUB.1 | 2026-01-24 | Substrate Layer v0.1.0 |

---

*Document generated: 2026-01-24*  
*Engine Version: 5.4.7-SUB.1*  
*Framework: Existence Symmetry Dynamic Equilibrium*  
*Philosophy: Aruism*
