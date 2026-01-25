# ESDE Glossary

**Version**: 5.4.8-MIG.2  
**Updated**: 2026-01-25  
**Spec**: Existence Symmetry Dynamic Equilibrium

---

## Core Philosophy

### Aruism (アルイズム)
The philosophical foundation of ESDE, based on the primordial recognition: "There is" (Aru wa, Aru). All understanding derives from this fundamental acknowledgment of existence.

### "Describe, but do not decide" (記述せよ、しかし決定するな)
Core principle for observation layers. Systems observe and record without making semantic judgments or classifications.

---

## Semantic Structure

### Atom
The indivisible unit of meaning in ESDE. The Foundation Layer defines 326 canonical atoms across 16 categories (ACT, EMO, REL, etc.). Atoms are the strong meaning system - stable reference points for observation.

### Molecule
A structured composition of atoms that represents observed meaning in context. Format:
```
{
  "active_atoms": [{"atom": "EMO.love", "axis": "ethical", "level": 3}],
  "formula": "EMO.love"
}
```

### Axis
One of 8 canonical axes that provide dimensional context: *cognitive*, *ethical*, *social*, *creative*, *ontological*, *temporal*, *spatial*, *physical*.

### Level
A 5-point scale (1-5) indicating intensity or degree along an axis.

### Synapse
The bridge between natural language and semantic atoms. Maps WordNet synsets to ESDE atoms with trigger words.

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
Handles tokens outside established semantic space. The weak meaning system - concepts that have not yet acquired stable semantic grounding.

### Phase 8: Introspective Engine
Self-reflection system monitoring concept processing patterns. Implements Rigidity detection and feedback loops.

### Phase 9: Weak Axis Statistics (W0-W6)
Statistical foundation for axis discovery without human labeling. Complete at W6.

| Layer | Name | Purpose |
|-------|------|---------|
| W0 | ContentGateway | External data normalization |
| W1 | Global Statistics | Unconditional token statistics |
| W2 | Conditional Statistics | Condition-sliced statistics |
| W3 | Axis Candidates | S-Score based candidate extraction |
| W4 | Structural Projection | Article → W3 resonance vectors |
| W5 | Structural Condensation | Clustering into islands |
| W6 | Structural Observation | Evidence extraction, topology |

**Note**: Phase 9 is complete at W6. Next development phase is Phase 10.

---

## Migration Phase 2: Policy-Based Statistics

### Overview
Migration Phase 2 introduces Policy-based condition signature generation, bridging Substrate Layer with W2 statistics.

### Components

#### BaseConditionPolicy
Abstract base class defining the interface for condition signature generation.

```python
class BaseConditionPolicy(ABC):
    policy_id: str
    version: str
    
    def compute_signature(self, record: ContextRecord) -> str:
        """Return full SHA256 hex (64 chars)"""
    
    def extract_factors(self, record: ContextRecord) -> Dict[str, Any]:
        """Return factors with types preserved"""
```

#### StandardConditionPolicy
Standard implementation extracting condition factors from Substrate traces.

```python
policy = StandardConditionPolicy(
    policy_id="legacy_migration_v1",
    target_keys=["legacy:source_type", "legacy:language_profile"],
    version="v1.0",
)
```

### P0 Requirements
| ID | Requirement | Description |
|----|-------------|-------------|
| P0-MIG-1 | Policy ID Mixing | policy_id included in hash to prevent collision |
| P0-MIG-2 | Type Preservation | No str() coercion; bool, int, float preserved |
| P0-MIG-3 | Canonical JSON | Unified with Substrate spec (no spaces) |
| P0-MIG-4 | Missing Key Handling | Explicit missing list, not empty factors |

### Data Flow
```
ArticleRecord.substrate_ref
    ↓
SubstrateRegistry.get(substrate_ref)
    ↓
ContextRecord.traces
    ↓
Policy.compute_signature()
    ↓
W2Aggregator condition signature (64 hex)
```

### Legacy Fallback
When `substrate_ref` is None or registry lookup fails, W2Aggregator falls back to legacy `source_meta` extraction.

---

## Key Metrics

### Rigidity (R)
Measures pattern fixation for a concept:
```
R = N_mode / N_total
```
| Range | Status | Strategy |
|-------|--------|----------|
| R < 0.3 | Volatile | STABILIZING |
| 0.3 ≤ R ≤ 0.9 | Healthy | NEUTRAL |
| R > 0.9 | Rigid | DISRUPTIVE |

### S-Score
Condition specificity measure for axis candidates:
```
S(token, condition) = log(P_cond / P_global)
```
Positive = condition-specific, Negative = condition-avoided.

### Resonance Vector
Per-article projection onto W3 axis candidates, computed by W4Projector.

---

## File Locations

| Component | Path |
|-----------|------|
| Glossary | esde_dictionary.json |
| Synapse | esde_synapses_v3.json |
| Substrate Registry | data/substrate/context_registry.jsonl |
| Substrate Schema | esde/substrate/schema.py |
| Substrate ID Generator | esde/substrate/id_generator.py |
| **Policy Base** | **esde/statistics/policies/base.py** |
| **Policy Standard** | **esde/statistics/policies/standard.py** |
| W0 ContentGateway | esde/integration/gateway.py |
| W1-W5 Modules | esde/statistics/ |
| W6 Modules | esde/discovery/ |

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

Phase 9 introduces the "Weak Axis Statistics" layer (W0-W6), providing statistical foundation for axis discovery without human labeling. **Phase 9 is complete at W6.**

**Substrate Layer** is a cross-cutting layer (Layer 0) that sits beneath all phases, providing machine-observable trace storage without semantic interpretation.

**Migration Phase 2** bridges Substrate Layer with W2 statistics through Policy-based condition signature generation.

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
| **MIG-2** | **v0.2.1** | **2026-01-25** | **Migration Phase 2 (Policy-Based Statistics)** |

**Note**: Phase 9 is complete. Next development phase is Phase 10.

---

*End of Glossary*
