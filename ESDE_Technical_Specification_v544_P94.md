# ESDE Technical Specification v5.4.4-P9.4

## Existence Symmetry Dynamic Equilibrium
### The Introspection Engine for AI

Version: 5.4.4-P9.4  
Date: 2026-01-21  
Status: Production (Phase 9-4 Complete - Weak Axis Statistics Layer)

---

## 0. Purpose and Scope

### 0.1 What ESDE Is

> ESDEは証明システムではない。  
> ESDEは生成言語モデルではない。  
> ESDEは意味構造上で動作する動的均衡エンジンである。  
> **ESDEはAIの内省エンジン（Introspection Engine）である。**

ESDEを使用するとは以下を意味する：

- 現実世界の言語やイベントを構造化された意味空間にマッピングする
- 不均衡と矛盾を明示的な誤差値（ε）として測定する
- 制約付き更新を通じて動的均衡を回復する
- 均衡が硬直した場合に制御されたRebootをトリガーする
- 単なる物語的応答ではなく、構造化された洞察を返す

### 0.2 Philosophical Foundation

ESDE is based on **Aruism Philosophy** and the **ESDE Framework**.

#### Aruism (The Root)

- **「ある（Aru）」の優先**: 世界は「定義」される前にまず「ある」
- ESDEは観測対象を既存の枠組みに無理やり当てはめるのではなく、「そこにある曖昧な状態」をそのまま保持・観測するための基盤（OS）である
- **二項対立の回避**: 「AかBか」の二者択一を迫ることは、Aruismの否定となる

#### ESDE Theory (The Logic)

- **公理T (Ternary Emergence)**: 「AとB」の二者関係だけでは存在は顕現しない。第三項（Observer）が介入することで初めて意味が立ち上がる
- **創造と破壊の対称性**: 「揺れ（Volatility）」や「未決（Winner=Null）」はエラーではなく、「破壊的創発（Destructive Emergence）」の兆候である可能性がある

#### Core Axioms (v3.3)

| Axiom | Name | Statement |
|-------|------|-----------|
| 0 | Aru (There Is) | The primordial fact of existence |
| E | Identification | E = {e1, e2, ..., en} |
| L | Linkage | L: E × E → [0,1] |
| Eq | Equality | All existences equal in status |
| C | Creativity | C ⊥ gradient(F) |
| U | Understanding | Ongoing participatory engagement |
| ε | Error | E = f(I) + ε, ε ≠ 0 |
| T | Ternary Emergence | A↔B↔C ⇒ Manifestation |

---

## 1. Semantic Hierarchy

### 1.1 Semantic Atoms

**326** semantic atoms are defined. These are the most primitive concepts closest to "Aru" (existence).

Characteristics:
- Irreducible fundamental meanings
- **163 symmetric pairs**
- Defined by Glossary (not by dictionary or kanji)

#### Examples of Symmetric Pairs

| Concept A | Concept B |
|-----------|-----------|
| EMO.love | EMO.hate |
| EXS.life | EXS.death |
| VAL.truth | VAL.falsehood |
| ACT.create | ACT.destroy |
| STA.peace | STA.war |
| ABS.bound | ABS.release |
| SOC.cooperate | SOC.conflict |

#### Category Codes (24 categories)

```
FND: Fundamental    EXS: Existence      EMO: Emotion
ACT: Action         CHG: Change         VAL: Value
STA: State          COG: Cognition      COM: Communication
PER: Perception     BOD: Body           BEI: Being
SOC: Social         ECO: Economic       SPC: Space
TIM: Time           ELM: Element        NAT: Nature
MAT: Material       REL: Relation       LOG: Logic
WLD: World          ABS: Abstract       PRP: Property
```

### 1.2 Axes and Levels (10 Axes, 48 Levels)

Semantic atoms alone do not "point to" anything specific. Only when axis and level are specified does a concrete position emerge.

| Axis | Levels | Count |
|------|--------|-------|
| temporal | emergence → indication → influence → transformation → establishment → continuation → permanence | 7 |
| scale | individual → community → society → ecosystem → stellar → cosmic | 6 |
| epistemological | perception → identification → understanding → experience → creation | 5 |
| ontological | material → informational → relational → structural → semantic | 5 |
| interconnection | independent → catalytic → chained → synchronous → resonant | 5 |
| resonance | superficial → structural → essential → existential | 4 |
| symmetry | destructive → inclusive → transformative → generative → cyclical | 5 |
| lawfulness | predictable → emergent → contingent → necessary | 4 |
| experience | discovery → creation → comprehension | 3 |
| value_generation | functional → aesthetic → ethical → sacred | 4 |

**Total: 48 levels**

### 1.3 Design Philosophy (Aruism Mapping)

The semantic architecture directly implements core Aruism concepts:

#### Equality of Existence → winner=null

> あらゆる存在が、その価値や重要性において等しく相互依存的であり、
> 欠けることのできない全体の構成要素として対等に扱われる。

Implementation: No hypothesis is privileged over another. `winner` remains `null` always. All routes (A/B/C/D) are evaluated with equal weight.

#### Symmetry of Existence → 163 Pairs

> 一つの存在に対して自然と対称的に生じる存在や概念の関係性。
> 異なる視点や状況によって反転しうる相補的な関係性。

Implementation: 326 atoms = 163 symmetric pairs. This is not binary opposition but mutual definition. `love` cannot be defined without `hate`. ε_sym measures imbalance between pairs.

#### Linkage of Existence → w_ij

> あらゆる存在が相互作用し合い、連動して現象や状況を生み出している。

Implementation: Linkage weight `w_ij` between concepts. Co-occurrence strengthens links. ε_link measures inconsistency in linked concepts.

#### Axis → 10 Axes

> 複雑な現実から関連性の強い存在群を見つけ出し、
> 意味ある一つのまとまりとして統合・囲い込む認識機能。

Implementation: 10 axes are not domain categories (emotion, economy, etc.) but **observation perspectives** applicable to any existence. They answer:

| Question | Axis |
|----------|------|
| When? How long? | temporal |
| Where? How big? | scale |
| How is it known? | epistemological |
| What is it? | ontological |
| How connected? | interconnection, resonance |
| What direction? | symmetry |
| How predictable? | lawfulness |
| What experience? | experience |
| What value? | value_generation |

### 1.4 State Space

```
State Vector: x_t[i, a]
    i = concept ID (326)
    a = axis ID (48)
    value = activation intensity (0.0〜1.0)

State Space Size: 326 × 48 = 15,648 dimensions
```

---

## 2. Architecture

### 2.1 Component Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    ESDE Framework                            │
│         (Existence Symmetry Dynamic Equilibrium)             │
│                   Aruism Philosophy                          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    ESDE Components                           │
├──────────────┬──────────────┬──────────────┬────────────────┤
│   Glossary   │   Synapse    │    Sensor    │     Engine     │
│  Atom Defs   │   Bridge     │   Input      │    State       │
│   326 atoms  │  WordNet     │  Operators   │   ε Calc       │
└──────────────┴──────────────┴──────────────┴────────────────┘
```

### 2.2 Component Status

| Component | Status | Implementation | Issue |
|-----------|--------|----------------|-------|
| Glossary | ✅ Stable | 326 atoms, 10×48 definitions | - |
| Synapse | ✅ Generated | 11,557 synsets, 22,285 edges, vector distance (raw_score) | - |
| Sensor | ✅ **V2 + Live** | Synapse lookup + Legacy fallback | - |
| Generator | ✅ **Live (QwQ)** | Phase 8-3 MoleculeGeneratorLive | - |
| Validator | ✅ **Production** | Phase 8-2 MoleculeValidator | - |
| **Ledger** | ✅ **Ephemeral** | Phase 8-5 EphemeralLedger | - |
| Engine | ✅ Working | State management, ε calculation | - |
| Audit Pipeline | ✅ Complete | 7A → 7D | - |
| Stability Audit | ✅ **PASS** | Phase 8-4 (3500 runs) | - |
| **Memory Audit** | ✅ **PASS** | Phase 8-5 (Integration Test) | - |
| **W0 (ContentGateway)** | ✅ **Production** | Phase 9-0 input normalization | - |
| **W1 (Global Stats)** | ✅ **Production** | Phase 9-1 token statistics | - |
| **W2 (Conditional Stats)** | ✅ **Production** | Phase 9-2 condition-sliced statistics | - |
| **W3 (Axis Candidates)** | ✅ **Production** | Phase 9-3 S-Score calculation | - |
| **W4 (Structural Projection)** | ✅ **Production** | Phase 9-4 Resonance vectors | - |

### 2.3 Integration Status

```
BEFORE (Disconnected):

┌─────────────────┐          ┌─────────────────┐
│   Sensor v1     │          │   Synapse       │
│   (Legacy)      │    ??    │   (v3.0)        │
│ Trigger-based   │◄────────►│ Vector distance │
└─────────────────┘   NOT    └─────────────────┘
                   CONNECTED

AFTER (Sensor V2 - Integrated):

┌─────────────────┐          ┌─────────────────┐
│   Input Text    │          │   Synapse       │
│                 │          │   (v3.0)        │
└────────┬────────┘          └────────┬────────┘
         │                            │
         ▼                            ▼
┌─────────────────────────────────────────────┐
│              Sensor V2 (Unified)             │
│                                              │
│  1. Tokenize → WordNet synset lookup        │
│  2. Synapse lookup → concept candidates     │
│  3. Vector distance (raw_score) ranking     │
│  4. Deterministic sort (score DESC, id ASC) │
│  5. Fallback: Legacy triggers (Hybrid mode) │
│  6. Output: concept_id + axis + level       │
└─────────────────────────────────────────────┘
```

**Implementation File:** `esde_sensor_v2.py`

### 2.4 Data Flow Pipeline

```
1. User Text → esde_sensor.py → Formula
2. Formula → esde-engine-v532.py → Known or Queue
3. Queue → resolve_unknown_queue_7bplus_v534_final.py ↔ aggregate_state.py
4. Resolver → online.py → hypothesis.py → Ledger & Aggregate Output
```

### 2.3 Key File Groups

#### Group A: The Brain (Resolver / Phase 7B+)

| File | Role |
|------|------|
| `resolve_unknown_queue_7bplus_v534_final.py` | Phase 7B+ main CLI. Maintains `winner=null` |
| `esde_engine/resolver/hypothesis.py` | 4-hypothesis evaluation (A/B/C/D) |
| `esde_engine/resolver/aggregate_state.py` | Observation state management |
| `esde_engine/resolver/online.py` | External evidence collection (v5.3.5: MultiSourceProvider) |

#### Group B: The Heart (Runtime Engine / Phase 7A+)

| File | Role |
|------|------|
| `esde-engine-v532.py` | Tokenization, Synapse activation, Routing |

#### Group C: The Senses (Input Sensor)

| File | Role | Status |
|------|------|--------|
| `esde_sensor.py` | Legacy: Trigger-based + LLM | Superseded by V2 |
| `esde_sensor_v2_modular.py` | **Facade: Synapse-integrated** | ✅ **Implemented** |
| `sensor/` | **Modular components package** | ✅ **Phase 8** |

**sensor/ Package Structure:**

| Module | Class | Role |
|--------|-------|------|
| `__init__.py` | - | Package exports (lazy import for 8-2/8-3) |
| `loader_synapse.py` | SynapseLoader | JSON load, singleton |
| `extract_synset.py` | SynsetExtractor | WordNet extraction |
| `rank_candidates.py` | CandidateRanker | Score aggregation |
| `legacy_trigger.py` | LegacyTriggerMatcher | v1 fallback |
| `audit_trace.py` | AuditTracer | Counters/hash/evidence |
| `molecule_validator.py` | MoleculeValidator | Phase 8-2 validation |
| `molecule_generator.py` | MoleculeGenerator | Phase 8-2 LLM generation (mock) |
| `molecule_generator_live.py` | MoleculeGeneratorLive | **Phase 8-3** Real LLM integration |

**Phase 8-3 Classes (molecule_generator_live.py):**

| Class | Role |
|-------|------|
| `MoleculeGeneratorLive` | Real LLM (QwQ-32B) integration with guardrails |
| `SpanCalculator` | System-calculated span (token proximity) |
| `CoordinateCoercer` | Invalid coordinate → null with logging |
| `FormulaValidator` | Formula syntax validation (consecutive operators) |
| `MockMoleculeGeneratorLive` | Test mock |

#### Group D: The Knowledge (Synapse Generator)

| File | Role |
|------|------|
| `generate_synapses_v2_1.py` | Connect ESDE concepts to WordNet |

#### Group E: The Memory (Ledger) - Phase 8-5

| File | Role |
|------|------|
| `ledger/` | **Semantic memory package** |

**ledger/ Package Structure:**

| Module | Class | Role |
|--------|-------|------|
| `__init__.py` | - | Package exports |
| `memory_math.py` | - | Decay, Reinforce, Tau Policy, Fingerprint |
| `ephemeral_ledger.py` | EphemeralLedger | In-memory semantic memory |

**Memory Math Functions:**

| Function | Formula | Description |
|----------|---------|-------------|
| `decay(w, dt, tau)` | w × exp(-dt/τ) | Exponential decay |
| `reinforce(w, alpha)` | w + α(1-w) | Asymptotic reinforcement |
| `should_purge(w, epsilon)` | w < ε | Oblivion check |
| `get_tau_from_molecule()` | - | Temporal axis → tau |
| `generate_fingerprint()` | SHA256 | Molecule identity |

**Constants:**

| Constant | Value | Description |
|----------|-------|-------------|
| ALPHA | 0.2 | Learning rate |
| EPSILON | 0.01 | Oblivion threshold |
| DEFAULT_TAU | 300 | Default time constant (5 min) |

**Tau Policy:**

| Temporal Level | Tau (sec) | Human |
|----------------|-----------|-------|
| permanence | 86400 | 24h |
| continuation | 3600 | 1h |
| establishment | 3600 | 1h |
| transformation | 1800 | 30m |
| indication | 300 | 5m |
| emergence | 60 | 1m |

---

## 3. Directory Structure

```
esde/
├── esde_engine/                     # Main package directory
│   ├── __init__.py                  # Package initialization & exports
│   ├── __main__.py                  # CLI entry point
│   ├── config.py                    # Configuration constants (SINGLE SOURCE OF TRUTH)
│   ├── utils.py                     # Utility functions
│   ├── loaders.py                   # Data loaders
│   ├── extractors.py                # Synset extraction
│   ├── collectors.py                # Activation collection
│   ├── routing.py                   # Unknown token routing (7A+)
│   ├── queue.py                     # Unknown queue writer (7A)
│   ├── engine.py                    # Main ESDEEngine class
│   │
│   └── resolver/                    # Phase 7B/7B+: Unknown Queue Resolver
│       ├── __init__.py              # Package exports
│       ├── state.py                 # Queue state management (legacy)
│       ├── aggregate_state.py       # [v5.3.4] Aggregate state management
│       ├── hypothesis.py            # Multi-hypothesis evaluation (A/B/C/D)
│       ├── online.py                # [v5.3.5] MultiSourceProvider
│       ├── ledger.py                # Evidence ledger (audit trail)
│       ├── cache.py                 # Search cache
│       ├── patches.py               # Patch output management
│       └── resolvers.py             # Route-specific resolvers
│
├── sensor/                          # [Phase 8] Modular Sensor Package
│   ├── __init__.py                  # Package exports (lazy import)
│   ├── loader_synapse.py            # SynapseLoader (singleton)
│   ├── extract_synset.py            # SynsetExtractor (WordNet)
│   ├── rank_candidates.py           # CandidateRanker (aggregation)
│   ├── legacy_trigger.py            # LegacyTriggerMatcher (v1 fallback)
│   ├── audit_trace.py               # AuditTracer (counters/hash)
│   ├── molecule_validator.py        # [Phase 8-2] MoleculeValidator
│   ├── molecule_generator.py        # [Phase 8-2] MoleculeGenerator (mock)
│   └── molecule_generator_live.py   # [Phase 8-3] MoleculeGeneratorLive (Real LLM)
│
├── ledger/                          # [Phase 8-5/8-6] Semantic Memory Package
│   ├── __init__.py                  # Package exports (v8.6.0)
│   ├── memory_math.py               # Decay, Reinforce, Tau, Fingerprint
│   ├── ephemeral_ledger.py          # EphemeralLedger (in-memory)
│   ├── canonical.py                 # [P8.6] JSON Canonicalization
│   ├── chain_crypto.py              # [P8.6] Hash Chain (SHA256)
│   └── persistent_ledger.py         # [P8.6] PersistentLedger (Hash Chain)
│
├── index/                           # [Phase 8-7] Semantic Index Package
│   ├── __init__.py                  # Package exports (v8.7.0)
│   ├── semantic_index.py            # L2 In-Memory Structure
│   ├── projector.py                 # L1→L2 Projection (rebuild/on_event)
│   ├── rigidity.py                  # Rigidity Calculation (R = N_mode/N_total)
│   └── query_api.py                 # External Query API
│
├── feedback/                        # [Phase 8-8] Feedback Loop Package
│   ├── __init__.py                  # Package exports (v8.8.0)
│   ├── strategies.py                # Strategy Definitions (NEUTRAL/DISRUPTIVE/STABILIZING)
│   └── modulator.py                 # Feedback Modulator (decide_strategy, check_alert)
│
├── pipeline/                        # [Phase 8-8] Core Pipeline Package
│   ├── __init__.py                  # Package exports (v8.8.0)
│   └── core_pipeline.py             # ESDEPipeline, ModulatedGenerator
│
├── monitor/                         # [Phase 8-9] Semantic Monitor Package
│   ├── __init__.py                  # Package exports (v8.9.0)
│   └── semantic_monitor.py          # TUI Dashboard (rich)
│
├── runner/                          # [Phase 8-9] Long-Run Runner Package
│   ├── __init__.py                  # Package exports (v8.9.0)
│   └── long_run.py                  # LongRunRunner, LongRunReport
│
├── integration/                     # [Phase 9-0] Content Gateway Package
│   ├── __init__.py                  # Package exports (v9.0.0)
│   └── content_gateway.py           # ContentGateway, ArticleRecord
│
├── statistics/                      # [Phase 9-1/9-2/9-3/9-4] Statistics Package
│   ├── __init__.py                  # Package exports (v9.4.0)
│   ├── schema.py                    # W1Record, W1GlobalStats
│   ├── schema_w2.py                 # W2Record, W2GlobalStats, ConditionEntry
│   ├── schema_w3.py                 # W3Record, CandidateToken
│   ├── schema_w4.py                 # W4Record (Phase 9-4)
│   ├── tokenizer.py                 # EnglishWordTokenizer, HybridTokenizer
│   ├── normalizer.py                # Token normalization (NFKC)
│   ├── w1_aggregator.py             # W1Aggregator (global stats)
│   ├── w2_aggregator.py             # W2Aggregator (conditional stats)
│   ├── w3_calculator.py             # W3Calculator (S-Score)
│   └── w4_projector.py              # W4Projector (Resonance)
│
├── esde_cli_live.py                 # [Phase 8-9] CLI Entry Point
├── esde_sensor.py                   # Semantic Operators v1.1.0 (Legacy)
├── esde_sensor_v2_modular.py        # [Phase 8] Sensor V2 Facade
├── esde-engine-v532.py              # Runtime Engine v5.3.2
├── resolve_unknown_queue_7bplus_v534_final.py  # Phase 7B+ CLI
├── generate_synapses_v2_1.py        # Synapse generator
├── esde_glossary_pipeline_v5_1.py   # Glossary pipeline
├── esde_meta_auditor.py             # Phase 7D Meta-Auditor
│
├── # [Phase 8-4] Stability Audit
├── esde_stability_audit.py          # Stability audit CLI
├── mode_a_quick_test.py             # Mode A drift test
├── audit_corpus.jsonl               # 100 test sentences (5 categories)
├── test_phase83_audit.py            # Phase 8-3 audit tests
│
├── # [Phase 8-5] Memory Tests
├── test_phase85_memory.py           # Memory math unit tests
├── test_phase85_integration.py      # E2E integration test
│
├── # [Phase 8-6] Ledger Tests
├── test_phase86_ledger.py           # Hash chain validation tests
│
├── # [Phase 8-7] Index Tests
├── test_phase87_index.py            # Parity, Rigidity, QueryAPI tests
│
├── # [Phase 8-8] Pipeline Tests
├── test_phase88_pipeline.py         # Feedback Loop, Modulator tests
│
├── # [Phase 8-9] Monitor Tests
├── test_phase89_monitor.py          # Monitor, Long-Run tests
│
├── # [Phase 9-4] W4 Tests
├── test_phase94_w4.py               # W4 Projector integration tests
│
└── data/                            # Runtime data directory
    │
    ├── audit_runs/                  # [Phase 8-4/8-5] Audit logs
    │   ├── mode_a_runs.jsonl        # Mode A raw logs
    │   ├── mode_b_runs.jsonl        # Mode B raw logs
    │   ├── mode_a_quick_runs.jsonl  # Quick test logs
    │   ├── mode_a_quick_drift_report.json  # Drift A/B/C report
    │   ├── phase84_stability_report.json   # Stability report
    │   └── phase85_integration_report.json # Integration report
    │
    ├── # [Phase 8-6] Persistent Semantic Ledger
    ├── semantic_ledger.jsonl        # Hash Chain (append-only, tamper-evident)
    │
    ├── # Phase 7A: Unknown Token Queue
    ├── unknown_queue.jsonl
    │
    ├── # Phase 7B+ (v5.3.4): Aggregate-Level Resolution
    ├── unknown_queue_state_7bplus.json
    ├── unknown_queue_7bplus.jsonl
    ├── evidence_ledger_7bplus.jsonl
    │
    ├── # Phase 7C/7C': Audit
    ├── audit_log_7c.jsonl
    ├── audit_votes_7cprime.jsonl
    ├── audit_drift_7cprime.jsonl
    │
    ├── # Phase 7D: Meta-Audit
    ├── audit_rules_review_7d.json
    │
    ├── # Patch Outputs (Human Review Required)
    ├── patch_alias_add.jsonl
    ├── patch_synapse_add.jsonl
    ├── patch_stopword_add.jsonl
    ├── patch_molecule_add.jsonl
    │
    ├── # [Phase 9] Statistics Data
    ├── stats/
    │   ├── w1_global.json            # W1 global statistics
    │   ├── w2_records.jsonl          # W2 condition-sliced records
    │   ├── w2_conditions.jsonl       # W2 condition registry
    │   ├── w3_candidates/            # W3 axis candidate outputs
    │   └── w4_projections/           # W4 per-article resonance vectors
    │
    └── cache/                       # Search cache directory
```

---

## 4. Binary Emergence Model

### 4.1 Two Information Sources

```
A. Prior Structure (経験則)
    - 326 concept definitions
    - 163 symmetric pair relationships
    - 10 axes × 48 levels
    - Structural knowledge: "war and peace should be symmetric"

B. Accumulated Data (蓄積データ)
    - Occurrence frequency (count)
    - Linkage weights (w_ij)
    - Temporal patterns

Emergence:
    (Prior Structure ↔ Accumulated Data) ⇒ Introspection
```

### 4.2 Ternary Emergence

Time enters as the third term:

```
(Prior Structure ↔ Accumulated Data) ↔ Time ⇒ Deep Introspection

1st observation: Time = 0, prior vs observation only
nth observation: Time becomes meaningful, ternary complete
```

### 4.3 Weight Calculation

```python
# Binary Composite Weight
weight[i] = α_t × data_weight[i] + (1 - α_t) × structure_weight[i]

# Composite Coefficient
α_t = min(0.7, n / 100)
    # n = 0:   α_t = 0.0 (prior only)
    # n = 100: α_t = 0.7 (data dominant, capped)
```

---

## 5. Observation Model

### 5.1 Principle

> Observation Principle: Record at maximum resolution

LLM Responsibility:
- Identify concept
- Specify axis and level
- Specify sub-level (when possible)
- Record evidence text

### 5.2 Observation vs Analysis Granularity

```
Storage: Save everything at maximum resolution
Analysis: Select granularity based on data volume

Phase 1 (~100 observations):    concept_id only
Phase 2 (100~1000):             concept_id + axis
Phase 3 (1000~):                concept_id + sub_level + axis + level
```

Benefits:
- Data doesn't decay (can analyze in detail later)
- No re-collection needed
- Scalable
- No unrealistic precision demands early on

---

## 6. Semantic Operators (Molecules)

### 6.1 Purpose

Structure semantic atoms into molecules via operators.

> Operators do NOT "determine" meaning. They only provide structure.  
> Do not generate winners.  
> Preserve volatility.

### 6.2 Operator List (v0.3)

| Operator | Name | Description |
|----------|------|-------------|
| × | Connection | A × B (connect two semantic units) |
| ▷ | Action | A ▷ B (A acts on B) |
| → | Transition | A → B (state/meaning change) |
| ⊕ | Juxtaposition | A ⊕ B (simultaneous presentation) |
| \| | Condition | A \| B (A under condition B) |
| ◯ | Target | A × ◯ (target of A) |
| ↺ | Recursion | A ↺ A (self-reference) |
| 〈〉 | Hierarchy | 〈A × B〉 (internal structure/scope) |
| ≡ | Equivalence | A ≡ B (theoretical identity) |
| ≃ | Practical Equivalence | A ≃ B (equivalence within ε) |
| ¬ | Negation | ¬A (meaning inversion) |
| ⇒ | Emergence | A ⇒ B (unspecified direction) |
| ⇒+ | Creative Emergence | New meaning generation |
| -\|> | Destructive Emergence | Structure reset/reboot |

### 6.3 Expression Capacity

```
Semantic Atoms: 326
Axes × Levels: 48 patterns
Base Patterns: ~15,000

+ Semantic Operators
= Billions of expressions (Molecules)
```

### 6.4 Sensor I/O Example (Current Legacy Behavior)

**Input:**
```
"I cannot forgive you."
```

**Processing:**
```
Step 1: Trigger match → "forgive" hits EMO.forgiveness
Step 2: Negation detected → "cannot"
Step 3: Entity detected → "you" = ◯ (other)
Step 4: Formula = ¬(EMO.forgiveness × ◯)
Step 5: Resolve ¬EMO.forgiveness → NEGATION_MAP → SOC.refuse
```

**Output:**
```json
{
  "concept_id": "SOC.refuse",
  "axis": "interconnection",
  "level": "independent",
  "evidence": "Formula: ¬(EMO.forgiveness × ◯)",
  "_formula": {"type": "EXPR", "op": "¬", "args": [...]},
  "_extracted": "EMO.forgiveness"
}
```

**Note:** This is legacy behavior. Future integration should use Synapse vector lookup before trigger matching.

---

## 7. Error (ε) Measurement

### 7.1 Symmetry Error

```python
ε_sym[i] = | weight[i] - weight[symmetric(i)] |
ε_sym_total = Σ ε_sym[i] / num_pairs
```

### 7.2 Linkage Error

```python
ε_link = Σ_{(i,j)} w_ij × || x[i,:] - x[j,:] ||²
```

### 7.3 Total Error

```python
ε_total = λ_sym × ε_sym_total + λ_link × ε_link
```

### 7.4 Interpretation

ε is not "error" but "evidence that structure is alive".

```
ε = 0:  Perfect equilibrium (rigid, dead)
ε > 0:  Dynamic equilibrium (alive)
ε >> threshold: Imbalance (sign of Reboot)
```

---

## 8. Audit Pipeline (Phase 7)

### 8.1 Philosophy: Volatility-First

> "Is this understood correctly?"  
> → "What does 'correct' even mean?"  
> → "Correctness always fluctuates"  
> → **Volatility Detection**

### 8.2 Pipeline Overview

```
Phase 7A:  Unknown Queue Collection
Phase 7B:  Aggregation & Priority
Phase 7B+: Evidence Collection (Web Search)
Phase 7C:  Structural Audit
Phase 7C': LLM Semantic Audit
Phase 7D:  Meta-Auditor (Rule Calibration)
```

### 8.3 Phase 7B+: Evidence Collection

External evidence collection for unknown tokens.

**v5.3.5 Update**: SearXNG → MultiSourceProvider

| Source | Type | Use Case |
|--------|------|----------|
| Free Dictionary API | Dictionary | Primary definitions |
| Wikipedia API | Encyclopedia | Proper nouns, concepts |
| Datamuse API | WordNet | Related words |
| Urban Dictionary | Slang | Informal language |
| DuckDuckGo | Web Search | Final fallback |

#### Performance Comparison

| Metric | v5.3.4 (SearXNG) | v5.3.5 (MultiSource) |
|--------|------------------|----------------------|
| Success Rate | 89% | **100%** |
| Quarantine | 45% | **6%** |
| Candidate | 20% | **49%** |
| Avg Volatility | 0.407 | **0.177** |

### 8.4 Four Hypothesis Routes

| Route | Name | Description |
|-------|------|-------------|
| A | Typo | Edit distance, spell check |
| B | Entity | Proper noun, searchable term |
| C | Novel | New concept, requires molecule |
| D | Noise | Discard or defer |

### 8.5 Classification Output

| Status | Symbol | Volatility | Action |
|--------|--------|------------|--------|
| Candidate | ○ | < 0.3 | Proceed |
| Deferred | ◐ | 0.3 - 0.6 | More observation |
| Quarantine | ● | > 0.6 | Human review |

### 8.6 Strict Invariants

1. **Winner MUST remain null** in 7B+ outputs
2. **Configuration is SINGLE SOURCE OF TRUTH**: Use `config.py` only
3. **Ledger is append-only**: Never rewrite past entries
4. **Patches are emitted but never auto-applied**: Human review mandatory
5. **State controls flow**: `aggregate_state.py` is the only place for reprocess/observation logic
6. **Determinism**: Resolver must be deterministic with MockSearchProvider

---

## 9. Development Environment

### Hardware

| Component | Specification |
|-----------|---------------|
| Client | MacBook Pro M3 / 8GB / 512GB |
| Host CPU | AMD Ryzen Threadripper PRO 7965WX 24-Cores (48 logical) |
| GPU | NVIDIA GeForce RTX 5090 x2 (32GB VRAM each) |
| Driver | 570.158.01 |
| CUDA | 12.8 |

### Software

| Component | Version |
|-----------|---------|
| Docker | 27.5.1 |
| Docker Compose | v2.38.2 |
| NVIDIA Container Toolkit | 1.17.8 |
| Tailscale | 1.92.1 |
| Python | 3.12.3 |
| TensorRT-LLM | 1.0.0rc2 |

### LLM Server

```bash
docker run -d --name qwq_llm --ipc=host --shm-size=2g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --gpus=all -p 8001:8001 \
  -v $HOME/models:/models \
  -v $HOME/Aruism-AI-Local/docker/engine:/engines \
  nvcr.io/nvidia/tensorrt-llm/release:1.0.0rc2 \
  trtllm-serve serve /models/qwq/engines/qwq32b_tp2_long32k_existing \
    --tokenizer /engines/qwen/QwQ-32B-AWQ \
    --max_batch_size 16 \
    --max_num_tokens 32768 \
    --host 0.0.0.0 \
    --port 8001 \
    --tp_size 2
```

Configuration:
```python
HOST = "http://100.107.6.119:8001/v1"
MODEL = "qwq32b_tp2_long32k_existing"
```

---

## 10. Parameters

### Core Configuration (config.py)

| Parameter | Value | Description |
|-----------|-------|-------------|
| VERSION | 5.3.5 | Engine version |
| COMPETE_TH | 0.15 | Competing hypothesis threshold |
| VOL_LOW_TH | 0.3 | Low volatility threshold |
| VOL_HIGH_TH | 0.6 | High volatility threshold |
| UNKNOWN_MARGIN_TH | 0.20 | Margin for abstain decision |
| UNKNOWN_ENTROPY_TH | 0.90 | Entropy for abstain decision |

### Dynamics

| Parameter | Value | Description |
|-----------|-------|-------------|
| α | 0.7 | Linkage weight frequency coefficient |
| τ | 86400 | Linkage weight decay constant (seconds) |
| γ | 0.1 | Observation injection rate |
| ε_0 | 0.01 | Initial state base value |
| δ | 0.005 | Initial state hierarchy bonus |
| λ_sym | 1.0 | Symmetry constraint weight |
| λ_link | 1.0 | Linkage constraint weight |

---

## 11. Code Skeleton (Key Classes)

### config.py

```python
VERSION: str
COMPETE_TH: float
ROUTE_A_MIN_SCORE: float
VOL_LOW_TH: float
VOL_HIGH_TH: float
SYNAPSE_FILE: str
GLOSSARY_FILE: str
QUEUE_FILE_PATH: str
```

### hypothesis.py

```python
class HypothesisResult:
    score: float
    reason: str  # mandatory
    volatility: float
    signals: Dict

class EvaluationReport:
    winner: None  # always null
    competing_routes: List[str]
    global_volatility: float

def evaluate_all_hypotheses(...) -> EvaluationReport: ...
def compute_global_volatility(...) -> float: ...
def determine_status(volatility: float) -> str: ...
```

### aggregate_state.py

```python
class AggregateStateManager:
    def compute_aggregate_key(token_norm, pos, route_set) -> str: ...
    def should_process(aggregate_key) -> bool: ...
    def upsert_observation(aggregate_key, ...) -> None: ...
    def mark_finalized(aggregate_key, reason) -> None: ...
```

### online.py (v5.3.5)

```python
class SearchProvider:
    def search(query: str, max_results: int) -> List[Dict]: ...

class MultiSourceProvider(SearchProvider):
    providers = [
        FreeDictionaryAPI(),
        WikipediaAPI(),
        DatamuseAPI(),
        UrbanDictionaryAPI(),
        DuckDuckGoAPI(),
    ]
    def search(query: str, max_results: int) -> List[Dict]: ...
```

### esde_sensor.py

```python
class ESDESensor:
    OP_NOT = "¬"
    OP_CONN = "×"
    OP_JUXT = "⊕"
    
    def analyze(text: str) -> Dict: ...
    def _build_formula(text, concept) -> Dict: ...
    def _resolve_formula(formula) -> str: ...
```

---

## 12. Reproduction Commands

```bash
# 1. Generate queue data
python esde-engine-v532.py

# 2. Process queue (Phase 7B+)
python resolve_unknown_queue_7bplus_v534_final.py --limit 50

# 3. Check output
cat ./data/unknown_queue_7bplus.jsonl | tail

# 4. Run 7C audit
python resolve_unknown_queue_7bplus_v534_final.py --mode 7c

# 5. Run 7C' audit
python resolve_unknown_queue_7bplus_v534_final.py --mode 7cprime
```

---

## 13. Acceptance Criteria

- Running 7B+ twice without new queue input must produce **0 processed** (State idempotency)
- When queue has 3 identical records, aggregation must output **1 aggregate with count=3**
- Candidate/Defer/Quarantine thresholds must strictly match `config.py`

---

## 14. Anti-Patterns (DO NOT)

- **DO NOT** introduce `web.run` or external browsing inside the runtime engine
- **DO NOT** "fix" ambiguity by choosing the most likely meaning (preserve volatility)
- **DO NOT** add new heuristics without corresponding audit signals
- **DO NOT** add new modules/files unless explicitly instructed
- **DO NOT** hardcode thresholds/paths outside `config.py`

---

## 15. Current Status & Next Steps

### 15.1 Completed Phases

| Phase | Status | Output |
|-------|--------|--------|
| 7A | ✅ Complete | unknown_queue.jsonl |
| 7B | ✅ Complete | Aggregation logic |
| 7B+ | ✅ Complete | MultiSourceProvider (v5.3.5, 100% success) |
| 7C | ✅ Complete | Structural audit |
| 7C' | ✅ Complete | LLM semantic audit |
| 7D | ✅ Complete | Meta-auditor, rule calibration |
| **8** | **✅ Complete** | **Introspective Engine v1** |
| **9-0** | **✅ Complete** | **W0 ContentGateway** |
| **9-1** | **✅ Complete** | **W1 Global Statistics** |
| **9-2** | **✅ Complete** | **W2 Conditional Statistics** |
| **9-3** | **✅ Complete** | **W3 Axis Candidates (S-Score)** |

### 15.2 Sensor V2 Implementation (Phase 8)

**File:** `esde_sensor_v2.py`

**Key Features:**
- Primary: Synapse vector lookup (raw_score from generate_synapses_v2_1.py)
- Fallback: Legacy trigger matching (Hybrid mode)
- ALLOWED_POS includes 's' (Satellite Adjective)
- Deterministic output (sorted by score DESC, concept_id ASC)
- Full audit trail (evidence, config_snapshot, determinism_hash)

**Config Requirements (add to config.py):**
```python
# Phase 8: Sensor V2
SENSOR_TOP_K = 5
SENSOR_MAX_SYNSETS_PER_TOKEN = 3
STRICT_SYNAPSE_ONLY = False
SENSOR_ALLOWED_POS = {'n', 'v', 'a', 'r', 's'}

# CRITICAL: Update existing ALLOWED_POS
ALLOWED_POS = {'n', 'v', 'a', 'r', 's'}  # Added 's'
```

**GPT Audit Compliance:**
- ✅ Config values injected explicitly
- ✅ ALLOWED_POS includes 's'
- ✅ determinism_hash includes config_snapshot
- ✅ Fallback to legacy triggers (Hybrid mode)

### 15.3 Modular Architecture (GPT Recommended)

Sensor V2 has been refactored into modular components for maintainability and testability.

**Directory Structure:**
```
esde/
├── esde_sensor_v2_modular.py     # Facade (thin orchestration)
└── sensor/
    ├── __init__.py               # Package exports
    ├── loader_synapse.py         # SynapseLoader
    ├── extract_synset.py         # SynsetExtractor  
    ├── rank_candidates.py        # CandidateRanker
    ├── legacy_trigger.py         # LegacyTriggerMatcher (v1 fallback)
    ├── audit_trace.py            # AuditTracer (counters/hash/evidence)
    ├── molecule_validator.py     # Phase 8-2 Validator
    └── molecule_generator.py     # Phase 8-2 Generator
```

**Module Responsibilities:**

| Module | Class | Responsibility |
|--------|-------|----------------|
| loader_synapse.py | SynapseLoader | JSON load, singleton, file hash |
| extract_synset.py | SynsetExtractor | WordNet synset extraction |
| rank_candidates.py | CandidateRanker | Score aggregation, deterministic sort |
| legacy_trigger.py | LegacyTriggerMatcher | v1 trigger fallback |
| audit_trace.py | AuditTracer | Counters, hash, evidence formatting |

### 15.4 Phase 8-2: Molecule Generation & Validation

**Spec v8.2.1** (GPT Audit Passed)

#### MoleculeValidator

Validates generated molecules for integrity.

| Check | Description | GPT Audit |
|-------|-------------|-----------|
| Atom Integrity | 2-tier: Sensor candidates + Glossary subset | ✅ v8.2.1 |
| Operator Valid | v0.3 operators (×, ▷, →, ⊕, ¬, etc.) | ✅ |
| Syntax Check | Bracket matching | ✅ |
| Coordinate Valid | axis/level exist in Glossary | ✅ |
| Evidence Linkage | span range valid + text_ref == text[span] | ✅ v8.2.1 |
| Coverage Policy | High confidence + null coords = warning | ✅ |

**Key Methods:**
```python
validator = MoleculeValidator(
    glossary=glossary,
    sensor_candidates=["EMO.love", "ACT.give"],
    allow_glossary_atoms=True  # v8.2.1: Allow Glossary subset atoms
)
result = validator.validate(molecule, original_text)
# result.valid, result.errors, result.warnings
```

#### MoleculeGenerator

LLM-based semantic molecule generation.

**Design Principles:**
- **Never Guess**: axis/level = null if uncertain
- **No New Atoms**: Only use atoms from candidates
- **Glossary Subset**: Only pass relevant definitions to LLM
- **Retry Policy**: Max 2 retries, then Abstain

**Prompting Policy:**
```
1. ONLY use atoms from the provided candidate list
2. If uncertain about axis/level, set them to null
3. text_ref must be EXACT substring from original text
4. span must be exact character positions [start, end)
```

**Data Schema (ActiveAtom) - v8.3 Canonical:**

> **IMPORTANT**: `coordinates` nesting is **DEPRECATED** as of v8.3.
> Canonical schema uses flat structure (axis/level at top level).

```json
{
  "active_atoms": [
    {
      "id": "aa_1",
      "atom": "EMO.love",
      "axis": "interconnection",
      "level": "resonant",
      "text_ref": "deeply in love",
      "span": [10, 24]
    }
  ],
  "formula": "aa_1 ▷ aa_2",
  "meta": {
    "generator": "MoleculeGeneratorLive",
    "generator_version": "v8.3",
    "validator_status": "ok",
    "coordinate_coercions": [],
    "span_warnings": [],
    "timestamp": "2026-01-20T..."
  }
}
```

**Prohibited Fields (DEPRECATED):**
- `coordinates` (nested structure) - Use flat `axis`, `level` instead
- `confidence` (selection criteria, not observation data)

**Schema Contract (INV-MOL-001):**
- Canonical schema is defined by v8.3 live generator output and the ledger write contract
- `coordinates` nesting must not appear in any data that crosses the Phase 8 boundary
- axis/level can be `null` (Never Guess principle preserved)

### 15.5 Test Results

**Sensor V2 (Modular):**
```
[Input] I love you
  Engine: v2_synapse
  Candidates: 1 (EMO.love via love.n.01)
  synsets_checked=3, with_edges=3 ✅

[Input] apprenticed to a master
  Engine: v2_synapse
  Candidates: 1 (ABS.bound via apprenticed.s.01) ← GPT Audit Key Test
  synsets_checked=5, with_edges=1 ✅

[Input] I cannot forgive you
  Engine: v2_synapse+v1_fallback
  Candidates: 0 (forgive.v not in synapses)
  synsets_checked=2, with_edges=0 ✅ Expected
```

**MoleculeValidator:**
```
Valid: True
Errors: []
Coordinate completeness: 1.0 ✅
```

**MoleculeGenerator (Mock):**
```
Success: True
Abstained: False
axis: null, level: null (Never Guess) ✅
validator_status: pass ✅
```

### 15.6 Phase 8-3: Live LLM Integration

**Implementation:** `sensor/molecule_generator_live.py`

#### 15.6.1 Spec v8.3 Guardrails

| Guardrail | Implementation | Status |
|-----------|----------------|--------|
| Strict Output Contract | "Return ONLY JSON" in system prompt | ✅ |
| Zero Chatter | QwQ `<think>` tag removal | ✅ |
| Fail-Closed Parsing | No fuzzy logic, markdown removal only | ✅ |
| System-Calculated Span | SpanCalculator (token proximity) | ✅ |
| Coordinate Coercion | CoordinateCoercer with logging | ✅ |
| Empty Check | Skip LLM if no candidates | ✅ |

#### 15.6.1.1 v8.3.1 Update: Legacy Dependency Removal

**Issue:** `molecule_generator_live.py` imported from legacy `molecule_validator.py`

**Fix:**
```python
# Before (legacy dependency)
from .molecule_validator import MoleculeValidator, ValidationResult, GlossaryValidator

# After (canonical, self-contained)
from .glossary_validator import GlossaryValidator
# ValidationResult defined locally in molecule_generator_live.py
```

**Principle (INV-MOL-LEG-001):**
- Canonical modules must not import from legacy modules
- Legacy modules are isolated in `sensor/legacy/`

#### 15.6.2 Live Test Results (QwQ-32B)

```
=== I love you ===
Success: True
Molecule: EMO.love@interconnection:catalytic
Formula: aa_1
Span: [2, 6] ✅

=== The law requires obedience ===
Success: True
Molecule: EMO.respect@value_generation:ethical
Formula: aa_1
Span: [17, 26] ✅

=== I cannot forgive you ===
Candidates: 0
LLM Called: False ✅ (Empty Check working)
```

#### 15.6.3 GPT Audit 4-Case Tests

| Test | Description | Result |
|------|-------------|--------|
| A | Same text_ref multiple occurrences | ✅ Token proximity |
| B | text_ref spelling variation | ✅ span=null + warning |
| C | Coordinate mismatch (Glossary外) | ✅ Coercion logged |
| D | JSON pollution (`<think>`, markdown) | ✅ Parsed correctly |

### 15.7 Phase 8-4: Semantic Stability & Drift Audit

**Implementation:** `esde_stability_audit.py`

#### 15.7.1 Audit Scope

| Item | In Scope | Out of Scope |
|------|----------|--------------|
| Stability verification | ✅ | Ledger implementation |
| Drift measurement | ✅ | Operator expansion |
| Pollution blocking | ✅ | Glossary update |

#### 15.7.2 Test Corpus

| Category | Count | Purpose |
|----------|-------|---------|
| human_emotion | 20 | Emotion word stability |
| human_logic | 20 | Logic structure reproduction |
| ai_text | 20 | AI-generated text immunity |
| noise | 20 | Hallucination test |
| edge_cases | 20 | Boundary value test |

#### 15.7.3 Audit Results (3500 runs)

**Mode A (temperature=0):**

| Category | Success Rate | Drift | Judgment |
|----------|--------------|-------|----------|
| edge_cases | 95% | 15.8% | ⚠️ |
| human_emotion | 72% | 5.0% | ✅ |
| ai_text | 75% | 11.7% | ⚠️ |
| human_logic | 70% | 10.7% | ⚠️ |
| noise | 15% | 0% | ✅ |

**Mode B (temperature=0.1):**

| Category | Success Rate | Drift | Judgment |
|----------|--------------|-------|----------|
| edge_cases | 95% | 17.4% | ⚠️ |
| human_emotion | 90% | 8.4% | ✅ |
| ai_text | 75% | 10.3% | ⚠️ |
| human_logic | 70% | 20.9% | ⚠️ |
| noise | 14% | 1.2% | ✅ |

**Immunity Metrics:**

| Metric | Count | Rate | Judgment |
|--------|-------|------|----------|
| Validator Blocks | 8 | 0.23% | ✅ Very Low |
| Coordinate Coercions | 5 | 0.14% | ✅ Very Low |
| Pollution Blocked | True | - | ✅ |

#### 15.7.4 Mode A Quick Test (Drift切り分け)

```
Temperature: 0.0
Samples: 10 items × 10 runs = 100 runs

Results:
  Drift-A (Atom): 0.0000 ✅ Perfect
  Drift-B (Coord): 0.0000 ✅ Perfect
  Drift-C (Formula): 0.0000 ✅ Perfect
  
Conclusion: temp=0 achieves deterministic output at Atom/Coord level.
Residual drift in full audit is Formula expression variation only.
```

#### 15.7.5 FormulaValidator Fix

**Issue:** `¬¬aa_1` (double negation) was blocked as unknown operator.

**Root Cause:** `extract_operators()` treated `¬¬` as single token.

**Fix:** Character-by-character parsing for consecutive operators.

```python
# Before: ['¬¬'] → Unknown operator error
# After:  ['¬', '¬'] → Valid (two negations)
```

### 15.8 Phase 8-5: Semantic Memory (Ephemeral Ledger)

**Implementation:** `ledger/ephemeral_ledger.py`

#### 15.8.1 Memory Constitution (記憶の憲法)

Phase 8-5は「記憶してもシステムが破綻しない条件」を確立するフェーズ。

**禁止事項 (Strict Prohibitions):**

| 項目 | 理由 |
|------|------|
| NO Persistence | 永続化は8-6以降 |
| NO Hash Chain | 改竄防止は時期尚早 |
| NO Semantic Logic | 演算子統合は対象外 |
| NO Emergence Classification | 創発判定は対象外 |

#### 15.8.2 Memory Math

**Decay (自然減衰):**
```
w = w_prev × exp(-dt / τ)
```

**Reinforcement (観測強化):**
```
w = w + α × (1 - w)
```
- α (Learning Rate) = 0.2

**Oblivion (忘却):**
```
if w < ε: purge
```
- ε (Threshold) = 0.01

#### 15.8.3 Tau Policy

| Temporal Level | τ (sec) | Human |
|----------------|---------|-------|
| permanence | 86400 | 24h |
| continuation | 3600 | 1h |
| establishment | 3600 | 1h |
| transformation | 1800 | 30m |
| indication | 300 | 5m |
| emergence | 60 | 1m |
| **Default** | **300** | **5m** |

#### 15.8.4 Fingerprint (同一性判定)

```
Key = SHA256( Sorted(AtomIDs) + "::" + OperatorType )
```

- `A ▷ B` と `A → B` は別の記憶
- 自然淘汰（Weight競争）に任せる

#### 15.8.5 Conflict Handling (矛盾の扱い)

**方針: 共存 (Co-existence)**

- Love と Hate は別Entryとして共存
- 相殺（引き算）は行わない
- Weight競争で自然淘汰

#### 15.8.6 Integration Test Results

**E2E Test (14 inputs):**

| Metric | Result |
|--------|--------|
| Success | 12 (85.7%) |
| Abstained | 2 (Noise) |
| Blocked | 0 (0%) |
| Coercions | 0 (0%) |
| Ledger Entries | 9 |
| Observations | 12 |

**Audit Tests:**

| Test | Result | Detail |
|------|--------|--------|
| Conflict Coexistence | ✅ PASS | Love + Hate 共存 |
| Reinforcement Stability | ✅ PASS | Weight ≤ 1.0 |
| Retention by Tau | ✅ PASS | 10分後9件残存 |
| No-Pollution | ✅ PASS | システム安定 |

**Reinforcement Observation:**
```
INT008: create (w=1.00)
INT009: reinforce (w=0.99)
INT010: reinforce (w=0.98)
```

### 15.9 Phase 8-6: Persistent Semantic Ledger

**Theme:** Semantic Time Crystallization (意味時間の結晶化)

Phase 8-6は、ESDEに**「不可逆な時間」**と**「正史（History）」**を物理的に実装する。
Phase 8-5の揮発性メモリを、**Hash Chain（ハッシュチェーン）** 技術を用いてディスクに刻み、
システムの再起動後も「自己の文脈」を維持可能にする。

#### 15.9.1 Spec v8.6.1 (GPT Audit Approved)

**GPT監査修正（4点）反映:**

| # | 要件 | 対応 |
|---|------|------|
| 1 | validate()は行文字列をcanonicalとして扱う | ✅ JSON再dumpなし |
| 2 | 最終行破損はsalvageなしで停止 | ✅ IntegrityError |
| 3 | event_idからmeta除外（意味同一性） | ✅ meta除外 |
| 4 | rehydrationはledger replay | ✅ 順次適用 |

#### 15.9.2 Data Schema (JSONL)

```json
{
  "v": 1,
  "ledger_id": "esde-semantic-ledger",
  "seq": 1024,
  "ts": "2026-01-14T10:00:00.000000Z",
  "event_type": "molecule.observe",
  "direction": "=>+",
  "data": {
    "source_text": "I love you",
    "molecule": { ... },
    "weight": 0.488,
    "audit": { "validator_pass": true }
  },
  "hash": {
    "algo": "sha256",
    "prev": "a1b2c3d4...",
    "event_id": "f9g0h1i2...",
    "self": "e5f6g7h8..."
  },
  "meta": {
    "engine_version": "5.3.6-P8.6",
    "actor": "SensorV2"
  }
}
```

#### 15.9.3 Hash Chain Logic

**Event ID (意味同一性):**
```
Target = {v, ledger_id, event_type, direction, data}
event_id = SHA256(Canonical(Target))
```
- `meta`を除外: 同じ意味イベントは同じID

**Self Hash (歴史連鎖):**
```
Target = {v, ledger_id, seq, ts, event_type, direction, data, meta, hash:{algo,prev,event_id}}
self = SHA256(prev + "\n" + Canonical(Target))
```

**Genesis Block:**
```
seq: 0
prev: "0" × 64
data: {"message": "Aru - There is"}
```

#### 15.9.4 Validation Tests (T861-T865)

| ID | テスト | 検証内容 |
|----|--------|----------|
| T861 | chain_validates | self hashの再計算一致 |
| T862 | prev_linkage | prev→self連鎖 |
| T863 | truncation | JSONパースエラー検知 |
| T864 | monotonic_seq | seq欠番なし |
| T865 | header_match | v, ledger_id, algo定数 |

#### 15.9.5 Test Results

```
============================================================
ESDE Phase 8-6: Ledger Test Suite
============================================================

Genesis Creation: ✅ PASS
Append and Validate: ✅ PASS
Molecule Observe: ✅ PASS
T861: Chain Validates: ✅ PASS
T862: Tamper Detected: ✅ PASS
T863: Truncation Detected: ✅ PASS
T864: Reorder Detected: ✅ PASS
T865: Header Mismatch: ✅ PASS
Rehydration: ✅ PASS
Sleep Decay: ✅ PASS
Direction Values: ✅ PASS
Event ID Excludes Meta: ✅ PASS
Conflict Coexistence: ✅ PASS

Results: 13/13 passed
✅ ALL TESTS PASSED - Phase 8-6 Ready
```

#### 15.9.6 New Files

| File | Description |
|------|-------------|
| `ledger/canonical.py` | JSON正規化（sort_keys, separators） |
| `ledger/chain_crypto.py` | Hash計算（event_id, self hash） |
| `ledger/persistent_ledger.py` | PersistentLedgerクラス |
| `test_phase86_ledger.py` | 監査テスト（T861-T865） |

### 15.10 Phase 8-7: Semantic Index (L2)

**Theme:** Projection of Understanding (理解の射影)

Phase 8-7は、L1 (Immutable Ledger) の上に**L2: Semantic Index**を構築する。
「検索（Index）」「硬直（Rigidity）」「傾向（Direction Balance）」を即答できるインメモリ構造。

> **Aruism Philosophy:**
> - **L1 (Ledger):** 「ある（Aru）」。変更不可能な事実の羅列。
> - **L2 (Index):** 「わかる（Understanding）」。事実を特定の視点で再構成した射影。

#### 15.10.1 Architecture (Dual Layer System)

```
[ Query API ] -> ユーザー/システムへの回答
      ^
      | (Read)
[ L2: Semantic Index + Rigidity Signals ] (Mutable / In-Memory)
      ^
      | (Project: Rebuild / Incremental)
[ L1: Immutable Ledger ] (Immutable / JSONL)
```

**絶対ルール:**
1. L2はL1の射影（独自情報を持たない）
2. L1不変（L2構築がL1を書き換えない）
3. Parity Consistency（rebuild結果 == 逐次更新結果）

#### 15.10.2 Rigidity Calculation

```
R = N_mode / N_total
```

- `N_total`: そのAtomが観測された総回数
- `N_mode`: 最頻出formula_signatureの出現回数

| R値 | 分類 | 解釈 |
|-----|------|------|
| ≥0.9 | crystallized | 結晶化（完全に固定） |
| ≥0.7 | rigid | 硬直 |
| ≥0.4 | stable | 安定 |
| ≥0.2 | fluid | 流動的 |
| <0.2 | volatile | 揮発的（探索中） |

#### 15.10.3 GPT監査v8.7.1対応

| 項目 | 実装 |
|------|------|
| formula_signature定義 | formula → atoms/ops連結 → `__unknown__` |
| L2肥大化防止 | `deque(maxlen=10000)` |
| window引数 | `get_frequency(atom_id, window=1000)` 等 |

#### 15.10.4 Query API

```python
from index import QueryAPI

api = QueryAPI(ledger)

# 硬直度
api.get_rigidity("EMO.love", window=1000)

# 出現頻度
api.get_frequency("EMO.love", window=1000)

# 方向性バランス
api.get_recent_directions(limit=1000)

# Atom詳細
api.get_atom_info("EMO.love")

# 共起
api.get_cooccurrence("EMO.love", "ACT.create")
```

#### 15.10.5 Test Results

```
============================================================
ESDE Phase 8-7: Index Test Suite
============================================================

Index Creation: ✅ PASS
Index Update: ✅ PASS
Formula Signature Extraction: ✅ PASS
Parity: Rebuild vs Incremental: ✅ PASS
Rigidity: Constant Formula (R=1.0): ✅ PASS
Rigidity: Varying Formula (R<1.0): ✅ PASS
Rigidity: Mixed Pattern (R=0.7): ✅ PASS
No Side Effect on Ledger: ✅ PASS
QueryAPI Basic: ✅ PASS
QueryAPI AtomInfo: ✅ PASS
QueryAPI Cooccurrence: ✅ PASS
QueryAPI Window: ✅ PASS
Direction Balance: ✅ PASS

Results: 13/13 passed
✅ ALL TESTS PASSED - Phase 8-7 Ready
```

#### 15.10.6 New Files

| File | Description |
|------|-------------|
| `index/__init__.py` | パッケージエクスポート |
| `index/semantic_index.py` | L2インメモリ構造 |
| `index/projector.py` | L1→L2投影 |
| `index/rigidity.py` | 硬直度計算 |
| `index/query_api.py` | 外部API |
| `test_phase87_index.py` | Parity/Rigidityテスト |

### 15.11 Next Steps

| Priority | Task | Status |
|----------|------|--------|
| 1 | ~~Sensor V2 implementation~~ | ✅ Complete |
| 2 | ~~Modular refactoring~~ | ✅ Complete |
| 3 | ~~Phase 8-2 Validator/Generator~~ | ✅ Complete |
| 4 | ~~Phase 8-3 Live LLM Integration~~ | ✅ Complete |
| 5 | ~~Phase 8-4 Stability Audit~~ | ✅ **PASS** |
| 6 | ~~Phase 8-5 Ephemeral Ledger~~ | ✅ **PASS** |
| 7 | ~~Phase 8-6 Persistent Ledger~~ | ✅ **PASS** |
| 8 | ~~Phase 8-7 Semantic Index~~ | ✅ **PASS** |
| 9 | ~~Phase 8-8 Feedback Loop~~ | ✅ **PASS** |
| 10 | ~~Phase 8-9 Monitor & Long-Run~~ | ✅ **PASS** |
| 11 | ~~Phase 8-10 Schema Consolidation~~ | ✅ **PASS** |
| 12 | ~~Phase 9-0 ContentGateway~~ | ✅ **PASS** |
| 13 | ~~Phase 9-1 W1 Global Stats~~ | ✅ **PASS** |
| 14 | ~~Phase 9-2 W2 Conditional Stats~~ | ✅ **PASS** |
| 15 | ~~Phase 9-3 W3 Axis Candidates~~ | ✅ **PASS** |
| 16 | ~~Phase 9-4 W4 Structural Projection~~ | ✅ **PASS** |
| 17 | Phase 9-5 W5 Axis Confirmation | **Next** |

### 15.12 Development Workflow

```
Gemini:  Design proposal (What to do)
   ↓
GPT:     Audit (Is it valid?)
   ↓
Claude:  Implementation (How to build)
   ↓
GPT:     Deliverable audit (Did we build it right?)
   ↓
All:     Phase 9-5 Scope Decision  ← Current
```

---

## 16. Version History

| Version | Date | Changes |
|---------|------|---------|
| 0.2.1 | 2024-12 | Core Specification established |
| 5.1.0 | 2024-12 | Glossary pipeline, synapse structure |
| 5.3.2 | 2025-01 | Phase 7A-7B implementation |
| 5.3.4 | 2025-01 | SearXNG integration, 7B+ pipeline |
| 5.3.5 | 2026-01-11 | MultiSourceProvider, 100% success rate |
| 5.3.5-P8 | 2026-01-12 | Phase 8: Sensor V2 + Modular Architecture |
| 5.3.5-P8.2 | 2026-01-12 | Phase 8-2: Molecule Generator/Validator |
| 5.3.5-P8.3 | 2026-01-12 | Phase 8-3: Live LLM Integration (QwQ-32B) |
| 5.3.5-P8.4 | 2026-01-13 | Phase 8-4: Stability Audit PASS |
| 5.3.5-P8.5 | 2026-01-13 | Phase 8-5: Semantic Memory (Ephemeral Ledger) |
| 5.3.6-P8.6 | 2026-01-14 | Phase 8-6: Persistent Ledger (Hash Chain) |
| 5.3.7-P8.7 | 2026-01-14 | Phase 8-7: Semantic Index (L2 + Rigidity) |
| 5.3.8-P8.8 | 2026-01-18 | Phase 8-8: Feedback Loop |
| 5.3.9-P8.9 | 2026-01-19 | Phase 8-9: Monitor & Long-Run |
| 5.3.10-P8.10 | 2026-01-20 | Phase 8-10: Schema Consolidation |
| **5.4.0-P9.0** | **2026-01-20** | **Phase 9-0: ContentGateway (W0)** |
| **5.4.1-P9.1** | **2026-01-20** | **Phase 9-1: W1 Global Statistics** |
| **5.4.2-P9.2** | **2026-01-20** | **Phase 9-2: W2 Conditional Statistics** |
| **5.4.2-P9.3** | **2026-01-21** | **Phase 9-3: W3 Axis Candidates (S-Score)** |
| **5.4.4-P9.4** | **2026-01-21** | **Phase 9-4: W4 Structural Projection (Resonance)** |

### v5.3.5-P8 Changelog (Phase 8)

1. **Sensor-Synapse Integration**
   - Sensor V2: Synapse vector lookup as primary method
   - Fallback: Legacy trigger matching (Hybrid mode)
   - ALLOWED_POS includes 's' (Satellite Adjective)

2. **GPT Audit Compliance**
   - Full determinism_hash (64 hex)
   - Fallback counters for debugging
   - Top evidence per candidate

3. **Modular Architecture** (GPT Recommended)
   - Facade: esde_sensor_v2_modular.py
   - Components: sensor/ package (8 modules)

### v5.3.5-P8.2 Changelog (Phase 8-2)

1. **MoleculeValidator**
   - Atom Integrity: 2-tier (Sensor + Glossary subset)
   - Coordinate Valid: axis/level existence check
   - Evidence Linkage: span exact match (v8.2.1)

2. **MoleculeGenerator**
   - Never Guess: null coordinates when uncertain
   - No New Atoms: only from candidates
   - Retry Policy: max 2, then Abstain

3. **Test Results**
   - Sensor V2: All tests passed
   - Validator: coordinate_completeness=1.0
   - Generator (Mock): abstained=False, pass

### v5.3.5-P8.3 Changelog (Phase 8-3)

1. **Live LLM Integration**
   - MoleculeGeneratorLive with QwQ-32B
   - QwQ `<think>` tag removal
   - max_tokens: 16000 (for long reasoning)

2. **GPT Audit Guardrails (v8.3)**
   - Strict Output Contract (Zero Chatter)
   - Fail-Closed Parsing
   - System-Calculated Span (token proximity)
   - Coordinate Coercion with logging

3. **New Components**
   - SpanCalculator: text_ref → span
   - CoordinateCoercer: invalid → null + log
   - FormulaValidator: consecutive operators fix (¬¬)

4. **Test Results**
   - Live Test: 3/3 PASS
   - GPT Audit 4-Case: 4/4 PASS

### v5.3.5-P8.4 Changelog (Phase 8-4)

1. **Semantic Stability Audit**
   - 3500 runs (Mode A: 500, Mode B: 3000)
   - 5 categories × 20 sentences = 100 corpus
   - Parallel execution (8 workers)

2. **Drift Analysis (A/B/C)**
   - Drift-A (Atom): 0% at temp=0 ✅
   - Drift-B (Coordinate): 0% at temp=0 ✅
   - Drift-C (Formula): ~5.6% (expression variation)

3. **Immunity Metrics**
   - Validator Blocks: 8 (0.23%)
   - Coercions: 5 (0.14%)
   - Pollution Blocked: True

4. **Audit Judgment**
   - **Conditional PASS**
   - System does not break
   - System does not pollute
   - System has courage to abstain (Noise: 85% empty)

### v5.3.5-P8.5 Changelog (Phase 8-5)

1. **Semantic Memory Package**
   - New `ledger/` package
   - `memory_math.py`: Decay, Reinforce, Tau, Fingerprint
   - `ephemeral_ledger.py`: EphemeralLedger class

2. **Memory Math Implementation**
   - Decay: w × exp(-dt/τ)
   - Reinforce: w + α(1-w), α=0.2
   - Oblivion: purge if w < ε, ε=0.01
   - Tau Policy: temporal axis → time constant

3. **Memory Constitution**
   - Conflict Co-existence (no cancellation)
   - Fingerprint-based identity
   - Natural selection via Weight competition

4. **Unit Tests (11/11 PASS)**
   - Decay Math ✅
   - Reinforcement Math ✅
   - Oblivion Threshold ✅
   - Tau Policy ✅
   - Tau Affects Lifespan ✅
   - Conflict Coexistence ✅
   - Retention Rate ✅
   - Reinforcement Stability ✅
   - Asymptotic Approach ✅
   - Fingerprint Identity ✅
   - Operator Extraction ✅

5. **Integration Test (E2E)**
   - 14 inputs, 12 success (85.7%)
   - Block rate: 0%, Coercion rate: 0%
   - Ledger: 9 entries, 12 observations
   - All audit tests PASS

### v5.3.6-P8.6 Changelog (Phase 8-6)

1. **Persistent Ledger Package**
   - `canonical.py`: JSON正規化（バイト一致保証）
   - `chain_crypto.py`: Hash Chain（event_id / self hash）
   - `persistent_ledger.py`: PersistentLedgerクラス

2. **Hash Chain Implementation**
   - Atomic Append: flush + fsync
   - Genesis Block: "Aru - There is"
   - Tamper Detection: 5種のIntegrity Check (T861-T865)

3. **GPT Audit v8.6.1 Compliance**
   - validate()は行文字列をcanonicalとして扱う
   - 最終行破損はsalvageなしで停止
   - event_idからmeta除外（意味同一性）
   - rehydrationはledger replay

4. **Emergence Directionality**
   - `=>` : 未確定
   - `=>+` : 創造的創発（connectivity増加）
   - `-|>` : 破壊的創発（Reboot/軸遷移）

5. **Test Results (13/13 PASS)**
   - Genesis Creation ✅
   - Append and Validate ✅
   - Molecule Observe ✅
   - T861: Chain Validates ✅
   - T862: Tamper Detected ✅
   - T863: Truncation Detected ✅
   - T864: Reorder Detected ✅
   - T865: Header Mismatch ✅
   - Rehydration ✅
   - Sleep Decay ✅
   - Direction Values ✅
   - Event ID Excludes Meta ✅
   - Conflict Coexistence ✅

### v5.3.7-P8.7 Changelog (Phase 8-7)

1. **Semantic Index Package (L2)**
   - `semantic_index.py`: AtomStats, DirectionStats構造
   - `projector.py`: L1→L2投影（rebuild / on_event）
   - `rigidity.py`: 硬直度計算
   - `query_api.py`: 外部クエリAPI

2. **Dual Layer Architecture**
   - L1 (Ledger): 不変の正史
   - L2 (Index): 再構築可能な射影
   - Parity Consistency保証

3. **Rigidity Signals**
   - R = N_mode / N_total
   - 分類: crystallized / rigid / stable / fluid / volatile

4. **GPT Audit v8.7.1 Compliance**
   - formula_signature: formula → atoms/ops連結 → `__unknown__`
   - L2肥大化防止: `deque(maxlen=10000)`
   - window引数: 各APIに実装

5. **Test Results (13/13 PASS)**
   - Index Creation ✅
   - Index Update ✅
   - Formula Signature Extraction ✅
   - Parity: Rebuild vs Incremental ✅
   - Rigidity: Constant Formula (R=1.0) ✅
   - Rigidity: Varying Formula (R<1.0) ✅
   - Rigidity: Mixed Pattern (R=0.7) ✅
   - No Side Effect on Ledger ✅
   - QueryAPI Basic ✅
   - QueryAPI AtomInfo ✅
   - QueryAPI Cooccurrence ✅
   - QueryAPI Window ✅
   - Direction Balance ✅

### v5.3.8-P8.8 Changelog (Phase 8-8: Feedback Loop)

1. **Feedback Package**
   - `strategies.py`: GenerationStrategy dataclass
   - `modulator.py`: Modulator class
   - Constants: RIGIDITY_HIGH=0.9, RIGIDITY_LOW=0.3, ALERT_THRESHOLD=0.98

2. **Pipeline Package**
   - `core_pipeline.py`: ESDEPipeline, ModulatedGenerator
   - Full loop: Sensor → Index → Modulator → Generator → Ledger → Index

3. **Strategy Modes**
   | Mode | Condition | Temperature | Action |
   |------|-----------|-------------|--------|
   | NEUTRAL | 0.3 ≤ R ≤ 0.9 | 0.1 | Normal operation |
   | DISRUPTIVE | R > 0.9 | 0.7 | "Doubt it. Find contradictions." |
   | STABILIZING | R < 0.3 | 0.0 | "Consolidate. Find commonality." |

4. **Alert System**
   - Condition: R ≥ 0.98 AND N ≥ 10
   - Output: `[ALERT] CONCEPT_CRYSTALLIZED: {atom_id}`

5. **Direction Adjustment**
   - DISRUPTIVE mode → direction = `-|>` (destructive emergence)

6. **Test Results (14/14 PASS)**
   - Constants Verification ✅
   - Neutral Strategy ✅
   - Disruptive Strategy ✅
   - Stabilizing Strategy ✅
   - Unknown Atom Handling ✅
   - Target Atom Extraction ✅
   - Alert Condition ✅
   - Alert Not Triggered (N<10) ✅
   - Neutral Loop ✅
   - Disruptive Feedback ✅
   - Stabilizing Feedback ✅
   - No Candidates ✅
   - Alert (No Index Update) ✅
   - Alert After Update ✅

### v5.3.9-P8.9 Changelog (Phase 8-9: Monitor & Long-Run)

1. **Monitor Package**
   - `semantic_monitor.py`: SemanticMonitor, MonitorState
   - TUI Dashboard using `rich` library
   - Fallback to plain text if `rich` unavailable

2. **Runner Package**
   - `long_run.py`: LongRunRunner, LongRunReport
   - Health Check: periodic Ledger.validate()
   - Error Policy: Fail Fast

3. **CLI Entry Point**
   - `esde_cli_live.py`: observe, monitor, longrun, status commands
   - Real LLM integration with QwQ-32B

4. **Monitor Display**
   | Panel | Content |
   |-------|---------|
   | Header | Version, Uptime, Ledger Seq, Alert Count |
   | Live Feed | Input, Target Atom, Rigidity, Strategy, Formula |
   | Rankings | Top Rigid (R→1.0), Top Volatile (R→0.0) |
   | Stats | Index Size, Direction Balance |

5. **Long-Run Report**
   - Execution: steps, duration, stopped_early
   - Results: successful, abstained
   - Alerts: total, atoms
   - Health: ledger_valid, validation_checks
   - Final State: ledger_seq, index_size, direction_balance

6. **GPT Audit v8.9.1 Compliance**
   - Fail Fast: エラー発生時は即座に停止
   - Mock LLM: テストでは実LLMを使わない
   - T895: Ledger Invariance Test（Genesis不変）
   - Rich Fallback: rich非依存でも動作

7. **Test Results (10/10 PASS)**
   - Monitor State Init ✅
   - Monitor Update ✅
   - Monitor Update with Alert ✅
   - Monitor Render ✅
   - LongRun Basic (steps=10) ✅
   - LongRun Alert Count ✅
   - LongRun Abstain Handling ✅
   - LongRun Report Structure ✅
   - T895: Ledger Invariance ✅
   - CLI Observe (Mock) ✅

8. **Live Long-Run Results (50 steps)**
   - Steps: 50/50 完走
   - Duration: 478秒
   - Successful: 38
   - Abstained: 12
   - Alerts: 1 (EMO.love crystallized)
   - Ledger Valid: ✅
   - Validation Checks: 6/6 PASS

### 15.10 Phase 8-10: Schema Consolidation & Integration Test

**Date:** 2026-01-20

#### 15.10.1 Issue: Schema Divergence

| Version | Schema | Example |
|---------|--------|---------|
| 8-2 (mock/validator) | nested | `coordinates.axis`, `coordinates.level` |
| 8-3 (live) | flat | `axis`, `level` |
| Synapse (data) | flat | `axis`, `level` |

**Root Cause:** 8-2 mock/validator were tested in isolation; integration tests validated 8-3 Live → Ledger only.

#### 15.10.2 Resolution

**Canonical = v8.3 flat structure** (INV-MOL-001)

Rationale:
- Synapse actual data is flat
- Ledger contract expects flat
- 8-2 nested was design-time artifact

#### 15.10.3 New Modules

| Module | Role |
|--------|------|
| `sensor/constants.py` | VALID_OPERATORS (single source of truth) |
| `sensor/glossary_validator.py` | GlossaryValidator (neutral, no legacy deps) |
| `sensor/validator_v83.py` | Canonical validator for v8.3 schema |
| `sensor/__init__.py` | Package API (canonical exports only) |

#### 15.10.4 MoleculeValidatorV83 Specification

```python
class MoleculeValidatorV83:
    def __init__(self, glossary, allowed_atoms, synapse_hash=None):
        ...
    
    def validate(self, molecule, original_text) -> ValidationResultV83:
        ...
```

**ValidationResultV83:**
```python
@dataclass
class ValidationResultV83:
    valid: bool
    errors: List[str]
    warnings: List[str]
    synapse_hash: Optional[str]  # For reproducibility
    atoms_checked: int
```

**Validation Rules:**

| Check | Severity | Rationale |
|-------|----------|-----------|
| Unknown atom | ERROR | Contract violation |
| Invalid axis | ERROR | Glossary contract |
| Invalid level | ERROR | Glossary contract |
| Span out of range | ERROR | Data integrity |
| Span text mismatch | WARNING | Normalization variance acceptable |
| Legacy nesting detected | WARNING | Deprecation notice |

#### 15.10.5 Integration Test Results

**Test:** `test_phase8_integration.py` (Real LLM: QwQ-32B)

```
Total: 5/5 passed
Schema compliance: 5/5
Synapse hash recorded: ✅ (3 tests)
```

| Input | Candidates | Formula | Result |
|-------|------------|---------|--------|
| "I love you" | 2 | `aa_1 × aa_2` | ✅ |
| "The law requires obedience" | 5 | `aa_1 ▷ aa_2 ▷ aa_5` | ✅ |
| "apprenticed to a master" | 1 | `aa_1` | ✅ |
| "" (empty) | 0 | Abstain (LLM not called) | ✅ |
| "asdfghjkl qwertyuiop" | 0 | Abstain | ✅ |

#### 15.10.6 File Structure (Final)

```
esde/
├── test_phase8_integration.py
├── glossary_results.json
├── esde_synapses_v3.json
│
├── Docs/
│   └── PHASE8_MOLECULE_SCHEMA_FIX.md
│
└── sensor/
    ├── __init__.py              # Package API (canonical only)
    ├── constants.py             # VALID_OPERATORS
    ├── glossary_validator.py    # GlossaryValidator (neutral)
    ├── validator_v83.py         # Canonical validator
    ├── molecule_generator_live.py  # v8.3.1 (no legacy dependency)
    ├── esde_sensor_v2_modular.py
    ├── loader_synapse.py
    ├── extract_synset.py
    ├── rank_candidates.py
    ├── audit_trace.py
    └── legacy/                  # Deprecated modules
        ├── __init__.py
        ├── molecule_generator.py
        ├── molecule_validator.py
        └── legacy_trigger.py
```

### Phase 8 Complete Summary

| Phase | Theme | Tests |
|-------|-------|-------|
| 8-1 | Sensor V2 + Modular | ✅ |
| 8-2 | Molecule Generator/Validator | ✅ |
| 8-3 | Live LLM Integration | ✅ |
| 8-4 | Stability Audit | ✅ |
| 8-5 | Ephemeral Ledger | 11/11 ✅ |
| 8-6 | Persistent Ledger | 13/13 ✅ |
| 8-7 | Semantic Index | 13/13 ✅ |
| 8-8 | Feedback Loop | 14/14 ✅ |
| 8-9 | Monitor & Long-Run | 10/10 ✅ |
| 8-10 | Schema Consolidation | 5/5 ✅ |

**Phase 8 Theme: 「内省エンジンの基盤構築」 - Complete**

---

## Phase 9: Weak Axis Statistics Layer

### Phase 9 Overview

Phase 9は「弱軸統計層（Weak Axis Statistics Layer）」を構築する。
人間によるラベリング無しに、データから「軸の影」を検出する統計的基盤を提供する。

```
W0 (ContentGateway) → 外部データの正規化・取り込み
    ↓
W1 (Global Statistics) → 条件を無視した全体統計
    ↓
W2 (Conditional Statistics) → 条件別のスライス統計
    ↓
W3 (Axis Candidates) → 条件特異性スコア（S-Score）による軸候補抽出
    ↓
W4 (Structural Projection) → 記事をW3軸候補に投影、共鳴ベクトル生成
    ↓
W5 (Axis Confirmation) → 人間レビューによる軸確定（将来）
```

### v5.4.2-P9.0 W0: ContentGateway

**File:** `integration/gateway.py`

**Purpose:** External data normalization into ArticleRecords

**Key Classes:**
- `ContentGateway`: Entry point for content observation
- `ArticleRecord`: Normalized external data container
- `ObservationEvent`: W0 observation unit (1 segment = 1 event)

**INV-W0-001:** ArticleRecord is immutable after creation

### v5.4.2-P9.1 W1: Global Statistics

**File:** `statistics/w1_aggregator.py`

**Purpose:** Condition-blind token statistics

**Key Classes:**
- `W1Aggregator`: Aggregates token statistics globally
- `W1Record`: Per-token statistics (total_count, document_frequency, entropy)
- `W1GlobalStats`: Global aggregation state

**INV-W1-001:** W1 can always be regenerated from W0
**INV-W1-002:** W1 input is ArticleRecord.raw_text sliced by segment_span

### v5.4.2-P9.2 W2: Conditional Statistics

**File:** `statistics/w2_aggregator.py`

**Purpose:** Token statistics sliced by condition factors

**Key Classes:**
- `W2Aggregator`: Aggregates token statistics per condition
- `W2Record`: Per-token statistics under a condition
- `ConditionEntry`: Condition metadata with denominators

**Condition Factors:**
- `source_type`: news | dialog | paper | social | unknown
- `language_profile`: en | ja | mixed | unknown
- `time_bucket`: YYYY-MM format

**INV-W2-001:** W2Aggregator reads condition factors, never writes/infers
**INV-W2-002:** time_bucket is YYYY-MM (monthly) in v9.x
**INV-W2-003:** ConditionEntry denominators written by W2Aggregator ONLY (W3 NEVER writes)

### v5.4.2-P9.3 W3: Axis Candidates

**File:** `statistics/w3_calculator.py`

**Purpose:** Per-token specificity analysis using S-Score

**Mathematical Model:**
```
S(t, C) = P(t|C) × log((P(t|C) + ε) / (P(t|G) + ε))

Where:
  P(t|C) = count_cond / total_cond
  P(t|G) = count_global / total_global
  ε = 1e-12 (fixed smoothing)
```

**Key Classes:**
- `W3Calculator`: Computes axis candidates
- `W3Record`: Analysis result with positive/negative candidates
- `CandidateToken`: Token with S-Score and probabilities

**INV-W3-001:** No Labeling (output is factual only)
**INV-W3-002:** Immutable Input (W3 does NOT modify W1/W2)
**INV-W3-003:** Deterministic (tie-break by token_norm asc)

### v5.4.4-P9.4 W4: Structural Projection

**Theme:** The Resonance of Weakness (弱さの共鳴・構造射影)

**File:** `statistics/w4_projector.py`, `statistics/schema_w4.py`

**Purpose:** Project ArticleRecords onto W3 axis candidates to produce resonance vectors

**Mathematical Model:**
```
R(A, C) = Σ count(t, A) × S(t, C)

Where:
  A = Article
  C = Condition
  count(t, A) = Token count in article A
  S(t, C) = S-Score from W3 for token t under condition C
```

**Key Classes:**
- `W4Projector`: Projects articles onto W3 candidates
- `W4Record`: Per-article resonance vector
- `resonance_vector`: Dict[condition_signature, float]

**Key Fields (W4Record):**

| Field | Type | Description |
|-------|------|-------------|
| article_id | str | Link to ArticleRecord |
| w4_analysis_id | str | Deterministic ID (P0-1) |
| resonance_vector | Dict[str, float] | Per-condition scores |
| used_w3 | Dict[str, str] | Traceability: cond_sig → w3_analysis_id |
| token_count | int | Total valid tokens (length bias awareness) |
| tokenizer_version | str | e.g., "hybrid_v1" |
| normalizer_version | str | e.g., "v9.1.0" |
| projection_norm | str | "raw" (v9.4: no normalization) |
| algorithm | str | "DotProduct-v1" |

**GPT Audit P0 Compliance:**
- P0-1: Deterministic w4_analysis_id = SHA256(article_id + sorted(w3_ids) + versions)
- P0-2: used_w3 is Dict (not List) for traceability
- P0-3: token_count recorded for length bias awareness

**Test Results (11/11 PASS):**
- Schema.1: W4Record creation ✅
- Schema.2: Deterministic analysis ID (P0-1) ✅
- Schema.3: JSON serialization ✅
- Schema.4: Canonical dict (P1-1) ✅
- Projector.1: News article projection ✅
- Projector.2: Dialog article projection ✅
- Projector.3: Determinism (INV-W4-002) ✅
- Projector.4: Traceability (P0-2) ✅
- Projector.5: Version tracking (P0-3) ✅
- Projector.6: Full S-Score usage (INV-W4-004) ✅
- Projector.7: Save and load ✅

**Example Output:**
```
News article: "The prime minister announced new government policy."
  → News resonance: +0.235
  → Dialog resonance: -0.130

Dialog article: "Hey! Yeah that's so cool lol."
  → News resonance: -0.095
  → Dialog resonance: +0.215
```

**INV-W4-001:** No Labeling (output keys are condition_signature only)
**INV-W4-002:** Deterministic (same article + same W3 → same scores)
**INV-W4-003:** Recomputable (W4 = f(W0, W3))
**INV-W4-004:** Full S-Score Usage (positive AND negative candidates)
**INV-W4-005:** Immutable Input (W4Projector does NOT modify ArticleRecord or W3Record)
**INV-W4-006:** Tokenization Canon (MUST use W1Tokenizer + normalize_token)

---

### Phase 9 Constants

| Parameter | Value | Description |
|-----------|-------|-------------|
| EPSILON | 1e-12 | S-Score smoothing constant |
| DEFAULT_MIN_COUNT_FOR_W3 | 2 | Minimum count filter for stability |
| DEFAULT_TOP_K | 100 | Default number of candidates |
| W3_ALGORITHM | "KL-PerToken-v1" | W3 algorithm identifier |
| W4_ALGORITHM | "DotProduct-v1" | W4 resonance calculation method |
| W4_PROJECTION_NORM | "raw" | v9.4: No length normalization |
| DEFAULT_W4_OUTPUT_DIR | "data/stats/w4_projections" | W4 output directory |

---

### Phase 9 Constitutional Articles

| Code | Layer | Description |
|------|-------|-------------|
| INV-W0-001 | W0 | ArticleRecord is immutable after creation |
| INV-W1-001 | W1 | W1 can always be regenerated from W0 |
| INV-W1-002 | W1 | W1 input is ArticleRecord.raw_text sliced by segment_span |
| INV-W2-001 | W2 | W2Aggregator reads condition factors, never writes/infers |
| INV-W2-002 | W2 | time_bucket is YYYY-MM (monthly) in v9.x |
| INV-W2-003 | W2 | ConditionEntry denominators written by W2Aggregator ONLY |
| INV-W3-001 | W3 | No Labeling: output is factual only |
| INV-W3-002 | W3 | Immutable Input: W3 does NOT modify W1/W2 |
| INV-W3-003 | W3 | Deterministic: tie-break by token_norm asc |
| INV-W4-001 | W4 | No Labeling: output keys are condition_signature only |
| INV-W4-002 | W4 | Deterministic: same article + same W3 → same scores |
| INV-W4-003 | W4 | Recomputable: W4 = f(W0, W3) |
| INV-W4-004 | W4 | Full S-Score Usage: positive AND negative candidates |
| INV-W4-005 | W4 | Immutable Input: W4Projector does NOT modify inputs |
| INV-W4-006 | W4 | Tokenization Canon: MUST use W1Tokenizer + normalize_token |

---

## 17. References

- ESDE Core Specification v0.2.1
- ESDE v3.3: Ternary Emergence and Dual Symmetry
- ESDE v3.3.1: Emergence Directionality
- Semantic Language Integrated v1.1
- ESDE Operator Spec v0.3
- Aruism Philosophy

---

*Document generated: 2026-01-21*  
*Engine Version: 5.4.4-P9.4*  
*Framework: Existence Symmetry Dynamic Equilibrium*  
*Philosophy: Aruism*
