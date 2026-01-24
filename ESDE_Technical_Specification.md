# ESDE Technical Specification v5.4.7-SUB.1

## Existence Symmetry Dynamic Equilibrium
### The Introspection Engine for AI

Version: 5.4.7-SUB.1  
Date: 2026-01-24  
Status: Production (Substrate Layer v0.1.0 Complete)

---

## 0. Purpose and Scope

### 0.1 What ESDE Is

> ESDEã¯è¨¼æ˜Žã‚·ã‚¹ãƒ†ãƒ ã§ã¯ãªã„ã€‚  
> ESDEã¯ç”Ÿæˆè¨€èªžãƒ¢ãƒ‡ãƒ«ã§ã¯ãªã„ã€‚  
> ESDEã¯æ„å‘³æ§‹é€ ä¸Šã§å‹•ä½œã™ã‚‹å‹•çš„å‡è¡¡ã‚¨ãƒ³ã‚¸ãƒ³ã§ã‚ã‚‹ã€‚  
> **ESDEã¯AIã®å†…çœã‚¨ãƒ³ã‚¸ãƒ³ï¼ˆIntrospection Engineï¼‰ã§ã‚ã‚‹ã€‚**

ESDEã‚’ä½¿ç”¨ã™ã‚‹ã¨ã¯ä»¥ä¸‹ã‚’æ„å‘³ã™ã‚‹ï¼š

- ç¾å®Ÿä¸–ç•Œã®è¨€èªžã‚„ã‚¤ãƒ™ãƒ³ãƒˆã‚’æ§‹é€ åŒ–ã•ã‚ŒãŸæ„å‘³ç©ºé–“ã«ãƒžãƒƒãƒ”ãƒ³ã‚°ã™ã‚‹
- ä¸å‡è¡¡ã¨çŸ›ç›¾ã‚’æ˜Žç¤ºçš„ãªèª¤å·®å€¤ï¼ˆÎµï¼‰ã¨ã—ã¦æ¸¬å®šã™ã‚‹
- åˆ¶ç´„ä»˜ãæ›´æ–°ã‚’é€šã˜ã¦å‹•çš„å‡è¡¡ã‚’å›žå¾©ã™ã‚‹
- å‡è¡¡ãŒç¡¬ç›´ã—ãŸå ´åˆã«åˆ¶å¾¡ã•ã‚ŒãŸRebootã‚’ãƒˆãƒªã‚¬ãƒ¼ã™ã‚‹
- å˜ãªã‚‹ç‰©èªžçš„å¿œç­”ã§ã¯ãªãã€æ§‹é€ åŒ–ã•ã‚ŒãŸæ´žå¯Ÿã‚’è¿”ã™

### 0.2 Philosophical Foundation

ESDE is based on **Aruism Philosophy** and the **ESDE Framework**.

#### Aruism (The Root)

- **ã€Œã‚ã‚‹ï¼ˆAruï¼‰ã€ã®å„ªå…ˆ**: ä¸–ç•Œã¯ã€Œå®šç¾©ã€ã•ã‚Œã‚‹å‰ã«ã¾ãšã€Œã‚ã‚‹ã€
- ESDEã¯è¦³æ¸¬å¯¾è±¡ã‚’æ—¢å­˜ã®æž çµ„ã¿ã«ç„¡ç†ã‚„ã‚Šå½“ã¦ã¯ã‚ã‚‹ã®ã§ã¯ãªãã€ã€Œãã“ã«ã‚ã‚‹æ›–æ˜§ãªçŠ¶æ…‹ã€ã‚’ãã®ã¾ã¾ä¿æŒãƒ»è¦³æ¸¬ã™ã‚‹ãŸã‚ã®åŸºç›¤ï¼ˆOSï¼‰ã§ã‚ã‚‹
- **äºŒé …å¯¾ç«‹ã®å›žé¿**: ã€ŒAã‹Bã‹ã€ã®äºŒè€…æŠžä¸€ã‚’è¿«ã‚‹ã“ã¨ã¯ã€Aruismã®å¦å®šã¨ãªã‚‹

#### ESDE Theory (The Logic)

- **å…¬ç†T (Ternary Emergence)**: ã€ŒAã¨Bã€ã®äºŒè€…é–¢ä¿‚ã ã‘ã§ã¯å­˜åœ¨ã¯é¡•ç¾ã—ãªã„ã€‚ç¬¬ä¸‰é …ï¼ˆObserverï¼‰ãŒä»‹å…¥ã™ã‚‹ã“ã¨ã§åˆã‚ã¦æ„å‘³ãŒç«‹ã¡ä¸ŠãŒã‚‹
- **å‰µé€ ã¨ç ´å£Šã®å¯¾ç§°æ€§**: ã€Œæºã‚Œï¼ˆVolatilityï¼‰ã€ã‚„ã€Œæœªæ±ºï¼ˆWinner=Nullï¼‰ã€ã¯ã‚¨ãƒ©ãƒ¼ã§ã¯ãªãã€ã€Œç ´å£Šçš„å‰µç™ºï¼ˆDestructive Emergenceï¼‰ã€ã®å…†å€™ã§ã‚ã‚‹å¯èƒ½æ€§ãŒã‚ã‚‹

#### Core Axioms (v3.3)

| Axiom | Name | Statement |
|-------|------|-----------|
| 0 | Aru (There Is) | The primordial fact of existence |
| E | Identification | E = {e1, e2, ..., en} |
| L | Linkage | L: E Ã— E â†’ [0,1] |
| Eq | Equality | All existences equal in status |
| C | Creativity | C âŠ¥ gradient(F) |
| U | Understanding | Ongoing participatory engagement |
| Îµ | Error | E = f(I) + Îµ, Îµ â‰  0 |
| T | Ternary Emergence | Aâ†”Bâ†”C â‡’ Manifestation |

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
| temporal | emergence â†’ indication â†’ influence â†’ transformation â†’ establishment â†’ continuation â†’ permanence | 7 |
| scale | individual â†’ community â†’ society â†’ ecosystem â†’ stellar â†’ cosmic | 6 |
| epistemological | perception â†’ identification â†’ understanding â†’ experience â†’ creation | 5 |
| ontological | material â†’ informational â†’ relational â†’ structural â†’ semantic | 5 |
| interconnection | independent â†’ catalytic â†’ chained â†’ synchronous â†’ resonant | 5 |
| resonance | superficial â†’ structural â†’ essential â†’ existential | 4 |
| symmetry | destructive â†’ inclusive â†’ transformative â†’ generative â†’ cyclical | 5 |
| lawfulness | predictable â†’ emergent â†’ contingent â†’ necessary | 4 |
| experience | discovery â†’ creation â†’ comprehension | 3 |
| value_generation | functional â†’ aesthetic â†’ ethical â†’ sacred | 4 |

**Total: 48 levels**

### 1.3 Design Philosophy (Aruism Mapping)

The semantic architecture directly implements core Aruism concepts:

#### Equality of Existence â†’ winner=null

> ã‚ã‚‰ã‚†ã‚‹å­˜åœ¨ãŒã€ãã®ä¾¡å€¤ã‚„é‡è¦æ€§ã«ãŠã„ã¦ç­‰ã—ãç›¸äº’ä¾å­˜çš„ã§ã‚ã‚Šã€
> æ¬ ã‘ã‚‹ã“ã¨ã®ã§ããªã„å…¨ä½“ã®æ§‹æˆè¦ç´ ã¨ã—ã¦å¯¾ç­‰ã«æ‰±ã‚ã‚Œã‚‹ã€‚

Implementation: No hypothesis is privileged over another. `winner` remains `null` always. All routes (A/B/C/D) are evaluated with equal weight.

#### Symmetry of Existence â†’ 163 Pairs

> ä¸€ã¤ã®å­˜åœ¨ã«å¯¾ã—ã¦è‡ªç„¶ã¨å¯¾ç§°çš„ã«ç”Ÿã˜ã‚‹å­˜åœ¨ã‚„æ¦‚å¿µã®é–¢ä¿‚æ€§ã€‚
> ç•°ãªã‚‹è¦–ç‚¹ã‚„çŠ¶æ³ã«ã‚ˆã£ã¦åè»¢ã—ã†ã‚‹ç›¸è£œçš„ãªé–¢ä¿‚æ€§ã€‚

Implementation: 326 atoms = 163 symmetric pairs. This is not binary opposition but mutual definition. `love` cannot be defined without `hate`. Îµ_sym measures imbalance between pairs.

#### Linkage of Existence â†’ w_ij

> ã‚ã‚‰ã‚†ã‚‹å­˜åœ¨ãŒç›¸äº’ä½œç”¨ã—åˆã„ã€é€£å‹•ã—ã¦ç¾è±¡ã‚„çŠ¶æ³ã‚’ç”Ÿã¿å‡ºã—ã¦ã„ã‚‹ã€‚

Implementation: Linkage weight `w_ij` between concepts. Co-occurrence strengthens links. Îµ_link measures inconsistency in linked concepts.

#### Axis â†’ 10 Axes

> è¤‡é›‘ãªç¾å®Ÿã‹ã‚‰é–¢é€£æ€§ã®å¼·ã„å­˜åœ¨ç¾¤ã‚’è¦‹ã¤ã‘å‡ºã—ã€
> æ„å‘³ã‚ã‚‹ä¸€ã¤ã®ã¾ã¨ã¾ã‚Šã¨ã—ã¦çµ±åˆãƒ»å›²ã„è¾¼ã‚€èªè­˜æ©Ÿèƒ½ã€‚

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
    value = activation intensity (0.0ã€œ1.0)

State Space Size: 326 Ã— 48 = 15,648 dimensions
```

---

## 2. Architecture

### 2.1 Component Overview

```
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚                    ESDE Framework                            â”‚
â”‚         (Existence Symmetry Dynamic Equilibrium)             â”‚
â”‚                   Aruism Philosophy                          â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                              â”‚
                              â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚                    ESDE Components                           â”‚
â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
â”‚   Glossary   â”‚   Synapse    â”‚    Sensor    â”‚     Engine     â”‚
â”‚  Atom Defs   â”‚   Bridge     â”‚   Input      â”‚    State       â”‚
â”‚   326 atoms  â”‚  WordNet     â”‚  Operators   â”‚   Îµ Calc       â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
```

### 2.2 Component Status

| Component | Status | Implementation | Issue |
|-----------|--------|----------------|-------|
| Glossary | âœ… Stable | 326 atoms, 10Ã—48 definitions | - |
| Synapse | âœ… Generated | 11,557 synsets, 22,285 edges, vector distance (raw_score) | - |
| Sensor | âœ… **V2 + Live** | Synapse lookup + Legacy fallback | - |
| Generator | âœ… **Live (QwQ)** | Phase 8-3 MoleculeGeneratorLive | - |
| Validator | âœ… **Production** | Phase 8-2 MoleculeValidator | - |
| **Ledger** | âœ… **Ephemeral** | Phase 8-5 EphemeralLedger | - |
| Engine | âœ… Working | State management, Îµ calculation | - |
| Audit Pipeline | âœ… Complete | 7A â†’ 7D | - |
| Stability Audit | âœ… **PASS** | Phase 8-4 (3500 runs) | - |
| **Memory Audit** | âœ… **PASS** | Phase 8-5 (Integration Test) | - |
| **W0 (ContentGateway)** | âœ… **Production** | Phase 9-0 input normalization | - |
| **W1 (Global Stats)** | âœ… **Production** | Phase 9-1 token statistics | - |
| **W2 (Conditional Stats)** | âœ… **Production** | Phase 9-2 condition-sliced statistics | - |
| **W3 (Axis Candidates)** | âœ… **Production** | Phase 9-3 S-Score calculation | - |
| **W4 (Structural Projection)** | ✅ **Production** | Phase 9-4 Resonance vectors | - |
| **W5 (Structural Condensation)** | ✅ **Production** | Phase 9-5 Island clustering | - |
| **W6 (Structural Observation)** | ✅ **Production** | Phase 9-6 Evidence extraction | - |

### 2.3 Integration Status

```
BEFORE (Disconnected):

â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”          â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚   Sensor v1     â”‚          â”‚   Synapse       â”‚
â”‚   (Legacy)      â”‚    ??    â”‚   (v3.0)        â”‚
â”‚ Trigger-based   â”‚â—„â”€â”€â”€â”€â”€â”€â”€â”€â–ºâ”‚ Vector distance â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜   NOT    â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                   CONNECTED

AFTER (Sensor V2 - Integrated):

â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”          â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚   Input Text    â”‚          â”‚   Synapse       â”‚
â”‚                 â”‚          â”‚   (v3.0)        â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”˜          â””â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”˜
         â”‚                            â”‚
         â–¼                            â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚              Sensor V2 (Unified)             â”‚
â”‚                                              â”‚
â”‚  1. Tokenize â†’ WordNet synset lookup        â”‚
â”‚  2. Synapse lookup â†’ concept candidates     â”‚
â”‚  3. Vector distance (raw_score) ranking     â”‚
â”‚  4. Deterministic sort (score DESC, id ASC) â”‚
â”‚  5. Fallback: Legacy triggers (Hybrid mode) â”‚
â”‚  6. Output: concept_id + axis + level       â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
```

**Implementation File:** `esde_sensor_v2.py`

### 2.4 Data Flow Pipeline

```
1. User Text â†’ esde_sensor.py â†’ Formula
2. Formula â†’ esde-engine-v532.py â†’ Known or Queue
3. Queue â†’ resolve_unknown_queue_7bplus_v534_final.py â†” aggregate_state.py
4. Resolver â†’ online.py â†’ hypothesis.py â†’ Ledger & Aggregate Output
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
| `esde_sensor_v2_modular.py` | **Facade: Synapse-integrated** | âœ… **Implemented** |
| `sensor/` | **Modular components package** | âœ… **Phase 8** |

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
| `CoordinateCoercer` | Invalid coordinate â†’ null with logging |
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
| `decay(w, dt, tau)` | w Ã— exp(-dt/Ï„) | Exponential decay |
| `reinforce(w, alpha)` | w + Î±(1-w) | Asymptotic reinforcement |
| `should_purge(w, epsilon)` | w < Îµ | Oblivion check |
| `get_tau_from_molecule()` | - | Temporal axis â†’ tau |
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
â”œâ”€â”€ esde_engine/                     # Main package directory
â”‚   â”œâ”€â”€ __init__.py                  # Package initialization & exports
â”‚   â”œâ”€â”€ __main__.py                  # CLI entry point
â”‚   â”œâ”€â”€ config.py                    # Configuration constants (SINGLE SOURCE OF TRUTH)
â”‚   â”œâ”€â”€ utils.py                     # Utility functions
â”‚   â”œâ”€â”€ loaders.py                   # Data loaders
â”‚   â”œâ”€â”€ extractors.py                # Synset extraction
â”‚   â”œâ”€â”€ collectors.py                # Activation collection
â”‚   â”œâ”€â”€ routing.py                   # Unknown token routing (7A+)
â”‚   â”œâ”€â”€ queue.py                     # Unknown queue writer (7A)
â”‚   â”œâ”€â”€ engine.py                    # Main ESDEEngine class
â”‚   â”‚
â”‚   â””â”€â”€ resolver/                    # Phase 7B/7B+: Unknown Queue Resolver
â”‚       â”œâ”€â”€ __init__.py              # Package exports
â”‚       â”œâ”€â”€ state.py                 # Queue state management (legacy)
â”‚       â”œâ”€â”€ aggregate_state.py       # [v5.3.4] Aggregate state management
â”‚       â”œâ”€â”€ hypothesis.py            # Multi-hypothesis evaluation (A/B/C/D)
â”‚       â”œâ”€â”€ online.py                # [v5.3.5] MultiSourceProvider
â”‚       â”œâ”€â”€ ledger.py                # Evidence ledger (audit trail)
â”‚       â”œâ”€â”€ cache.py                 # Search cache
â”‚       â”œâ”€â”€ patches.py               # Patch output management
â”‚       â””â”€â”€ resolvers.py             # Route-specific resolvers
â”‚
â”œâ”€â”€ sensor/                          # [Phase 8] Modular Sensor Package
â”‚   â”œâ”€â”€ __init__.py                  # Package exports (lazy import)
â”‚   â”œâ”€â”€ loader_synapse.py            # SynapseLoader (singleton)
â”‚   â”œâ”€â”€ extract_synset.py            # SynsetExtractor (WordNet)
â”‚   â”œâ”€â”€ rank_candidates.py           # CandidateRanker (aggregation)
â”‚   â”œâ”€â”€ legacy_trigger.py            # LegacyTriggerMatcher (v1 fallback)
â”‚   â”œâ”€â”€ audit_trace.py               # AuditTracer (counters/hash)
â”‚   â”œâ”€â”€ molecule_validator.py        # [Phase 8-2] MoleculeValidator
â”‚   â”œâ”€â”€ molecule_generator.py        # [Phase 8-2] MoleculeGenerator (mock)
â”‚   â””â”€â”€ molecule_generator_live.py   # [Phase 8-3] MoleculeGeneratorLive (Real LLM)
â”‚
â”œâ”€â”€ ledger/                          # [Phase 8-5/8-6] Semantic Memory Package
â”‚   â”œâ”€â”€ __init__.py                  # Package exports (v8.6.0)
â”‚   â”œâ”€â”€ memory_math.py               # Decay, Reinforce, Tau, Fingerprint
â”‚   â”œâ”€â”€ ephemeral_ledger.py          # EphemeralLedger (in-memory)
â”‚   â”œâ”€â”€ canonical.py                 # [P8.6] JSON Canonicalization
â”‚   â”œâ”€â”€ chain_crypto.py              # [P8.6] Hash Chain (SHA256)
â”‚   â””â”€â”€ persistent_ledger.py         # [P8.6] PersistentLedger (Hash Chain)
â”‚
â”œâ”€â”€ index/                           # [Phase 8-7] Semantic Index Package
â”‚   â”œâ”€â”€ __init__.py                  # Package exports (v8.7.0)
â”‚   â”œâ”€â”€ semantic_index.py            # L2 In-Memory Structure
â”‚   â”œâ”€â”€ projector.py                 # L1â†’L2 Projection (rebuild/on_event)
â”‚   â”œâ”€â”€ rigidity.py                  # Rigidity Calculation (R = N_mode/N_total)
â”‚   â””â”€â”€ query_api.py                 # External Query API
â”‚
â”œâ”€â”€ feedback/                        # [Phase 8-8] Feedback Loop Package
â”‚   â”œâ”€â”€ __init__.py                  # Package exports (v8.8.0)
â”‚   â”œâ”€â”€ strategies.py                # Strategy Definitions (NEUTRAL/DISRUPTIVE/STABILIZING)
â”‚   â””â”€â”€ modulator.py                 # Feedback Modulator (decide_strategy, check_alert)
â”‚
â”œâ”€â”€ pipeline/                        # [Phase 8-8] Core Pipeline Package
â”‚   â”œâ”€â”€ __init__.py                  # Package exports (v8.8.0)
â”‚   â””â”€â”€ core_pipeline.py             # ESDEPipeline, ModulatedGenerator
â”‚
â”œâ”€â”€ monitor/                         # [Phase 8-9] Semantic Monitor Package
â”‚   â”œâ”€â”€ __init__.py                  # Package exports (v8.9.0)
â”‚   â””â”€â”€ semantic_monitor.py          # TUI Dashboard (rich)
â”‚
â”œâ”€â”€ runner/                          # [Phase 8-9] Long-Run Runner Package
â”‚   â”œâ”€â”€ __init__.py                  # Package exports (v8.9.0)
â”‚   â””â”€â”€ long_run.py                  # LongRunRunner, LongRunReport
â”‚
â”œâ”€â”€ integration/                     # [Phase 9-0] Content Gateway Package
â”‚   â”œâ”€â”€ __init__.py                  # Package exports (v9.0.0)
â”‚   â””â”€â”€ content_gateway.py           # ContentGateway, ArticleRecord
â”‚
â”œâ”€â”€ statistics/                      # [Phase 9-1/9-2/9-3/9-4] Statistics Package
â”‚   â”œâ”€â”€ __init__.py                  # Package exports (v9.4.0)
â”‚   â”œâ”€â”€ schema.py                    # W1Record, W1GlobalStats
â”‚   â”œâ”€â”€ schema_w2.py                 # W2Record, W2GlobalStats, ConditionEntry
â”‚   â”œâ”€â”€ schema_w3.py                 # W3Record, CandidateToken
â”‚   â”œâ”€â”€ schema_w4.py                 # W4Record (Phase 9-4)
â”‚   â”œâ”€â”€ tokenizer.py                 # EnglishWordTokenizer, HybridTokenizer
â”‚   â”œâ”€â”€ normalizer.py                # Token normalization (NFKC)
â”‚   â”œâ”€â”€ w1_aggregator.py             # W1Aggregator (global stats)
â”‚   â”œâ”€â”€ w2_aggregator.py             # W2Aggregator (conditional stats)
â”‚   â”œâ”€â”€ w3_calculator.py             # W3Calculator (S-Score)
â”‚   â””â”€â”€ w4_projector.py              # W4Projector (Resonance)
│
├── discovery/                       # [Phase 9-5/9-6] Discovery Package
│   ├── __init__.py                  # Package exports (v9.6.0)
│   ├── schema_w5.py                 # W5Structure, W5Island
│   ├── schema_w6.py                 # W6Observatory, W6IslandDetail
│   ├── w5_condensator.py            # W5Condensator (Island clustering)
│   ├── w6_analyzer.py               # W6Analyzer (Evidence extraction)
│   └── w6_exporter.py               # W6Exporter (MD/CSV/JSON export)
â”‚
â”œâ”€â”€ esde_cli_live.py                 # [Phase 8-9] CLI Entry Point
â”œâ”€â”€ esde_sensor.py                   # Semantic Operators v1.1.0 (Legacy)
â”œâ”€â”€ esde_sensor_v2_modular.py        # [Phase 8] Sensor V2 Facade
â”œâ”€â”€ esde-engine-v532.py              # Runtime Engine v5.3.2
â”œâ”€â”€ resolve_unknown_queue_7bplus_v534_final.py  # Phase 7B+ CLI
â”œâ”€â”€ generate_synapses_v2_1.py        # Synapse generator
â”œâ”€â”€ esde_glossary_pipeline_v5_1.py   # Glossary pipeline
â”œâ”€â”€ esde_meta_auditor.py             # Phase 7D Meta-Auditor
â”‚
â”œâ”€â”€ # [Phase 8-4] Stability Audit
â”œâ”€â”€ esde_stability_audit.py          # Stability audit CLI
â”œâ”€â”€ mode_a_quick_test.py             # Mode A drift test
â”œâ”€â”€ audit_corpus.jsonl               # 100 test sentences (5 categories)
â”œâ”€â”€ test_phase83_audit.py            # Phase 8-3 audit tests
â”‚
â”œâ”€â”€ # [Phase 8-5] Memory Tests
â”œâ”€â”€ test_phase85_memory.py           # Memory math unit tests
â”œâ”€â”€ test_phase85_integration.py      # E2E integration test
â”‚
â”œâ”€â”€ # [Phase 8-6] Ledger Tests
â”œâ”€â”€ test_phase86_ledger.py           # Hash chain validation tests
â”‚
â”œâ”€â”€ # [Phase 8-7] Index Tests
â”œâ”€â”€ test_phase87_index.py            # Parity, Rigidity, QueryAPI tests
â”‚
â”œâ”€â”€ # [Phase 8-8] Pipeline Tests
â”œâ”€â”€ test_phase88_pipeline.py         # Feedback Loop, Modulator tests
â”‚
â”œâ”€â”€ # [Phase 8-9] Monitor Tests
â”œâ”€â”€ test_phase89_monitor.py          # Monitor, Long-Run tests
â”‚
â”œâ”€â”€ # [Phase 9-4] W4 Tests
â”œâ”€â”€ test_phase94_w4.py               # W4 Projector integration tests
â”‚
â””â”€â”€ data/                            # Runtime data directory
    â”‚
    â”œâ”€â”€ audit_runs/                  # [Phase 8-4/8-5] Audit logs
    â”‚   â”œâ”€â”€ mode_a_runs.jsonl        # Mode A raw logs
    â”‚   â”œâ”€â”€ mode_b_runs.jsonl        # Mode B raw logs
    â”‚   â”œâ”€â”€ mode_a_quick_runs.jsonl  # Quick test logs
    â”‚   â”œâ”€â”€ mode_a_quick_drift_report.json  # Drift A/B/C report
    â”‚   â”œâ”€â”€ phase84_stability_report.json   # Stability report
    â”‚   â””â”€â”€ phase85_integration_report.json # Integration report
    â”‚
    â”œâ”€â”€ # [Phase 8-6] Persistent Semantic Ledger
    â”œâ”€â”€ semantic_ledger.jsonl        # Hash Chain (append-only, tamper-evident)
    â”‚
    â”œâ”€â”€ # Phase 7A: Unknown Token Queue
    â”œâ”€â”€ unknown_queue.jsonl
    â”‚
    â”œâ”€â”€ # Phase 7B+ (v5.3.4): Aggregate-Level Resolution
    â”œâ”€â”€ unknown_queue_state_7bplus.json
    â”œâ”€â”€ unknown_queue_7bplus.jsonl
    â”œâ”€â”€ evidence_ledger_7bplus.jsonl
    â”‚
    â”œâ”€â”€ # Phase 7C/7C': Audit
    â”œâ”€â”€ audit_log_7c.jsonl
    â”œâ”€â”€ audit_votes_7cprime.jsonl
    â”œâ”€â”€ audit_drift_7cprime.jsonl
    â”‚
    â”œâ”€â”€ # Phase 7D: Meta-Audit
    â”œâ”€â”€ audit_rules_review_7d.json
    â”‚
    â”œâ”€â”€ # Patch Outputs (Human Review Required)
    â”œâ”€â”€ patch_alias_add.jsonl
    â”œâ”€â”€ patch_synapse_add.jsonl
    â”œâ”€â”€ patch_stopword_add.jsonl
    â”œâ”€â”€ patch_molecule_add.jsonl
    â”‚
    â”œâ”€â”€ # [Phase 9] Statistics Data
    â”œâ”€â”€ stats/
    â”‚   â”œâ”€â”€ w1_global.json            # W1 global statistics
    â”‚   â”œâ”€â”€ w2_records.jsonl          # W2 condition-sliced records
    â”‚   â”œâ”€â”€ w2_conditions.jsonl       # W2 condition registry
    â”‚   â”œâ”€â”€ w3_candidates/            # W3 axis candidate outputs
    â”‚   â””â”€â”€ w4_projections/           # W4 per-article resonance vectors
    â”‚
    â””â”€â”€ cache/                       # Search cache directory
```

---

## 4. Binary Emergence Model

### 4.1 Two Information Sources

```
A. Prior Structure (çµŒé¨“å‰‡)
    - 326 concept definitions
    - 163 symmetric pair relationships
    - 10 axes Ã— 48 levels
    - Structural knowledge: "war and peace should be symmetric"

B. Accumulated Data (è“„ç©ãƒ‡ãƒ¼ã‚¿)
    - Occurrence frequency (count)
    - Linkage weights (w_ij)
    - Temporal patterns

Emergence:
    (Prior Structure â†” Accumulated Data) â‡’ Introspection
```

### 4.2 Ternary Emergence

Time enters as the third term:

```
(Prior Structure â†” Accumulated Data) â†” Time â‡’ Deep Introspection

1st observation: Time = 0, prior vs observation only
nth observation: Time becomes meaningful, ternary complete
```

### 4.3 Weight Calculation

```python
# Binary Composite Weight
weight[i] = Î±_t Ã— data_weight[i] + (1 - Î±_t) Ã— structure_weight[i]

# Composite Coefficient
Î±_t = min(0.7, n / 100)
    # n = 0:   Î±_t = 0.0 (prior only)
    # n = 100: Î±_t = 0.7 (data dominant, capped)
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
| Ã— | Connection | A Ã— B (connect two semantic units) |
| â–· | Action | A â–· B (A acts on B) |
| â†’ | Transition | A â†’ B (state/meaning change) |
| âŠ• | Juxtaposition | A âŠ• B (simultaneous presentation) |
| \| | Condition | A \| B (A under condition B) |
| â—¯ | Target | A Ã— â—¯ (target of A) |
| â†º | Recursion | A â†º A (self-reference) |
| ã€ˆã€‰ | Hierarchy | ã€ˆA Ã— Bã€‰ (internal structure/scope) |
| â‰¡ | Equivalence | A â‰¡ B (theoretical identity) |
| â‰ƒ | Practical Equivalence | A â‰ƒ B (equivalence within Îµ) |
| Â¬ | Negation | Â¬A (meaning inversion) |
| â‡’ | Emergence | A â‡’ B (unspecified direction) |
| â‡’+ | Creative Emergence | New meaning generation |
| -\|> | Destructive Emergence | Structure reset/reboot |

### 6.3 Expression Capacity

```
Semantic Atoms: 326
Axes Ã— Levels: 48 patterns
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
Step 1: Trigger match â†’ "forgive" hits EMO.forgiveness
Step 2: Negation detected â†’ "cannot"
Step 3: Entity detected â†’ "you" = â—¯ (other)
Step 4: Formula = Â¬(EMO.forgiveness Ã— â—¯)
Step 5: Resolve Â¬EMO.forgiveness â†’ NEGATION_MAP â†’ SOC.refuse
```

**Output:**
```json
{
  "concept_id": "SOC.refuse",
  "axis": "interconnection",
  "level": "independent",
  "evidence": "Formula: Â¬(EMO.forgiveness Ã— â—¯)",
  "_formula": {"type": "EXPR", "op": "Â¬", "args": [...]},
  "_extracted": "EMO.forgiveness"
}
```

**Note:** This is legacy behavior. Future integration should use Synapse vector lookup before trigger matching.

---

## 7. Error (Îµ) Measurement

### 7.1 Symmetry Error

```python
Îµ_sym[i] = | weight[i] - weight[symmetric(i)] |
Îµ_sym_total = Î£ Îµ_sym[i] / num_pairs
```

### 7.2 Linkage Error

```python
Îµ_link = Î£_{(i,j)} w_ij Ã— || x[i,:] - x[j,:] ||Â²
```

### 7.3 Total Error

```python
Îµ_total = Î»_sym Ã— Îµ_sym_total + Î»_link Ã— Îµ_link
```

### 7.4 Interpretation

Îµ is not "error" but "evidence that structure is alive".

```
Îµ = 0:  Perfect equilibrium (rigid, dead)
Îµ > 0:  Dynamic equilibrium (alive)
Îµ >> threshold: Imbalance (sign of Reboot)
```

---

## 8. Audit Pipeline (Phase 7)

### 8.1 Philosophy: Volatility-First

> "Is this understood correctly?"  
> â†’ "What does 'correct' even mean?"  
> â†’ "Correctness always fluctuates"  
> â†’ **Volatility Detection**

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

**v5.3.5 Update**: SearXNG â†’ MultiSourceProvider

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
| Candidate | â—‹ | < 0.3 | Proceed |
| Deferred | â— | 0.3 - 0.6 | More observation |
| Quarantine | â— | > 0.6 | Human review |

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
| Î± | 0.7 | Linkage weight frequency coefficient |
| Ï„ | 86400 | Linkage weight decay constant (seconds) |
| Î³ | 0.1 | Observation injection rate |
| Îµ_0 | 0.01 | Initial state base value |
| Î´ | 0.005 | Initial state hierarchy bonus |
| Î»_sym | 1.0 | Symmetry constraint weight |
| Î»_link | 1.0 | Linkage constraint weight |

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
    OP_NOT = "Â¬"
    OP_CONN = "Ã—"
    OP_JUXT = "âŠ•"
    
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
| 7A | âœ… Complete | unknown_queue.jsonl |
| 7B | âœ… Complete | Aggregation logic |
| 7B+ | âœ… Complete | MultiSourceProvider (v5.3.5, 100% success) |
| 7C | âœ… Complete | Structural audit |
| 7C' | âœ… Complete | LLM semantic audit |
| 7D | âœ… Complete | Meta-auditor, rule calibration |
| **8** | **âœ… Complete** | **Introspective Engine v1** |
| **9-0** | **âœ… Complete** | **W0 ContentGateway** |
| **9-1** | **âœ… Complete** | **W1 Global Statistics** |
| **9-2** | **âœ… Complete** | **W2 Conditional Statistics** |
| **9-3** | **âœ… Complete** | **W3 Axis Candidates (S-Score)** |

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
- âœ… Config values injected explicitly
- âœ… ALLOWED_POS includes 's'
- âœ… determinism_hash includes config_snapshot
- âœ… Fallback to legacy triggers (Hybrid mode)

### 15.3 Modular Architecture (GPT Recommended)

Sensor V2 has been refactored into modular components for maintainability and testability.

**Directory Structure:**
```
esde/
â”œâ”€â”€ esde_sensor_v2_modular.py     # Facade (thin orchestration)
â””â”€â”€ sensor/
    â”œâ”€â”€ __init__.py               # Package exports
    â”œâ”€â”€ loader_synapse.py         # SynapseLoader
    â”œâ”€â”€ extract_synset.py         # SynsetExtractor  
    â”œâ”€â”€ rank_candidates.py        # CandidateRanker
    â”œâ”€â”€ legacy_trigger.py         # LegacyTriggerMatcher (v1 fallback)
    â”œâ”€â”€ audit_trace.py            # AuditTracer (counters/hash/evidence)
    â”œâ”€â”€ molecule_validator.py     # Phase 8-2 Validator
    â””â”€â”€ molecule_generator.py     # Phase 8-2 Generator
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
| Atom Integrity | 2-tier: Sensor candidates + Glossary subset | âœ… v8.2.1 |
| Operator Valid | v0.3 operators (Ã—, â–·, â†’, âŠ•, Â¬, etc.) | âœ… |
| Syntax Check | Bracket matching | âœ… |
| Coordinate Valid | axis/level exist in Glossary | âœ… |
| Evidence Linkage | span range valid + text_ref == text[span] | âœ… v8.2.1 |
| Coverage Policy | High confidence + null coords = warning | âœ… |

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
  "formula": "aa_1 â–· aa_2",
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
  synsets_checked=3, with_edges=3 âœ…

[Input] apprenticed to a master
  Engine: v2_synapse
  Candidates: 1 (ABS.bound via apprenticed.s.01) â† GPT Audit Key Test
  synsets_checked=5, with_edges=1 âœ…

[Input] I cannot forgive you
  Engine: v2_synapse+v1_fallback
  Candidates: 0 (forgive.v not in synapses)
  synsets_checked=2, with_edges=0 âœ… Expected
```

**MoleculeValidator:**
```
Valid: True
Errors: []
Coordinate completeness: 1.0 âœ…
```

**MoleculeGenerator (Mock):**
```
Success: True
Abstained: False
axis: null, level: null (Never Guess) âœ…
validator_status: pass âœ…
```

### 15.6 Phase 8-3: Live LLM Integration

**Implementation:** `sensor/molecule_generator_live.py`

#### 15.6.1 Spec v8.3 Guardrails

| Guardrail | Implementation | Status |
|-----------|----------------|--------|
| Strict Output Contract | "Return ONLY JSON" in system prompt | âœ… |
| Zero Chatter | QwQ `<think>` tag removal | âœ… |
| Fail-Closed Parsing | No fuzzy logic, markdown removal only | âœ… |
| System-Calculated Span | SpanCalculator (token proximity) | âœ… |
| Coordinate Coercion | CoordinateCoercer with logging | âœ… |
| Empty Check | Skip LLM if no candidates | âœ… |

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
Span: [2, 6] âœ…

=== The law requires obedience ===
Success: True
Molecule: EMO.respect@value_generation:ethical
Formula: aa_1
Span: [17, 26] âœ…

=== I cannot forgive you ===
Candidates: 0
LLM Called: False âœ… (Empty Check working)
```

#### 15.6.3 GPT Audit 4-Case Tests

| Test | Description | Result |
|------|-------------|--------|
| A | Same text_ref multiple occurrences | âœ… Token proximity |
| B | text_ref spelling variation | âœ… span=null + warning |
| C | Coordinate mismatch (Glossaryå¤–) | âœ… Coercion logged |
| D | JSON pollution (`<think>`, markdown) | âœ… Parsed correctly |

### 15.7 Phase 8-4: Semantic Stability & Drift Audit

**Implementation:** `esde_stability_audit.py`

#### 15.7.1 Audit Scope

| Item | In Scope | Out of Scope |
|------|----------|--------------|
| Stability verification | âœ… | Ledger implementation |
| Drift measurement | âœ… | Operator expansion |
| Pollution blocking | âœ… | Glossary update |

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
| edge_cases | 95% | 15.8% | âš ï¸ |
| human_emotion | 72% | 5.0% | âœ… |
| ai_text | 75% | 11.7% | âš ï¸ |
| human_logic | 70% | 10.7% | âš ï¸ |
| noise | 15% | 0% | âœ… |

**Mode B (temperature=0.1):**

| Category | Success Rate | Drift | Judgment |
|----------|--------------|-------|----------|
| edge_cases | 95% | 17.4% | âš ï¸ |
| human_emotion | 90% | 8.4% | âœ… |
| ai_text | 75% | 10.3% | âš ï¸ |
| human_logic | 70% | 20.9% | âš ï¸ |
| noise | 14% | 1.2% | âœ… |

**Immunity Metrics:**

| Metric | Count | Rate | Judgment |
|--------|-------|------|----------|
| Validator Blocks | 8 | 0.23% | âœ… Very Low |
| Coordinate Coercions | 5 | 0.14% | âœ… Very Low |
| Pollution Blocked | True | - | âœ… |

#### 15.7.4 Mode A Quick Test (Driftåˆ‡ã‚Šåˆ†ã‘)

```
Temperature: 0.0
Samples: 10 items Ã— 10 runs = 100 runs

Results:
  Drift-A (Atom): 0.0000 âœ… Perfect
  Drift-B (Coord): 0.0000 âœ… Perfect
  Drift-C (Formula): 0.0000 âœ… Perfect
  
Conclusion: temp=0 achieves deterministic output at Atom/Coord level.
Residual drift in full audit is Formula expression variation only.
```

#### 15.7.5 FormulaValidator Fix

**Issue:** `Â¬Â¬aa_1` (double negation) was blocked as unknown operator.

**Root Cause:** `extract_operators()` treated `Â¬Â¬` as single token.

**Fix:** Character-by-character parsing for consecutive operators.

```python
# Before: ['Â¬Â¬'] â†’ Unknown operator error
# After:  ['Â¬', 'Â¬'] â†’ Valid (two negations)
```

### 15.8 Phase 8-5: Semantic Memory (Ephemeral Ledger)

**Implementation:** `ledger/ephemeral_ledger.py`

#### 15.8.1 Memory Constitution (è¨˜æ†¶ã®æ†²æ³•)

Phase 8-5ã¯ã€Œè¨˜æ†¶ã—ã¦ã‚‚ã‚·ã‚¹ãƒ†ãƒ ãŒç ´ç¶»ã—ãªã„æ¡ä»¶ã€ã‚’ç¢ºç«‹ã™ã‚‹ãƒ•ã‚§ãƒ¼ã‚ºã€‚

**ç¦æ­¢äº‹é … (Strict Prohibitions):**

| é …ç›® | ç†ç”± |
|------|------|
| NO Persistence | æ°¸ç¶šåŒ–ã¯8-6ä»¥é™ |
| NO Hash Chain | æ”¹ç«„é˜²æ­¢ã¯æ™‚æœŸå°šæ—© |
| NO Semantic Logic | æ¼”ç®—å­çµ±åˆã¯å¯¾è±¡å¤– |
| NO Emergence Classification | å‰µç™ºåˆ¤å®šã¯å¯¾è±¡å¤– |

#### 15.8.2 Memory Math

**Decay (è‡ªç„¶æ¸›è¡°):**
```
w = w_prev Ã— exp(-dt / Ï„)
```

**Reinforcement (è¦³æ¸¬å¼·åŒ–):**
```
w = w + Î± Ã— (1 - w)
```
- Î± (Learning Rate) = 0.2

**Oblivion (å¿˜å´):**
```
if w < Îµ: purge
```
- Îµ (Threshold) = 0.01

#### 15.8.3 Tau Policy

| Temporal Level | Ï„ (sec) | Human |
|----------------|---------|-------|
| permanence | 86400 | 24h |
| continuation | 3600 | 1h |
| establishment | 3600 | 1h |
| transformation | 1800 | 30m |
| indication | 300 | 5m |
| emergence | 60 | 1m |
| **Default** | **300** | **5m** |

#### 15.8.4 Fingerprint (åŒä¸€æ€§åˆ¤å®š)

```
Key = SHA256( Sorted(AtomIDs) + "::" + OperatorType )
```

- `A â–· B` ã¨ `A â†’ B` ã¯åˆ¥ã®è¨˜æ†¶
- è‡ªç„¶æ·˜æ±°ï¼ˆWeightç«¶äº‰ï¼‰ã«ä»»ã›ã‚‹

#### 15.8.5 Conflict Handling (çŸ›ç›¾ã®æ‰±ã„)

**æ–¹é‡: å…±å­˜ (Co-existence)**

- Love ã¨ Hate ã¯åˆ¥Entryã¨ã—ã¦å…±å­˜
- ç›¸æ®ºï¼ˆå¼•ãç®—ï¼‰ã¯è¡Œã‚ãªã„
- Weightç«¶äº‰ã§è‡ªç„¶æ·˜æ±°

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
| Conflict Coexistence | âœ… PASS | Love + Hate å…±å­˜ |
| Reinforcement Stability | âœ… PASS | Weight â‰¤ 1.0 |
| Retention by Tau | âœ… PASS | 10åˆ†å¾Œ9ä»¶æ®‹å­˜ |
| No-Pollution | âœ… PASS | ã‚·ã‚¹ãƒ†ãƒ å®‰å®š |

**Reinforcement Observation:**
```
INT008: create (w=1.00)
INT009: reinforce (w=0.99)
INT010: reinforce (w=0.98)
```

### 15.9 Phase 8-6: Persistent Semantic Ledger

**Theme:** Semantic Time Crystallization (æ„å‘³æ™‚é–“ã®çµæ™¶åŒ–)

Phase 8-6ã¯ã€ESDEã«**ã€Œä¸å¯é€†ãªæ™‚é–“ã€**ã¨**ã€Œæ­£å²ï¼ˆHistoryï¼‰ã€**ã‚’ç‰©ç†çš„ã«å®Ÿè£…ã™ã‚‹ã€‚
Phase 8-5ã®æ®ç™ºæ€§ãƒ¡ãƒ¢ãƒªã‚’ã€**Hash Chainï¼ˆãƒãƒƒã‚·ãƒ¥ãƒã‚§ãƒ¼ãƒ³ï¼‰** æŠ€è¡“ã‚’ç”¨ã„ã¦ãƒ‡ã‚£ã‚¹ã‚¯ã«åˆ»ã¿ã€
ã‚·ã‚¹ãƒ†ãƒ ã®å†èµ·å‹•å¾Œã‚‚ã€Œè‡ªå·±ã®æ–‡è„ˆã€ã‚’ç¶­æŒå¯èƒ½ã«ã™ã‚‹ã€‚

#### 15.9.1 Spec v8.6.1 (GPT Audit Approved)

**GPTç›£æŸ»ä¿®æ­£ï¼ˆ4ç‚¹ï¼‰åæ˜ :**

| # | è¦ä»¶ | å¯¾å¿œ |
|---|------|------|
| 1 | validate()ã¯è¡Œæ–‡å­—åˆ—ã‚’canonicalã¨ã—ã¦æ‰±ã† | âœ… JSONå†dumpãªã— |
| 2 | æœ€çµ‚è¡Œç ´æã¯salvageãªã—ã§åœæ­¢ | âœ… IntegrityError |
| 3 | event_idã‹ã‚‰metaé™¤å¤–ï¼ˆæ„å‘³åŒä¸€æ€§ï¼‰ | âœ… metaé™¤å¤– |
| 4 | rehydrationã¯ledger replay | âœ… é †æ¬¡é©ç”¨ |

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

**Event ID (æ„å‘³åŒä¸€æ€§):**
```
Target = {v, ledger_id, event_type, direction, data}
event_id = SHA256(Canonical(Target))
```
- `meta`ã‚’é™¤å¤–: åŒã˜æ„å‘³ã‚¤ãƒ™ãƒ³ãƒˆã¯åŒã˜ID

**Self Hash (æ­´å²é€£éŽ–):**
```
Target = {v, ledger_id, seq, ts, event_type, direction, data, meta, hash:{algo,prev,event_id}}
self = SHA256(prev + "\n" + Canonical(Target))
```

**Genesis Block:**
```
seq: 0
prev: "0" Ã— 64
data: {"message": "Aru - There is"}
```

#### 15.9.4 Validation Tests (T861-T865)

| ID | ãƒ†ã‚¹ãƒˆ | æ¤œè¨¼å†…å®¹ |
|----|--------|----------|
| T861 | chain_validates | self hashã®å†è¨ˆç®—ä¸€è‡´ |
| T862 | prev_linkage | prevâ†’selfé€£éŽ– |
| T863 | truncation | JSONãƒ‘ãƒ¼ã‚¹ã‚¨ãƒ©ãƒ¼æ¤œçŸ¥ |
| T864 | monotonic_seq | seqæ¬ ç•ªãªã— |
| T865 | header_match | v, ledger_id, algoå®šæ•° |

#### 15.9.5 Test Results

```
============================================================
ESDE Phase 8-6: Ledger Test Suite
============================================================

Genesis Creation: âœ… PASS
Append and Validate: âœ… PASS
Molecule Observe: âœ… PASS
T861: Chain Validates: âœ… PASS
T862: Tamper Detected: âœ… PASS
T863: Truncation Detected: âœ… PASS
T864: Reorder Detected: âœ… PASS
T865: Header Mismatch: âœ… PASS
Rehydration: âœ… PASS
Sleep Decay: âœ… PASS
Direction Values: âœ… PASS
Event ID Excludes Meta: âœ… PASS
Conflict Coexistence: âœ… PASS

Results: 13/13 passed
âœ… ALL TESTS PASSED - Phase 8-6 Ready
```

#### 15.9.6 New Files

| File | Description |
|------|-------------|
| `ledger/canonical.py` | JSONæ­£è¦åŒ–ï¼ˆsort_keys, separatorsï¼‰ |
| `ledger/chain_crypto.py` | Hashè¨ˆç®—ï¼ˆevent_id, self hashï¼‰ |
| `ledger/persistent_ledger.py` | PersistentLedgerã‚¯ãƒ©ã‚¹ |
| `test_phase86_ledger.py` | ç›£æŸ»ãƒ†ã‚¹ãƒˆï¼ˆT861-T865ï¼‰ |

### 15.10 Phase 8-7: Semantic Index (L2)

**Theme:** Projection of Understanding (ç†è§£ã®å°„å½±)

Phase 8-7ã¯ã€L1 (Immutable Ledger) ã®ä¸Šã«**L2: Semantic Index**ã‚’æ§‹ç¯‰ã™ã‚‹ã€‚
ã€Œæ¤œç´¢ï¼ˆIndexï¼‰ã€ã€Œç¡¬ç›´ï¼ˆRigidityï¼‰ã€ã€Œå‚¾å‘ï¼ˆDirection Balanceï¼‰ã€ã‚’å³ç­”ã§ãã‚‹ã‚¤ãƒ³ãƒ¡ãƒ¢ãƒªæ§‹é€ ã€‚

> **Aruism Philosophy:**
> - **L1 (Ledger):** ã€Œã‚ã‚‹ï¼ˆAruï¼‰ã€ã€‚å¤‰æ›´ä¸å¯èƒ½ãªäº‹å®Ÿã®ç¾…åˆ—ã€‚
> - **L2 (Index):** ã€Œã‚ã‹ã‚‹ï¼ˆUnderstandingï¼‰ã€ã€‚äº‹å®Ÿã‚’ç‰¹å®šã®è¦–ç‚¹ã§å†æ§‹æˆã—ãŸå°„å½±ã€‚

#### 15.10.1 Architecture (Dual Layer System)

```
[ Query API ] -> ãƒ¦ãƒ¼ã‚¶ãƒ¼/ã‚·ã‚¹ãƒ†ãƒ ã¸ã®å›žç­”
      ^
      | (Read)
[ L2: Semantic Index + Rigidity Signals ] (Mutable / In-Memory)
      ^
      | (Project: Rebuild / Incremental)
[ L1: Immutable Ledger ] (Immutable / JSONL)
```

**çµ¶å¯¾ãƒ«ãƒ¼ãƒ«:**
1. L2ã¯L1ã®å°„å½±ï¼ˆç‹¬è‡ªæƒ…å ±ã‚’æŒãŸãªã„ï¼‰
2. L1ä¸å¤‰ï¼ˆL2æ§‹ç¯‰ãŒL1ã‚’æ›¸ãæ›ãˆãªã„ï¼‰
3. Parity Consistencyï¼ˆrebuildçµæžœ == é€æ¬¡æ›´æ–°çµæžœï¼‰

#### 15.10.2 Rigidity Calculation

```
R = N_mode / N_total
```

- `N_total`: ãã®AtomãŒè¦³æ¸¬ã•ã‚ŒãŸç·å›žæ•°
- `N_mode`: æœ€é »å‡ºformula_signatureã®å‡ºç¾å›žæ•°

| Rå€¤ | åˆ†é¡ž | è§£é‡ˆ |
|-----|------|------|
| â‰¥0.9 | crystallized | çµæ™¶åŒ–ï¼ˆå®Œå…¨ã«å›ºå®šï¼‰ |
| â‰¥0.7 | rigid | ç¡¬ç›´ |
| â‰¥0.4 | stable | å®‰å®š |
| â‰¥0.2 | fluid | æµå‹•çš„ |
| <0.2 | volatile | æ®ç™ºçš„ï¼ˆæŽ¢ç´¢ä¸­ï¼‰ |

#### 15.10.3 GPTç›£æŸ»v8.7.1å¯¾å¿œ

| é …ç›® | å®Ÿè£… |
|------|------|
| formula_signatureå®šç¾© | formula â†’ atoms/opsé€£çµ â†’ `__unknown__` |
| L2è‚¥å¤§åŒ–é˜²æ­¢ | `deque(maxlen=10000)` |
| windowå¼•æ•° | `get_frequency(atom_id, window=1000)` ç­‰ |

#### 15.10.4 Query API

```python
from index import QueryAPI

api = QueryAPI(ledger)

# ç¡¬ç›´åº¦
api.get_rigidity("EMO.love", window=1000)

# å‡ºç¾é »åº¦
api.get_frequency("EMO.love", window=1000)

# æ–¹å‘æ€§ãƒãƒ©ãƒ³ã‚¹
api.get_recent_directions(limit=1000)

# Atomè©³ç´°
api.get_atom_info("EMO.love")

# å…±èµ·
api.get_cooccurrence("EMO.love", "ACT.create")
```

#### 15.10.5 Test Results

```
============================================================
ESDE Phase 8-7: Index Test Suite
============================================================

Index Creation: âœ… PASS
Index Update: âœ… PASS
Formula Signature Extraction: âœ… PASS
Parity: Rebuild vs Incremental: âœ… PASS
Rigidity: Constant Formula (R=1.0): âœ… PASS
Rigidity: Varying Formula (R<1.0): âœ… PASS
Rigidity: Mixed Pattern (R=0.7): âœ… PASS
No Side Effect on Ledger: âœ… PASS
QueryAPI Basic: âœ… PASS
QueryAPI AtomInfo: âœ… PASS
QueryAPI Cooccurrence: âœ… PASS
QueryAPI Window: âœ… PASS
Direction Balance: âœ… PASS

Results: 13/13 passed
âœ… ALL TESTS PASSED - Phase 8-7 Ready
```

#### 15.10.6 New Files

| File | Description |
|------|-------------|
| `index/__init__.py` | ãƒ‘ãƒƒã‚±ãƒ¼ã‚¸ã‚¨ã‚¯ã‚¹ãƒãƒ¼ãƒˆ |
| `index/semantic_index.py` | L2ã‚¤ãƒ³ãƒ¡ãƒ¢ãƒªæ§‹é€  |
| `index/projector.py` | L1â†’L2æŠ•å½± |
| `index/rigidity.py` | ç¡¬ç›´åº¦è¨ˆç®— |
| `index/query_api.py` | å¤–éƒ¨API |
| `test_phase87_index.py` | Parity/Rigidityãƒ†ã‚¹ãƒˆ |

### 15.11 Next Steps

| Priority | Task | Status |
|----------|------|--------|
| 1 | ~~Sensor V2 implementation~~ | âœ… Complete |
| 2 | ~~Modular refactoring~~ | âœ… Complete |
| 3 | ~~Phase 8-2 Validator/Generator~~ | âœ… Complete |
| 4 | ~~Phase 8-3 Live LLM Integration~~ | âœ… Complete |
| 5 | ~~Phase 8-4 Stability Audit~~ | âœ… **PASS** |
| 6 | ~~Phase 8-5 Ephemeral Ledger~~ | âœ… **PASS** |
| 7 | ~~Phase 8-6 Persistent Ledger~~ | âœ… **PASS** |
| 8 | ~~Phase 8-7 Semantic Index~~ | âœ… **PASS** |
| 9 | ~~Phase 8-8 Feedback Loop~~ | âœ… **PASS** |
| 10 | ~~Phase 8-9 Monitor & Long-Run~~ | âœ… **PASS** |
| 11 | ~~Phase 8-10 Schema Consolidation~~ | âœ… **PASS** |
| 12 | ~~Phase 9-0 ContentGateway~~ | âœ… **PASS** |
| 13 | ~~Phase 9-1 W1 Global Stats~~ | âœ… **PASS** |
| 14 | ~~Phase 9-2 W2 Conditional Stats~~ | âœ… **PASS** |
| 15 | ~~Phase 9-3 W3 Axis Candidates~~ | âœ… **PASS** |
| 16 | ~~Phase 9-4 W4 Structural Projection~~ | âœ… **PASS** |
| 17 | ~~Phase 9-5 W5 Structural Condensation~~ | ✅ **PASS** |
| 18 | ~~Phase 9-6 W6 Structural Observation~~ | ✅ **PASS** |
| 19 | Phase 9-7 W7 Judgment Layer | **Next** |

### 15.12 Development Workflow

```
Gemini:  Design proposal (What to do)
   â†“
GPT:     Audit (Is it valid?)
   â†“
Claude:  Implementation (How to build)
   â†“
GPT:     Deliverable audit (Did we build it right?)
   â†“
All:     Phase 9-7 Scope Decision  â† Current
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
| **5.4.5-P9.5** | **2026-01-22** | **Phase 9-5: W5 Structural Condensation (Islands)** |
| **5.4.6-P9.6** | **2026-01-22** | **Phase 9-6: W6 Structural Observation (Evidence)** |

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
   - SpanCalculator: text_ref â†’ span
   - CoordinateCoercer: invalid â†’ null + log
   - FormulaValidator: consecutive operators fix (Â¬Â¬)

4. **Test Results**
   - Live Test: 3/3 PASS
   - GPT Audit 4-Case: 4/4 PASS

### v5.3.5-P8.4 Changelog (Phase 8-4)

1. **Semantic Stability Audit**
   - 3500 runs (Mode A: 500, Mode B: 3000)
   - 5 categories Ã— 20 sentences = 100 corpus
   - Parallel execution (8 workers)

2. **Drift Analysis (A/B/C)**
   - Drift-A (Atom): 0% at temp=0 âœ…
   - Drift-B (Coordinate): 0% at temp=0 âœ…
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
   - Decay: w Ã— exp(-dt/Ï„)
   - Reinforce: w + Î±(1-w), Î±=0.2
   - Oblivion: purge if w < Îµ, Îµ=0.01
   - Tau Policy: temporal axis â†’ time constant

3. **Memory Constitution**
   - Conflict Co-existence (no cancellation)
   - Fingerprint-based identity
   - Natural selection via Weight competition

4. **Unit Tests (11/11 PASS)**
   - Decay Math âœ…
   - Reinforcement Math âœ…
   - Oblivion Threshold âœ…
   - Tau Policy âœ…
   - Tau Affects Lifespan âœ…
   - Conflict Coexistence âœ…
   - Retention Rate âœ…
   - Reinforcement Stability âœ…
   - Asymptotic Approach âœ…
   - Fingerprint Identity âœ…
   - Operator Extraction âœ…

5. **Integration Test (E2E)**
   - 14 inputs, 12 success (85.7%)
   - Block rate: 0%, Coercion rate: 0%
   - Ledger: 9 entries, 12 observations
   - All audit tests PASS

### v5.3.6-P8.6 Changelog (Phase 8-6)

1. **Persistent Ledger Package**
   - `canonical.py`: JSONæ­£è¦åŒ–ï¼ˆãƒã‚¤ãƒˆä¸€è‡´ä¿è¨¼ï¼‰
   - `chain_crypto.py`: Hash Chainï¼ˆevent_id / self hashï¼‰
   - `persistent_ledger.py`: PersistentLedgerã‚¯ãƒ©ã‚¹

2. **Hash Chain Implementation**
   - Atomic Append: flush + fsync
   - Genesis Block: "Aru - There is"
   - Tamper Detection: 5ç¨®ã®Integrity Check (T861-T865)

3. **GPT Audit v8.6.1 Compliance**
   - validate()ã¯è¡Œæ–‡å­—åˆ—ã‚’canonicalã¨ã—ã¦æ‰±ã†
   - æœ€çµ‚è¡Œç ´æã¯salvageãªã—ã§åœæ­¢
   - event_idã‹ã‚‰metaé™¤å¤–ï¼ˆæ„å‘³åŒä¸€æ€§ï¼‰
   - rehydrationã¯ledger replay

4. **Emergence Directionality**
   - `=>` : æœªç¢ºå®š
   - `=>+` : å‰µé€ çš„å‰µç™ºï¼ˆconnectivityå¢—åŠ ï¼‰
   - `-|>` : ç ´å£Šçš„å‰µç™ºï¼ˆReboot/è»¸é·ç§»ï¼‰

5. **Test Results (13/13 PASS)**
   - Genesis Creation âœ…
   - Append and Validate âœ…
   - Molecule Observe âœ…
   - T861: Chain Validates âœ…
   - T862: Tamper Detected âœ…
   - T863: Truncation Detected âœ…
   - T864: Reorder Detected âœ…
   - T865: Header Mismatch âœ…
   - Rehydration âœ…
   - Sleep Decay âœ…
   - Direction Values âœ…
   - Event ID Excludes Meta âœ…
   - Conflict Coexistence âœ…

### v5.3.7-P8.7 Changelog (Phase 8-7)

1. **Semantic Index Package (L2)**
   - `semantic_index.py`: AtomStats, DirectionStatsæ§‹é€ 
   - `projector.py`: L1â†’L2æŠ•å½±ï¼ˆrebuild / on_eventï¼‰
   - `rigidity.py`: ç¡¬ç›´åº¦è¨ˆç®—
   - `query_api.py`: å¤–éƒ¨ã‚¯ã‚¨ãƒªAPI

2. **Dual Layer Architecture**
   - L1 (Ledger): ä¸å¤‰ã®æ­£å²
   - L2 (Index): å†æ§‹ç¯‰å¯èƒ½ãªå°„å½±
   - Parity Consistencyä¿è¨¼

3. **Rigidity Signals**
   - R = N_mode / N_total
   - åˆ†é¡ž: crystallized / rigid / stable / fluid / volatile

4. **GPT Audit v8.7.1 Compliance**
   - formula_signature: formula â†’ atoms/opsé€£çµ â†’ `__unknown__`
   - L2è‚¥å¤§åŒ–é˜²æ­¢: `deque(maxlen=10000)`
   - windowå¼•æ•°: å„APIã«å®Ÿè£…

5. **Test Results (13/13 PASS)**
   - Index Creation âœ…
   - Index Update âœ…
   - Formula Signature Extraction âœ…
   - Parity: Rebuild vs Incremental âœ…
   - Rigidity: Constant Formula (R=1.0) âœ…
   - Rigidity: Varying Formula (R<1.0) âœ…
   - Rigidity: Mixed Pattern (R=0.7) âœ…
   - No Side Effect on Ledger âœ…
   - QueryAPI Basic âœ…
   - QueryAPI AtomInfo âœ…
   - QueryAPI Cooccurrence âœ…
   - QueryAPI Window âœ…
   - Direction Balance âœ…

### v5.3.8-P8.8 Changelog (Phase 8-8: Feedback Loop)

1. **Feedback Package**
   - `strategies.py`: GenerationStrategy dataclass
   - `modulator.py`: Modulator class
   - Constants: RIGIDITY_HIGH=0.9, RIGIDITY_LOW=0.3, ALERT_THRESHOLD=0.98

2. **Pipeline Package**
   - `core_pipeline.py`: ESDEPipeline, ModulatedGenerator
   - Full loop: Sensor â†’ Index â†’ Modulator â†’ Generator â†’ Ledger â†’ Index

3. **Strategy Modes**
   | Mode | Condition | Temperature | Action |
   |------|-----------|-------------|--------|
   | NEUTRAL | 0.3 â‰¤ R â‰¤ 0.9 | 0.1 | Normal operation |
   | DISRUPTIVE | R > 0.9 | 0.7 | "Doubt it. Find contradictions." |
   | STABILIZING | R < 0.3 | 0.0 | "Consolidate. Find commonality." |

4. **Alert System**
   - Condition: R â‰¥ 0.98 AND N â‰¥ 10
   - Output: `[ALERT] CONCEPT_CRYSTALLIZED: {atom_id}`

5. **Direction Adjustment**
   - DISRUPTIVE mode â†’ direction = `-|>` (destructive emergence)

6. **Test Results (14/14 PASS)**
   - Constants Verification âœ…
   - Neutral Strategy âœ…
   - Disruptive Strategy âœ…
   - Stabilizing Strategy âœ…
   - Unknown Atom Handling âœ…
   - Target Atom Extraction âœ…
   - Alert Condition âœ…
   - Alert Not Triggered (N<10) âœ…
   - Neutral Loop âœ…
   - Disruptive Feedback âœ…
   - Stabilizing Feedback âœ…
   - No Candidates âœ…
   - Alert (No Index Update) âœ…
   - Alert After Update âœ…

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
   | Rankings | Top Rigid (Râ†’1.0), Top Volatile (Râ†’0.0) |
   | Stats | Index Size, Direction Balance |

5. **Long-Run Report**
   - Execution: steps, duration, stopped_early
   - Results: successful, abstained
   - Alerts: total, atoms
   - Health: ledger_valid, validation_checks
   - Final State: ledger_seq, index_size, direction_balance

6. **GPT Audit v8.9.1 Compliance**
   - Fail Fast: ã‚¨ãƒ©ãƒ¼ç™ºç”Ÿæ™‚ã¯å³åº§ã«åœæ­¢
   - Mock LLM: ãƒ†ã‚¹ãƒˆã§ã¯å®ŸLLMã‚’ä½¿ã‚ãªã„
   - T895: Ledger Invariance Testï¼ˆGenesisä¸å¤‰ï¼‰
   - Rich Fallback: richéžä¾å­˜ã§ã‚‚å‹•ä½œ

7. **Test Results (10/10 PASS)**
   - Monitor State Init âœ…
   - Monitor Update âœ…
   - Monitor Update with Alert âœ…
   - Monitor Render âœ…
   - LongRun Basic (steps=10) âœ…
   - LongRun Alert Count âœ…
   - LongRun Abstain Handling âœ…
   - LongRun Report Structure âœ…
   - T895: Ledger Invariance âœ…
   - CLI Observe (Mock) âœ…

8. **Live Long-Run Results (50 steps)**
   - Steps: 50/50 å®Œèµ°
   - Duration: 478ç§’
   - Successful: 38
   - Abstained: 12
   - Alerts: 1 (EMO.love crystallized)
   - Ledger Valid: âœ…
   - Validation Checks: 6/6 PASS

### 15.10 Phase 8-10: Schema Consolidation & Integration Test

**Date:** 2026-01-20

#### 15.10.1 Issue: Schema Divergence

| Version | Schema | Example |
|---------|--------|---------|
| 8-2 (mock/validator) | nested | `coordinates.axis`, `coordinates.level` |
| 8-3 (live) | flat | `axis`, `level` |
| Synapse (data) | flat | `axis`, `level` |

**Root Cause:** 8-2 mock/validator were tested in isolation; integration tests validated 8-3 Live â†’ Ledger only.

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
Synapse hash recorded: âœ… (3 tests)
```

| Input | Candidates | Formula | Result |
|-------|------------|---------|--------|
| "I love you" | 2 | `aa_1 Ã— aa_2` | âœ… |
| "The law requires obedience" | 5 | `aa_1 â–· aa_2 â–· aa_5` | âœ… |
| "apprenticed to a master" | 1 | `aa_1` | âœ… |
| "" (empty) | 0 | Abstain (LLM not called) | âœ… |
| "asdfghjkl qwertyuiop" | 0 | Abstain | âœ… |

#### 15.10.6 File Structure (Final)

```
esde/
â”œâ”€â”€ test_phase8_integration.py
â”œâ”€â”€ glossary_results.json
â”œâ”€â”€ esde_synapses_v3.json
â”‚
â”œâ”€â”€ Docs/
â”‚   â””â”€â”€ PHASE8_MOLECULE_SCHEMA_FIX.md
â”‚
â””â”€â”€ sensor/
    â”œâ”€â”€ __init__.py              # Package API (canonical only)
    â”œâ”€â”€ constants.py             # VALID_OPERATORS
    â”œâ”€â”€ glossary_validator.py    # GlossaryValidator (neutral)
    â”œâ”€â”€ validator_v83.py         # Canonical validator
    â”œâ”€â”€ molecule_generator_live.py  # v8.3.1 (no legacy dependency)
    â”œâ”€â”€ esde_sensor_v2_modular.py
    â”œâ”€â”€ loader_synapse.py
    â”œâ”€â”€ extract_synset.py
    â”œâ”€â”€ rank_candidates.py
    â”œâ”€â”€ audit_trace.py
    â””â”€â”€ legacy/                  # Deprecated modules
        â”œâ”€â”€ __init__.py
        â”œâ”€â”€ molecule_generator.py
        â”œâ”€â”€ molecule_validator.py
        â””â”€â”€ legacy_trigger.py
```

### Phase 8 Complete Summary

| Phase | Theme | Tests |
|-------|-------|-------|
| 8-1 | Sensor V2 + Modular | âœ… |
| 8-2 | Molecule Generator/Validator | âœ… |
| 8-3 | Live LLM Integration | âœ… |
| 8-4 | Stability Audit | âœ… |
| 8-5 | Ephemeral Ledger | 11/11 âœ… |
| 8-6 | Persistent Ledger | 13/13 âœ… |
| 8-7 | Semantic Index | 13/13 âœ… |
| 8-8 | Feedback Loop | 14/14 âœ… |
| 8-9 | Monitor & Long-Run | 10/10 âœ… |
| 8-10 | Schema Consolidation | 5/5 âœ… |

**Phase 8 Theme: ã€Œå†…çœã‚¨ãƒ³ã‚¸ãƒ³ã®åŸºç›¤æ§‹ç¯‰ã€ - Complete**

---

## Phase 9: Weak Axis Statistics Layer

### Phase 9 Overview

Phase 9ã¯ã€Œå¼±è»¸çµ±è¨ˆå±¤ï¼ˆWeak Axis Statistics Layerï¼‰ã€ã‚’æ§‹ç¯‰ã™ã‚‹ã€‚
äººé–“ã«ã‚ˆã‚‹ãƒ©ãƒ™ãƒªãƒ³ã‚°ç„¡ã—ã«ã€ãƒ‡ãƒ¼ã‚¿ã‹ã‚‰ã€Œè»¸ã®å½±ã€ã‚’æ¤œå‡ºã™ã‚‹çµ±è¨ˆçš„åŸºç›¤ã‚’æä¾›ã™ã‚‹ã€‚

```
W0 (ContentGateway) â†’ å¤–éƒ¨ãƒ‡ãƒ¼ã‚¿ã®æ­£è¦åŒ–ãƒ»å–ã‚Šè¾¼ã¿
    â†“
W1 (Global Statistics) â†’ æ¡ä»¶ã‚’ç„¡è¦–ã—ãŸå…¨ä½“çµ±è¨ˆ
    â†“
W2 (Conditional Statistics) â†’ æ¡ä»¶åˆ¥ã®ã‚¹ãƒ©ã‚¤ã‚¹çµ±è¨ˆ
    â†“
W3 (Axis Candidates) â†’ æ¡ä»¶ç‰¹ç•°æ€§ã‚¹ã‚³ã‚¢ï¼ˆS-Scoreï¼‰ã«ã‚ˆã‚‹è»¸å€™è£œæŠ½å‡º
    â†“
W4 (Structural Projection) â†’ è¨˜äº‹ã‚’W3è»¸å€™è£œã«æŠ•å½±ã€å…±é³´ãƒ™ã‚¯ãƒˆãƒ«ç”Ÿæˆ
    â†“
W5 (Weak Structural Condensation) → 共鳴ベクトルをクラスタリング、島構造抽出
    ↓
W6 (Weak Structural Observation) → 島からEvidence抽出、Topology計算、観測出力
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
S(t, C) = P(t|C) Ã— log((P(t|C) + Îµ) / (P(t|G) + Îµ))

Where:
  P(t|C) = count_cond / total_cond
  P(t|G) = count_global / total_global
  Îµ = 1e-12 (fixed smoothing)
```

**Key Classes:**
- `W3Calculator`: Computes axis candidates
- `W3Record`: Analysis result with positive/negative candidates
- `CandidateToken`: Token with S-Score and probabilities

**INV-W3-001:** No Labeling (output is factual only)
**INV-W3-002:** Immutable Input (W3 does NOT modify W1/W2)
**INV-W3-003:** Deterministic (tie-break by token_norm asc)

### v5.4.4-P9.4 W4: Structural Projection

**Theme:** The Resonance of Weakness (å¼±ã•ã®å…±é³´ãƒ»æ§‹é€ å°„å½±)

**File:** `statistics/w4_projector.py`, `statistics/schema_w4.py`

**Purpose:** Project ArticleRecords onto W3 axis candidates to produce resonance vectors

**Mathematical Model:**
```
R(A, C) = Î£ count(t, A) Ã— S(t, C)

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
| used_w3 | Dict[str, str] | Traceability: cond_sig â†’ w3_analysis_id |
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
- Schema.1: W4Record creation âœ…
- Schema.2: Deterministic analysis ID (P0-1) âœ…
- Schema.3: JSON serialization âœ…
- Schema.4: Canonical dict (P1-1) âœ…
- Projector.1: News article projection âœ…
- Projector.2: Dialog article projection âœ…
- Projector.3: Determinism (INV-W4-002) âœ…
- Projector.4: Traceability (P0-2) âœ…
- Projector.5: Version tracking (P0-3) âœ…
- Projector.6: Full S-Score usage (INV-W4-004) âœ…
- Projector.7: Save and load âœ…

**Example Output:**
```
News article: "The prime minister announced new government policy."
  â†’ News resonance: +0.235
  â†’ Dialog resonance: -0.130

Dialog article: "Hey! Yeah that's so cool lol."
  â†’ News resonance: -0.095
  â†’ Dialog resonance: +0.215
```

**INV-W4-001:** No Labeling (output keys are condition_signature only)
**INV-W4-002:** Deterministic (same article + same W3 â†’ same scores)
**INV-W4-003:** Recomputable (W4 = f(W0, W3))
**INV-W4-004:** Full S-Score Usage (positive AND negative candidates)
**INV-W4-005:** Immutable Input (W4Projector does NOT modify ArticleRecord or W3Record)
**INV-W4-006:** Tokenization Canon (MUST use W1Tokenizer + normalize_token)

---

### v5.4.5-P9.5 W5: Weak Structural Condensation

**Theme:** The Deterministic Shape of Resonance (共鳴の決定論的形状)

**File:** `statistics/w5_condensator.py`, `statistics/schema_w5.py`

**Purpose:** Condense W4 resonance vectors into structural clusters (islands) based on similarity

**Algorithm: Resonance Condensation**
```
1. Validation: Batch size check, duplicate article_id check (P0-A)
2. Preprocessing: L2 normalize all vectors
3. Similarity Matrix: Cosine similarity with fixed rounding (P0-B)
4. Graph Linkage: Edge if similarity >= threshold
5. Component Detection: Connected components via DFS
6. Filtering: Size < min_island_size → noise
7. Centroid: Raw mean with rounding (INV-W5-005)
8. Cohesion: Edge average within island (P1-1)
9. ID Generation: Canonical JSON hash (INV-W5-006)
```

**Key Classes:**
- `W5Condensator`: Resonance-based structural condensation
- `W5Structure`: Snapshot of condensation results
- `W5Island`: Condensation unit (connected component)

**Key Fields (W5Structure):**

| Field | Type | Description |
|-------|------|-------------|
| structure_id | str | Deterministic ID (INV-W5-004) |
| islands | List[W5Island] | Extracted clusters |
| noise_ids | List[str] | Articles not forming islands |
| input_count | int | Total input records |
| island_count | int | Number of islands found |
| noise_count | int | Number of noise items |
| threshold | float | Similarity threshold used (default: 0.70) |
| min_island_size | int | Minimum size filter (default: 3) |
| algorithm | str | "ResonanceCondensation-v1" |
| vector_policy | str | "mean_raw_v1" |
| created_at | str | Timestamp (excluded from canonical check) |

**Key Fields (W5Island):**

| Field | Type | Description |
|-------|------|-------------|
| island_id | str | SHA256({"members": sorted_member_ids}) |
| member_ids | List[str] | Sorted article_ids in island |
| size | int | Number of members |
| representative_vector | Dict[str, float] | Raw mean, rounded to 9 decimals |
| cohesion_score | float | Average edge similarity within island |

**GPT Audit P0/P1 Compliance:**
- P0-A: Duplicate article_id check → ValueError
- P0-B: Similarity rounded to 12 decimals before threshold comparison
- P1-1: Cohesion = average of edges that formed the island (not all pairs)
- P1-4: Batch size limit (2000) to prevent O(N²) explosion

**Test Results:**
- Schema.1: W5Island creation ✅
- Schema.2: Canonical JSON (INV-W5-006) ✅
- Schema.3: W5Structure creation ✅
- Schema.4: Canonical dict excludes created_at (INV-W5-008) ✅
- Schema.5: JSON round-trip ✅
- Condensator.1: Empty input ✅
- Condensator.2: P0-A Duplicate article_id detection ✅
- Condensator.3: P0-B Boundary similarity stability ✅
- Condensator.4: Basic condensation with clusters ✅
- Condensator.5: P1-4 Batch size limit ✅
- Condensator.6: INV-W5-008 created_at excluded ✅

**INV-W5-001:** No Naming (output must not contain natural language labels)
**INV-W5-002:** Topological Identity (island_id from member IDs only, no floating-point)
**INV-W5-003:** Fixed Metric (similarity = L2-normalized Cosine, operator >=)
**INV-W5-004:** Parameter Traceability (structure_id includes input w4_analysis_ids + params)
**INV-W5-005:** Canonical Vector (vector values must be rounded before storage)
**INV-W5-006:** ID Collision Safety (Canonical JSON hash, string join forbidden)
**INV-W5-007:** Structure Identity (uses w4_analysis_id, not article_id)
**INV-W5-008:** Canonical Output (created_at excluded from identity check)

---

### v5.4.6-P9.6 W6: Weak Structural Observation

**Theme:** The Observatory - From Structure to Evidence (構造から証拠へ)

**File:** `discovery/w6_analyzer.py`, `discovery/w6_exporter.py`, `discovery/schema_w6.py`

**Purpose:** Extract evidence and topology from W5 structures for human review

**Core Principle:** W6 is an observation window, NOT a computation layer.

**Algorithm: Evidence Extraction (P0-X1: mean_s_score_v1)**
```
evidence(token) = mean_r( S(token, cond(r)) * I[token in article(r)] )

Where:
  r = iterates over all articles in the island
  cond(r) = condition_signature used by article r
  I[...] = 1 if token exists in article, else 0
  denominator = total island articles (including zeros)
```

**Algorithm: Topology Calculation**
```
For each island pair (A, B):
  1. Get representative_vector from W5 (INV-W6-004)
  2. Compute cosine similarity
  3. Distance = 1.0 - round(cos_sim, 12)
  4. Sort by (-distance, pair_id) for most distant first
```

**Key Classes:**
- `W6Analyzer`: Evidence extraction + topology calculation
- `W6Exporter`: Multi-format export (JSON/MD/CSV)
- `W6Observatory`: Main observation output

**Key Fields (W6Observatory):**

| Field | Type | Description |
|-------|------|-------------|
| observation_id | str | SHA256(structure_id + w4_ids + params) |
| input_structure_id | str | Reference to W5 Structure |
| islands | List[W6IslandDetail] | Detailed observations per island |
| topology_pairs | List[W6TopologyPair] | Inter-island distances |
| noise_count | int | Number of unclustered articles |
| params | Dict | Fixed policies (evidence, snippet, metric, digest) |

**Key Fields (W6IslandDetail):**

| Field | Type | Description |
|-------|------|-------------|
| island_id | str | Reference to W5Island |
| size | int | Number of members |
| cohesion_score | float | From W5 (read-only) |
| evidence_tokens | List[W6EvidenceToken] | Top-K tokens for island |
| representative_articles | List[W6RepresentativeArticle] | Sample articles |

**Export Formats:**
- JSON: Full observation data (machine-readable)
- Markdown: Human-readable report with tables
- CSV: 4 files (islands, topology, evidence, members)

**GPT Audit P0 Compliance:**
- P0-X1: Evidence formula fixed (mean_s_score_v1)
- P0-X2: Scope closure enforced (INV-W6-009)
- P0-1: Topology uses W5 vectors only
- P0-4: observation_id excludes floating-point values

**Test Results (6/6 PASS):**
- Pipeline: W5 → W6 conversion ✅
- Invariants: All 9 INV validated ✅
- Determinism: 5 runs identical output ✅
- Export: 6 files generated ✅
- Evidence (P0-X1): Formula verified ✅
- Scope (P0-X2): Closure enforced ✅

**INV-W6-001:** No Synthetic Labels (no natural language categories or LLM summaries)
**INV-W6-002:** Deterministic Export (same input produces bit-identical output)
**INV-W6-003:** Read Only (W1-W5 data is never modified)
**INV-W6-004:** No New Math (only extraction/transformation, no new statistics)
**INV-W6-005:** Evidence Provenance (all evidence tokens traceable to W3)
**INV-W6-006:** Stable Ordering (all lists have complete tie-break rules)
**INV-W6-007:** No Hypothesis (no "Axis Hypothesis" or judgment logic)
**INV-W6-008:** Strict Versioning (tokenizer, normalizer, W3 version tracked)
**INV-W6-009:** Scope Closure (W5 members must match W4/Article sets exactly)


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
| W5_ALGORITHM | "ResonanceCondensation-v1" | W5 condensation algorithm |
| W5_VECTOR_POLICY | "mean_raw_v1" | W5 centroid calculation |
| W5_DEFAULT_THRESHOLD | 0.70 | Similarity threshold for edges |
| W5_DEFAULT_MIN_ISLAND_SIZE | 3 | Minimum island size |
| W5_MAX_BATCH_SIZE | 2000 | P1-4: Prevent O(N²) explosion |
| W5_VECTOR_ROUNDING | 9 | Decimal places for vector values |
| W5_SIMILARITY_ROUNDING | 12 | Decimal places for similarity (P0-B) |
| DEFAULT_W5_OUTPUT_DIR | "data/stats/w5_structures" | W5 output directory |
| W6_EVIDENCE_POLICY | "mean_s_score_v1" | Evidence formula (P0-X1) |
| W6_SNIPPET_POLICY | "head_chars_v1" | Snippet extraction (first 200 chars) |
| W6_METRIC_POLICY | "cosine_dist_v1" | Topology distance metric |
| W6_DIGEST_POLICY | "abs_val_desc_v1" | Vector digest sorting |
| W6_EVIDENCE_K | 20 | Top-K evidence tokens per island |
| W6_TOPOLOGY_K | 10 | Top-K topology pairs to export |
| W6_DISTANCE_ROUNDING | 12 | Decimal places for distance |
| W6_EVIDENCE_ROUNDING | 8 | Decimal places for evidence score |
| DEFAULT_W6_OUTPUT_DIR | "data/discovery/w6_observations" | W6 output directory |

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
| INV-W4-002 | W4 | Deterministic: same article + same W3 â†’ same scores |
| INV-W4-003 | W4 | Recomputable: W4 = f(W0, W3) |
| INV-W4-004 | W4 | Full S-Score Usage: positive AND negative candidates |
| INV-W4-005 | W4 | Immutable Input: W4Projector does NOT modify inputs |
| INV-W4-006 | W4 | Tokenization Canon: MUST use W1Tokenizer + normalize_token |
| INV-W5-001 | W5 | No Naming: output must not contain natural language labels |
| INV-W5-002 | W5 | Topological Identity: island_id from member IDs only |
| INV-W5-003 | W5 | Fixed Metric: similarity = L2-normalized Cosine, operator >= |
| INV-W5-004 | W5 | Parameter Traceability: structure_id includes w4_analysis_ids + params |
| INV-W5-005 | W5 | Canonical Vector: vector values must be rounded before storage |
| INV-W5-006 | W5 | ID Collision Safety: Canonical JSON hash, string join forbidden |
| INV-W5-007 | W5 | Structure Identity: uses w4_analysis_id, not article_id |
| INV-W5-008 | W5 | Canonical Output: created_at excluded from identity check |
| INV-W6-001 | W6 | No Synthetic Labels: no natural language categories or LLM summaries |
| INV-W6-002 | W6 | Deterministic Export: same input produces bit-identical output |
| INV-W6-003 | W6 | Read Only: W1-W5 data is never modified |
| INV-W6-004 | W6 | No New Math: only extraction/transformation |
| INV-W6-005 | W6 | Evidence Provenance: all evidence tokens traceable to W3 |
| INV-W6-006 | W6 | Stable Ordering: all lists have complete tie-break rules |
| INV-W6-007 | W6 | No Hypothesis: no "Axis Hypothesis" or judgment logic |
| INV-W6-008 | W6 | Strict Versioning: version compatibility tracked |
| INV-W6-009 | W6 | Scope Closure: W5 members must match W4/Article sets exactly |

---

## 16. Substrate Layer (Layer 0) - NEW

### 16.1 Overview

The Substrate Layer (Context Fabric) provides permanent, append-only storage for machine-observable traces without semantic interpretation.

**Philosophy:** "Describe, but do not decide."

**Version:** v0.1.0

### 16.2 Package Structure

```
esde/substrate/
├── __init__.py          # Package exports
├── schema.py            # ContextRecord (frozen dataclass)
├── registry.py          # SubstrateRegistry (JSONL storage)
├── id_generator.py      # Deterministic ID generation
├── traces.py            # Trace validation and normalization
└── NAMESPACES.md        # Namespace definitions
```

### 16.3 ContextRecord Schema

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

### 16.4 Context ID Generation

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

### 16.5 Trace Validation Rules

#### 16.5.1 Key Format

```
namespace:name
```

- Namespace: `[a-z][a-z0-9_]*`
- Name: `[a-z][a-z0-9_]*`
- Example: `html:tag_count`, `meta:domain`

#### 16.5.2 Forbidden Namespaces

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

#### 16.5.3 Forbidden Key Names (P1-SUB-002)

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

#### 16.5.4 Value Type Constraints

| Type | Constraints |
|------|-------------|
| `str` | Max 4096 characters |
| `int` | Range: [-2³¹, 2³¹-1] |
| `float` | 9 decimal precision, no NaN/Inf |
| `bool` | True/False |
| `None` | Null |
| `list` | **FORBIDDEN** |
| `dict` | **FORBIDDEN** |

### 16.6 SubstrateRegistry API

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
    
    def export_canonical(self, output_path: str) -> int:
        """Export sorted by context_id (INV-SUB-007)."""
```

### 16.7 File I/O Constants

```python
FILE_ENCODING = "utf-8"     # Canonical encoding
FILE_NEWLINE = "\n"         # Unix LF only
```

### 16.8 Substrate Invariants

| ID | Name | Definition |
|----|------|------------|
| INV-SUB-001 | Upper Read-Only | Upper layers can read only; no update/delete (append-only) |
| INV-SUB-002 | No Semantic Transform | Raw observation values only, no interpretation |
| INV-SUB-003 | Machine-Observable | Only machine-computable values (no human judgment) |
| INV-SUB-004 | No Inference | No ML inference, no probabilistic values |
| INV-SUB-005 | Append-Only | Records can only be added, never updated/deleted |
| INV-SUB-006 | ID Determinism | context_id computed from inputs only (no timestamp/random) |
| INV-SUB-007 | Canonical Export | Output order and format must be deterministic |

### 16.9 Substrate Constants

| Parameter | Value | Description |
|-----------|-------|-------------|
| FLOAT_PRECISION | 9 | Decimal places for float normalization |
| CONTEXT_ID_LENGTH | 32 | Hex characters in context_id |
| STRING_MAX_LENGTH | 4096 | Max trace string value length |
| INT_MIN | -2147483648 | Minimum integer |
| INT_MAX | 2147483647 | Maximum integer |
| FILE_ENCODING | "utf-8" | Canonical encoding |
| FILE_NEWLINE | "\n" | Canonical newline |

### 16.10 GPT Audit Compliance

**Audit Status:** PASS (Conditional)

| Issue | Severity | Description | Resolution |
|-------|----------|-------------|------------|
| P0-SUB-001 | P0 | INV-SUB-001 description inaccurate | Fixed: "no update/delete (append-only)" |
| P1-SUB-002 | P1 | Forbidden key names incomplete | Added FORBIDDEN_KEY_NAMES set |
| P1-SUB-003 | P1 | File encoding not standardized | Added FILE_ENCODING, FILE_NEWLINE constants |
| P1-SUB-004 | P1 | Float normalization order unclear | Documented: normalize_traces() → canonical_json() |

---

## 17. References

- ESDE Core Specification v0.2.1
- ESDE v3.3: Ternary Emergence and Dual Symmetry
- ESDE v3.3.1: Emergence Directionality
- Semantic Language Integrated v1.1
- ESDE Operator Spec v0.3
- Aruism Philosophy
- **Substrate Layer Specification v0.1.0 (Gemini Design)**

---

*Document generated: 2026-01-24*  
*Engine Version: 5.4.7-SUB.1*  
*Framework: Existence Symmetry Dynamic Equilibrium*  
*Philosophy: Aruism*
