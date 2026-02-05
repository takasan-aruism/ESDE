# ESDE Integration Briefing: Synapse Expansion via Phase 7
## For: Gemini (Design) / GPT (Audit)
## From: Claude (Implementation) + Taka (Architecture Decision)
## Date: 2026-02-05

---

## 0. Purpose

This document brings Gemini and GPT up to speed on an architectural insight
that emerged from implementing and testing the Relation Pipeline.
It proposes a new integration path: **Phase 7 → Synapse Expansion**.

The insight requires understanding how Phase 7, Phase 8, Phase 9,
and the Relation Pipeline interact at the implementation level.
This document provides that context.

---

## 1. Background: What Was Built and What Was Found

### 1.1 The Relation Pipeline (new)

We built an SVO (Subject-Verb-Object) extraction pipeline that:
1. Parses Wikipedia articles into SVO triples using spaCy
2. Grounds the **verb** of each triple onto ESDE Atoms via Synapse
3. Produces relation edges for entity graph construction

This was tested on 15 articles (5 military figures, 5 scholars, 5 cities),
producing 5,342 SVO triples.

### 1.2 What the Diagnostics Revealed

Three categories of grounding failure were found:

| Category | Example | Root Cause |
|----------|---------|------------|
| CATEGORY_MISMATCH | `include → PRP.dirty` | Synapse doesn't distinguish POS |
| CONSISTENT_MISGROUND | `have → EMO.like` | Light verbs are too polysemous |
| SYNAPSE_COVERAGE_GAP | `kill` ungrounded 24x | Synapse has no edge for this verb |

### 1.3 What We Fixed (v0.2.0)

We implemented three filters in the grounding logic (not Synapse itself):

| Filter | Effect |
|--------|--------|
| Light Verb Stoplist | 13 functional verbs → `UNGROUNDED_LIGHTVERB` (edge preserved, atom suppressed) |
| POS Guard | Block noun-category Atoms (NAT/MAT/PRP/SPA) from verb grounding |
| Score Threshold | Drop candidates below 0.45 (CLI-configurable) |

Results: CATEGORY_MISMATCH → 0. Old misgrounds → eliminated.
But grounding rate dropped from 89% to 55%, exposing the **true** coverage gaps.

### 1.4 The Remaining Problem

After filtering, these verbs are cleanly identified as genuinely ungroundable:

```
write (41x), host (39x), defeat (34x), serve (34x),
join (29x), hold (29x), attack (27x), contain (25x),
kill (24x), occupy (24x)
```

These are real, important verbs with clear WordNet synsets and obvious
Atom counterparts (kill → EXS.death, attack → ACT.destroy, etc.).
The gap is purely in Synapse: no edge connects these synsets to Atoms.

---

## 2. Why Synapse Has This Gap

### 2.1 How Synapse Was Generated

`generate_synapses_v3.py` works as follows:

1. For each of the 326 Atoms, compute an embedding of its Glossary definition
2. For each WordNet synset, compute an embedding of its WordNet definition
3. Calculate cosine similarity between all pairs
4. Keep pairs where similarity ≥ 0.3 AND the synset wins the global top-K competition

Key parameters:
- `MIN_SCORE_THRESHOLD = 0.3`
- `GLOBAL_TOP_K = 3` (max atoms connected to one synset)
- `ALLOWED_POS = {'n', 'v', 'a', 's'}` (verbs are included)

### 2.2 The Structural Bias

The Glossary definitions were written to describe **concepts** (nouns/adjectives).

```
EMO.love:  "Deep affection and care for another"
EXS.death: "Cessation of biological life"
ACT.destroy: "To cause the complete ruin of something"
```

WordNet verb definitions describe **actions**:

```
kill.v.01: "cause to die; put to death"
defeat.v.01: "win a victory over"
attack.v.01: "launch an assault upon"
```

Even though "kill" and "EXS.death" are semantically related,
their **definitions** are phrased differently enough that
cosine similarity in embedding space can fall below 0.3.

"Cause to die" ←→ "Cessation of biological life"
   (action frame)      (state description)

This is not a bug in Synapse. It's a structural mismatch between
how concepts are defined (noun-centric) and how actions are defined (verb-centric).

### 2.3 Implication

**The 326 Atoms are fine. Synapse just needs more edges.**

Specifically, it needs edges from verb synsets to action/change/relation Atoms.
The Atom inventory (ACT.attack, ACT.destroy, EXS.death, SOC.official, etc.)
already has natural landing spots for most verbs.

---

## 3. The Phase 7 Connection

### 3.1 What Phase 7 Already Does

Phase 7 handles tokens that fall outside the established semantic space.
It evaluates unknown tokens against four competing hypotheses:

| Route | Name | Criteria | Current Action |
|-------|------|----------|----------------|
| A | Typo | Edit distance ≤ 2 from known word | Suggest correction |
| B | Entity | Capitalized, proper noun pattern | External search |
| C | Novel | Valid morphology, no existing entry | Queue for Molecule generation |
| D | Noise | Invalid pattern, random characters | Discard |

Route C currently queues tokens for Phase 8 Molecule generation.
The original AI proposal was to eventually **add new Atoms** based on
Route C evidence. Taka rejected this because Atoms are the coordinate system
and must remain immutable.

### 3.2 The Insight: Route C → Synapse Expansion

Taka's new position:

> Adding Atoms = changing the coordinate system → rejected (as before)
> Adding Synapse edges = enriching the dictionary → acceptable

Phase 7's evidence accumulation mechanism (volatility tracking, observation counting,
human review requirement) is exactly the governance structure needed for
Synapse expansion.

### 3.3 What This Looks Like Concretely

```
Current (Phase 7 alone):
  "kill" encountered → synsets exist → Synapse miss → Route C → queue
  → accumulate evidence → ... → dead end (Atom addition rejected)

Proposed (Phase 7 + Synapse Expansion):
  "kill" encountered → synsets exist → Synapse miss → Route C → queue
  → accumulate evidence (N occurrences across corpus)
  → evidence threshold met
  → propose Synapse edge: kill.v.01 → EXS.death (with computed score)
  → human review
  → approved → Synapse edge added
  → future encounters of "kill" are grounded
```

### 3.4 What the Relation Pipeline Adds

The Relation Pipeline diagnostic report is essentially Phase 7 Route C
executed at corpus scale:

| Mechanism | Scope | Speed |
|-----------|-------|-------|
| Phase 7 Route C | Single token per encounter | Slow accumulation |
| Relation Pipeline diagnostics | Entire corpus at once | Immediate |

They produce the same output: a ranked list of (verb, frequency) pairs
that need Synapse edges. The Relation Pipeline just does it faster
because it processes all articles in one pass.

---

## 4. Architecture: Two Layers, Two Rates of Change

Taka's key architectural principle:

```
326 Atoms (Glossary)     = Coordinate system  = IMMUTABLE
    ↕ connected via
Synapse edges             = Dictionary         = GROWS over time
    ↕ used by
Phase 8 Sensor            = Lookup engine      = Stable code
Phase 7 Route C           = Gap detector       = Stable code
Relation Pipeline         = Corpus-scale gap detector = New tool
```

| Layer | Analogy | Change Rate | Governance |
|-------|---------|-------------|------------|
| 326 Atoms | Periodic table elements | Never | Founding design decision |
| Synapse | Compound/reaction database | Grows | Phase 7 evidence + human review |
| Code | Lab instruments | Versioned | 3AI workflow |

This aligns with Aruism's dynamic equilibrium:
the skeleton (Atoms) is fixed, but the connective tissue (Synapse)
is alive and grows as more language is observed.

---

## 5. Proposal for Gemini (Design)

Design a **Synapse Expansion Pipeline** that:

### 5.1 Input Sources

1. **Phase 7 Route C accumulations** — tokens that repeatedly miss Synapse
2. **Relation Pipeline SYNAPSE_COVERAGE_GAP** — corpus-scale gap detection

### 5.2 Expansion Process

```
Gap Detection (automatic)
    ↓
Candidate Edge Generation (automatic)
  - verb lemma → WordNet synsets (POS=VERB)
  - For each synset, compute embedding similarity to all 326 Atoms
  - Propose top-N edges with scores
    ↓
Evidence Threshold (automatic)
  - Must appear ≥ T times across corpus (prevents one-off noise)
  - Proposed score must be ≥ MIN_SCORE (same as Synapse generation)
    ↓
Human Review (manual)
  - Taka reviews proposed edges
  - Approve / reject / modify
    ↓
Synapse Patch (versioned)
  - Append new edges to Synapse
  - Increment Synapse version
  - Re-run diagnostics to verify improvement
```

### 5.3 Key Design Questions for Gemini

1. **Threshold T**: How many corpus occurrences before proposing an edge?
   (Currently we see kill=24x, host=39x — what's the minimum?)

2. **Score computation**: Reuse `generate_synapses_v3.py` embedding approach?
   Or allow manual assignment?

3. **Versioning**: Synapse v3.0 → v3.1 with patch log?
   Or separate override file that layers on top?

4. **Phase 7 integration**: Where in the existing Route C flow does the
   "propose Synapse edge" step go? After what evidence threshold?

5. **Verb-specific treatment**: Should verb synsets get a separate
   Synapse generation pass with verb-optimized parameters?
   (e.g., lower MIN_SCORE_THRESHOLD for verb definitions)

---

## 6. Audit Points for GPT

### 6.1 Verify the Diagnosis

- The 3 symptom categories (CATEGORY_MISMATCH, CONSISTENT_MISGROUND,
  SYNAPSE_COVERAGE_GAP) are supported by data from 5,342 triples
  across 15 articles in 3 domains.
- The v0.2.0 filters eliminated the first two categories cleanly
  (CATEGORY_MISMATCH: 5→0, old misgrounds: all resolved).
- The remaining COVERAGE_GAP verbs are genuine Synapse gaps, not filter artifacts.

### 6.2 Verify the Architecture

- Atom immutability is preserved (no change to 326 Atoms).
- Synapse expansion is append-only (existing edges unaffected).
- Phase 7's evidence accumulation provides governance (not arbitrary addition).
- Human review is required before any Synapse modification.

### 6.3 Risk Assessment

| Risk | Mitigation |
|------|------------|
| Over-expansion: too many weak edges added | Score threshold + human review |
| Verb bias: Synapse becomes verb-heavy | Separate verb edges from noun edges in audit |
| Regression: new edges cause new misgrounds | Re-run diagnostic pipeline after each patch |
| Phase 7 / Relation Pipeline disagreement | Both feed into same proposal queue |

---

## 7. Current File Inventory (for reference)

### Relation Pipeline (new, Claude-implemented)
```
integration/relations/
  parser_adapter.py      — spaCy SVO extraction
  relation_logger.py     — Synapse grounding (v0.2.0 with 3 filters)
  run_relations.py       — CLI runner + diagnostic report generation
  __init__.py
```

### Phase 7 (existing)
```
esde_engine/
  routing.py             — UnknownTokenRouter (4-hypothesis)
  resolver/
    hypothesis.py        — A/B/C/D evaluation
    aggregate_state.py   — Evidence accumulation
    online.py            — External source lookup
    ledger.py            — Decision audit trail
```

### Phase 8 Sensor (existing)
```
sensor/
  loader_synapse.py      — Synapse JSON loader
  extract_synset.py      — WordNet synset extraction
  rank_candidates.py     — Score aggregation
  validator_v83.py       — Molecule validator
  molecule_generator_live.py — LLM-based molecule generation
```

### Synapse Generation (existing)
```
generate_synapses_v3.py  — Embedding-based Synapse generation
esde_synapses_v3.json    — Current Synapse (11,557 synsets, 22,285 edges)
```

---

## 8. Summary

| Item | Status |
|------|--------|
| Problem | Synapse has structural verb coverage gaps |
| Root cause | Noun-centric Glossary definitions vs verb-centric WordNet definitions |
| Wrong fix | Adding new Atoms (breaks coordinate system) |
| Right fix | Adding Synapse edges (enriches dictionary) |
| Governance | Phase 7 evidence accumulation + human review |
| Detection | Relation Pipeline diagnostics (corpus-scale) + Phase 7 Route C (token-scale) |
| Atoms touched | Zero |
| Synapse touched | Append-only new edges |
| Reversibility | Full (remove edges, re-run) |

**Decision needed from Gemini**: Design the Synapse Expansion Pipeline (Section 5).
**Decision needed from GPT**: Audit the diagnosis and architecture (Section 6).
**Decision needed from Taka**: Approve the overall direction before Gemini proceeds.
