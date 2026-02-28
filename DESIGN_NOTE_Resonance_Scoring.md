# ESDE Design Note: Resonance Scoring Model

**Date:** 2026-02-15  
**Approved by:** Taka  
**Status:** Approved — A1 パイプラインで実装済み

---

## Decision

The Lexicon's slot-word relationship model is officially **continuous resonance scoring (0-10)**, replacing the originally conceived binary membership model.

## Background

### Original Design (Binary Membership)

```
ACT_arrive × temporal.emergence = {arrive, land, reach, ...}
```

Each word either belongs to a slot or does not. The internal structure of words (POS, frequency, semantic relations) was delegated to external databases (WordNet, etc.).

### Current Design (Continuous Resonance)

```
ACT_arrive × arrive × temporal.emergence = 8
ACT_arrive × arrive × scale.cosmic      = 0
```

Each word has a 48-dimensional resonance profile within its Atom. Scores range from 0 (no resonance) to 10 (strong resonance), observed by LLM.

### How This Emerged

The A1 pipeline was implemented with per-slot scoring as the observation mechanism. This was not a deliberate design pivot but a natural consequence of asking an LLM to "observe resonance" rather than "classify membership." The continuous-value structure emerged from the implementation.

Taka reviewed the implications on 2026-02-15 and approved the direction:

> "共鳴度という概念に切り替えたことで数学的な演算を可能にしている。既存DBとの接続に関しても共鳴の度合いが検知されるというのは面白みがある。"

## Implementation Details (A1 Pipeline)

### Post-Processing Pipeline

Raw scores (0-10 integer) from QwQ-32B undergo the following processing:

```
raw_scores (48 slots, 0-10 int)
  → softmax normalization (τ=1.0) → normalized_scores (probability distribution, Σ=1)
  → Shannon entropy → entropy_norm (0=certainty, 1=uniform)
  → focus_rate = 1.0 - entropy_norm (high=focused, low=diffuse)
  → status classification:
      F ≥ 0.30 → OK
      F < 0.30 → Diffuse_Observation (flagged for Phase 7)
```

### Actual Output Record

```json
{
  "word": "bask",
  "pos": "v",
  "atom": "EMO.like",
  "raw_scores": {"temporal.emergence": 0, "temporal.indication": 0, ...},
  "normalized_scores": {"epistemological.experience": 0.463, ...},
  "entropy_norm": 0.4277,
  "focus_rate": 0.5723,
  "status": "OK",
  "top5": [
    {"slot": "epistemological.experience", "p": 0.463},
    {"slot": "scale.individual", "p": 0.170},
    {"slot": "value_generation.aesthetic", "p": 0.170}
  ],
  "evidence": "Bask strongly resonates with personal, sensory experiences..."
}
```

### Quality Assurance (Auditor C1-C5)

Score inflation problem identified during EMO.like pilot: QwQ-32B assigned nonzero scores to 39/48 slots where only 8-15 should be relevant.

Five structural checks implemented in `auditor_a1.py`:

| Check | Detection Target | Key Threshold |
|-------|-----------------|---------------|
| C1: Distribution Anomaly | all-zero, inflation, spread | sum ≥ 150, NZ ≥ 25 |
| C2: Symmetric Pair Leak | antonym scoring on wrong side | non-destructive > 3 |
| C3: Evidence-Score Mismatch | evidence text ≠ top slots | keyword absence |
| C4: Axis-Generic Inflation | entire axis uniformly high | mean ≥ 4.0, spread ≤ 2 |
| C5: POS Coherence | POS vs slot mismatch | adj/adv + material ≥ 6 |

Key insight: **detection works, correction fails** with same model. Auditor only flags; Writer re-observes with quantitative constraints.

### Before/After Fix Results

Prompt engineering (mapper_a1.py) + threshold tightening (auditor_a1.py) の効果:

| Metric | Before Fix (.bak) | After Fix | Target | |
|--------|-------------------|-----------|--------|---|
| nz_mean | 38.7 | **13.6** | ≤25 | ✅ |
| Diffuse remaining | 18.0% | **0.0%** | ≤5% | ✅ |
| Diffuse → REVISE capture | 0/13 | **12/12** | 全捕捉 | ✅ |
| OK rate | 78.4% | **97.3%** | — | ✅ |

INFLATION_NONZERO_THRESHOLD adjusted from 40 → 25. Diffuse observations now reliably routed to REVISE for re-observation.

## Implications

### What This Enables

1. **Mathematical operations on meaning**
   - Cosine similarity between word profiles (interpretable distance)
   - Clustering within Atoms (discovering sub-groups)
   - Cross-Atom comparison (word profiles that span multiple Atoms)

2. **Interpretable differences**
   - "arrive and reach differ primarily on epistemological.experience (reach: 7, arrive: 5)"
   - Not just "how far apart" but "in what way"

3. **Weighted external DB connections**
   - WordNet synset connections carry resonance weight, not just presence/absence
   - Enables graded integration with any external knowledge source

4. **Graceful uncertainty**
   - Score of 2-3 = weak but present resonance (not forced to binary yes/no)
   - Score of 0 = explicit non-resonance (a meaningful observation, not missing data)
   - Aligns with "Describe, but do not decide" — observation admits gradation

### What This Requires

1. **Quality control on score distribution** ✅ Implemented & Verified
   - LLMs tend to inflate scores — confirmed and fixed in A1 pipeline
   - Before fix: nz_mean=38.7, Diffuse残存=18%, OK率=78.4%
   - After fix: nz_mean=13.6, Diffuse残存=0%, OK率=97.3%
   - Prompt engineering enforces zero-default scoring (mapper_a1.py)
   - Auditor C1-C5 pre-screen + LLM audit catches inflation (auditor_a1.py)
   - Diffuse→REVISE 完全捕捉 (0/13 → 12/12)

2. **Normalization policy** — Partially resolved
   - softmax normalization (τ=1.0) implemented for per-word comparison
   - Cross-Atom normalization policy TBD: awaiting full 326-atom run data
   - Per-word normalization in production; per-Atom/global normalization pending

3. **Constitution update** ✅ Resolved
   - Constitution v1.0 locked with 17 proposals approved (2026-02-11)
   - Constitution was written under the evolving model; formal resonance amendment deferred to v1.1 after full A1 data is available

## Consistency with Aruism

The continuous model is arguably more aligned with Aruism than the binary model:

- **"Describe, but do not decide"**: A score of 4 describes partial resonance without deciding membership. The Auditor embodies this too — it flags anomalies but does not correct them.
- **Existence Symmetry**: The symmetric pair relationship is preserved — low scores on one Atom's profile may correspond to high scores on the symmetric pair's profile, creating a continuous symmetry rather than a binary boundary. Auditor C2 (Symmetric Pair Leak) validates this structural property.
- **Dynamic Equilibrium**: Score distributions across words create an equilibrium that can shift as the Lexicon grows, rather than static set membership.

## Action Items

- [x] Implement A1 pipeline with resonance scoring (mapper_a1.py)
- [x] Implement structural quality audit (auditor_a1.py, C1-C5)
- [x] Address score inflation (INFLATION_NONZERO_THRESHOLD: 40→25, prompt改善)
- [x] Confirm Constitution v1.0 (17 proposals approved)
- [ ] Amend Constitution v1.1 to formally reference resonance scoring model
- [ ] Update Glossary entry for "Slot" to include scoring semantics
- [ ] Define cross-Atom normalization policy after full 326-atom run
- [ ] Update Technical Specification to document 48-dimensional profile structure
