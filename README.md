# ESDE Phase 8 — Projection Operator Experiment

**Version**: 0.1.0  
**Date**: 2026-03-03  
**Status**: Ground truth template ready. Awaiting Taka annotation.

## Overview

Tests 4 projection operators (Base/B/C/BC) that reweight Synapse atom candidates
using sentence-level context embeddings, addressing the WSD bottleneck discovered
in Phase 8 Pilot v2 (99.5% LLM FAIL rate due to `capital→STA.wealth` type errors).

## Files

| File | Role | Status |
|------|------|--------|
| `esde/projection.py` | 4 Operator implementations (B/C/BC + embedder backends) | ✅ |
| `tools/projection_eval.py` | Recall@1/@3 scorer | ✅ |
| `scripts/run_projection_experiment.py` | Experiment runner (all modes) | ✅ |
| `output/projection_eval/ground_truth_50.jsonl` | **Taka annotation** | ⚠ 3 examples, needs 47 more |
| `output/projection_eval/report.md` | Auto-generated comparison report | Pending |

## Ground Truth Annotation Guide

### File: `ground_truth_50.jsonl`

One JSON object per line. Each object:

```json
{
  "id": "berlin_NNNN",
  "sentence": "The actual sentence text.",
  "targets": [
    {
      "span_text": "capital",
      "span": null,
      "pos": "NOUN",
      "atoms_top3": ["SOC.official", "SPC.place", "SOC.nation"]
    }
  ],
  "notes": "Optional context notes"
}
```

### Rules

1. **Select 50 sentences** from Berlin article (already cached in Harvester)
2. **Focus on polysemous words** — words where Synapse is likely wrong:
   - `capital` (STA.wealth vs SOC.official/SPC.place)
   - `state` (ACT.descend vs SOC.nation)
   - `area` (NAT.sea vs SPC.place)
   - `bank` (ECO.finance vs NAT.river)
   - `fall` (ACT.descend vs CHG.collapse/EXS.end)
   - `party` (EMO.fun vs SOC.political)
   - `power` (STA.energy vs SOC.authority)
   - `court` (SOC.legal vs SPC.place)
   - `major` (SOC.military vs PRP.important)
   - `field` (SPC.place vs WLD.domain)
3. **atoms_top3**: List 1-3 acceptable atoms. Order = preference (first is best).
   These are the "correct" atoms **in this sentence's context**.
4. **winner=null**: Multiple atoms can be correct. This is expected.
5. **span**: Character offsets. Set `null` if tedious — matcher uses `span_text`.
6. **pos**: NOUN, VERB, ADJ, or PROPN.

### How many targets per sentence?

Not every word needs annotation. Focus on:
- Words with **high polysemy** (multiple Synapse atoms)
- Words where **context clearly disambiguates** meaning
- Skip function words (be, have, do, etc.)

Typical: 2-4 targets per sentence × 50 sentences = ~100-200 target words.

## Running Experiments

```bash
# Step 1: Prepare Berlin sentences (extract from Harvester cache)
# (Script will provide berlin_sentences.jsonl)

# Step 2: Run all 4 modes
python3 scripts/run_projection_experiment.py \
  --mode all \
  --sentences output/projection_eval/berlin_sentences.jsonl \
  --dictionary lexicon_wn/esde_dictionary.json \
  --synapse esde_synapses_v3.json \
  --gt output/projection_eval/ground_truth_50.jsonl \
  --out output/projection_eval/

# Step 3: Check report
cat output/projection_eval/report.md

# Step 4: Detailed scoring for single mode
python3 tools/projection_eval.py \
  --pred-jsonl output/projection_eval/BC/pred_50.jsonl \
  --gt-jsonl output/projection_eval/ground_truth_50.jsonl \
  --mode BC
```

## Success Criteria

| Metric | Current (Base) | Target |
|--------|---------------|--------|
| Recall@3 | ~25% (est.) | ≥ 50% |
| Phase β FAIL | 99.5% | < 80% |
| Overhead | 0s | < 5s |

## Embedder Backends

| Backend | Requires | Quality | Speed |
|---------|----------|---------|-------|
| `tfidf` | sklearn (built-in) | Lower (lexical only) | Fast |
| `minilm` | sentence-transformers + PyTorch | Higher (semantic) | Fast (CPU OK) |

Default: `tfidf`. For production: set `--embedder minilm` or `EMBEDDER_BACKEND=minilm`.

## Architecture

```
Natural Language (superposition)
       ↓ (Sentence Embedding: MiniLM or TF-IDF)
Continuous Semantic Space (z_s ∈ R^384)
       ↓ cos(z_s, z_i) for all 326 atoms
Atom Field Gravity (g_i per atom)
       ↓ Projection Operator (B: gate, C: Bayes, BC: hybrid)
Reweighted atom candidates
       ↓
Rule Generator (existing Phase α)
```

---

*「記述せよ、しかし決定するな」*
