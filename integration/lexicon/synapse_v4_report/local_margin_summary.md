# ESDE Synapse v4 — Task 1: Local Margin Analysis

**Date**: 2026-03-01 16:42 UTC
**Phase**: Analysis-only (no Synapse modification)
**Score basis**: raw_scores (0-10 integers)

## Global vs Local Margin Comparison

| Metric | Global | Local (Neighborhood) |
|--------|--------|---------------------|
| Negative margin % | **82.1%** | **77.1%** |
| Avg margin | -0.0789 | -0.0507 |
| Total words | 32666 | 32666 |

**Flipped to positive** (negative global → positive local): **1776** (5.4%)

## Interpretation

Global negative margin: 82.1%
Local negative margin:  77.1%
Reduction:              5.0 percentage points

**Result**: High local negative margin indicates substantial overlap between neighboring atoms in 48D space. Investigate problem atoms below.

## Local Margin Distribution

| Range | Count |
|-------|-------|
| <-0.20 | 1492 |
| -0.20..-0.10 | 5030 |
| -0.10..0.00 | 18672 |
| 0.00..0.05 | 5945 |
| 0.05..0.10 | 1104 |
| 0.10..0.20 | 363 |
| >=0.20 | 60 |

## Problem Atoms (local negative margin > 50%): 279

| Atom | Total Words | Neg Local | Neg % | Avg Margin Local |
|------|-------------|-----------|-------|------------------|
| PRP.large | 103 | 103 | 100.0% | -0.0748 |
| TIM.moment | 38 | 38 | 100.0% | -0.1216 |
| WLD.technique | 82 | 81 | 98.8% | -0.0936 |
| ECO.money | 68 | 67 | 98.5% | -0.0831 |
| SOC.work | 361 | 355 | 98.3% | -0.1077 |
| PRP.cold | 57 | 56 | 98.2% | -0.0561 |
| ACT.take | 155 | 152 | 98.1% | -0.0904 |
| COM.answer | 152 | 149 | 98.0% | -0.1062 |
| EMO.pleasure | 41 | 40 | 97.6% | -0.0618 |
| PRP.low | 42 | 41 | 97.6% | -0.062 |
| PRP.heavy | 33 | 32 | 97.0% | -0.0619 |
| ACT.descend | 96 | 93 | 96.9% | -0.0668 |
| BEI.female | 162 | 157 | 96.9% | -0.0749 |
| ACT.obtain | 244 | 236 | 96.7% | -0.0951 |
| EMO.like | 24 | 23 | 95.8% | -0.0622 |
| PER.feel | 119 | 114 | 95.8% | -0.0763 |
| BEI.beast | 141 | 135 | 95.7% | -0.0531 |
| EMO.satisfaction | 46 | 44 | 95.7% | -0.0526 |
| ACT.abandon | 60 | 57 | 95.0% | -0.0695 |
| EXS.absence | 20 | 19 | 95.0% | -0.0588 |
| PRP.long | 79 | 75 | 94.9% | -0.1077 |
| PRP.bright | 77 | 73 | 94.8% | -0.051 |
| PER.salty | 18 | 17 | 94.4% | -0.0364 |
| ABS.bound | 184 | 172 | 93.5% | -0.0861 |
| ELM.light | 162 | 151 | 93.2% | -0.0504 |
| FND.labor | 263 | 245 | 93.2% | -0.0892 |
| FND.temporality | 72 | 67 | 93.1% | -0.1396 |
| ACT.exit | 86 | 80 | 93.0% | -0.0825 |
| ECO.sell | 85 | 79 | 92.9% | -0.0766 |
| PRP.smooth | 55 | 51 | 92.7% | -0.0509 |
| WLD.nonscience | 230 | 213 | 92.6% | -0.0681 |
| LOG.cause | 465 | 430 | 92.5% | -0.0698 |
| SOC.homeless | 642 | 594 | 92.5% | -0.0954 |
| MAT.food | 251 | 232 | 92.4% | -0.0375 |
| ACT.sink | 38 | 35 | 92.1% | -0.0447 |
| FND.unconscious | 38 | 35 | 92.1% | -0.0656 |
| PRP.soft | 75 | 69 | 92.0% | -0.0803 |
| EMO.wish | 37 | 34 | 91.9% | -0.0538 |
| SPC.direction | 83 | 76 | 91.6% | -0.093 |
| PRP.small | 82 | 75 | 91.5% | -0.0623 |
| ACT.move | 1250 | 1143 | 91.4% | -0.0608 |
| PER.sweet | 35 | 32 | 91.4% | -0.0505 |
| SPC.place | 102 | 93 | 91.2% | -0.0722 |
| VAL.correct | 91 | 83 | 91.2% | -0.0709 |
| BEI.male | 78 | 71 | 91.0% | -0.0748 |
| TIM.now | 32 | 29 | 90.6% | -0.0903 |
| ACT.arrive | 72 | 65 | 90.3% | -0.0448 |
| VAL.truth | 31 | 28 | 90.3% | -0.0769 |
| ECO.price | 82 | 74 | 90.2% | -0.0725 |
| PER.smell | 30 | 27 | 90.0% | -0.0297 |
