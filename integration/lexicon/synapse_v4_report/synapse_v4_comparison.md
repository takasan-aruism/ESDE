# ESDE Synapse v4 Comparison Report

**Date**: 2026-03-01 15:44 UTC
**Phase**: Analysis-only (no Synapse modification)
**Score basis**: raw_scores (0-10 integers, not softmax)
**Thresholds**: embedding ≥ 0.55, A1_cosine ≥ 0.65, margin ≥ 0.05

**Lemma matching**: nltk_full_lemma
**Total lemmas indexed**: 25303 (from 11575 synsets)

## Score Mode Comparison (normalized vs raw)

| Mode | Avg cos_self | Avg margin | Negative margin % |
|------|-------------|------------|-------------------|
| normalized_scores (softmax) | 0.7362 | -0.1801 | 90.1% |
| **raw_scores (0-10)** | **0.7792** | **-0.0789** | **82.1%** |

All subsequent analysis uses **raw_scores** to avoid softmax simplex compression.

## Section A: Overall Correlation

- Matched word-edge pairs: **2893**
- Pearson r (embedding_score vs A1_cosine): **0.1854**
- Unmatched Synapse edges (no A1 word): 33951

### Confusion Matrix

|  | A1 High | A1 Low |
|--|---------|--------|
| **Emb High** | TP: 45 | FP: 967 |
| **Emb Low**  | FN: 64 | TN: 1817 |

- Accuracy: 0.644
- Precision (Emb High → A1 High): 0.044
- Recall (A1 High captured by Emb): 0.413

### Centroid Construction

- Atoms with centroids: **325**
- Total Core words used: **32666**
- Atoms with no Core match (used all words): 3
  - ACT.build
  - FND.temporality
  - PRP.old

## Section B: Top 50 FN Candidates

These are words where embedding scored LOW but A1 measured HIGH.
→ Synapse is missing these connections.

| # | Word | Atom | Emb Score | A1 Cosine | A1 Margin | Focus |
|---|------|------|-----------|-----------|-----------|-------|
| 1 | hallowed | VAL.sacred | 0.4375 | 0.9737 | 0.0646 | 0.707 |
| 2 | defunctness | EXS.death | 0.4176 | 0.9698 | 0.0908 | 0.529 |
| 3 | mortification | EXS.death | 0.4356 | 0.9648 | 0.0966 | 0.424 |
| 4 | decease | EXS.death | 0.4715 | 0.9609 | 0.1164 | 0.500 |
| 5 | disintegrate | CHG.decay | 0.4008 | 0.9597 | 0.0701 | 0.784 |
| 6 | intuition | FND.intuition | 0.5075 | 0.9594 | 0.0659 | 0.527 |
| 7 | clearness | PRP.clear | 0.5211 | 0.9563 | 0.0660 | 0.563 |
| 8 | lucid | PRP.clear | 0.4880 | 0.9474 | 0.1051 | 0.626 |
| 9 | crystal clear | PRP.clear | 0.4880 | 0.9474 | 0.0645 | 0.504 |
| 10 | falsify | VAL.falsehood | 0.5093 | 0.9416 | 0.0540 | 0.313 |
| 11 | expiry | EXS.death | 0.4715 | 0.9415 | 0.1124 | 0.518 |
| 12 | pellucid | PRP.clear | 0.4880 | 0.9374 | 0.0646 | 0.542 |
| 13 | innate | COG.instinct | 0.4658 | 0.9371 | 0.0639 | 0.490 |
| 14 | info | FND.information | 0.4623 | 0.9343 | 0.0884 | 0.931 |
| 15 | unclouded | PRP.clear | 0.3531 | 0.9325 | 0.1084 | 0.518 |
| 16 | ahistorical | FND.ahistorical | 0.5068 | 0.9322 | 0.1257 | 0.771 |
| 17 | misrepresent | VAL.falsehood | 0.5093 | 0.9317 | 0.0622 | 0.565 |
| 18 | obliteration | ACT.destroy | 0.3805 | 0.9264 | 0.0658 | 0.403 |
| 19 | interior | SPC.inside | 0.4650 | 0.9256 | 0.0971 | 0.613 |
| 20 | break | ACT.destroy | 0.5249 | 0.9248 | 0.0534 | 0.399 |
| 21 | pellucidity | PRP.clear | 0.5211 | 0.9231 | 0.0701 | 0.465 |
| 22 | lead off | CHG.begin | 0.4993 | 0.9199 | 0.0574 | 0.802 |
| 23 | hunch | FND.intuition | 0.5075 | 0.9189 | 0.0938 | 0.577 |
| 24 | surface | PRP.shallow | 0.5101 | 0.9159 | 0.0755 | 0.740 |
| 25 | foster | BEI.parent | 0.5059 | 0.9138 | 0.0688 | 0.454 |
| 26 | static | FND.unchanging | 0.4560 | 0.9137 | 0.1021 | 0.687 |
| 27 | side by side | PRP.near | 0.5402 | 0.9088 | 0.0828 | 0.813 |
| 28 | get going | CHG.begin | 0.4525 | 0.9063 | 0.0647 | 0.616 |
| 29 | take in | ACT.receive | 0.4773 | 0.9025 | 0.0943 | 0.324 |
| 30 | have | ACT.receive | 0.4844 | 0.9023 | 0.0559 | 0.520 |
| 31 | emerge | TIM.appear | 0.4670 | 0.9000 | 0.1142 | 0.612 |
| 32 | beginning | CHG.begin | 0.5213 | 0.8990 | 0.0656 | 0.561 |
| 33 | myriad | FND.numberless | 0.4389 | 0.8954 | 0.0629 | 0.569 |
| 34 | continuous | FND.time | 0.4403 | 0.8945 | 0.1013 | 0.531 |
| 35 | superficial | PRP.shallow | 0.3222 | 0.8926 | 0.2027 | 0.984 |
| 36 | tooth root | BEI.root | 0.3071 | 0.8872 | 0.0548 | 0.470 |
| 37 | manifest | PRP.clear | 0.3666 | 0.8858 | 0.1080 | 0.525 |
| 38 | unbounded | FND.numberless | 0.4621 | 0.8848 | 0.0640 | 0.540 |
| 39 | burst | ELM.fire | 0.4094 | 0.8837 | 0.0795 | 0.595 |
| 40 | open | PRP.clear | 0.3880 | 0.8830 | 0.0582 | 0.490 |
| 41 | internal | SPC.inside | 0.5211 | 0.8759 | 0.0859 | 0.784 |
| 42 | limpid | PRP.clear | 0.4880 | 0.8757 | 0.0884 | 0.533 |
| 43 | tongueless | FND.languageless | 0.4052 | 0.8655 | 0.0940 | 0.344 |
| 44 | remote | PRP.far | 0.5489 | 0.8627 | 0.0573 | 0.404 |
| 45 | transformation | FND.transformation | 0.5314 | 0.8597 | 0.0517 | 0.682 |
| 46 | embark on | CHG.begin | 0.3974 | 0.8541 | 0.0552 | 0.558 |
| 47 | appear | TIM.appear | 0.4091 | 0.8533 | 0.1052 | 0.632 |
| 48 | come along | TIM.appear | 0.4075 | 0.8494 | 0.1279 | 0.610 |
| 49 | unspoken | FND.languageless | 0.4052 | 0.8426 | 0.0569 | 0.303 |
| 50 | skirmish | STA.war | 0.3420 | 0.8393 | 0.0664 | 0.719 |

## Section C: Top 50 FP Candidates

These are words where embedding scored HIGH but A1 measured LOW.
→ Synapse may have spurious connections.

| # | Word | Atom | Emb Score | A1 Cosine | A1 Margin | Top2 Atom | Focus |
|---|------|------|-----------|-----------|-----------|-----------|-------|
| 1 | virtue | VAL.good | 0.9035 | 0.9357 | 0.0188 | WLD.nonscience | 0.521 |
| 2 | separate | COG.separate | 0.8948 | 0.7404 | -0.0675 | CHG.advance | 0.506 |
| 3 | divide | COG.separate | 0.8948 | 0.7963 | -0.0999 | ACT.dissolve | 0.397 |
| 4 | darkness | PRP.dark | 0.8876 | 0.8910 | -0.0229 | ELM.darkness | 0.534 |
| 5 | dark | PRP.dark | 0.8876 | 0.8366 | -0.0412 | ACT.dissolve | 0.481 |
| 6 | clash | COM.conflict | 0.8764 | 0.9504 | 0.0055 | SOC.attack | 0.486 |
| 7 | friction | COM.conflict | 0.8764 | 0.8043 | 0.0120 | SOC.family | 0.438 |
| 8 | halt | ACT.stop | 0.8744 | 0.9086 | 0.0273 | SPC.reverse | 0.799 |
| 9 | good | VAL.good | 0.8361 | 0.8297 | -0.0552 | EMO.satisfaction | 0.613 |
| 10 | goodness | VAL.good | 0.8361 | 0.8976 | 0.0062 | EMO.love | 0.417 |
| 11 | hear | PER.hear | 0.8208 | 0.8767 | -0.0034 | PER.see | 0.811 |
| 12 | flavorlessness | PER.tasteless | 0.8140 | 0.9015 | -0.0093 | PRP.dirty | 0.514 |
| 13 | flavourlessness | PER.tasteless | 0.8140 | 0.8753 | -0.0884 | PER.untouched | 0.405 |
| 14 | savorlessness | PER.tasteless | 0.8140 | 0.8681 | -0.0658 | PRP.weak | 0.525 |
| 15 | savourlessness | PER.tasteless | 0.8140 | 0.9639 | -0.0191 | PER.bitter | 0.493 |
| 16 | tastelessness | PER.tasteless | 0.8140 | 0.7459 | -0.1402 | PRP.shallow | 0.460 |
| 17 | darkness | ELM.darkness | 0.8121 | 0.8664 | 0.0029 | ACT.destroy | 0.694 |
| 18 | dark | ELM.darkness | 0.8121 | 0.8708 | 0.0022 | PRP.dark | 0.592 |
| 19 | iniquity | PRP.dark | 0.8121 | 0.7053 | -0.1991 | VAL.evil | 0.319 |
| 20 | wickedness | PRP.dark | 0.8121 | 0.7088 | -0.2220 | VAL.evil | 0.439 |
| 21 | iniquity | ELM.darkness | 0.8121 | 0.6462 | -0.2370 | VAL.evil | 0.589 |
| 22 | wickedness | ELM.darkness | 0.8121 | 0.6876 | -0.2440 | VAL.evil | 0.813 |
| 23 | descent | ACT.descend | 0.8004 | 0.8381 | -0.0310 | ACT.fall | 0.337 |
| 24 | ego | FND.mind | 0.7982 | 0.8268 | -0.1059 | EMO.sorrow | 0.710 |
| 25 | giving | ACT.give | 0.7943 | 0.8866 | -0.0328 | SOC.defend | 0.500 |
| 26 | dark | ELM.night | 0.7902 | 0.9141 | 0.0258 | ELM.moon | 0.534 |
| 27 | salt | PER.salty | 0.7881 | 0.8767 | -0.0694 | PRP.small | 0.497 |
| 28 | saltiness | PER.salty | 0.7881 | 0.9413 | -0.0124 | PER.sweet | 0.552 |
| 29 | salinity | PER.salty | 0.7881 | 0.8672 | -0.0927 | PER.untouched | 0.405 |
| 30 | night | PRP.dark | 0.7842 | 0.6150 | -0.2024 | ELM.morning | 0.461 |
| 31 | lesion | STA.wound | 0.7802 | 0.9150 | -0.0111 | STA.illness | 0.707 |
| 32 | smell | PER.smell | 0.7778 | 0.9143 | -0.0461 | PER.untouched | 0.432 |
| 33 | smelling | PER.smell | 0.7778 | 0.8601 | -0.0165 | ACT.go | 0.579 |
| 34 | duration | FND.temporality | 0.7770 | 0.8174 | -0.0525 | FND.numberless | 0.563 |
| 35 | continuance | FND.temporality | 0.7770 | 0.5948 | -0.1678 | FND.unchanging | 0.851 |
| 36 | antipathy | EMO.dislike | 0.7762 | 0.9582 | -0.0056 | EMO.unhappiness | 0.785 |
| 37 | aversion | EMO.dislike | 0.7762 | 0.9211 | -0.0085 | EMO.anxiety | 0.792 |
| 38 | distaste | EMO.dislike | 0.7762 | 0.8876 | -0.0183 | EMO.pride | 0.954 |
| 39 | sweet | PER.sweet | 0.7750 | 0.9560 | -0.0048 | VAL.beauty | 0.449 |
| 40 | sweetness | PER.sweet | 0.7750 | 0.9369 | -0.0117 | PER.fragrance | 0.555 |
| 41 | sugariness | PER.sweet | 0.7750 | 0.9391 | 0.0211 | PER.fragrance | 0.627 |
| 42 | stance | ACT.stand | 0.7731 | 0.9556 | 0.0044 | ACT.rise | 0.529 |
| 43 | black | PRP.dark | 0.7716 | 0.7808 | -0.1620 | PER.taste | 0.605 |
| 44 | blackness | PRP.dark | 0.7716 | 0.7223 | -0.1902 | BOD.hand | 0.469 |
| 45 | total darkness | PRP.dark | 0.7716 | 0.9416 | -0.0041 | ELM.darkness | 0.624 |
| 46 | lightlessness | PRP.dark | 0.7716 | 0.8003 | -0.0080 | ACT.destroy | 0.438 |
| 47 | pitch blackness | PRP.dark | 0.7716 | 0.8894 | -0.0175 | ELM.darkness | 0.734 |
| 48 | human | BEI.human | 0.7688 | 0.8689 | -0.0339 | EXS.presence | 0.586 |
| 49 | terror | EMO.fear | 0.7679 | 0.9239 | -0.0390 | EMO.worry | 0.677 |
| 50 | panic | EMO.fear | 0.7679 | 0.8862 | 0.0087 | STA.danger | 0.499 |

## Section D: Atom-level Statistics

### Overall Word Distance Summary

- Total words evaluated: **32666**
- Avg cos_self: **0.7792**
- Avg margin: **-0.0789**
- Words with negative margin: **26828** (82.1%)

### Cosine Similarity Distribution (cos_self)

| Bin | Count |
|-----|-------|
| 0.0 | 58 |
| 0.1 | 34 |
| 0.2 | 93 |
| 0.3 | 209 |
| 0.4 | 528 |
| 0.5 | 1213 |
| 0.6 | 2664 |
| 0.7 | 5674 |
| 0.8 | 10723 |
| 0.9 | 10740 |
| 1.0 | 730 |

### Margin Distribution

| Range | Count |
|-------|-------|
| <-0.10 | 10455 |
| -0.10..0.00 | 16373 |
| 0.00..0.05 | 4837 |
| 0.05..0.10 | 803 |
| 0.10..0.20 | 189 |
| >=0.20 | 9 |

### Top 50 Negative Margin Words (potential misassignments)

| # | Word | Assigned | cos_self | Top2 Atom | cos_top2 | Margin |
|---|------|----------|----------|-----------|----------|--------|
| 1 | exposit | CHG.grow | 0.0498 | FND.information | 0.7811 | -0.7313 |
| 2 | peaceful | STA.peace | 0.0645 | PRP.wide | 0.7773 | -0.7128 |
| 3 | copy | ACT.make | 0.0744 | FND.information | 0.7811 | -0.7067 |
| 4 | plan of attack | SOC.attack | 0.0776 | FND.information | 0.7811 | -0.7035 |
| 5 | Thatch | COM.teach | 0.1886 | EXS.matter | 0.8557 | -0.6671 |
| 6 | stool | COG.learn | 0.1797 | ELM.water | 0.8391 | -0.6594 |
| 7 | stamp | SOC.family | 0.1335 | ELM.earth | 0.7890 | -0.6554 |
| 8 | uncomplicated | PRP.easy | 0.0544 | REL.different | 0.7063 | -0.6519 |
| 9 | vacation spot | SOC.nation | 0.2173 | ELM.earth | 0.8685 | -0.6512 |
| 10 | wintry | TIM.period | 0.1849 | ELM.earth | 0.8353 | -0.6503 |
| 11 | fill in | REL.same | 0.1587 | ECO.currency | 0.7754 | -0.6167 |
| 12 | choker | COG.forget | 0.2470 | ELM.earth | 0.8636 | -0.6166 |
| 13 | incapable | PRP.impossible | 0.2968 | ACT.sit | 0.9134 | -0.6166 |
| 14 | clapper | SOC.praise | 0.2581 | ELM.water | 0.8700 | -0.6119 |
| 15 | malediction | ACT.build | 0.2435 | VAL.evil | 0.8553 | -0.6119 |
| 16 | pod | COG.learn | 0.1416 | REL.together | 0.7481 | -0.6065 |
| 17 | transcendental number | LOG.unreason | 0.0574 | FND.information | 0.6570 | -0.5997 |
| 18 | vernal | PRP.young | 0.2368 | ELM.earth | 0.8353 | -0.5985 |
| 19 | solvent | COG.know | 0.2831 | ELM.water | 0.8805 | -0.5974 |
| 20 | specify | CHG.grow | 0.1387 | FND.information | 0.7338 | -0.5951 |
| 21 | topic | EXS.matter | 0.2798 | VAL.sacred | 0.8722 | -0.5924 |
| 22 | ostracism | ACT.move | 0.2708 | COM.conflict | 0.8541 | -0.5833 |
| 23 | constitutionalize | ACT.go | 0.3090 | WLD.culture | 0.8896 | -0.5806 |
| 24 | literacy | WLD.technique | 0.2033 | FND.information | 0.7811 | -0.5778 |
| 25 | shrubbery | SOC.nation | 0.3011 | ELM.earth | 0.8771 | -0.5760 |
| 26 | indexing | ACT.give | 0.2288 | FND.information | 0.8046 | -0.5758 |
| 27 | game bird | FND.play | 0.2855 | ELM.earth | 0.8593 | -0.5739 |
| 28 | FTP | SOC.homeless | 0.0397 | ACT.receive | 0.6091 | -0.5694 |
| 29 | other | TIM.past | 0.1600 | PRP.wide | 0.7286 | -0.5686 |
| 30 | present | WLD.nonscience | 0.2901 | ACT.emit | 0.8568 | -0.5666 |
| 31 | damnation | ACT.build | 0.3333 | PRP.impossible | 0.8993 | -0.5660 |
| 32 | some | PRP.multiple | 0.2693 | ELM.earth | 0.8353 | -0.5660 |
| 33 | drought | TIM.period | 0.1944 | ECO.waste | 0.7598 | -0.5654 |
| 34 | backup | REL.same | 0.2738 | FND.information | 0.8365 | -0.5627 |
| 35 | geographical | VAL.truth | 0.2396 | ELM.earth | 0.8003 | -0.5606 |
| 36 | annual | EXS.being | 0.2050 | ELM.night | 0.7629 | -0.5579 |
| 37 | turn | FND.number | 0.0134 | ELM.night | 0.5700 | -0.5566 |
| 38 | famine | EMO.wish | 0.3168 | PRP.murky | 0.8732 | -0.5564 |
| 39 | inaccurate | SPC.outside | 0.2971 | FND.information | 0.8518 | -0.5548 |
| 40 | lick | COG.enlightenment | 0.3527 | ACT.emit | 0.9072 | -0.5545 |
| 41 | predict | FND.mind | 0.2273 | FND.information | 0.7811 | -0.5538 |
| 42 | floor | VAL.correct | 0.3051 | FND.spatiality | 0.8570 | -0.5518 |
| 43 | reprint | FND.labor | 0.2657 | FND.information | 0.8168 | -0.5511 |
| 44 | dullard | FND.transformation | 0.3082 | ABS.foolish | 0.8576 | -0.5494 |
| 45 | young | PRP.old | 0.1809 | CHG.begin | 0.7302 | -0.5493 |
| 46 | ascomycetous | EXS.being | 0.2866 | ELM.earth | 0.8353 | -0.5487 |
| 47 | micro-organism | EXS.being | 0.2866 | ELM.earth | 0.8353 | -0.5487 |
| 48 | wedding night | ELM.night | 0.4112 | EMO.forgiveness | 0.9587 | -0.5475 |
| 49 | indexer | ACT.give | 0.1915 | FND.information | 0.7388 | -0.5473 |
| 50 | survey | BOD.eye | 0.1668 | FND.information | 0.7089 | -0.5421 |

### Atom-level Precision/Recall

| Atom | TP | FP | FN | TN | Precision | Recall | Total |
|------|----|----|----|----|-----------|--------|-------|
| ABS.bound | 0 | 0 | 0 | 10 | N/A | N/A | 10 |
| ABS.exempt | 0 | 7 | 0 | 0 | 0.000 | N/A | 7 |
| ABS.foolish | 0 | 1 | 0 | 7 | 0.000 | N/A | 8 |
| ABS.other | 0 | 0 | 0 | 1 | N/A | N/A | 1 |
| ABS.release | 0 | 7 | 0 | 14 | 0.000 | N/A | 21 |
| ABS.responsibility | 0 | 3 | 0 | 3 | 0.000 | N/A | 6 |
| ABS.self | 0 | 1 | 0 | 0 | 0.000 | N/A | 1 |
| ACT.abandon | 0 | 0 | 0 | 5 | N/A | N/A | 5 |
| ACT.agitate | 0 | 9 | 0 | 3 | 0.000 | N/A | 12 |
| ACT.arrive | 0 | 4 | 0 | 5 | 0.000 | N/A | 9 |
| ACT.ascend | 0 | 0 | 0 | 6 | N/A | N/A | 6 |
| ACT.build | 0 | 1 | 0 | 16 | 0.000 | N/A | 17 |
| ACT.calm | 0 | 1 | 0 | 19 | 0.000 | N/A | 20 |
| ACT.create | 0 | 8 | 0 | 24 | 0.000 | N/A | 32 |
| ACT.descend | 0 | 7 | 0 | 14 | 0.000 | N/A | 21 |
| ACT.destroy | 0 | 1 | 2 | 7 | 0.000 | 0.000 | 10 |
| ACT.dissolve | 0 | 3 | 0 | 9 | 0.000 | N/A | 12 |
| ACT.emit | 0 | 6 | 0 | 2 | 0.000 | N/A | 8 |
| ACT.exit | 0 | 2 | 0 | 27 | 0.000 | N/A | 29 |
| ACT.fall | 0 | 1 | 0 | 2 | 0.000 | N/A | 3 |
| ACT.give | 0 | 3 | 0 | 22 | 0.000 | N/A | 25 |
| ACT.go | 0 | 2 | 0 | 4 | 0.000 | N/A | 6 |
| ACT.leave | 0 | 2 | 0 | 26 | 0.000 | N/A | 28 |
| ACT.lie_down | 0 | 1 | 0 | 11 | 0.000 | N/A | 12 |
| ACT.make | 0 | 7 | 0 | 21 | 0.000 | N/A | 28 |
| ACT.move | 0 | 6 | 0 | 40 | 0.000 | N/A | 46 |
| ACT.obtain | 0 | 5 | 0 | 18 | 0.000 | N/A | 23 |
| ACT.receive | 0 | 0 | 2 | 3 | N/A | 0.000 | 5 |
| ACT.rise | 0 | 3 | 0 | 19 | 0.000 | N/A | 22 |
| ACT.sink | 0 | 5 | 0 | 2 | 0.000 | N/A | 7 |
| ACT.sit | 0 | 32 | 0 | 5 | 0.000 | N/A | 37 |
| ACT.stand | 0 | 1 | 0 | 10 | 0.000 | N/A | 11 |
| ACT.stop | 0 | 3 | 0 | 7 | 0.000 | N/A | 10 |
| ACT.take | 0 | 0 | 0 | 9 | N/A | N/A | 9 |
| BEI.beast | 0 | 0 | 0 | 8 | N/A | N/A | 8 |
| BEI.child | 0 | 3 | 0 | 12 | 0.000 | N/A | 15 |
| BEI.female | 0 | 0 | 0 | 5 | N/A | N/A | 5 |
| BEI.human | 0 | 1 | 0 | 5 | 0.000 | N/A | 6 |
| BEI.male | 0 | 2 | 0 | 1 | 0.000 | N/A | 3 |
| BEI.parent | 0 | 4 | 1 | 7 | 0.000 | 0.000 | 12 |
| BEI.plant | 0 | 2 | 0 | 7 | 0.000 | N/A | 9 |
| BEI.root | 0 | 2 | 1 | 0 | 0.000 | 0.000 | 3 |
| BOD.ear | 0 | 4 | 0 | 0 | 0.000 | N/A | 4 |
| BOD.eye | 0 | 3 | 0 | 3 | 0.000 | N/A | 6 |
| BOD.face | 0 | 6 | 0 | 4 | 0.000 | N/A | 10 |
| BOD.foot | 0 | 6 | 0 | 10 | 0.000 | N/A | 16 |
| BOD.hand | 0 | 2 | 0 | 10 | 0.000 | N/A | 12 |
| BOD.head | 0 | 2 | 0 | 4 | 0.000 | N/A | 6 |
| BOD.hip | 2 | 2 | 0 | 3 | 0.500 | 1.000 | 7 |
| BOD.mouth | 0 | 1 | 0 | 32 | 0.000 | N/A | 33 |
| CHG.advance | 0 | 11 | 0 | 4 | 0.000 | N/A | 15 |
| CHG.begin | 0 | 3 | 4 | 15 | 0.000 | 0.000 | 22 |
| CHG.decay | 0 | 9 | 2 | 6 | 0.000 | 0.000 | 17 |
| CHG.end | 1 | 7 | 0 | 12 | 0.125 | 1.000 | 20 |
| CHG.grow | 0 | 0 | 0 | 2 | N/A | N/A | 2 |
| CHG.retreat | 0 | 0 | 0 | 12 | N/A | N/A | 12 |
| CHG.stay | 0 | 0 | 0 | 16 | N/A | N/A | 16 |
| COG.confusion | 0 | 0 | 0 | 9 | N/A | N/A | 9 |
| COG.enlightenment | 0 | 1 | 0 | 2 | 0.000 | N/A | 3 |
| COG.forget | 0 | 0 | 0 | 2 | N/A | N/A | 2 |
| COG.ignorance | 0 | 1 | 0 | 1 | 0.000 | N/A | 2 |
| COG.instinct | 0 | 2 | 1 | 2 | 0.000 | 0.000 | 5 |
| COG.intention | 0 | 5 | 0 | 14 | 0.000 | N/A | 19 |
| COG.know | 0 | 0 | 0 | 6 | N/A | N/A | 6 |
| COG.learn | 0 | 1 | 0 | 19 | 0.000 | N/A | 20 |
| COG.mindless | 0 | 2 | 0 | 4 | 0.000 | N/A | 6 |
| COG.remember | 0 | 1 | 0 | 9 | 0.000 | N/A | 10 |
| COG.separate | 0 | 6 | 0 | 12 | 0.000 | N/A | 18 |
| COG.think | 0 | 4 | 0 | 13 | 0.000 | N/A | 17 |
| COG.unlearned | 0 | 6 | 0 | 1 | 0.000 | N/A | 7 |
| COM.announce | 0 | 2 | 0 | 4 | 0.000 | N/A | 6 |
| COM.answer | 0 | 1 | 0 | 7 | 0.000 | N/A | 8 |
| COM.conduct | 0 | 0 | 0 | 6 | N/A | N/A | 6 |
| COM.conflict | 0 | 14 | 0 | 0 | 0.000 | N/A | 14 |
| COM.cooperate | 0 | 1 | 1 | 2 | 0.000 | 0.000 | 4 |
| COM.muteness | 0 | 5 | 0 | 1 | 0.000 | N/A | 6 |
| COM.practice | 0 | 2 | 0 | 1 | 0.000 | N/A | 3 |
| COM.question | 2 | 5 | 0 | 6 | 0.286 | 1.000 | 13 |
| COM.secret | 0 | 9 | 0 | 3 | 0.000 | N/A | 12 |
| COM.silence | 0 | 6 | 0 | 1 | 0.000 | N/A | 7 |
| COM.speak | 0 | 8 | 0 | 18 | 0.000 | N/A | 26 |
| COM.teach | 0 | 2 | 0 | 12 | 0.000 | N/A | 14 |
| ECO.buy | 0 | 1 | 0 | 2 | 0.000 | N/A | 3 |
| ECO.currency | 0 | 0 | 0 | 13 | N/A | N/A | 13 |
| ECO.earn | 0 | 0 | 0 | 11 | N/A | N/A | 11 |
| ECO.income | 0 | 0 | 0 | 19 | N/A | N/A | 19 |
| ECO.loss | 0 | 0 | 0 | 7 | N/A | N/A | 7 |
| ECO.money | 0 | 1 | 0 | 3 | 0.000 | N/A | 4 |
| ECO.pay | 0 | 5 | 0 | 3 | 0.000 | N/A | 8 |
| ECO.price | 0 | 5 | 0 | 5 | 0.000 | N/A | 10 |
| ECO.save | 0 | 1 | 0 | 0 | 0.000 | N/A | 1 |
| ECO.sell | 0 | 6 | 0 | 7 | 0.000 | N/A | 13 |
| ECO.waste | 0 | 13 | 0 | 4 | 0.000 | N/A | 17 |
| ECO.withdraw | 0 | 1 | 0 | 1 | 0.000 | N/A | 2 |
| ELM.darkness | 0 | 8 | 0 | 5 | 0.000 | N/A | 13 |
| ELM.earth | 0 | 5 | 0 | 11 | 0.000 | N/A | 16 |
| ELM.fire | 0 | 10 | 1 | 1 | 0.000 | 0.000 | 12 |
| ELM.light | 0 | 3 | 0 | 0 | 0.000 | N/A | 3 |
| ELM.moon | 0 | 2 | 0 | 3 | 0.000 | N/A | 5 |
| ELM.morning | 9 | 3 | 0 | 0 | 0.750 | 1.000 | 12 |
| ELM.night | 2 | 6 | 0 | 0 | 0.250 | 1.000 | 8 |
| ELM.sky | 0 | 2 | 0 | 2 | 0.000 | N/A | 4 |
| ELM.star | 0 | 3 | 0 | 0 | 0.000 | N/A | 3 |
| ELM.sun | 0 | 2 | 0 | 3 | 0.000 | N/A | 5 |
| ELM.water | 0 | 0 | 0 | 7 | N/A | N/A | 7 |
| ELM.wind | 2 | 5 | 0 | 8 | 0.286 | 1.000 | 15 |
| EMO.anger | 0 | 6 | 0 | 2 | 0.000 | N/A | 8 |
| EMO.anxiety | 0 | 0 | 0 | 4 | N/A | N/A | 4 |
| EMO.apology | 0 | 1 | 0 | 2 | 0.000 | N/A | 3 |
| EMO.compassion | 0 | 3 | 0 | 0 | 0.000 | N/A | 3 |
| EMO.contempt | 0 | 5 | 0 | 4 | 0.000 | N/A | 9 |
| EMO.courage | 0 | 0 | 0 | 8 | N/A | N/A | 8 |
| EMO.desire | 0 | 1 | 0 | 8 | 0.000 | N/A | 9 |
| EMO.dislike | 0 | 4 | 0 | 8 | 0.000 | N/A | 12 |
| EMO.doubt | 0 | 12 | 0 | 4 | 0.000 | N/A | 16 |
| EMO.fear | 0 | 12 | 0 | 5 | 0.000 | N/A | 17 |
| EMO.forgiveness | 0 | 1 | 0 | 1 | 0.000 | N/A | 2 |
| EMO.happiness | 0 | 3 | 0 | 0 | 0.000 | N/A | 3 |
| EMO.hate | 0 | 0 | 0 | 3 | N/A | N/A | 3 |
| EMO.hope | 0 | 1 | 0 | 1 | 0.000 | N/A | 2 |
| EMO.joy | 0 | 10 | 0 | 1 | 0.000 | N/A | 11 |
| EMO.like | 0 | 1 | 0 | 3 | 0.000 | N/A | 4 |
| EMO.love | 0 | 5 | 0 | 1 | 0.000 | N/A | 6 |
| EMO.manifest | 0 | 1 | 0 | 10 | 0.000 | N/A | 11 |
| EMO.patience | 0 | 3 | 0 | 1 | 0.000 | N/A | 4 |
| EMO.pleasure | 0 | 4 | 0 | 0 | 0.000 | N/A | 4 |
| EMO.pride | 0 | 2 | 0 | 0 | 0.000 | N/A | 2 |
| EMO.respect | 0 | 2 | 0 | 8 | 0.000 | N/A | 10 |
| EMO.satisfaction | 0 | 2 | 0 | 4 | 0.000 | N/A | 6 |
| EMO.shame | 0 | 1 | 0 | 4 | 0.000 | N/A | 5 |
| EMO.sorrow | 0 | 5 | 0 | 5 | 0.000 | N/A | 10 |
| EMO.trust | 0 | 4 | 0 | 3 | 0.000 | N/A | 7 |
| EMO.unhappiness | 0 | 3 | 0 | 3 | 0.000 | N/A | 6 |
| EMO.wish | 0 | 4 | 0 | 3 | 0.000 | N/A | 7 |
| EMO.worry | 0 | 4 | 0 | 3 | 0.000 | N/A | 7 |
| EXS.absence | 0 | 1 | 0 | 2 | 0.000 | N/A | 3 |
| EXS.being | 0 | 1 | 0 | 6 | 0.000 | N/A | 7 |
| EXS.death | 1 | 1 | 4 | 4 | 0.500 | 0.200 | 10 |
| EXS.life | 0 | 4 | 0 | 0 | 0.000 | N/A | 4 |
| EXS.matter | 0 | 0 | 0 | 9 | N/A | N/A | 9 |
| EXS.nonbeing | 0 | 0 | 0 | 2 | N/A | N/A | 2 |
| EXS.physical_body | 0 | 16 | 0 | 0 | 0.000 | N/A | 16 |
| EXS.presence | 0 | 1 | 0 | 1 | 0.000 | N/A | 2 |
| EXS.spirit | 0 | 2 | 0 | 6 | 0.000 | N/A | 8 |
| EXS.vitality | 0 | 2 | 0 | 12 | 0.000 | N/A | 14 |
| EXS.void | 0 | 5 | 0 | 7 | 0.000 | N/A | 12 |
| FND.ahistorical | 2 | 0 | 2 | 0 | 1.000 | 0.500 | 4 |
| FND.body | 0 | 0 | 0 | 3 | N/A | N/A | 3 |
| FND.consciousness | 0 | 3 | 0 | 1 | 0.000 | N/A | 4 |
| FND.history | 0 | 2 | 0 | 2 | 0.000 | N/A | 4 |
| FND.information | 2 | 2 | 1 | 4 | 0.500 | 0.667 | 9 |
| FND.intuition | 2 | 2 | 2 | 3 | 0.500 | 0.500 | 9 |
| FND.labor | 0 | 3 | 0 | 9 | 0.000 | N/A | 12 |
| FND.language | 0 | 2 | 0 | 9 | 0.000 | N/A | 11 |
| FND.languageless | 0 | 0 | 2 | 1 | N/A | 0.000 | 3 |
| FND.logic | 0 | 4 | 0 | 1 | 0.000 | N/A | 5 |
| FND.memory | 0 | 9 | 0 | 7 | 0.000 | N/A | 16 |
| FND.mind | 0 | 12 | 0 | 10 | 0.000 | N/A | 22 |
| FND.number | 0 | 2 | 0 | 9 | 0.000 | N/A | 11 |
| FND.numberless | 1 | 0 | 2 | 13 | 1.000 | 0.333 | 16 |
| FND.play | 0 | 3 | 0 | 8 | 0.000 | N/A | 11 |
| FND.spatiality | 0 | 2 | 0 | 6 | 0.000 | N/A | 8 |
| FND.temporality | 0 | 5 | 0 | 0 | 0.000 | N/A | 5 |
| FND.time | 0 | 1 | 1 | 4 | 0.000 | 0.000 | 6 |
| FND.timeless | 0 | 0 | 0 | 4 | N/A | N/A | 4 |
| FND.transformation | 0 | 0 | 3 | 5 | N/A | 0.000 | 8 |
| FND.unchanging | 0 | 3 | 2 | 4 | 0.000 | 0.000 | 9 |
| FND.unconscious | 0 | 1 | 0 | 4 | 0.000 | N/A | 5 |
| FND.uninformed | 0 | 3 | 0 | 0 | 0.000 | N/A | 3 |
| LOG.cause | 0 | 7 | 0 | 9 | 0.000 | N/A | 16 |
| LOG.effect | 0 | 1 | 0 | 8 | 0.000 | N/A | 9 |
| LOG.reason | 0 | 3 | 0 | 2 | 0.000 | N/A | 5 |
| LOG.unreason | 0 | 2 | 0 | 1 | 0.000 | N/A | 3 |
| MAT.clothing | 0 | 14 | 0 | 23 | 0.000 | N/A | 37 |
| MAT.drink | 0 | 4 | 0 | 16 | 0.000 | N/A | 20 |
| MAT.food | 0 | 9 | 0 | 19 | 0.000 | N/A | 28 |
| MAT.naked | 0 | 0 | 0 | 5 | N/A | N/A | 5 |
| MAT.tool | 0 | 1 | 0 | 4 | 0.000 | N/A | 5 |
| NAT.flower | 0 | 5 | 0 | 3 | 0.000 | N/A | 8 |
| NAT.river | 0 | 3 | 0 | 3 | 0.000 | N/A | 6 |
| NAT.sea | 0 | 1 | 0 | 1 | 0.000 | N/A | 2 |
| NAT.tree | 0 | 3 | 0 | 3 | 0.000 | N/A | 6 |
| PER.bitter | 0 | 2 | 0 | 0 | 0.000 | N/A | 2 |
| PER.blind | 0 | 4 | 0 | 0 | 0.000 | N/A | 4 |
| PER.deaf | 0 | 1 | 0 | 1 | 0.000 | N/A | 2 |
| PER.delicious | 0 | 1 | 0 | 0 | 0.000 | N/A | 1 |
| PER.feel | 0 | 6 | 0 | 2 | 0.000 | N/A | 8 |
| PER.fragrance | 0 | 11 | 0 | 4 | 0.000 | N/A | 15 |
| PER.hear | 0 | 1 | 0 | 2 | 0.000 | N/A | 3 |
| PER.numb | 0 | 1 | 0 | 6 | 0.000 | N/A | 7 |
| PER.odorless | 0 | 3 | 0 | 0 | 0.000 | N/A | 3 |
| PER.salty | 0 | 4 | 0 | 1 | 0.000 | N/A | 5 |
| PER.see | 0 | 3 | 0 | 6 | 0.000 | N/A | 9 |
| PER.smell | 0 | 8 | 0 | 6 | 0.000 | N/A | 14 |
| PER.sound | 0 | 6 | 0 | 5 | 0.000 | N/A | 11 |
| PER.soundless | 0 | 4 | 0 | 5 | 0.000 | N/A | 9 |
| PER.stench | 0 | 1 | 0 | 1 | 0.000 | N/A | 2 |
| PER.sweet | 0 | 8 | 0 | 10 | 0.000 | N/A | 18 |
| PER.taste | 0 | 14 | 0 | 1 | 0.000 | N/A | 15 |
| PER.tasteless | 0 | 5 | 0 | 8 | 0.000 | N/A | 13 |
| PER.touch | 0 | 3 | 0 | 11 | 0.000 | N/A | 14 |
| PER.untouched | 0 | 0 | 0 | 3 | N/A | N/A | 3 |
| PRP.aged | 0 | 8 | 0 | 1 | 0.000 | N/A | 9 |
| PRP.blunt | 0 | 4 | 0 | 0 | 0.000 | N/A | 4 |
| PRP.bright | 0 | 4 | 0 | 7 | 0.000 | N/A | 11 |
| PRP.clean | 0 | 1 | 0 | 2 | 0.000 | N/A | 3 |
| PRP.clear | 2 | 3 | 9 | 2 | 0.400 | 0.182 | 16 |
| PRP.cold | 0 | 1 | 0 | 7 | 0.000 | N/A | 8 |
| PRP.dark | 0 | 12 | 0 | 3 | 0.000 | N/A | 15 |
| PRP.deep | 0 | 3 | 0 | 3 | 0.000 | N/A | 6 |
| PRP.difficult | 0 | 1 | 0 | 6 | 0.000 | N/A | 7 |
| PRP.dirty | 0 | 3 | 0 | 5 | 0.000 | N/A | 8 |
| PRP.dry | 0 | 5 | 0 | 3 | 0.000 | N/A | 8 |
| PRP.easy | 0 | 0 | 0 | 8 | N/A | N/A | 8 |
| PRP.far | 0 | 1 | 1 | 4 | 0.000 | 0.000 | 6 |
| PRP.fast | 0 | 4 | 0 | 3 | 0.000 | N/A | 7 |
| PRP.hard | 0 | 0 | 0 | 1 | N/A | N/A | 1 |
| PRP.heavy | 0 | 1 | 0 | 3 | 0.000 | N/A | 4 |
| PRP.high | 0 | 0 | 0 | 2 | N/A | N/A | 2 |
| PRP.hot | 0 | 2 | 0 | 2 | 0.000 | N/A | 4 |
| PRP.impossible | 1 | 0 | 0 | 3 | 1.000 | 1.000 | 4 |
| PRP.large | 0 | 0 | 0 | 4 | N/A | N/A | 4 |
| PRP.light | 0 | 2 | 0 | 1 | 0.000 | N/A | 3 |
| PRP.long | 0 | 1 | 0 | 13 | 0.000 | N/A | 14 |
| PRP.low | 0 | 0 | 0 | 4 | N/A | N/A | 4 |
| PRP.multiple | 0 | 0 | 0 | 1 | N/A | N/A | 1 |
| PRP.murky | 0 | 3 | 0 | 6 | 0.000 | N/A | 9 |
| PRP.narrow | 0 | 1 | 0 | 5 | 0.000 | N/A | 6 |
| PRP.near | 1 | 3 | 2 | 3 | 0.250 | 0.333 | 9 |
| PRP.new | 0 | 0 | 3 | 3 | N/A | 0.000 | 6 |
| PRP.part | 0 | 3 | 0 | 8 | 0.000 | N/A | 11 |
| PRP.possible | 0 | 0 | 0 | 8 | N/A | N/A | 8 |
| PRP.rough | 1 | 1 | 0 | 20 | 0.500 | 1.000 | 22 |
| PRP.shallow | 0 | 0 | 2 | 2 | N/A | 0.000 | 4 |
| PRP.sharp | 0 | 1 | 0 | 1 | 0.000 | N/A | 2 |
| PRP.short | 0 | 0 | 0 | 1 | N/A | N/A | 1 |
| PRP.single | 0 | 0 | 0 | 1 | N/A | N/A | 1 |
| PRP.slow | 0 | 3 | 0 | 9 | 0.000 | N/A | 12 |
| PRP.small | 0 | 0 | 0 | 8 | N/A | N/A | 8 |
| PRP.smooth | 0 | 3 | 0 | 3 | 0.000 | N/A | 6 |
| PRP.soft | 0 | 0 | 0 | 3 | N/A | N/A | 3 |
| PRP.strong | 0 | 1 | 0 | 3 | 0.000 | N/A | 4 |
| PRP.weak | 0 | 8 | 0 | 2 | 0.000 | N/A | 10 |
| PRP.wet | 0 | 4 | 0 | 0 | 0.000 | N/A | 4 |
| PRP.whole | 0 | 0 | 0 | 8 | N/A | N/A | 8 |
| PRP.wide | 0 | 2 | 0 | 1 | 0.000 | N/A | 3 |
| PRP.young | 1 | 4 | 0 | 3 | 0.200 | 1.000 | 8 |
| REL.alone | 2 | 0 | 0 | 1 | 1.000 | 1.000 | 3 |
| REL.different | 1 | 0 | 0 | 6 | 1.000 | 1.000 | 7 |
| REL.same | 0 | 1 | 0 | 12 | 0.000 | N/A | 13 |
| REL.together | 0 | 0 | 0 | 2 | N/A | N/A | 2 |
| SOC.attack | 0 | 2 | 0 | 2 | 0.000 | N/A | 4 |
| SOC.awake | 0 | 1 | 0 | 10 | 0.000 | N/A | 11 |
| SOC.citizen | 1 | 1 | 0 | 4 | 0.500 | 1.000 | 6 |
| SOC.criticize | 0 | 5 | 0 | 1 | 0.000 | N/A | 6 |
| SOC.defend | 0 | 3 | 0 | 10 | 0.000 | N/A | 13 |
| SOC.family | 0 | 6 | 1 | 7 | 0.000 | 0.000 | 14 |
| SOC.homeless | 0 | 0 | 0 | 7 | N/A | N/A | 7 |
| SOC.individual | 0 | 1 | 0 | 10 | 0.000 | N/A | 11 |
| SOC.nation | 0 | 0 | 0 | 7 | N/A | N/A | 7 |
| SOC.obstruct | 0 | 0 | 0 | 31 | N/A | N/A | 31 |
| SOC.official | 1 | 1 | 0 | 7 | 0.500 | 1.000 | 9 |
| SOC.pass | 0 | 1 | 0 | 3 | 0.000 | N/A | 4 |
| SOC.praise | 0 | 1 | 0 | 3 | 0.000 | N/A | 4 |
| SOC.public | 0 | 0 | 0 | 1 | N/A | N/A | 1 |
| SOC.refuse | 0 | 1 | 0 | 0 | 0.000 | N/A | 1 |
| SOC.request | 0 | 3 | 1 | 8 | 0.000 | 0.000 | 12 |
| SOC.rest | 0 | 0 | 0 | 4 | N/A | N/A | 4 |
| SOC.sleep | 0 | 4 | 0 | 2 | 0.000 | N/A | 6 |
| SOC.stateless | 0 | 2 | 0 | 0 | 0.000 | N/A | 2 |
| SOC.village | 2 | 4 | 0 | 2 | 0.333 | 1.000 | 8 |
| SOC.work | 0 | 3 | 0 | 26 | 0.000 | N/A | 29 |
| SPC.direction | 0 | 4 | 0 | 0 | 0.000 | N/A | 4 |
| SPC.inside | 0 | 0 | 2 | 3 | N/A | 0.000 | 5 |
| SPC.nowhere | 0 | 0 | 0 | 4 | N/A | N/A | 4 |
| SPC.outside | 0 | 2 | 0 | 3 | 0.000 | N/A | 5 |
| SPC.place | 0 | 0 | 0 | 6 | N/A | N/A | 6 |
| SPC.reverse | 0 | 0 | 0 | 5 | N/A | N/A | 5 |
| STA.comfort | 0 | 3 | 0 | 9 | 0.000 | N/A | 12 |
| STA.danger | 0 | 11 | 0 | 3 | 0.000 | N/A | 14 |
| STA.healing | 0 | 1 | 0 | 7 | 0.000 | N/A | 8 |
| STA.health | 0 | 1 | 0 | 1 | 0.000 | N/A | 2 |
| STA.illness | 0 | 3 | 0 | 16 | 0.000 | N/A | 19 |
| STA.pain | 0 | 12 | 0 | 10 | 0.000 | N/A | 22 |
| STA.peace | 0 | 3 | 0 | 0 | 0.000 | N/A | 3 |
| STA.poverty | 0 | 7 | 0 | 0 | 0.000 | N/A | 7 |
| STA.war | 2 | 0 | 1 | 3 | 1.000 | 0.667 | 6 |
| STA.wealth | 0 | 6 | 0 | 1 | 0.000 | N/A | 7 |
| STA.wound | 1 | 10 | 0 | 13 | 0.091 | 1.000 | 24 |
| TIM.appear | 0 | 0 | 3 | 1 | N/A | 0.000 | 4 |
| TIM.come | 0 | 0 | 0 | 19 | N/A | N/A | 19 |
| TIM.indefinite | 0 | 0 | 0 | 1 | N/A | N/A | 1 |
| TIM.moment | 0 | 1 | 0 | 13 | 0.000 | N/A | 14 |
| TIM.now | 0 | 1 | 0 | 2 | 0.000 | N/A | 3 |
| TIM.past | 0 | 1 | 0 | 3 | 0.000 | N/A | 4 |
| TIM.period | 0 | 5 | 2 | 8 | 0.000 | 0.000 | 15 |
| VAL.beauty | 0 | 1 | 0 | 12 | 0.000 | N/A | 13 |
| VAL.correct | 0 | 3 | 0 | 5 | 0.000 | N/A | 8 |
| VAL.evil | 0 | 5 | 0 | 6 | 0.000 | N/A | 11 |
| VAL.falsehood | 3 | 8 | 2 | 10 | 0.273 | 0.600 | 23 |
| VAL.good | 0 | 6 | 0 | 9 | 0.000 | N/A | 15 |
| VAL.incorrect | 0 | 2 | 0 | 1 | 0.000 | N/A | 3 |
| VAL.profane | 0 | 2 | 0 | 8 | 0.000 | N/A | 10 |
| VAL.sacred | 0 | 4 | 1 | 0 | 0.000 | 0.000 | 5 |
| VAL.truth | 0 | 4 | 0 | 2 | 0.000 | N/A | 6 |
| VAL.ugliness | 0 | 2 | 0 | 3 | 0.000 | N/A | 5 |
| WLD.art | 0 | 8 | 0 | 7 | 0.000 | N/A | 15 |
| WLD.artless | 0 | 0 | 0 | 10 | N/A | N/A | 10 |
| WLD.culture | 0 | 1 | 0 | 2 | 0.000 | N/A | 3 |
| WLD.nonscience | 0 | 0 | 0 | 1 | N/A | N/A | 1 |
| WLD.outer_realm | 0 | 0 | 0 | 2 | N/A | N/A | 2 |
| WLD.realm | 0 | 0 | 0 | 3 | N/A | N/A | 3 |
| WLD.religion | 0 | 6 | 0 | 0 | 0.000 | N/A | 6 |
| WLD.science | 0 | 2 | 0 | 2 | 0.000 | N/A | 4 |
| WLD.technique | 0 | 3 | 0 | 4 | 0.000 | N/A | 7 |
| WLD.uncultured | 0 | 0 | 0 | 3 | N/A | N/A | 3 |
| WLD.unskilled | 0 | 7 | 0 | 1 | 0.000 | N/A | 8 |

## Section E: Preliminary Structural Verdict

1. Embedding-A1 correlation: r=0.1854 — weak linear relationship.
2. FN rate: 2.2% — embedding misses these valid connections.
3. FP rate: 33.4% — embedding assigns these spuriously.
4. Negative-margin words: 82.1% — words closer to another atom than their assigned one.
5. A1 empirical data available for Synapse v4 evidence-based reconstruction.
