# ESDE Phase 9: Psycholinguistic Dictionaries

This directory contains psycholinguistic dictionaries used for token feature extraction.

## Required Files

Place downloaded files in this directory. The system auto-detects multiple filename patterns.

| Dictionary | Primary Filename | Alternative |
|------------|-----------------|-------------|
| Brysbaert | `13428_2013_403_MOESM1_ESM.xlsx` | `brysbaert_concreteness.csv` |
| Kuperman | `AoA_51715_words.csv` | `kuperman_aoa.csv` |
| Lancaster | `Lancaster_sensorimotor_norms_for_39707_words.csv` | `lancaster_sensorimotor.csv` |
| Warriner | `BRM-emot-submit.csv` | `warriner_valence.csv` |

## Download Instructions

### 1. Brysbaert Concreteness Ratings (XLSX)

- **Paper**: Brysbaert, M., Warriner, A.B., & Kuperman, V. (2014)
- **URL**: https://link.springer.com/article/10.3758/s13428-013-0403-5
- **Download**: Supplementary materials → `13428_2013_403_MOESM1_ESM.xlsx`
- **Note**: Requires `openpyxl` package: `pip install openpyxl`

### 2. Kuperman Age of Acquisition

- **Paper**: Kuperman, V., Stadthagen-Gonzalez, H., & Brysbaert, M. (2012)
- **URL**: https://link.springer.com/article/10.3758/s13428-012-0210-4
- **Download**: Supplementary materials

### 3. Lancaster Sensorimotor Norms

- **Paper**: Lynott et al. (2020)
- **URL**: https://osf.io/7emr6/
- **Download**: `Lancaster_sensorimotor_norms_for_39707_words.csv` (~5MB summary version)
- **Note**: Do NOT use the 1GB individual ratings file

### 4. Warriner Emotional Valence

- **Paper**: Warriner, A.B., Kuperman, V., & Brysbaert, M. (2013)
- **URL**: https://link.springer.com/article/10.3758/s13428-012-0314-x
- **Download**: `BRM-emot-submit.csv`

## Fallback Behavior

If dictionary files are missing:
- The system will print a warning
- All lookups for that dictionary will return `-1.0` (NULL)
- Feature extraction will continue with available dictionaries

## Score Interpretation

| Dictionary | Raw Range | Normalized | Meaning |
|------------|-----------|------------|---------|
| Concreteness | 1-5 | 0-1 | 0=abstract, 1=concrete |
| AoA | 0-25 years | 0-1 | Lower=earlier acquisition |
| Sensorimotor | 0-5 | 0-1 | Higher=more perceptual |
| Valence | 1-9 | 0-1 | 0=negative, 1=positive |
| NULL | - | -1.0 | Out of vocabulary |

## Dependencies

```bash
pip install openpyxl  # Required for XLSX files (Brysbaert)
```
