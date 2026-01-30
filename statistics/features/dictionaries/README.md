# ESDE Phase 9: Psycholinguistic Dictionaries

This directory contains psycholinguistic dictionaries used for token feature extraction.

## Required Files

| File | Source | Description |
|------|--------|-------------|
| `brysbaert_concreteness.csv` | Brysbaert et al. (2014) | Concreteness ratings (40k words) |
| `kuperman_aoa.csv` | Kuperman et al. (2012) | Age of Acquisition norms (30k words) |
| `lancaster_sensorimotor.csv` | Lancaster Sensorimotor Norms | Perceptual/motor ratings (40k words) |
| `warriner_valence.csv` | Warriner et al. (2013) | Emotional valence ratings (14k words) |

## Download Instructions

### 1. Brysbaert Concreteness Ratings

- **Paper**: Brysbaert, M., Warriner, A.B., & Kuperman, V. (2014). Concreteness ratings for 40 thousand generally known English word lemmas.
- **URL**: https://link.springer.com/article/10.3758/s13428-013-0403-5
- **Format Expected**:
  ```
  Word,Conc.M,Conc.SD,Unknown,Total,Percent_known,SUBTLEX,Dom_Pos
  aardvark,4.97,0.18,0,30,100,0.01,Noun
  ```

### 2. Kuperman Age of Acquisition

- **Paper**: Kuperman, V., Stadthagen-Gonzalez, H., & Brysbaert, M. (2012). Age-of-acquisition ratings for 30,000 English words.
- **URL**: https://link.springer.com/article/10.3758/s13428-012-0210-4
- **Format Expected**:
  ```
  Word,Rating.Mean,Rating.SD,Dunlosky,Freq_pm,Dom_PoS_SUBTLEX,Nletters,Nphon,Nsyll
  a,2.58,1.58,2.73,22012.64,Determiner,1,1,1
  ```

### 3. Lancaster Sensorimotor Norms

- **Paper**: Lynott et al. (2020). The Lancaster Sensorimotor Norms.
- **URL**: https://osf.io/7emr6/
- **Format Expected**:
  ```
  Word,Auditory.mean,Gustatory.mean,Haptic.mean,Interoceptive.mean,Olfactory.mean,Visual.mean,...
  ability,1.65,0.35,1.15,1.65,0.40,2.15,...
  ```

### 4. Warriner Emotional Valence

- **Paper**: Warriner, A.B., Kuperman, V., & Brysbaert, M. (2013). Norms of valence, arousal, and dominance for 13,915 English lemmas.
- **URL**: https://link.springer.com/article/10.3758/s13428-012-0314-x
- **Format Expected**:
  ```
  Word,V.Mean.Sum,V.SD.Sum,A.Mean.Sum,A.SD.Sum,D.Mean.Sum,D.SD.Sum
  aardvark,5.71,1.20,3.29,1.86,4.86,2.23
  ```

## Fallback Behavior

If dictionary files are missing:
- The system will print a warning
- All lookups for that dictionary will return `-1.0` (NULL)
- Feature extraction will continue with available dictionaries

## Score Interpretation

| Dictionary | Range | Normalized | Meaning |
|------------|-------|------------|---------|
| Concreteness | 1-5 | 0-1 | 1=abstract, 5=concrete |
| AoA | 0-25 | 0-1 | Lower=earlier acquisition |
| Sensorimotor | 0-5 | 0-1 | Higher=more perceptual |
| Valence | 1-9 | 0-1 | 1=negative, 9=positive |
| NULL | - | -1.0 | Out of vocabulary |
