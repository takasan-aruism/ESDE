# ESDE Phase 9: Internal Condition Pipeline

## Overview

This pipeline solves the "1-condition death" problem by extracting conditions from **internal structure** rather than external metadata.

### Problem (従来)
```
source_meta条件 (wiki/ja) → 同種データで1条件に潰れる → W3-W6死亡
```

### Solution (解決)
```
内部構造条件 (section/passive/quote) → データから軸が自動生成 → W3-W6復活
```

## Quick Start

```bash
cd /home/claude
python esde_phase9_v2/statistics/pipeline/run_full_pipeline.py --axis section
```

## Available Condition Axes

| Axis | Description | Typical Conditions |
|------|-------------|-------------------|
| `section` | Section names | Lead, Military, Death, Legacy |
| `passive` | Voice mode | passive:0, passive:1 |
| `paren` | Parentheses | paren:0, paren:1 |
| `quote` | Quotations | quote:0, quote:1 |
| `propn` | Proper nouns | propn:0, propn:1 |

## Pipeline Flow

```
W1: Feature Extraction (20-dim token vectors)
  ↓
W2: Conditional Statistics (by internal condition)
  ↓
W3: S-Score Calculation (specificity per condition)
  ↓
W4: Article Projection (resonance vectors)
  ↓
W5: Clustering (island formation)
  ↓
W6: Export (Markdown + JSON)
```

## Output Files

- `output/report.md` - Human-readable analysis report
- `output/analysis.json` - Machine-readable data

## Key Components

| File | Description |
|------|-------------|
| `condition_provider.py` | 5 condition axis providers |
| `w2_aggregator.py` | Conditional statistics aggregation |
| `w3_calculator.py` | S-Score calculation |
| `w4_projector.py` | Article vector projection |
| `w5_w6_adapter.py` | Clustering and export |
| `run_full_pipeline.py` | Complete pipeline runner |

## Test Results

### Section Axis (3 Sengoku warlords)
- **Conditions**: 6 (Lead, Early life, Military, Policies, Death, Legacy)
- **Islands**: 1 (all 3 articles clustered together)
- **Cohesion**: 0.95

### Passive Axis
- **Conditions**: 2 (passive:0, passive:1)
- **Islands**: 1
- **Cohesion**: 0.99

## API Usage

```python
from esde_phase9_v2.statistics.features import FeatureExtractor
from esde_phase9_v2.statistics.pipeline import (
    W2Aggregator,
    W3Calculator,
    W4Projector,
    SimpleCondensator,
)

# Extract features
extractor = FeatureExtractor()
features = extractor.extract_text("Your text here")

# W2: Aggregate by condition
aggregator = W2Aggregator(axis="section")
aggregator.process_article("article_id", features, sections)
w2_stats = aggregator.get_stats()

# W3: Calculate S-Scores
calculator = W3Calculator(w2_stats)
w3_result = calculator.calculate_all()

# W4: Project articles
projector = W4Projector(w3_result)
w4_result = projector.project_all(articles)

# W5: Cluster
condensator = SimpleCondensator(threshold=0.9)
structure = condensator.condense(records)
```

## Philosophy

> "Describe, but do not decide."

This pipeline discovers patterns emergently from internal structure, 
rather than imposing predefined categories.

## Version

Phase 9 Pipeline v1.0.0
