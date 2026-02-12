#!/usr/bin/env python3
"""
Tests for mapper_a1.py — post-processing logic.
Run: python3 -m pytest test_mapper_a1.py -v
"""

import json
import math
import pytest
from mapper_a1 import (
    SLOT_IDS, softmax, shannon_entropy, focus_rate,
    classify_status, process_raw_scores, parse_qwq_response,
    DIFFUSE_THRESHOLD
)


# ============================================================
# Slot structure
# ============================================================

def test_slot_count():
    assert len(SLOT_IDS) == 48


def test_slot_names_unique():
    assert len(set(SLOT_IDS)) == 48


def test_slot_format():
    for s in SLOT_IDS:
        parts = s.split(".")
        assert len(parts) == 2, f"Bad format: {s}"


# ============================================================
# Softmax
# ============================================================

def test_softmax_sums_to_one():
    raw = {s: 0.0 for s in SLOT_IDS}
    raw["resonance.essential"] = 8.0
    raw["temporal.establishment"] = 5.0
    result = softmax(raw)
    total = sum(result.values())
    assert abs(total - 1.0) < 1e-4, f"Sum = {total}"


def test_softmax_all_zeros():
    raw = {s: 0.0 for s in SLOT_IDS}
    result = softmax(raw)
    # Uniform distribution
    expected = 1.0 / 48
    for v in result.values():
        assert abs(v - expected) < 1e-4


def test_softmax_one_hot():
    """One very high score should dominate."""
    raw = {s: 0.0 for s in SLOT_IDS}
    raw["resonance.essential"] = 100.0
    result = softmax(raw)
    assert result["resonance.essential"] > 0.99


def test_softmax_preserves_order():
    raw = {s: 0.0 for s in SLOT_IDS}
    raw["temporal.emergence"] = 3.0
    raw["temporal.indication"] = 5.0
    raw["temporal.influence"] = 1.0
    result = softmax(raw)
    assert result["temporal.indication"] > result["temporal.emergence"]
    assert result["temporal.emergence"] > result["temporal.influence"]


# ============================================================
# Entropy
# ============================================================

def test_entropy_uniform():
    """Uniform distribution → max entropy → 1.0."""
    probs = {s: 1.0 / 48 for s in SLOT_IDS}
    h = shannon_entropy(probs)
    assert abs(h - 1.0) < 0.01, f"H = {h}"


def test_entropy_one_hot():
    """One-hot → min entropy → 0.0."""
    probs = {s: 0.0 for s in SLOT_IDS}
    probs["resonance.essential"] = 1.0
    h = shannon_entropy(probs)
    assert h < 0.01, f"H = {h}"


def test_entropy_between():
    """Partially concentrated → 0 < H < 1."""
    raw = {s: 0.0 for s in SLOT_IDS}
    raw["resonance.essential"] = 8.0
    raw["temporal.establishment"] = 5.0
    raw["scale.individual"] = 3.0
    probs = softmax(raw)
    h = shannon_entropy(probs)
    assert 0.0 < h < 1.0, f"H = {h}"


# ============================================================
# Focus Rate
# ============================================================

def test_focus_rate_range():
    for h in [0.0, 0.3, 0.5, 0.7, 1.0]:
        f = focus_rate(h)
        assert 0.0 <= f <= 1.0


def test_focus_rate_inverse():
    assert focus_rate(0.0) == 1.0
    assert focus_rate(1.0) == 0.0


# ============================================================
# Status Classification
# ============================================================

def test_diffuse_threshold():
    assert classify_status(DIFFUSE_THRESHOLD - 0.01) == "Diffuse_Observation"
    assert classify_status(DIFFUSE_THRESHOLD) == "OK"
    assert classify_status(0.8) == "OK"


# ============================================================
# Full Pipeline
# ============================================================

def test_process_raw_scores_typical():
    """Typical EMO word: few high slots, mostly zero."""
    raw = {s: 0 for s in SLOT_IDS}
    raw["resonance.essential"] = 7
    raw["temporal.establishment"] = 5
    raw["scale.individual"] = 6
    raw["ontological.relational"] = 4
    raw["interconnection.resonant"] = 5
    
    result = process_raw_scores(raw)
    
    assert result["status"] in ("OK", "Diffuse_Observation")
    assert 0.0 < result["entropy_norm"] < 1.0
    assert 0.0 < result["focus_rate"] < 1.0
    assert len(result["top5"]) == 5
    assert result["top5"][0]["slot"] == "resonance.essential"
    assert abs(sum(result["normalized_scores"].values()) - 1.0) < 1e-4


def test_process_raw_scores_all_zeros():
    """All zeros = uniform = diffuse."""
    raw = {s: 0 for s in SLOT_IDS}
    result = process_raw_scores(raw)
    assert result["status"] == "Diffuse_Observation"
    assert result["focus_rate"] < 0.01


def test_process_raw_scores_missing_slots():
    """Missing slots should default to 0."""
    raw = {"resonance.essential": 8}  # Only 1 slot provided
    result = process_raw_scores(raw)
    assert len(result["normalized_scores"]) == 48
    assert result["normalized_scores"]["resonance.essential"] > 0.5


# ============================================================
# QwQ Response Parsing
# ============================================================

def test_parse_json_direct():
    text = '{"word": "fondness", "pos": "n", "atom": "EMO.like", "raw_scores": {"temporal.emergence": 2}}'
    result = parse_qwq_response(text)
    assert result["word"] == "fondness"


def test_parse_json_with_think_block():
    text = """<think>
Let me think about this word...
The word "fondness" relates to liking.
</think>

```json
{"word": "fondness", "pos": "n", "atom": "EMO.like", "raw_scores": {"temporal.emergence": 2}}
```"""
    result = parse_qwq_response(text)
    assert result["word"] == "fondness"


def test_parse_json_with_preamble():
    text = """Here is my analysis:

{
  "word": "fondness",
  "pos": "n",
  "atom": "EMO.like",
  "raw_scores": {"temporal.emergence": 2}
}"""
    result = parse_qwq_response(text)
    assert result["word"] == "fondness"


def test_parse_no_json_raises():
    with pytest.raises(ValueError):
        parse_qwq_response("No JSON here at all!")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
