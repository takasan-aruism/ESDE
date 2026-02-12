#!/usr/bin/env python3
"""
Tests for auditor_a1.py — pre-screening logic.
Run: python3 -m pytest test_auditor_a1.py -v
"""

import pytest
from mapper_a1 import SLOT_IDS
from auditor_a1 import pre_screen, PreScreenResult


def make_record(word="test", pos="n", overrides=None, evidence=""):
    """Helper to create a test record with default zero scores."""
    raw = {s: 0 for s in SLOT_IDS}
    if overrides:
        raw.update(overrides)
    return {
        "word": word,
        "pos": pos,
        "atom": "EMO.like",
        "raw_scores": raw,
        "evidence": evidence,
    }


# ============================================================
# C1: Distribution Anomaly
# ============================================================

class TestC1:
    def test_all_zero(self):
        rec = make_record()
        r = pre_screen(rec, set())
        assert "C1_all_zero" in r.flags
    
    def test_deflation(self):
        """Only 3 nonzero slots."""
        rec = make_record(overrides={
            "resonance.essential": 5,
            "scale.individual": 3,
            "epistemological.experience": 7,
        })
        r = pre_screen(rec, set())
        assert "C1_deflation" in r.flags
    
    def test_inflation_sum(self):
        """Very high total sum."""
        overrides = {s: 4 for s in SLOT_IDS}  # 48 * 4 = 192
        rec = make_record(overrides=overrides)
        r = pre_screen(rec, set())
        assert "C1_inflation_sum" in r.flags
    
    def test_inflation_spread(self):
        """Too many nonzero slots (info level)."""
        overrides = {s: 1 for s in SLOT_IDS}  # all 48 nonzero
        rec = make_record(overrides=overrides)
        r = pre_screen(rec, set())
        assert "C1_inflation_spread" in r.flags
    
    def test_inflation_high(self):
        """Too many high-scoring slots."""
        overrides = {}
        for i, s in enumerate(SLOT_IDS):
            overrides[s] = 8 if i < 16 else 0  # 16 slots at 8
        rec = make_record(overrides=overrides)
        r = pre_screen(rec, set())
        assert "C1_inflation_high" in r.flags
    
    def test_normal_distribution_clean(self):
        """Typical healthy distribution — no C1 flags."""
        overrides = {
            "epistemological.experience": 8,
            "scale.individual": 7,
            "resonance.essential": 6,
            "value_generation.aesthetic": 7,
            "ontological.semantic": 5,
            "temporal.continuation": 4,
            "symmetry.inclusive": 3,
            "lawfulness.contingent": 3,
            "epistemological.perception": 2,
            "interconnection.independent": 2,
        }
        rec = make_record(overrides=overrides)
        r = pre_screen(rec, set())
        c1_flags = [f for f in r.flags if f.startswith("C1")]
        assert len(c1_flags) == 0


# ============================================================
# C2: Symmetric Pair Leak
# ============================================================

class TestC2:
    def test_antonym_with_leak(self):
        """Antonym 'dislike' scoring high on non-destructive slots."""
        overrides = {
            "symmetry.destructive": 2,
            "epistemological.experience": 7,  # Leak!
            "scale.individual": 6,           # Leak!
            "resonance.essential": 5,        # Leak!
        }
        rec = make_record(word="dislike", overrides=overrides)
        r = pre_screen(rec, {"dislike", "hate", "aversion"})
        assert "C2_antonym_leak" in r.flags
    
    def test_antonym_correct(self):
        """Antonym 'dislike' correctly concentrated on destructive."""
        overrides = {
            "symmetry.destructive": 8,
            "resonance.superficial": 2,
            "temporal.emergence": 1,
        }
        rec = make_record(word="dislike", overrides=overrides)
        r = pre_screen(rec, {"dislike", "hate"})
        assert "C2_antonym_leak" not in r.flags
    
    def test_antonym_low_destructive(self):
        """Antonym with low destructive score."""
        overrides = {
            "symmetry.destructive": 1,
            "resonance.superficial": 2,
        }
        rec = make_record(word="aversion", overrides=overrides)
        r = pre_screen(rec, {"aversion"})
        assert "C2_antonym_low_destructive" in r.flags
    
    def test_non_antonym_no_c2(self):
        """Normal word — no C2 flags regardless of score pattern."""
        overrides = {
            "symmetry.destructive": 0,
            "epistemological.experience": 9,
        }
        rec = make_record(word="fondness", overrides=overrides)
        r = pre_screen(rec, {"dislike"})
        c2_flags = [f for f in r.flags if f.startswith("C2")]
        assert len(c2_flags) == 0


# ============================================================
# C3: Evidence-Score Mismatch
# ============================================================

class TestC3:
    def test_evidence_matches_top(self):
        """Evidence mentions top axis keywords — no flag."""
        overrides = {
            "epistemological.experience": 9,
            "value_generation.aesthetic": 7,
        }
        rec = make_record(
            overrides=overrides,
            evidence="This resonates with experiential knowing and aesthetic value."
        )
        r = pre_screen(rec, set())
        assert "C3_evidence_mismatch" not in r.flags
    
    def test_evidence_no_match(self):
        """Evidence doesn't mention any top keywords."""
        overrides = {
            "epistemological.experience": 9,
            "value_generation.aesthetic": 7,
            "scale.individual": 6,
        }
        rec = make_record(
            overrides=overrides,
            evidence="This word relates to warmth and comfort in daily life."
        )
        r = pre_screen(rec, set())
        assert "C3_evidence_mismatch" in r.flags
    
    def test_evidence_empty_no_crash(self):
        """Empty evidence — should not crash."""
        overrides = {"epistemological.experience": 9}
        rec = make_record(overrides=overrides, evidence="")
        r = pre_screen(rec, set())
        # No crash is the test


# ============================================================
# C4: Axis-Generic Inflation
# ============================================================

class TestC4:
    def test_uniform_high_axis(self):
        """All levels of an axis uniformly high → generic flag."""
        overrides = {
            "resonance.superficial": 5,
            "resonance.structural": 5,
            "resonance.essential": 6,
            "resonance.existential": 5,
        }
        rec = make_record(overrides=overrides)
        r = pre_screen(rec, set())
        assert any("C4_generic_resonance" in f for f in r.flags)
    
    def test_peaked_high_axis(self):
        """One level high, others low → no generic flag even if mean is high."""
        overrides = {
            "resonance.superficial": 1,
            "resonance.structural": 0,
            "resonance.essential": 9,
            "resonance.existential": 1,
        }
        rec = make_record(overrides=overrides)
        r = pre_screen(rec, set())
        c4_resonance = [f for f in r.flags if "C4_generic_resonance" in f]
        assert len(c4_resonance) == 0
    
    def test_low_axis_no_flag(self):
        """Low axis mean → no generic flag."""
        overrides = {
            "resonance.superficial": 1,
            "resonance.structural": 0,
            "resonance.essential": 2,
            "resonance.existential": 0,
        }
        rec = make_record(overrides=overrides)
        r = pre_screen(rec, set())
        c4_flags = [f for f in r.flags if f.startswith("C4")]
        assert len(c4_flags) == 0


# ============================================================
# C5: POS Coherence
# ============================================================

class TestC5:
    def test_adj_high_material(self):
        """Adjective with high material score → flag."""
        overrides = {"ontological.material": 7}
        rec = make_record(word="pleasant", pos="adj", overrides=overrides)
        r = pre_screen(rec, set())
        assert "C5_pos_material" in r.flags
    
    def test_verb_high_material_no_flag(self):
        """Verb with high material score → no C5 flag (verbs can be material)."""
        overrides = {"ontological.material": 7}
        rec = make_record(word="bask", pos="v", overrides=overrides)
        r = pre_screen(rec, set())
        assert "C5_pos_material" not in r.flags
    
    def test_noun_high_creation(self):
        """Noun 'fondness' with high creation score → flag."""
        overrides = {"experience.creation": 8}
        rec = make_record(word="fondness", pos="n", overrides=overrides)
        r = pre_screen(rec, set())
        assert "C5_pos_creation" in r.flags
    
    def test_noun_creation_word_no_flag(self):
        """Noun 'creation' with high creation score → no flag (name matches)."""
        overrides = {"experience.creation": 8}
        rec = make_record(word="creation", pos="n", overrides=overrides)
        r = pre_screen(rec, set())
        assert "C5_pos_creation" not in r.flags


# ============================================================
# Integration: Real pilot data
# ============================================================

class TestPilotData:
    def test_bask_clean(self):
        """bask should pass pre-screen cleanly."""
        raw = {"temporal.emergence": 2, "temporal.indication": 1, "temporal.influence": 1,
               "temporal.transformation": 0, "temporal.establishment": 0, "temporal.continuation": 5,
               "temporal.permanence": 0, "scale.individual": 7, "scale.community": 0,
               "scale.society": 0, "scale.ecosystem": 0, "scale.stellar": 0, "scale.cosmic": 0,
               "epistemological.perception": 5, "epistemological.identification": 2,
               "epistemological.understanding": 2, "epistemological.experience": 8,
               "epistemological.creation": 0, "ontological.material": 7,
               "ontological.informational": 1, "ontological.relational": 3,
               "ontological.structural": 1, "ontological.semantic": 4,
               "interconnection.independent": 7, "interconnection.catalytic": 0,
               "interconnection.chained": 0, "interconnection.synchronous": 0,
               "interconnection.resonant": 3, "resonance.superficial": 4,
               "resonance.structural": 2, "resonance.essential": 7, "resonance.existential": 0,
               "symmetry.destructive": 0, "symmetry.inclusive": 6, "symmetry.transformative": 0,
               "symmetry.generative": 3, "symmetry.cyclical": 1, "lawfulness.predictable": 7,
               "lawfulness.emergent": 0, "lawfulness.contingent": 5, "lawfulness.necessary": 0,
               "experience.discovery": 2, "experience.creation": 3, "experience.comprehension": 0,
               "value_generation.functional": 0, "value_generation.aesthetic": 9,
               "value_generation.ethical": 2, "value_generation.sacred": 0}
        rec = {"word": "bask", "pos": "v", "atom": "EMO.like", "raw_scores": raw,
               "evidence": "Bask resonates with individual, material, experiential. High aesthetic and essential."}
        r = pre_screen(rec, set())
        assert len(r.flags) == 0
    
    def test_enjoy_has_spread_flag(self):
        """enjoy has 44/48 nonzero — should get info flag."""
        raw = {"temporal.emergence": 1, "temporal.indication": 0, "temporal.influence": 3,
               "temporal.transformation": 1, "temporal.establishment": 2, "temporal.continuation": 5,
               "temporal.permanence": 1, "scale.individual": 8, "scale.community": 2,
               "scale.society": 1, "scale.ecosystem": 0, "scale.stellar": 0, "scale.cosmic": 0,
               "epistemological.perception": 5, "epistemological.identification": 3,
               "epistemological.understanding": 2, "epistemological.experience": 9,
               "epistemological.creation": 1, "ontological.material": 4,
               "ontological.informational": 2, "ontological.relational": 3,
               "ontological.structural": 1, "ontological.semantic": 6,
               "interconnection.independent": 5, "interconnection.catalytic": 3,
               "interconnection.chained": 2, "interconnection.synchronous": 1,
               "interconnection.resonant": 4, "resonance.superficial": 4,
               "resonance.structural": 2, "resonance.essential": 7, "resonance.existential": 5,
               "symmetry.destructive": 1, "symmetry.inclusive": 3, "symmetry.transformative": 2,
               "symmetry.generative": 3, "symmetry.cyclical": 1, "lawfulness.predictable": 4,
               "lawfulness.emergent": 2, "lawfulness.contingent": 5, "lawfulness.necessary": 1,
               "experience.discovery": 2, "experience.creation": 1, "experience.comprehension": 3,
               "value_generation.functional": 2, "value_generation.aesthetic": 7,
               "value_generation.ethical": 3, "value_generation.sacred": 4}
        rec = {"word": "enjoy", "pos": "v", "atom": "EMO.like", "raw_scores": raw,
               "evidence": "Enjoy is experiential and aesthetic."}
        r = pre_screen(rec, set())
        assert "C1_inflation_spread" in r.flags
        # But it should NOT trigger LLM (info-level only)
        assert not r.needs_llm_audit


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
