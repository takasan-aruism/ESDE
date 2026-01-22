"""
ESDE Phase 9-5: W5 Test Suite
=============================

Comprehensive tests for W5 Weak Structural Condensation.

Test Categories:
  - Schema tests (W5Island, W5Structure)
  - Condensator tests (algorithm correctness)
  - GPT Audit compliance (P0-A, P0-B, P1-1, P1-4)
  - Invariant verification (INV-W5-001 through INV-W5-008)

Usage:
    python test_phase95_w5.py
    
Expected: All tests PASS
"""

import sys
import time
import math
from typing import List, Dict

# Import W5 modules
from statistics.schema_w5 import (
    W5Island,
    W5Structure,
    get_canonical_json,
    compute_canonical_hash,
    compare_w5_structures,
    W5_ALGORITHM,
    W5_VECTOR_POLICY,
    W5_SIMILARITY_ROUNDING,
    W5_VECTOR_ROUNDING,
)

from statistics.w5_condensator import (
    W5Condensator,
    condense_batch,
)

from statistics.schema_w4 import (
    W4Record,
    compute_w4_analysis_id,
)


# ==========================================
# Test Helpers
# ==========================================

def make_w4(article_id: str, vector: Dict[str, float]) -> W4Record:
    """Helper to create test W4Record."""
    w4_analysis_id = compute_w4_analysis_id(
        article_id=article_id,
        used_w3={"cond_a": "w3_test_001"},
        tokenizer_version="test_v1",
        normalizer_version="test_v1",
    )
    
    return W4Record(
        article_id=article_id,
        w4_analysis_id=w4_analysis_id,
        resonance_vector=vector,
        used_w3={"cond_a": "w3_test_001"},
        token_count=100,
        tokenizer_version="test_v1",
        normalizer_version="test_v1",
    )


class TestCounter:
    """Simple test counter."""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.tests = []
    
    def record(self, name: str, passed: bool, detail: str = ""):
        if passed:
            self.passed += 1
            status = "✅ PASS"
        else:
            self.failed += 1
            status = "❌ FAIL"
        
        self.tests.append((name, passed, detail))
        print(f"  {status}: {name}")
        if detail and not passed:
            print(f"         {detail}")
    
    def summary(self):
        print(f"\n{'=' * 60}")
        print(f"Results: {self.passed} passed, {self.failed} failed")
        print(f"{'=' * 60}")
        return self.failed == 0


# ==========================================
# Schema Tests
# ==========================================

class TestSchema:
    """Tests for W5 Schema (schema_w5.py)."""
    
    @staticmethod
    def test_canonical_json(counter: TestCounter):
        """INV-W5-006: Canonical JSON is order-independent."""
        data1 = {"z": 3, "a": 1, "m": 2}
        data2 = {"a": 1, "m": 2, "z": 3}
        data3 = {"m": 2, "z": 3, "a": 1}
        
        json1 = get_canonical_json(data1)
        json2 = get_canonical_json(data2)
        json3 = get_canonical_json(data3)
        
        passed = (json1 == json2 == json3)
        counter.record("INV-W5-006: Canonical JSON order-independent", passed)
    
    @staticmethod
    def test_canonical_hash(counter: TestCounter):
        """Hash of canonical JSON is deterministic."""
        data = {"members": ["art_001", "art_002", "art_003"]}
        
        hash1 = compute_canonical_hash(data)
        hash2 = compute_canonical_hash(data)
        
        passed = (hash1 == hash2) and (len(hash1) == 64)
        counter.record("Canonical hash deterministic", passed)
    
    @staticmethod
    def test_w5_island_creation(counter: TestCounter):
        """W5Island can be created and serialized."""
        island_id = compute_canonical_hash({"members": ["a", "b", "c"]})
        
        island = W5Island(
            island_id=island_id,
            member_ids=["a", "b", "c"],
            size=3,
            representative_vector={"dim1": 0.5, "dim2": -0.3},
            cohesion_score=0.85,
        )
        
        d = island.to_dict()
        restored = W5Island.from_dict(d)
        
        passed = (
            island.island_id == restored.island_id and
            island.size == restored.size and
            island.member_ids == restored.member_ids
        )
        counter.record("W5Island creation and serialization", passed)
    
    @staticmethod
    def test_w5_structure_canonical_dict(counter: TestCounter):
        """INV-W5-008: created_at excluded from canonical dict."""
        structure = W5Structure(
            structure_id="test_id",
            islands=[],
            noise_ids=[],
            input_count=0,
            island_count=0,
            noise_count=0,
            threshold=0.70,
            min_island_size=3,
            algorithm=W5_ALGORITHM,
            vector_policy=W5_VECTOR_POLICY,
            created_at="2026-01-22T12:00:00Z",
        )
        
        canonical = structure.get_canonical_dict()
        
        passed = "created_at" not in canonical and "structure_id" in canonical
        counter.record("INV-W5-008: created_at excluded from canonical", passed)
    
    @staticmethod
    def test_structure_comparison(counter: TestCounter):
        """compare_w5_structures works correctly."""
        s1 = W5Structure(
            structure_id="id_1",
            islands=[],
            noise_ids=["n1"],
            input_count=1,
            island_count=0,
            noise_count=1,
            threshold=0.70,
            min_island_size=3,
            algorithm=W5_ALGORITHM,
            vector_policy=W5_VECTOR_POLICY,
            created_at="time_1",
        )
        
        s2 = W5Structure(
            structure_id="id_1",
            islands=[],
            noise_ids=["n1"],
            input_count=1,
            island_count=0,
            noise_count=1,
            threshold=0.70,
            min_island_size=3,
            algorithm=W5_ALGORITHM,
            vector_policy=W5_VECTOR_POLICY,
            created_at="time_2",  # Different!
        )
        
        comparison = compare_w5_structures(s1, s2)
        
        passed = comparison["match"]  # Should match despite different created_at
        counter.record("Structure comparison ignores created_at", passed)


# ==========================================
# Condensator Tests
# ==========================================

class TestCondensator:
    """Tests for W5 Condensator (w5_condensator.py)."""
    
    @staticmethod
    def test_empty_input(counter: TestCounter):
        """Empty input produces empty structure."""
        condensator = W5Condensator()
        result = condensator.condense([])
        
        passed = (
            result.input_count == 0 and
            result.island_count == 0 and
            result.noise_count == 0 and
            len(result.structure_id) == 64
        )
        counter.record("Empty input handling", passed)
    
    @staticmethod
    def test_p0a_duplicate_detection(counter: TestCounter):
        """P0-A: Duplicate article_id raises ValueError."""
        condensator = W5Condensator()
        
        r1 = make_w4("dup_001", {"a": 1.0})
        r2 = make_w4("dup_001", {"a": 0.9})  # Same article_id!
        
        try:
            condensator.condense([r1, r2])
            passed = False
            detail = "Should have raised ValueError"
        except ValueError as e:
            passed = "Duplicate" in str(e)
            detail = "" if passed else f"Wrong error: {e}"
        
        counter.record("P0-A: Duplicate article_id detection", passed, detail)
    
    @staticmethod
    def test_p0b_boundary_determinism(counter: TestCounter):
        """P0-B: Boundary similarity is deterministic after rounding."""
        condensator = W5Condensator(threshold=0.70)
        
        # Create vectors with borderline similarity
        r1 = make_w4("border_001", {"x": 1.0, "y": 0.0})
        r2 = make_w4("border_002", {"x": 0.7, "y": 0.7141428})  # ~cos=0.70
        r3 = make_w4("border_003", {"x": 0.71, "y": 0.7041428})
        
        # Multiple runs should produce identical results
        results = [condensator.condense([r1, r2, r3]) for _ in range(5)]
        
        structure_ids = [r.structure_id for r in results]
        island_counts = [r.island_count for r in results]
        
        passed = len(set(structure_ids)) == 1 and len(set(island_counts)) == 1
        counter.record("P0-B: Boundary determinism", passed)
    
    @staticmethod
    def test_p14_batch_size_limit(counter: TestCounter):
        """P1-4: Batch size exceeding limit raises ValueError."""
        condensator = W5Condensator(max_batch_size=5)
        
        records = [make_w4(f"batch_{i:03d}", {"x": float(i)}) for i in range(10)]
        
        try:
            condensator.condense(records)
            passed = False
            detail = "Should have raised ValueError"
        except ValueError as e:
            passed = "exceeds limit" in str(e)
            detail = "" if passed else f"Wrong error: {e}"
        
        counter.record("P1-4: Batch size limit", passed, detail)
    
    @staticmethod
    def test_basic_clustering(counter: TestCounter):
        """Basic clustering produces expected islands."""
        condensator = W5Condensator(threshold=0.70, min_island_size=3)
        
        # Cluster A: similar vectors (should form island)
        a1 = make_w4("a_001", {"d1": 1.0, "d2": 0.1})
        a2 = make_w4("a_002", {"d1": 0.95, "d2": 0.15})
        a3 = make_w4("a_003", {"d1": 0.9, "d2": 0.2})
        
        # Cluster B: different direction (should form separate island)
        b1 = make_w4("b_001", {"d1": -0.8, "d2": 0.9})
        b2 = make_w4("b_002", {"d1": -0.85, "d2": 0.85})
        b3 = make_w4("b_003", {"d1": -0.75, "d2": 0.95})
        
        # Noise: isolated
        noise = make_w4("noise_001", {"d3": 1.0})
        
        result = condensator.condense([a1, a2, a3, b1, b2, b3, noise])
        
        passed = (
            result.input_count == 7 and
            result.island_count == 2 and
            result.noise_count == 1 and
            "noise_001" in result.noise_ids
        )
        counter.record("Basic clustering", passed)
    
    @staticmethod
    def test_island_cohesion(counter: TestCounter):
        """P1-1: Cohesion is average of edge similarities."""
        condensator = W5Condensator(threshold=0.70, min_island_size=2)
        
        # Create tight cluster
        r1 = make_w4("tight_001", {"x": 1.0})
        r2 = make_w4("tight_002", {"x": 1.0})  # Identical = similarity 1.0
        
        result = condensator.condense([r1, r2])
        
        passed = False
        if result.island_count == 1:
            cohesion = result.islands[0].cohesion_score
            passed = cohesion > 0.99  # Should be ~1.0 for identical vectors
        
        counter.record("P1-1: Cohesion calculation", passed)
    
    @staticmethod
    def test_inv_w5_002_island_id_topology(counter: TestCounter):
        """INV-W5-002: island_id depends only on members."""
        # Same members should produce same island_id regardless of order
        members1 = ["art_c", "art_a", "art_b"]
        members2 = ["art_a", "art_b", "art_c"]
        
        hash1 = compute_canonical_hash({"members": sorted(members1)})
        hash2 = compute_canonical_hash({"members": sorted(members2)})
        
        passed = hash1 == hash2
        counter.record("INV-W5-002: island_id from members only", passed)
    
    @staticmethod
    def test_inv_w5_007_structure_uses_w4_analysis_id(counter: TestCounter):
        """INV-W5-007: structure_id uses w4_analysis_id, not article_id."""
        condensator = W5Condensator()
        
        # Same article_id but different w4_analysis_id
        r1 = W4Record(
            article_id="art_001",
            w4_analysis_id="w4_analysis_AAA",
            resonance_vector={"x": 1.0},
            used_w3={},
            token_count=100,
            tokenizer_version="v1",
            normalizer_version="v1",
        )
        
        r2 = W4Record(
            article_id="art_002",
            w4_analysis_id="w4_analysis_BBB",
            resonance_vector={"x": 0.9},
            used_w3={},
            token_count=100,
            tokenizer_version="v1",
            normalizer_version="v1",
        )
        
        result1 = condensator.condense([r1, r2])
        
        # Change w4_analysis_id
        r1_alt = W4Record(
            article_id="art_001",
            w4_analysis_id="w4_analysis_CCC",  # Different!
            resonance_vector={"x": 1.0},
            used_w3={},
            token_count=100,
            tokenizer_version="v1",
            normalizer_version="v1",
        )
        
        result2 = condensator.condense([r1_alt, r2])
        
        # structure_id should differ because w4_analysis_id differs
        passed = result1.structure_id != result2.structure_id
        counter.record("INV-W5-007: structure_id uses w4_analysis_id", passed)
    
    @staticmethod
    def test_representative_vector_rounding(counter: TestCounter):
        """INV-W5-005: Representative vectors are rounded."""
        condensator = W5Condensator(threshold=0.70, min_island_size=2)
        
        r1 = make_w4("round_001", {"x": 0.123456789012345})
        r2 = make_w4("round_002", {"x": 0.123456789012345})
        
        result = condensator.condense([r1, r2])
        
        passed = False
        if result.island_count == 1:
            vec = result.islands[0].representative_vector
            if "x" in vec:
                # Should be rounded to W5_VECTOR_ROUNDING decimals
                value = vec["x"]
                # Check it's not the full precision
                passed = value == round(0.123456789012345, W5_VECTOR_ROUNDING)
        
        counter.record("INV-W5-005: Vector rounding", passed)


# ==========================================
# Main
# ==========================================

def main():
    print("=" * 60)
    print("ESDE Phase 9-5: W5 Test Suite")
    print("=" * 60)
    
    counter = TestCounter()
    
    # Schema Tests
    print("\n[Schema Tests]")
    TestSchema.test_canonical_json(counter)
    TestSchema.test_canonical_hash(counter)
    TestSchema.test_w5_island_creation(counter)
    TestSchema.test_w5_structure_canonical_dict(counter)
    TestSchema.test_structure_comparison(counter)
    
    # Condensator Tests
    print("\n[Condensator Tests]")
    TestCondensator.test_empty_input(counter)
    TestCondensator.test_p0a_duplicate_detection(counter)
    TestCondensator.test_p0b_boundary_determinism(counter)
    TestCondensator.test_p14_batch_size_limit(counter)
    TestCondensator.test_basic_clustering(counter)
    TestCondensator.test_island_cohesion(counter)
    TestCondensator.test_inv_w5_002_island_id_topology(counter)
    TestCondensator.test_inv_w5_007_structure_uses_w4_analysis_id(counter)
    TestCondensator.test_representative_vector_rounding(counter)
    
    # Summary
    success = counter.summary()
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
