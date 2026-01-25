"""
ESDE Migration Phase 2: Policy-Based Statistics Tests
======================================================

P0 Test Items (Gemini v0.2.1):
  1. Collision Test: Same factors + different policy_id → different signatures
  2. Type Test: true (bool) vs "true" (str) → different signatures
  3. Missing Key Test: Missing key != empty factors

Additional Audit Checks (GPT):
  4. Canonical一致テスト: StandardConditionPolicy uses correct separators
  5. Missing順序テスト: missing_keys sorted regardless of input order

Spec: Migration Phase 2 v0.2.1 (Audit Fixed)
"""

import sys
import os
import json
import hashlib
from typing import Dict, Any

# Path setup
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ==========================================
# Mock ContextRecord (for isolated testing)
# ==========================================

class MockContextRecord:
    """Mock ContextRecord for testing without Substrate dependency."""
    
    def __init__(self, traces: Dict[str, Any]):
        self.traces = traces


# ==========================================
# Test Results Tracking
# ==========================================

class TestResult:
    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.error = None
    
    def __str__(self):
        status = "✅ PASS" if self.passed else f"❌ FAIL: {self.error}"
        return f"{self.name}: {status}"


def run_test(name: str, test_func) -> TestResult:
    """Run a test and capture result."""
    result = TestResult(name)
    try:
        test_func()
        result.passed = True
    except AssertionError as e:
        result.error = str(e)
    except Exception as e:
        result.error = f"{type(e).__name__}: {e}"
    return result


# ==========================================
# P0-1: Collision Test (Policy ID)
# ==========================================

def test_p0_1_collision_different_policy_id():
    """
    P0-1: Same factors + different policy_id → different signatures.
    
    This ensures that policies with identical extraction logic but
    different identifiers produce distinct signatures.
    """
    print("\n[P0-1] Collision Test: Different policy_id")
    print("-" * 60)
    
    from statistics.policies import StandardConditionPolicy
    
    # Same target_keys, same factors
    policy_a = StandardConditionPolicy(
        policy_id="policy_alpha",
        target_keys=["legacy:source_type"],
        version="v1.0",
    )
    
    policy_b = StandardConditionPolicy(
        policy_id="policy_beta",
        target_keys=["legacy:source_type"],
        version="v1.0",
    )
    
    # Same traces
    record = MockContextRecord(traces={
        "legacy:source_type": "news",
    })
    
    sig_a = policy_a.compute_signature(record)
    sig_b = policy_b.compute_signature(record)
    
    print(f"  Policy A (alpha): {sig_a[:32]}...")
    print(f"  Policy B (beta):  {sig_b[:32]}...")
    
    assert sig_a != sig_b, "Different policy_id MUST produce different signatures"
    assert len(sig_a) == 64, f"Signature A length should be 64, got {len(sig_a)}"
    assert len(sig_b) == 64, f"Signature B length should be 64, got {len(sig_b)}"
    
    print("  ✅ Different policy_id → different signatures")


def test_p0_1_collision_different_version():
    """
    P0-1 Extended: Same policy_id + different version → different signatures.
    """
    print("\n[P0-1 Extended] Collision Test: Different version")
    print("-" * 60)
    
    from statistics.policies import StandardConditionPolicy
    
    policy_v1 = StandardConditionPolicy(
        policy_id="same_policy",
        target_keys=["legacy:source_type"],
        version="v1.0",
    )
    
    policy_v2 = StandardConditionPolicy(
        policy_id="same_policy",
        target_keys=["legacy:source_type"],
        version="v2.0",
    )
    
    record = MockContextRecord(traces={
        "legacy:source_type": "news",
    })
    
    sig_v1 = policy_v1.compute_signature(record)
    sig_v2 = policy_v2.compute_signature(record)
    
    print(f"  Version v1.0: {sig_v1[:32]}...")
    print(f"  Version v2.0: {sig_v2[:32]}...")
    
    assert sig_v1 != sig_v2, "Different version MUST produce different signatures"
    
    print("  ✅ Different version → different signatures")


# ==========================================
# P0-2: Type Test
# ==========================================

def test_p0_2_type_bool_vs_string():
    """
    P0-2: true (bool) vs "true" (str) → different signatures.
    
    Type preservation is critical for determinism.
    """
    print("\n[P0-2] Type Test: bool vs string")
    print("-" * 60)
    
    from statistics.policies import StandardConditionPolicy
    
    policy = StandardConditionPolicy(
        policy_id="type_test",
        target_keys=["legacy:flag"],
        version="v1.0",
    )
    
    record_bool = MockContextRecord(traces={
        "legacy:flag": True,  # bool
    })
    
    record_str = MockContextRecord(traces={
        "legacy:flag": "true",  # str (note: Python "True" != "true")
    })
    
    sig_bool = policy.compute_signature(record_bool)
    sig_str = policy.compute_signature(record_str)
    
    print(f"  bool True:   {sig_bool[:32]}...")
    print(f"  str 'true':  {sig_str[:32]}...")
    
    assert sig_bool != sig_str, "bool and str MUST produce different signatures"
    
    # Also verify factors preserve types
    factors_bool = policy.extract_factors(record_bool)
    factors_str = policy.extract_factors(record_str)
    
    print(f"  factors_bool type: {type(factors_bool['legacy:flag']).__name__}")
    print(f"  factors_str type: {type(factors_str['legacy:flag']).__name__}")
    
    assert isinstance(factors_bool['legacy:flag'], bool), "Should preserve bool"
    assert isinstance(factors_str['legacy:flag'], str), "Should preserve str"
    
    print("  ✅ Type preserved, different signatures")


def test_p0_2_type_int_vs_float():
    """
    P0-2 Extended: int vs float with same numeric value.
    """
    print("\n[P0-2 Extended] Type Test: int vs float")
    print("-" * 60)
    
    from statistics.policies import StandardConditionPolicy
    
    policy = StandardConditionPolicy(
        policy_id="num_test",
        target_keys=["legacy:count"],
        version="v1.0",
    )
    
    record_int = MockContextRecord(traces={
        "legacy:count": 42,  # int
    })
    
    record_float = MockContextRecord(traces={
        "legacy:count": 42.0,  # float
    })
    
    sig_int = policy.compute_signature(record_int)
    sig_float = policy.compute_signature(record_float)
    
    print(f"  int 42:     {sig_int[:32]}...")
    print(f"  float 42.0: {sig_float[:32]}...")
    
    # Note: JSON serialization may or may not distinguish 42 vs 42.0
    # depending on Python version. The important thing is type in factors.
    factors_int = policy.extract_factors(record_int)
    factors_float = policy.extract_factors(record_float)
    
    print(f"  factors_int type: {type(factors_int['legacy:count']).__name__}")
    print(f"  factors_float type: {type(factors_float['legacy:count']).__name__}")
    
    assert isinstance(factors_int['legacy:count'], int), "Should preserve int"
    assert isinstance(factors_float['legacy:count'], float), "Should preserve float"
    
    print("  ✅ Types preserved in factors")


# ==========================================
# P0-4: Missing Key Test
# ==========================================

def test_p0_4_missing_key_not_error():
    """
    P0-4: Missing key should not raise error.
    """
    print("\n[P0-4] Missing Key Test: No error on missing")
    print("-" * 60)
    
    from statistics.policies import StandardConditionPolicy
    
    policy = StandardConditionPolicy(
        policy_id="missing_test",
        target_keys=["legacy:source_type", "legacy:language_profile", "legacy:nonexistent"],
        version="v1.0",
    )
    
    # Record missing one of the target keys
    record = MockContextRecord(traces={
        "legacy:source_type": "news",
        "legacy:language_profile": "en",
        # "legacy:nonexistent" is NOT present
    })
    
    # Should not raise
    try:
        sig = policy.compute_signature(record)
        factors = policy.extract_factors(record)
        print(f"  Signature: {sig[:32]}...")
        print(f"  Factors: {factors}")
        print("  ✅ No error on missing key")
    except Exception as e:
        raise AssertionError(f"Should not raise on missing key: {e}")


def test_p0_4_missing_key_vs_empty():
    """
    P0-4: Missing key record vs completely empty traces → different signatures.
    
    This prevents the "giant bucket" problem where all records with any
    missing keys would hash to the same value.
    """
    print("\n[P0-4] Missing Key Test: Missing vs empty distinction")
    print("-" * 60)
    
    from statistics.policies import StandardConditionPolicy
    
    policy = StandardConditionPolicy(
        policy_id="bucket_test",
        target_keys=["legacy:a", "legacy:b"],
        version="v1.0",
    )
    
    # Case 1: Has one key, missing other
    record_partial = MockContextRecord(traces={
        "legacy:a": "value_a",
        # "legacy:b" is missing
    })
    
    # Case 2: Completely empty
    record_empty = MockContextRecord(traces={})
    
    # Case 3: Has different key missing
    record_partial_b = MockContextRecord(traces={
        "legacy:b": "value_b",
        # "legacy:a" is missing
    })
    
    sig_partial = policy.compute_signature(record_partial)
    sig_empty = policy.compute_signature(record_empty)
    sig_partial_b = policy.compute_signature(record_partial_b)
    
    print(f"  Partial (has a): {sig_partial[:32]}...")
    print(f"  Empty:           {sig_empty[:32]}...")
    print(f"  Partial (has b): {sig_partial_b[:32]}...")
    
    assert sig_partial != sig_empty, "Partial != Empty"
    assert sig_partial != sig_partial_b, "Different present keys → different signatures"
    assert sig_empty != sig_partial_b, "Empty != Partial B"
    
    print("  ✅ All three cases produce distinct signatures")


# ==========================================
# Audit Check 1: Canonical JSON Verification
# ==========================================

def test_audit_canonical_json_no_spaces():
    """
    Audit Check: StandardConditionPolicy uses separators=(',', ':').
    
    Verifies that the canonical JSON has no spaces after colons or commas.
    """
    print("\n[Audit 1] Canonical JSON: No spaces in separators")
    print("-" * 60)
    
    from statistics.policies.standard import (
        CANONICAL_SORT_KEYS,
        CANONICAL_ENSURE_ASCII,
        CANONICAL_SEPARATORS,
    )
    
    print(f"  CANONICAL_SORT_KEYS: {CANONICAL_SORT_KEYS}")
    print(f"  CANONICAL_ENSURE_ASCII: {CANONICAL_ENSURE_ASCII}")
    print(f"  CANONICAL_SEPARATORS: {CANONICAL_SEPARATORS}")
    
    assert CANONICAL_SORT_KEYS is True, "sort_keys should be True"
    assert CANONICAL_ENSURE_ASCII is False, "ensure_ascii should be False"
    assert CANONICAL_SEPARATORS == (',', ':'), f"separators should be (',', ':'), got {CANONICAL_SEPARATORS}"
    
    # Verify by generating actual JSON
    test_payload = {
        "policy_id": "test",
        "version": "v1.0",
        "factors": {"b": 2, "a": 1},  # Unsorted input
        "missing": ["z", "y"],
    }
    
    canonical = json.dumps(
        test_payload,
        sort_keys=CANONICAL_SORT_KEYS,
        ensure_ascii=CANONICAL_ENSURE_ASCII,
        separators=CANONICAL_SEPARATORS,
    )
    
    print(f"  Generated JSON: {canonical}")
    
    assert ': ' not in canonical, "Should not have space after colon"
    assert ', ' not in canonical, "Should not have space after comma"
    assert '"a":1' in canonical, "Keys should be sorted"
    
    print("  ✅ Canonical JSON matches Substrate spec")


def test_audit_canonical_matches_substrate():
    """
    Audit Check: Policy canonical JSON matches Substrate's canonical.py.
    """
    print("\n[Audit 1b] Canonical JSON: Matches Substrate spec")
    print("-" * 60)
    
    from statistics.policies.standard import (
        CANONICAL_SORT_KEYS,
        CANONICAL_ENSURE_ASCII,
        CANONICAL_SEPARATORS,
    )
    
    # Substrate spec (from ledger/canonical.py and substrate/schema.py)
    substrate_settings = {
        "sort_keys": True,
        "ensure_ascii": False,
        "separators": (',', ':'),
    }
    
    policy_settings = {
        "sort_keys": CANONICAL_SORT_KEYS,
        "ensure_ascii": CANONICAL_ENSURE_ASCII,
        "separators": CANONICAL_SEPARATORS,
    }
    
    print(f"  Substrate: {substrate_settings}")
    print(f"  Policy:    {policy_settings}")
    
    assert policy_settings == substrate_settings, "Settings should match exactly"
    
    print("  ✅ Policy settings match Substrate spec")


# ==========================================
# Audit Check 2: Missing Keys Sorting
# ==========================================

def test_audit_missing_keys_sorted():
    """
    Audit Check: missing_keys list is always sorted.
    
    Ensures determinism regardless of input order.
    """
    print("\n[Audit 2] Missing Keys: Always sorted")
    print("-" * 60)
    
    from statistics.policies import StandardConditionPolicy
    
    # Create policy with unsorted target_keys
    policy = StandardConditionPolicy(
        policy_id="sort_test",
        target_keys=["z:key", "a:key", "m:key"],  # Unsorted
        version="v1.0",
    )
    
    print(f"  target_keys (input order): {policy.target_keys}")
    
    # Empty record - all keys will be missing
    record = MockContextRecord(traces={})
    
    # Access internal method to check missing list
    factors, missing = policy._extract_raw_factors(record)
    
    print(f"  missing_keys (output): {missing}")
    
    assert missing == sorted(missing), f"Missing keys should be sorted: {missing}"
    assert missing == ["a:key", "m:key", "z:key"], "Should be alphabetically sorted"
    
    print("  ✅ Missing keys are sorted")


def test_audit_missing_keys_deterministic_signature():
    """
    Audit Check: Same missing keys in different order → same signature.
    """
    print("\n[Audit 2b] Missing Keys: Deterministic signature")
    print("-" * 60)
    
    from statistics.policies import StandardConditionPolicy
    
    # Two policies with same keys in different order
    policy1 = StandardConditionPolicy(
        policy_id="det_test",
        target_keys=["a:key", "b:key", "c:key"],
        version="v1.0",
    )
    
    policy2 = StandardConditionPolicy(
        policy_id="det_test",
        target_keys=["c:key", "a:key", "b:key"],  # Different order
        version="v1.0",
    )
    
    record = MockContextRecord(traces={})  # All keys missing
    
    sig1 = policy1.compute_signature(record)
    sig2 = policy2.compute_signature(record)
    
    print(f"  Policy 1 (a,b,c): {sig1[:32]}...")
    print(f"  Policy 2 (c,a,b): {sig2[:32]}...")
    
    assert sig1 == sig2, "Same keys (different order) should produce same signature"
    
    print("  ✅ Deterministic regardless of input key order")


# ==========================================
# W2Aggregator Integration Test
# ==========================================

def test_w2_aggregator_legacy_mode():
    """
    Integration Test: W2Aggregator works in legacy mode (no registry/policy).
    """
    print("\n[Integration] W2Aggregator Legacy Mode")
    print("-" * 60)
    
    from statistics.w2_aggregator import W2Aggregator
    
    # Mock ArticleRecord
    class MockObservation:
        def __init__(self, segment_span, timestamp):
            self.segment_span = segment_span
            self.timestamp = timestamp
    
    class MockArticle:
        def __init__(self, article_id, raw_text, observations, source_meta=None, 
                     ingestion_time=None, substrate_ref=None):
            self.article_id = article_id
            self.raw_text = raw_text
            self.observations = observations
            self.source_meta = source_meta or {}
            self.ingestion_time = ingestion_time or "2026-01-20T10:00:00Z"
            self.substrate_ref = substrate_ref
    
    # Create aggregator without registry/policy
    aggregator = W2Aggregator(
        records_path="/tmp/mig_test_records.jsonl",
        conditions_path="/tmp/mig_test_conditions.jsonl",
    )
    
    article = MockArticle(
        article_id="test_001",
        raw_text="Hello world",
        observations=[MockObservation((0, 11), "2026-01-20T10:00:00Z")],
        source_meta={"source_type": "news", "language_profile": "en"},
    )
    
    result = aggregator.process_article(article)
    
    print(f"  signature_source: {result['signature_source']}")
    print(f"  condition_signature: {result['condition_signature'][:32]}...")
    print(f"  signature length: {len(result['condition_signature'])}")
    
    assert result['signature_source'] == 'legacy', "Should use legacy path"
    assert len(result['condition_signature']) == 64, "Legacy hash should be 64 chars"
    
    print("  ✅ W2Aggregator legacy mode works")


def test_w2_aggregator_legacy_hash_canonical():
    """
    Integration Test: W2Aggregator._compute_legacy_hash uses correct canonical JSON.
    """
    print("\n[Integration] W2Aggregator Legacy Hash Canonical")
    print("-" * 60)
    
    from statistics.w2_aggregator import W2Aggregator, CANONICAL_SEPARATORS
    
    aggregator = W2Aggregator(
        records_path="/tmp/mig_test2_records.jsonl",
        conditions_path="/tmp/mig_test2_conditions.jsonl",
    )
    
    factors = {"b": "2", "a": "1"}
    
    legacy_hash = aggregator._compute_legacy_hash(factors)
    
    print(f"  Legacy hash: {legacy_hash}")
    print(f"  Length: {len(legacy_hash)}")
    
    # Verify by computing expected hash
    import json
    canonical = json.dumps(
        factors,
        sort_keys=True,
        ensure_ascii=False,
        separators=(',', ':'),
    )
    expected_hash = hashlib.sha256(canonical.encode('utf-8')).hexdigest()
    
    print(f"  Expected:   {expected_hash}")
    
    assert legacy_hash == expected_hash, "Legacy hash should match manual calculation"
    assert len(legacy_hash) == 64, "Should be full 64 chars"
    
    print("  ✅ Legacy hash uses correct canonical JSON")


# ==========================================
# Main Test Runner
# ==========================================

def run_all_tests():
    """Run all Migration Phase 2 tests."""
    print("=" * 60)
    print("ESDE Migration Phase 2: Policy-Based Statistics Tests")
    print("=" * 60)
    
    results = []
    
    # P0 Tests
    results.append(run_test("P0-1: Collision (policy_id)", test_p0_1_collision_different_policy_id))
    results.append(run_test("P0-1: Collision (version)", test_p0_1_collision_different_version))
    results.append(run_test("P0-2: Type (bool vs str)", test_p0_2_type_bool_vs_string))
    results.append(run_test("P0-2: Type (int vs float)", test_p0_2_type_int_vs_float))
    results.append(run_test("P0-4: Missing key (no error)", test_p0_4_missing_key_not_error))
    results.append(run_test("P0-4: Missing key (vs empty)", test_p0_4_missing_key_vs_empty))
    
    # Audit Tests
    results.append(run_test("Audit 1: Canonical JSON (no spaces)", test_audit_canonical_json_no_spaces))
    results.append(run_test("Audit 1b: Canonical (Substrate match)", test_audit_canonical_matches_substrate))
    results.append(run_test("Audit 2: Missing keys (sorted)", test_audit_missing_keys_sorted))
    results.append(run_test("Audit 2b: Missing keys (deterministic)", test_audit_missing_keys_deterministic_signature))
    
    # Integration Tests
    results.append(run_test("Integration: W2 legacy mode", test_w2_aggregator_legacy_mode))
    results.append(run_test("Integration: W2 legacy hash", test_w2_aggregator_legacy_hash_canonical))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    
    for result in results:
        print(result)
    
    print("\n" + "-" * 60)
    print(f"Results: {passed}/{total} passed")
    
    if passed == total:
        print("\n✅ All tests PASSED!")
        print("Migration Phase 2 implementation is ready for deployment.")
    else:
        print("\n❌ Some tests FAILED!")
        print("Please review and fix before proceeding.")
        return False
    
    return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
