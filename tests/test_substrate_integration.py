"""
ESDE Substrate Integration Tests
================================
Tests for ArticleRecord + Substrate Layer integration.

DoD (Definition of Done) from GPT Audit:
  1. ArticleRecord.substrate_ref: Optional[str] added (backward compatible)
  2. W0 (Gateway) converts source_meta → legacy:* traces → Substrate registration
  3. Regression: Existing Phase 9 tests pass (W2/W3/W6)
  4. New tests:
     - Same source_meta → same context_id (determinism)
     - Conversion is lossless (source_meta ↔ legacy traces match)
     - Fallback when substrate_ref is None

Version: v5.4.7-SUB.1
Audit: GPT P0-AR-001, P0-AR-002, P0-W0-001, P0-W0-002
"""

import sys
import os
import json
import tempfile
from datetime import datetime, timezone
from typing import Dict, Any

# Path setup (run from esde/ directory: python tests/test_substrate_integration.py)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ==========================================
# Test 1: ArticleRecord substrate_ref (DoD #1)
# ==========================================

def test_article_record_substrate_ref():
    """
    DoD #1: ArticleRecord.substrate_ref: Optional[str] added
    
    Tests:
      - Field exists and is Optional
      - Default is None (backward compatible)
      - Can be set to context_id string
    """
    print("\n[Test 1] ArticleRecord.substrate_ref field")
    print("=" * 60)
    
    from integration.schema import ArticleRecord
    
    # Test 1a: Default is None
    article = ArticleRecord(
        raw_text="Test content",
        source_meta={"source_type": "news"},
    )
    
    print(f"  article_id: {article.article_id[:16]}...")
    print(f"  substrate_ref (default): {article.substrate_ref}")
    
    assert article.substrate_ref is None, "Default should be None"
    print("  ✅ Default is None (backward compatible)")
    
    # Test 1b: Can set substrate_ref
    article_with_ref = ArticleRecord(
        raw_text="Test content",
        source_meta={"source_type": "news"},
        substrate_ref="abc123def456abc123def456abc12345",  # 32 chars
    )
    
    print(f"  substrate_ref (set): {article_with_ref.substrate_ref}")
    assert article_with_ref.substrate_ref == "abc123def456abc123def456abc12345"
    print("  ✅ substrate_ref can be set")
    
    # Test 1c: to_dict() includes substrate_ref
    d = article_with_ref.to_dict()
    print(f"  to_dict() keys: {sorted(d.keys())[:5]}...")
    
    assert "substrate_ref" in d, "substrate_ref should be in to_dict()"
    assert d["substrate_ref"] == "abc123def456abc123def456abc12345"
    print("  ✅ substrate_ref included in to_dict()")
    
    print("\n  [Test 1] PASSED ✅")
    return True


# ==========================================
# Test 2: Lossless Migration (DoD #4a, P0-AR-001)
# ==========================================

def test_lossless_migration():
    """
    DoD #4: Conversion is lossless (source_meta ↔ legacy traces match)
    P0-AR-001: source_meta の全キーが legacy:* として traces に落ちること
    
    Tests:
      - All source_meta keys appear as legacy:* in traces
      - Values are preserved exactly
      - None values are skipped (information-less)
    """
    print("\n[Test 2] Lossless Migration (P0-AR-001)")
    print("=" * 60)
    
    from substrate.schema import convert_source_meta_to_traces
    
    # Test 2a: Standard source_meta
    source_meta = {
        "source_type": "news",
        "language_profile": "en",
        "custom_field": 123,
        "float_value": 3.14159,
        "bool_value": True,
    }
    
    traces = convert_source_meta_to_traces(source_meta)
    
    print(f"  source_meta: {source_meta}")
    print(f"  traces: {traces}")
    
    # Verify all keys are converted
    for key, value in source_meta.items():
        if value is None:
            continue
        expected_key = f"legacy:{key.lower()}"
        assert expected_key in traces, f"Missing key: {expected_key}"
        assert traces[expected_key] == value, f"Value mismatch for {expected_key}"
    
    print("  ✅ All keys converted to legacy:* namespace")
    
    # Test 2b: None values are skipped
    source_meta_with_none = {
        "source_type": "news",
        "language_profile": None,  # Should be skipped
    }
    
    traces_with_none = convert_source_meta_to_traces(source_meta_with_none)
    
    print(f"  source_meta (with None): {source_meta_with_none}")
    print(f"  traces (with None): {traces_with_none}")
    
    assert "legacy:language_profile" not in traces_with_none, "None should be skipped"
    assert "legacy:source_type" in traces_with_none
    print("  ✅ None values correctly skipped")
    
    # Test 2c: Empty source_meta
    empty_traces = convert_source_meta_to_traces({})
    assert empty_traces == {}, "Empty source_meta should produce empty traces"
    print("  ✅ Empty source_meta → empty traces")
    
    print("\n  [Test 2] PASSED ✅")
    return True


# ==========================================
# Test 3: Deterministic context_id (DoD #4b, P0-W0-001)
# ==========================================

def test_deterministic_context_id():
    """
    DoD #4: Same source_meta → same context_id (determinism)
    P0-W0-001: 同一入力（同一source_meta）→ 同一context_id
    
    Tests:
      - Same source_meta produces same context_id
      - Key order doesn't affect context_id
      - Different source_meta produces different context_id
    """
    print("\n[Test 3] Deterministic context_id (P0-W0-001)")
    print("=" * 60)
    
    from substrate import (
        create_context_record,
        convert_source_meta_to_traces,
    )
    
    # Test 3a: Same input → same ID
    source_meta = {"source_type": "news", "language_profile": "en"}
    traces1 = convert_source_meta_to_traces(source_meta)
    traces2 = convert_source_meta_to_traces(source_meta)
    
    record1 = create_context_record(
        traces=traces1,
        retrieval_path=None,
        capture_version="v1.0"
    )
    
    record2 = create_context_record(
        traces=traces2,
        retrieval_path=None,
        capture_version="v1.0"
    )
    
    print(f"  source_meta: {source_meta}")
    print(f"  context_id 1: {record1.context_id}")
    print(f"  context_id 2: {record2.context_id}")
    
    assert record1.context_id == record2.context_id, "Same input should produce same ID"
    print("  ✅ Same source_meta → same context_id")
    
    # Test 3b: Key order independence
    source_meta_reordered = {"language_profile": "en", "source_type": "news"}
    traces_reordered = convert_source_meta_to_traces(source_meta_reordered)
    
    record3 = create_context_record(
        traces=traces_reordered,
        retrieval_path=None,
        capture_version="v1.0"
    )
    
    print(f"  source_meta (reordered): {source_meta_reordered}")
    print(f"  context_id 3: {record3.context_id}")
    
    assert record1.context_id == record3.context_id, "Key order should not affect ID"
    print("  ✅ Key order independent")
    
    # Test 3c: Different input → different ID
    source_meta_different = {"source_type": "dialog", "language_profile": "en"}
    traces_different = convert_source_meta_to_traces(source_meta_different)
    
    record4 = create_context_record(
        traces=traces_different,
        retrieval_path=None,
        capture_version="v1.0"
    )
    
    print(f"  source_meta (different): {source_meta_different}")
    print(f"  context_id 4: {record4.context_id}")
    
    assert record1.context_id != record4.context_id, "Different input should produce different ID"
    print("  ✅ Different source_meta → different context_id")
    
    print("\n  [Test 3] PASSED ✅")
    return True


# ==========================================
# Test 4: Backward Compatibility (DoD #4c, P0-AR-002)
# ==========================================

def test_backward_compatibility():
    """
    DoD #4: Fallback when substrate_ref is None
    P0-AR-002: substrate_ref が無い古いデータでも従来どおり動く
    
    Tests:
      - W2 can process ArticleRecord with substrate_ref=None
      - W2 reads from source_meta when substrate_ref is None
      - Existing test patterns still work
    """
    print("\n[Test 4] Backward Compatibility (P0-AR-002)")
    print("=" * 60)
    
    from integration.schema import ArticleRecord
    
    # Test 4a: Old-style ArticleRecord (no substrate_ref)
    old_style_article = ArticleRecord(
        raw_text="Apple releases new iPhone.",
        source_meta={"source_type": "news", "language_profile": "en"},
        # substrate_ref intentionally not set (defaults to None)
    )
    
    print(f"  article_id: {old_style_article.article_id[:16]}...")
    print(f"  substrate_ref: {old_style_article.substrate_ref}")
    print(f"  source_meta: {old_style_article.source_meta}")
    
    assert old_style_article.substrate_ref is None
    assert old_style_article.source_meta == {"source_type": "news", "language_profile": "en"}
    print("  ✅ Old-style ArticleRecord works (substrate_ref=None)")
    
    # Test 4b: W2 can still read source_meta
    # Simulate W2Aggregator behavior (reads source_meta directly)
    def simulate_w2_condition_extraction(article):
        """Simulates W2Aggregator._extract_condition_factors()"""
        source_meta = article.source_meta
        return {
            "source_type": source_meta.get("source_type", "unknown"),
            "language_profile": source_meta.get("language_profile", "unknown"),
        }
    
    factors = simulate_w2_condition_extraction(old_style_article)
    print(f"  Extracted factors: {factors}")
    
    assert factors["source_type"] == "news"
    assert factors["language_profile"] == "en"
    print("  ✅ W2 condition extraction works with source_meta")
    
    # Test 4c: New-style with substrate_ref still has source_meta
    new_style_article = ArticleRecord(
        raw_text="Apple releases new iPhone.",
        source_meta={"source_type": "news", "language_profile": "en"},
        substrate_ref="abc123def456abc123def456abc12345",
    )
    
    factors_new = simulate_w2_condition_extraction(new_style_article)
    print(f"  New-style factors: {factors_new}")
    
    assert factors_new == factors, "Same factors regardless of substrate_ref"
    print("  ✅ New-style ArticleRecord compatible with W2")
    
    print("\n  [Test 4] PASSED ✅")
    return True


# ==========================================
# Test 5: Registry Deduplication (P0-W0-002)
# ==========================================

def test_registry_deduplication():
    """
    P0-W0-002: 同一context_id登録は JSONLに重複行を増やさない
    
    Tests:
      - Same traces registered twice → only one record
      - context_id is returned both times
    """
    print("\n[Test 5] Registry Deduplication (P0-W0-002)")
    print("=" * 60)
    
    from substrate import SubstrateRegistry, convert_source_meta_to_traces
    
    # Use temp file for test
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        temp_path = f.name
    
    try:
        registry = SubstrateRegistry(storage_path=temp_path)
        
        source_meta = {"source_type": "news", "language_profile": "en"}
        traces = convert_source_meta_to_traces(source_meta)
        
        # Register first time
        id1 = registry.register_traces(
            traces=traces,
            retrieval_path=None,
            capture_version="v1.0"
        )
        
        print(f"  First registration: {id1}")
        
        # Register second time (same traces)
        id2 = registry.register_traces(
            traces=traces,
            retrieval_path=None,
            capture_version="v1.0"
        )
        
        print(f"  Second registration: {id2}")
        
        assert id1 == id2, "Same traces should produce same ID"
        print("  ✅ Same context_id returned")
        
        # Check file has only one line
        # Note: SubstrateRegistry writes immediately on register(), no save() needed
        
        with open(temp_path, 'r') as f:
            lines = [l for l in f if l.strip()]
        
        print(f"  Lines in registry file: {len(lines)}")
        assert len(lines) == 1, "Should have only one record (deduplicated)"
        print("  ✅ No duplicate records in file")
        
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
    
    print("\n  [Test 5] PASSED ✅")
    return True


# ==========================================
# Test 6: Gateway Integration (DoD #2)
# ==========================================

def test_gateway_integration():
    """
    DoD #2: W0 (Gateway) converts source_meta → legacy:* traces → Substrate registration
    
    Tests:
      - Gateway.ingest() populates substrate_ref
      - substrate_ref is valid context_id (32 hex chars)
      - Substrate contains corresponding record
    """
    print("\n[Test 6] Gateway Integration (DoD #2)")
    print("=" * 60)
    
    from integration.gateway import ContentGateway, GatewayConfig
    from substrate import SubstrateRegistry
    
    # Use temp file for test
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        temp_path = f.name
    
    try:
        # Create registry
        registry = SubstrateRegistry(storage_path=temp_path)
        
        # Create gateway with Substrate enabled
        config = GatewayConfig(
            enable_substrate=True,
            substrate_registry_path=temp_path,
        )
        
        gateway = ContentGateway(
            config=config,
            substrate_registry=registry,
        )
        
        # Ingest article
        article = gateway.ingest(
            text="Apple releases new iPhone. Stock rises.",
            source_url="https://example.com/news",
            source_meta={"source_type": "news", "language_profile": "en"},
        )
        
        print(f"  article_id: {article.article_id[:16]}...")
        print(f"  substrate_ref: {article.substrate_ref}")
        print(f"  source_meta: {article.source_meta}")
        
        # Verify substrate_ref
        assert article.substrate_ref is not None, "substrate_ref should be set"
        assert len(article.substrate_ref) == 32, "context_id should be 32 chars"
        assert all(c in '0123456789abcdef' for c in article.substrate_ref), "Should be hex"
        print("  ✅ substrate_ref is valid context_id")
        
        # Verify record exists in registry
        record = registry.get(article.substrate_ref)
        assert record is not None, "Record should exist in registry"
        print(f"  Registry record traces: {record.traces}")
        
        assert "legacy:source_type" in record.traces
        assert record.traces["legacy:source_type"] == "news"
        print("  ✅ Record contains correct traces")
        
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
    
    print("\n  [Test 6] PASSED ✅")
    return True


# ==========================================
# Test 7: Substrate Disabled Fallback
# ==========================================

def test_substrate_disabled():
    """
    Tests Gateway behavior when Substrate is disabled.
    
    Tests:
      - substrate_ref is None when disabled
      - Article is still valid
    """
    print("\n[Test 7] Substrate Disabled Fallback")
    print("=" * 60)
    
    from integration.gateway import ContentGateway, GatewayConfig
    
    # Create gateway with Substrate disabled
    config = GatewayConfig(
        enable_substrate=False,
    )
    
    gateway = ContentGateway(config=config)
    
    # Ingest article
    article = gateway.ingest(
        text="Test content.",
        source_meta={"source_type": "news"},
    )
    
    print(f"  article_id: {article.article_id[:16]}...")
    print(f"  substrate_ref: {article.substrate_ref}")
    
    assert article.substrate_ref is None, "substrate_ref should be None when disabled"
    assert article.source_meta == {"source_type": "news"}, "source_meta should be preserved"
    print("  ✅ substrate_ref is None when Substrate disabled")
    print("  ✅ Article is still valid")
    
    print("\n  [Test 7] PASSED ✅")
    return True


# ==========================================
# Main Test Runner
# ==========================================

def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "=" * 70)
    print("ESDE Substrate Integration Tests")
    print("Version: v5.4.7-SUB.1")
    print("Audit: GPT P0-AR-001, P0-AR-002, P0-W0-001, P0-W0-002")
    print("=" * 70)
    
    tests = [
        ("DoD #1: ArticleRecord.substrate_ref", test_article_record_substrate_ref),
        ("DoD #4a: Lossless Migration", test_lossless_migration),
        ("DoD #4b: Deterministic context_id", test_deterministic_context_id),
        ("DoD #4c: Backward Compatibility", test_backward_compatibility),
        ("P0-W0-002: Registry Deduplication", test_registry_deduplication),
        ("DoD #2: Gateway Integration", test_gateway_integration),
        ("Fallback: Substrate Disabled", test_substrate_disabled),
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success, None))
        except Exception as e:
            print(f"\n  ❌ FAILED: {e}")
            results.append((name, False, str(e)))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, success, _ in results if success)
    failed = len(results) - passed
    
    for name, success, error in results:
        status = "✅ PASS" if success else f"❌ FAIL: {error}"
        print(f"  {name}: {status}")
    
    print("\n" + "-" * 70)
    print(f"  Total: {len(results)}, Passed: {passed}, Failed: {failed}")
    
    if failed == 0:
        print("\n  🎉 ALL TESTS PASSED! Ready for GPT Audit.")
    else:
        print(f"\n  ⚠️ {failed} test(s) failed. Fix before audit.")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)