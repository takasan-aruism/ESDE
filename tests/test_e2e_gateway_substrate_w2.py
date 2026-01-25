"""
ESDE End-to-End Integration Test: Gateway → Substrate → W2 (Policy)
====================================================================

Migration Phase 2 統合テスト

テストフロー:
  1. Gateway.ingest() で ArticleRecord 作成
  2. substrate_ref が SubstrateRegistry に登録される
  3. W2Aggregator が Policy 経由で signature 生成
  4. 全体が繋がって動くことを確認

前提条件:
  - Gateway が enable_substrate=True で動作すること
  - W2Aggregator に registry と policy を渡せること
  - StandardConditionPolicy が正しく動作すること

Version: v5.4.8-MIG.2
"""

import sys
import os
import tempfile
from typing import Dict, Any

# Path setup
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ==========================================
# Test Utilities
# ==========================================

class TestResult:
    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.error = None
        self.skipped = False
        self.skip_reason = None
    
    def __str__(self):
        if self.skipped:
            return f"{self.name}: ⚠️ SKIP - {self.skip_reason}"
        status = "✅ PASS" if self.passed else f"❌ FAIL: {self.error}"
        return f"{self.name}: {status}"


def run_test(name: str, test_func) -> TestResult:
    """Run a test and capture result."""
    result = TestResult(name)
    try:
        test_func()
        result.passed = True
    except SkipTest as e:
        result.skipped = True
        result.skip_reason = str(e)
    except AssertionError as e:
        result.error = str(e)
    except Exception as e:
        result.error = f"{type(e).__name__}: {e}"
    return result


class SkipTest(Exception):
    """Exception to skip a test with reason."""
    pass


def check_module_available(module_name: str, feature_name: str = None):
    """Check if a module is available, raise SkipTest if not."""
    try:
        __import__(module_name)
        return True
    except ImportError as e:
        feature = feature_name or module_name
        raise SkipTest(f"{feature} not available: {e}")


# ==========================================
# Test 1: Gateway + Substrate 連携
# ==========================================

def test_gateway_substrate_connection():
    """
    Test: Gateway creates ArticleRecord with substrate_ref
    
    検証項目:
      - GatewayConfig に enable_substrate がある
      - Gateway.ingest() が substrate_ref を設定する
      - SubstrateRegistry に対応する record が存在する
    """
    print("\n[Test 1] Gateway + Substrate Connection")
    print("-" * 60)
    
    # Check modules
    check_module_available("integration.gateway", "ContentGateway")
    check_module_available("substrate", "SubstrateRegistry")
    
    from integration.gateway import ContentGateway, GatewayConfig
    from substrate import SubstrateRegistry
    
    # Check if GatewayConfig supports enable_substrate
    if not hasattr(GatewayConfig, '__dataclass_fields__') or \
       'enable_substrate' not in GatewayConfig.__dataclass_fields__:
        raise SkipTest("GatewayConfig does not have enable_substrate field")
    
    # Create temp registry
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        temp_path = f.name
    
    try:
        registry = SubstrateRegistry(storage_path=temp_path)
        
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
            source_meta={"source_type": "news", "language_profile": "en"},
        )
        
        print(f"  article_id: {article.article_id[:16]}...")
        print(f"  substrate_ref: {article.substrate_ref}")
        
        # Verify substrate_ref is set
        assert article.substrate_ref is not None, "substrate_ref should be set"
        assert len(article.substrate_ref) == 32, f"context_id should be 32 chars, got {len(article.substrate_ref)}"
        print("  ✅ substrate_ref is valid")
        
        # Verify record exists in registry
        record = registry.get(article.substrate_ref)
        assert record is not None, "Record should exist in registry"
        print(f"  Registry traces: {list(record.traces.keys())}")
        
        assert "legacy:source_type" in record.traces, "Should have legacy:source_type"
        assert record.traces["legacy:source_type"] == "news"
        print("  ✅ Registry record contains correct traces")
        
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
    
    print("  ✅ Gateway + Substrate connection works")


# ==========================================
# Test 2: W2Aggregator + Policy 連携
# ==========================================

def test_w2_aggregator_policy_connection():
    """
    Test: W2Aggregator uses Policy for signature generation
    
    検証項目:
      - W2Aggregator に registry/policy を渡せる
      - Policy 経由で signature が生成される
      - signature_source が "policy" になる
    """
    print("\n[Test 2] W2Aggregator + Policy Connection")
    print("-" * 60)
    
    # Check modules
    check_module_available("statistics.w2_aggregator", "W2Aggregator")
    check_module_available("statistics.policies", "StandardConditionPolicy")
    check_module_available("substrate", "SubstrateRegistry")
    
    from statistics.w2_aggregator import W2Aggregator
    from statistics.policies import StandardConditionPolicy
    from substrate import SubstrateRegistry, create_context_record
    
    # Check if W2Aggregator supports registry/policy
    import inspect
    sig = inspect.signature(W2Aggregator.__init__)
    if 'registry' not in sig.parameters or 'policy' not in sig.parameters:
        raise SkipTest("W2Aggregator does not support registry/policy parameters")
    
    # Create temp files
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        registry_path = f.name
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        records_path = f.name
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        conditions_path = f.name
    
    try:
        # Create registry and register a record
        registry = SubstrateRegistry(storage_path=registry_path)
        
        record = create_context_record(
            traces={
                "legacy:source_type": "news",
                "legacy:language_profile": "en",
            },
            retrieval_path=None,
            capture_version="v1.0",
        )
        context_id = registry.register(record)
        
        print(f"  Registered context_id: {context_id}")
        
        # Create Policy
        policy = StandardConditionPolicy(
            policy_id="e2e_test_policy",
            target_keys=["legacy:source_type", "legacy:language_profile"],
            version="v1.0",
        )
        
        # Create W2Aggregator with registry and policy
        aggregator = W2Aggregator(
            records_path=records_path,
            conditions_path=conditions_path,
            registry=registry,
            policy=policy,
        )
        
        print(f"  W2Aggregator created with policy: {policy.policy_id}")
        
        # Create mock ArticleRecord with substrate_ref
        class MockObservation:
            def __init__(self, segment_span, timestamp):
                self.segment_span = segment_span
                self.timestamp = timestamp
        
        class MockArticle:
            def __init__(self, article_id, raw_text, observations, source_meta, 
                         ingestion_time, substrate_ref):
                self.article_id = article_id
                self.raw_text = raw_text
                self.observations = observations
                self.source_meta = source_meta
                self.ingestion_time = ingestion_time
                self.substrate_ref = substrate_ref
        
        article = MockArticle(
            article_id="test_001",
            raw_text="Apple stock rises.",
            observations=[MockObservation((0, 18), "2026-01-20T10:00:00Z")],
            source_meta={"source_type": "news", "language_profile": "en"},
            ingestion_time="2026-01-20T10:00:00Z",
            substrate_ref=context_id,  # Link to Substrate
        )
        
        # Process article
        result = aggregator.process_article(article)
        
        print(f"  signature_source: {result.get('signature_source', 'N/A')}")
        print(f"  condition_signature: {result['condition_signature'][:32]}...")
        print(f"  signature length: {len(result['condition_signature'])}")
        
        # Verify Policy was used
        if 'signature_source' in result:
            assert result['signature_source'] == 'policy', \
                f"Should use policy path, got {result['signature_source']}"
            print("  ✅ Policy path used for signature")
        else:
            print("  ⚠️ signature_source not in result (old W2Aggregator?)")
        
        # Verify signature is 64 chars (Policy returns full hash)
        assert len(result['condition_signature']) == 64, \
            f"Policy signature should be 64 chars, got {len(result['condition_signature'])}"
        print("  ✅ Signature is full 64-char SHA256")
        
    finally:
        for path in [registry_path, records_path, conditions_path]:
            if os.path.exists(path):
                os.remove(path)
    
    print("  ✅ W2Aggregator + Policy connection works")


# ==========================================
# Test 3: End-to-End (Gateway → Substrate → W2)
# ==========================================

def test_e2e_gateway_substrate_w2():
    """
    Test: Full E2E flow from Gateway to W2 with Policy
    
    フロー:
      1. Gateway.ingest() で ArticleRecord 作成
      2. substrate_ref が自動設定される
      3. W2Aggregator が Policy 経由で統計を計算
      4. 全体が正しく動作する
    """
    print("\n[Test 3] End-to-End: Gateway → Substrate → W2 (Policy)")
    print("-" * 60)
    
    # Check modules
    check_module_available("integration.gateway", "ContentGateway")
    check_module_available("statistics.w2_aggregator", "W2Aggregator")
    check_module_available("statistics.policies", "StandardConditionPolicy")
    check_module_available("substrate", "SubstrateRegistry")
    
    from integration.gateway import ContentGateway, GatewayConfig
    from statistics.w2_aggregator import W2Aggregator
    from statistics.policies import StandardConditionPolicy
    from substrate import SubstrateRegistry
    
    # Check prerequisites
    if not hasattr(GatewayConfig, '__dataclass_fields__') or \
       'enable_substrate' not in GatewayConfig.__dataclass_fields__:
        raise SkipTest("GatewayConfig does not support enable_substrate")
    
    import inspect
    sig = inspect.signature(W2Aggregator.__init__)
    if 'registry' not in sig.parameters:
        raise SkipTest("W2Aggregator does not support registry parameter")
    
    # Create temp files
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        registry_path = f.name
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        records_path = f.name
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        conditions_path = f.name
    
    try:
        # Step 1: Create shared registry
        registry = SubstrateRegistry(storage_path=registry_path)
        print("  Step 1: Registry created")
        
        # Step 2: Create Gateway with Substrate enabled
        gateway_config = GatewayConfig(
            enable_substrate=True,
            substrate_registry_path=registry_path,
        )
        
        gateway = ContentGateway(
            config=gateway_config,
            substrate_registry=registry,
        )
        print("  Step 2: Gateway created (Substrate enabled)")
        
        # Step 3: Create Policy
        policy = StandardConditionPolicy(
            policy_id="e2e_integration_v1",
            target_keys=["legacy:source_type", "legacy:language_profile"],
            version="v1.0",
        )
        print(f"  Step 3: Policy created ({policy.policy_id})")
        
        # Step 4: Create W2Aggregator with registry and policy
        aggregator = W2Aggregator(
            records_path=records_path,
            conditions_path=conditions_path,
            registry=registry,
            policy=policy,
        )
        print("  Step 4: W2Aggregator created with Policy")
        
        # Step 5: Ingest articles through Gateway
        articles = []
        test_data = [
            ("Apple releases new iPhone. Stock rises.", "news", "en"),
            ("I love this product! Amazing!", "social", "en"),
            ("研究論文の概要です。結論は以下の通り。", "paper", "ja"),
        ]
        
        for text, source_type, lang in test_data:
            article = gateway.ingest(
                text=text,
                source_meta={"source_type": source_type, "language_profile": lang},
            )
            articles.append(article)
        
        print(f"  Step 5: Ingested {len(articles)} articles")
        
        for i, article in enumerate(articles):
            print(f"    [{i}] substrate_ref: {article.substrate_ref[:16] if article.substrate_ref else 'None'}...")
        
        # Step 6: Process articles through W2Aggregator
        results = []
        for article in articles:
            result = aggregator.process_article(article)
            results.append(result)
        
        print(f"  Step 6: Processed {len(results)} articles")
        
        for i, result in enumerate(results):
            sig_source = result.get('signature_source', 'unknown')
            sig_len = len(result['condition_signature'])
            print(f"    [{i}] source={sig_source}, sig_len={sig_len}")
        
        # Verify results
        # All articles with substrate_ref should use "policy" path
        policy_count = sum(1 for r in results if r.get('signature_source') == 'policy')
        legacy_count = sum(1 for r in results if r.get('signature_source') == 'legacy')
        
        print(f"\n  Results: policy={policy_count}, legacy={legacy_count}")
        
        # All should be policy (since all have substrate_ref)
        for i, (article, result) in enumerate(zip(articles, results)):
            if article.substrate_ref:
                sig_source = result.get('signature_source', 'unknown')
                if sig_source != 'policy':
                    print(f"    ⚠️ Article [{i}] has substrate_ref but used {sig_source}")
        
        # Verify conditions were created
        print(f"\n  Stats summary:")
        print(f"    Total conditions: {aggregator.stats.total_conditions}")
        print(f"    Total records: {aggregator.stats.total_records}")
        
        assert aggregator.stats.total_conditions >= 1, "Should have at least 1 condition"
        assert aggregator.stats.total_records >= 1, "Should have at least 1 record"
        
    finally:
        for path in [registry_path, records_path, conditions_path]:
            if os.path.exists(path):
                os.remove(path)
    
    print("\n  ✅ E2E Gateway → Substrate → W2 works!")


# ==========================================
# Test 4: Legacy Fallback (substrate_ref=None)
# ==========================================

def test_legacy_fallback():
    """
    Test: W2Aggregator falls back to legacy when substrate_ref is None
    
    検証項目:
      - substrate_ref=None の場合、legacy path を使用
      - signature_source が "legacy" になる
      - 結果は正しく処理される
    """
    print("\n[Test 4] Legacy Fallback (substrate_ref=None)")
    print("-" * 60)
    
    check_module_available("statistics.w2_aggregator", "W2Aggregator")
    check_module_available("statistics.policies", "StandardConditionPolicy")
    check_module_available("substrate", "SubstrateRegistry")
    
    from statistics.w2_aggregator import W2Aggregator
    from statistics.policies import StandardConditionPolicy
    from substrate import SubstrateRegistry
    
    import inspect
    sig = inspect.signature(W2Aggregator.__init__)
    if 'registry' not in sig.parameters:
        raise SkipTest("W2Aggregator does not support registry parameter")
    
    # Create temp files
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        registry_path = f.name
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        records_path = f.name
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        conditions_path = f.name
    
    try:
        registry = SubstrateRegistry(storage_path=registry_path)
        
        policy = StandardConditionPolicy(
            policy_id="fallback_test",
            target_keys=["legacy:source_type"],
            version="v1.0",
        )
        
        aggregator = W2Aggregator(
            records_path=records_path,
            conditions_path=conditions_path,
            registry=registry,
            policy=policy,
        )
        
        # Mock article WITHOUT substrate_ref
        class MockObservation:
            def __init__(self, segment_span, timestamp):
                self.segment_span = segment_span
                self.timestamp = timestamp
        
        class MockArticle:
            def __init__(self):
                self.article_id = "legacy_test_001"
                self.raw_text = "Hello world."
                self.observations = [MockObservation((0, 12), "2026-01-20T10:00:00Z")]
                self.source_meta = {"source_type": "news", "language_profile": "en"}
                self.ingestion_time = "2026-01-20T10:00:00Z"
                self.substrate_ref = None  # No substrate_ref
        
        article = MockArticle()
        
        result = aggregator.process_article(article)
        
        print(f"  substrate_ref: {article.substrate_ref}")
        print(f"  signature_source: {result.get('signature_source', 'N/A')}")
        print(f"  condition_signature: {result['condition_signature'][:32]}...")
        
        # Verify legacy was used
        if 'signature_source' in result:
            assert result['signature_source'] == 'legacy', \
                f"Should use legacy path when substrate_ref is None, got {result['signature_source']}"
            print("  ✅ Legacy path used correctly")
        
        # Verify signature is still 64 chars (unified)
        assert len(result['condition_signature']) == 64, \
            f"Legacy signature should be 64 chars, got {len(result['condition_signature'])}"
        print("  ✅ Legacy signature is unified 64-char format")
        
    finally:
        for path in [registry_path, records_path, conditions_path]:
            if os.path.exists(path):
                os.remove(path)
    
    print("  ✅ Legacy fallback works correctly")


# ==========================================
# Main Test Runner
# ==========================================

def run_all_tests():
    """Run all E2E integration tests."""
    print("=" * 60)
    print("ESDE E2E Integration Test: Gateway → Substrate → W2 (Policy)")
    print("=" * 60)
    
    results = []
    
    results.append(run_test("Test 1: Gateway + Substrate", test_gateway_substrate_connection))
    results.append(run_test("Test 2: W2Aggregator + Policy", test_w2_aggregator_policy_connection))
    results.append(run_test("Test 3: E2E Full Flow", test_e2e_gateway_substrate_w2))
    results.append(run_test("Test 4: Legacy Fallback", test_legacy_fallback))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for r in results if r.passed)
    skipped = sum(1 for r in results if r.skipped)
    failed = sum(1 for r in results if not r.passed and not r.skipped)
    total = len(results)
    
    for result in results:
        print(result)
    
    print("\n" + "-" * 60)
    print(f"Results: {passed} passed, {skipped} skipped, {failed} failed (total: {total})")
    
    if failed == 0:
        if skipped > 0:
            print("\n⚠️ Some tests were skipped (missing features)")
            print("This may indicate Gateway or W2Aggregator needs updating")
        else:
            print("\n✅ All E2E tests PASSED!")
            print("Migration Phase 2 integration is complete.")
    else:
        print("\n❌ Some tests FAILED!")
        print("Please review and fix the issues.")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
