"""
ESDE Migration Phase 3: Test Suite
===================================

Test coverage for:
  - P0-1: unknown_scope禁止 (Policy mode で scope が必須)
  - P0-2: パストラバーサル対策 (ID サニタイズ)
  - P0-3: Policy isolation (Legacy パス汚染防止)
  - P1-4: policy_signature_hash 生成
  - P1-5: ExecutionContext 管理

Spec: Migration Phase 3 v0.3.1 (Audit Fixed)
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ==========================================
# Test 1: Scope Validation (P0-1)
# ==========================================

def test_scope_validation_policy_mode():
    """
    P0-1: unknown_scope 禁止
    
    Policy mode で analysis_scope_id が None/空なら即エラー。
    """
    print("\n[Test 1] Scope Validation in Policy Mode (P0-1)")
    print("-" * 60)
    
    from statistics.utils import (
        validate_scope_id,
        ScopeValidationError,
        resolve_stats_dir,
    )
    
    # Mock policy
    class MockPolicy:
        policy_id = "test_policy"
        version = "v1.0"
    
    policy = MockPolicy()
    
    # Test 1a: None scope should raise error
    print("  Test 1a: None scope in policy mode")
    try:
        validate_scope_id(None, policy_mode=True)
        print("    ❌ FAIL: Should have raised ScopeValidationError")
        return False
    except ScopeValidationError as e:
        assert "REQUIRED" in str(e)
        print("    ✅ PASS: None rejected")
    
    # Test 1b: Empty scope should raise error
    print("  Test 1b: Empty scope in policy mode")
    try:
        validate_scope_id("", policy_mode=True)
        print("    ❌ FAIL: Should have raised ScopeValidationError")
        return False
    except ScopeValidationError as e:
        assert "REQUIRED" in str(e)
        print("    ✅ PASS: Empty rejected")
    
    # Test 1c: resolve_stats_dir with None scope should raise error
    print("  Test 1c: resolve_stats_dir without scope")
    try:
        resolve_stats_dir("data/stats", policy, None)
        print("    ❌ FAIL: Should have raised ScopeValidationError")
        return False
    except ScopeValidationError:
        print("    ✅ PASS: Path resolution rejected")
    
    # Test 1d: Valid scope should work
    print("  Test 1d: Valid scope")
    result = validate_scope_id("my_scope", policy_mode=True)
    assert result == "my_scope"
    print(f"    ✅ PASS: '{result}'")
    
    # Test 1e: Legacy mode ignores scope
    print("  Test 1e: Legacy mode ignores scope")
    result = validate_scope_id(None, policy_mode=False)
    assert result == ""
    print("    ✅ PASS: None accepted in legacy")
    
    print("  ✅ All scope validation tests PASS")
    return True


# ==========================================
# Test 2: Path Traversal Prevention (P0-2)
# ==========================================

def test_path_traversal_prevention():
    """
    P0-2: パストラバーサル対策
    
    ID類に ../, /, \\ などが含まれる場合は即エラー。
    """
    print("\n[Test 2] Path Traversal Prevention (P0-2)")
    print("-" * 60)
    
    from statistics.utils import (
        validate_id,
        validate_policy_id,
        validate_scope_id,
        PathTraversalError,
        PolicyIdValidationError,
        ScopeValidationError,
    )
    
    # Dangerous inputs
    dangerous_inputs = [
        ("../etc/passwd", "path traversal .."),
        ("..\\windows\\system32", "path traversal ..\\"),
        ("foo/../bar", "embedded .."),
        ("foo/bar", "forward slash"),
        ("foo\\bar", "backslash"),
        ("/etc/passwd", "leading slash"),
    ]
    
    print("  Testing dangerous inputs:")
    for value, reason in dangerous_inputs:
        try:
            validate_id(value, "test")
            print(f"    ❌ FAIL: '{value}' should be rejected ({reason})")
            return False
        except (PathTraversalError, ValueError):
            print(f"    ✅ '{value}' rejected ({reason})")
    
    # Test with policy_id validator
    print("\n  Testing policy_id validator:")
    try:
        validate_policy_id("../hack")
        print("    ❌ FAIL: Should reject path traversal in policy_id")
        return False
    except PolicyIdValidationError:
        print("    ✅ policy_id rejects path traversal")
    
    # Test with scope validator
    print("\n  Testing scope validator:")
    try:
        validate_scope_id("../hack", policy_mode=True)
        print("    ❌ FAIL: Should reject path traversal in scope")
        return False
    except ScopeValidationError:
        print("    ✅ scope_id rejects path traversal")
    
    print("  ✅ All path traversal tests PASS")
    return True


# ==========================================
# Test 3: Reserved Words (P0-2 extension)
# ==========================================

def test_reserved_words():
    """
    P0-2 extension: 予約語禁止
    
    "legacy", "policies", "global", "unknown" などは使用不可。
    """
    print("\n[Test 3] Reserved Words Rejection")
    print("-" * 60)
    
    from statistics.utils import validate_id, RESERVED_WORDS
    
    print(f"  Reserved words ({len(RESERVED_WORDS)}): {sorted(RESERVED_WORDS)[:5]}...")
    
    for word in ["legacy", "global", "unknown_scope", "policies"]:
        try:
            validate_id(word, "test")
            print(f"    ❌ FAIL: '{word}' should be rejected")
            return False
        except ValueError as e:
            assert "reserved" in str(e).lower()
            print(f"    ✅ '{word}' rejected")
    
    print("  ✅ All reserved word tests PASS")
    return True


# ==========================================
# Test 4: Path Resolution Logic
# ==========================================

def test_path_resolution():
    """
    Test resolve_stats_dir() produces correct paths.
    """
    print("\n[Test 4] Path Resolution Logic")
    print("-" * 60)
    
    from statistics.utils import resolve_stats_dir, resolve_w2_paths
    
    class MockPolicy:
        policy_id = "my_policy"
        version = "v2.0"
    
    # Test 4a: Legacy mode
    print("  Test 4a: Legacy mode path")
    path = resolve_stats_dir("data/stats", None, None)
    assert str(path) == "data/stats"
    print(f"    Path: {path}")
    print("    ✅ PASS")
    
    # Test 4b: Policy mode
    print("  Test 4b: Policy mode path")
    path = resolve_stats_dir("data/stats", MockPolicy(), "run_20260125")
    expected = "data/stats/policies/my_policy/v2.0/run_20260125"
    assert str(path) == expected, f"Expected {expected}, got {path}"
    print(f"    Path: {path}")
    print("    ✅ PASS")
    
    # Test 4c: W2 paths
    print("  Test 4c: W2 file paths")
    paths = resolve_w2_paths("data/stats", MockPolicy(), "run_20260125")
    assert "policies/my_policy/v2.0/run_20260125" in paths['records_path']
    assert paths['records_path'].endswith("w2_records.jsonl")
    print(f"    Records: {paths['records_path']}")
    print(f"    Conditions: {paths['conditions_path']}")
    print("    ✅ PASS")
    
    print("  ✅ All path resolution tests PASS")
    return True


# ==========================================
# Test 5: Policy Signature Hash (P1-4)
# ==========================================

def test_policy_signature_hash():
    """
    P1-4: policy_signature_hash 生成
    """
    print("\n[Test 5] Policy Signature Hash (P1-4)")
    print("-" * 60)
    
    from statistics.utils import compute_policy_signature_hash
    
    class MockPolicy:
        policy_id = "test_policy"
        version = "v1.0"
        target_keys = ["legacy:source_type", "legacy:language_profile"]
    
    # Test 5a: Compute hash
    print("  Test 5a: Compute hash")
    policy = MockPolicy()
    hash1 = compute_policy_signature_hash(policy)
    
    assert hash1 is not None
    assert len(hash1) == 64  # SHA256 hex
    print(f"    Hash: {hash1[:32]}...")
    print("    ✅ PASS")
    
    # Test 5b: Deterministic
    print("  Test 5b: Determinism")
    hash2 = compute_policy_signature_hash(MockPolicy())
    assert hash1 == hash2, "Hash should be deterministic"
    print("    ✅ PASS")
    
    # Test 5c: None policy returns None
    print("  Test 5c: None policy")
    result = compute_policy_signature_hash(None)
    assert result is None
    print("    ✅ PASS")
    
    # Test 5d: Different policy produces different hash
    print("  Test 5d: Different policy")
    class OtherPolicy:
        policy_id = "other_policy"
        version = "v1.0"
        target_keys = ["different"]
    
    hash3 = compute_policy_signature_hash(OtherPolicy())
    assert hash3 != hash1, "Different policies should have different hashes"
    print("    ✅ PASS")
    
    print("  ✅ All signature hash tests PASS")
    return True


# ==========================================
# Test 6: ExecutionContext (P1-5)
# ==========================================

def test_execution_context():
    """
    P1-5: ExecutionContext 管理
    """
    print("\n[Test 6] ExecutionContext (P1-5)")
    print("-" * 60)
    
    from statistics.utils import ExecutionContext, ScopeValidationError
    
    class MockPolicy:
        policy_id = "test_policy"
        version = "v1.0"
        target_keys = ["a", "b"]
    
    # Test 6a: Policy mode context
    print("  Test 6a: Policy mode context")
    ctx = ExecutionContext(
        analysis_scope_id="run_20260125",
        policy=MockPolicy(),
        signature_source="policy",
    )
    
    assert ctx.analysis_scope_id == "run_20260125"
    assert ctx.policy_id == "test_policy"
    assert ctx.policy_version == "v1.0"
    assert ctx.signature_source == "policy"
    assert ctx.policy_signature_hash is not None
    print(f"    Context: {ctx}")
    print("    ✅ PASS")
    
    # Test 6b: Metadata export
    print("  Test 6b: Metadata export")
    meta = ctx.to_metadata_dict()
    
    assert "analysis_scope_id" in meta
    assert "policy_id" in meta
    assert "policy_version" in meta
    assert "policy_signature_hash" in meta
    assert "signature_source" in meta
    print(f"    Keys: {list(meta.keys())}")
    print("    ✅ PASS")
    
    # Test 6c: Legacy mode context
    print("  Test 6c: Legacy mode context")
    ctx_legacy = ExecutionContext(
        analysis_scope_id="",
        policy=None,
        signature_source="legacy",
    )
    
    assert ctx_legacy.policy_id is None
    assert ctx_legacy.signature_source == "legacy"
    print("    ✅ PASS")
    
    # Test 6d: Policy mode without scope should fail
    print("  Test 6d: Policy mode without scope")
    try:
        ExecutionContext(
            analysis_scope_id=None,
            policy=MockPolicy(),
        )
        print("    ❌ FAIL: Should raise ScopeValidationError")
        return False
    except ScopeValidationError:
        print("    ✅ PASS: Rejected")
    
    print("  ✅ All ExecutionContext tests PASS")
    return True


# ==========================================
# Test 7: Scope ID Generation
# ==========================================

def test_scope_id_generation():
    """
    Test auto-generation of scope IDs.
    """
    print("\n[Test 7] Scope ID Generation")
    print("-" * 60)
    
    from statistics.utils import generate_scope_id, validate_scope_id
    import time
    
    # Generate and validate
    scope1 = generate_scope_id()
    print(f"  Generated: {scope1}")
    
    # Should be valid
    result = validate_scope_id(scope1, policy_mode=True)
    assert result == scope1
    print("  ✅ Format valid")
    
    # Should start with "run_"
    assert scope1.startswith("run_")
    print("  ✅ Prefix correct")
    
    # Generate another (should be different after small delay)
    time.sleep(0.1)
    scope2 = generate_scope_id()
    # Note: might be same if within same second, that's OK
    print(f"  Second: {scope2}")
    
    print("  ✅ All generation tests PASS")
    return True


# ==========================================
# Test 8: File System Isolation (Integration)
# ==========================================

def test_filesystem_isolation():
    """
    Integration test: Verify Policy mode creates files in isolated directory.
    """
    print("\n[Test 8] File System Isolation (Integration)")
    print("-" * 60)
    
    from statistics.utils import resolve_stats_dir
    
    class MockPolicy:
        policy_id = "isolation_test"
        version = "v1.0"
    
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = tmpdir
        
        # Legacy path
        legacy_path = resolve_stats_dir(base_dir, None, None)
        
        # Policy path
        policy_path = resolve_stats_dir(base_dir, MockPolicy(), "test_scope")
        
        # Create both directories
        legacy_path.mkdir(parents=True, exist_ok=True)
        policy_path.mkdir(parents=True, exist_ok=True)
        
        # Create files
        (legacy_path / "legacy_file.txt").write_text("legacy")
        (policy_path / "policy_file.txt").write_text("policy")
        
        # Verify isolation
        print(f"  Legacy path: {legacy_path}")
        print(f"  Policy path: {policy_path}")
        
        assert legacy_path != policy_path, "Paths should be different"
        assert "policies" not in str(legacy_path), "Legacy should not have 'policies'"
        assert "policies" in str(policy_path), "Policy should have 'policies'"
        
        # Verify files are separate
        assert (legacy_path / "legacy_file.txt").exists()
        assert not (legacy_path / "policy_file.txt").exists()
        assert (policy_path / "policy_file.txt").exists()
        assert not (policy_path / "legacy_file.txt").exists()
        
        print("  ✅ Files are isolated")
    
    print("  ✅ All isolation tests PASS")
    return True


# ==========================================
# Main
# ==========================================

def main():
    print("\n" + "=" * 70)
    print("  ESDE Migration Phase 3 Test Suite")
    print("  GPT Audit P0 Compliance Tests")
    print("=" * 70)
    
    tests = [
        ("P0-1: Scope Validation", test_scope_validation_policy_mode),
        ("P0-2: Path Traversal", test_path_traversal_prevention),
        ("P0-2: Reserved Words", test_reserved_words),
        ("Path Resolution", test_path_resolution),
        ("P1-4: Signature Hash", test_policy_signature_hash),
        ("P1-5: ExecutionContext", test_execution_context),
        ("Scope Generation", test_scope_id_generation),
        ("File Isolation", test_filesystem_isolation),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, passed, None))
        except Exception as e:
            import traceback
            traceback.print_exc()
            results.append((name, False, str(e)))
    
    # Summary
    print("\n" + "=" * 70)
    print("  Test Summary")
    print("=" * 70)
    
    passed = sum(1 for _, p, _ in results if p)
    failed = len(results) - passed
    
    for name, p, err in results:
        status = "✅ PASS" if p else f"❌ FAIL: {err}"
        print(f"  {name}: {status}")
    
    print()
    print(f"  Total: {passed}/{len(results)} passed")
    
    if failed > 0:
        print("\n  ⚠️ Some tests FAILED")
        return 1
    else:
        print("\n  🎉 All tests PASSED")
        return 0


if __name__ == "__main__":
    sys.exit(main())
