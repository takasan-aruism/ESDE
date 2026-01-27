"""
ESDE Migration Phase 3: Statistics Utilities
=============================================

Path resolution and validation for Policy-based statistics isolation.

GPT Audit P0 Compliance:
  P0-1: unknown_scope禁止 - Policy mode で scope が None/空なら即エラー
  P0-2: パストラバーサル対策 - ID類のサニタイズ必須
  P0-3: 予約語禁止 - "legacy", "policies" 等は使用不可

Design:
  - resolve_stats_dir() は Single Source of Truth
  - すべてのID検証は validate_*() 経由で行う
  - Legacy mode では scope_id は無視される

Spec: Migration Phase 3 v0.3.1 (Audit Fixed)
"""

import re
import hashlib
import json
from pathlib import Path
from typing import Optional, TYPE_CHECKING, Any, Dict
from datetime import datetime

if TYPE_CHECKING:
    from .policies.base import BaseConditionPolicy


# ==========================================
# Constants
# ==========================================

# Version
UTILS_VERSION = "0.3.1"

# Allowed pattern for IDs (policy_id, version, scope_id)
# Only alphanumeric, underscore, hyphen, and dot allowed
ID_PATTERN = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._-]*$')

# Maximum ID length
MAX_ID_LENGTH = 128

# Reserved words that cannot be used as IDs
RESERVED_WORDS = frozenset({
    'legacy',
    'policies',
    'global',
    'unknown',
    'unknown_scope',
    'default',
    'system',
    'admin',
    'root',
    'config',
    'data',
    'stats',
    'temp',
    'tmp',
    'null',
    'none',
    'undefined',
})

# Directory names in the hierarchy
POLICIES_DIR = "policies"


# ==========================================
# Validation Errors
# ==========================================

class ScopeValidationError(ValueError):
    """Raised when analysis_scope_id validation fails."""
    pass


class PolicyIdValidationError(ValueError):
    """Raised when policy_id or version validation fails."""
    pass


class PathTraversalError(ValueError):
    """Raised when path traversal attempt is detected."""
    pass


# ==========================================
# ID Validation (P0-2: Sanitization)
# ==========================================

def validate_id(
    value: str,
    field_name: str,
    allow_empty: bool = False,
) -> str:
    """
    Validate an ID string for path safety.
    
    P0-2 Compliance:
      - Rejects path traversal attempts (../, ./, /, \\)
      - Only allows alphanumeric + underscore + hyphen + dot
      - Rejects reserved words
      - Enforces length limits
    
    Args:
        value: ID string to validate
        field_name: Field name for error messages
        allow_empty: Whether to allow empty string
        
    Returns:
        Validated ID string (stripped)
        
    Raises:
        PathTraversalError: If path traversal detected
        ValueError: If validation fails
    """
    # Handle None
    if value is None:
        if allow_empty:
            return ""
        raise ValueError(f"{field_name} is required and cannot be None")
    
    # Strip whitespace
    value = str(value).strip()
    
    # Check empty
    if not value:
        if allow_empty:
            return ""
        raise ValueError(f"{field_name} is required and cannot be empty")
    
    # P0-2: Path traversal detection
    if '..' in value or '/' in value or '\\' in value:
        raise PathTraversalError(
            f"{field_name} contains path traversal characters: '{value}'"
        )
    
    # Length check
    if len(value) > MAX_ID_LENGTH:
        raise ValueError(
            f"{field_name} exceeds maximum length ({MAX_ID_LENGTH}): {len(value)} chars"
        )
    
    # Pattern check
    if not ID_PATTERN.match(value):
        raise ValueError(
            f"{field_name} contains invalid characters: '{value}'. "
            f"Only alphanumeric, underscore, hyphen, and dot are allowed."
        )
    
    # Reserved word check
    if value.lower() in RESERVED_WORDS:
        raise ValueError(
            f"{field_name} is a reserved word and cannot be used: '{value}'"
        )
    
    return value


def validate_scope_id(scope_id: Optional[str], policy_mode: bool = False) -> str:
    """
    Validate analysis_scope_id.
    
    P0-1 Compliance:
      - In Policy mode, scope_id is REQUIRED (no unknown_scope fallback)
      - In Legacy mode, scope_id is ignored (returns empty string)
    
    Args:
        scope_id: Analysis scope ID
        policy_mode: Whether Policy is being used
        
    Returns:
        Validated scope_id or empty string for legacy mode
        
    Raises:
        ScopeValidationError: If validation fails in policy mode
    """
    if not policy_mode:
        # Legacy mode: scope is ignored
        return ""
    
    # Policy mode: scope is REQUIRED
    if not scope_id:
        raise ScopeValidationError(
            "analysis_scope_id is REQUIRED in Policy mode. "
            "Cannot use Policy without specifying a scope. "
            "Use --analysis-scope <ID> or let the system auto-generate one."
        )
    
    try:
        return validate_id(scope_id, "analysis_scope_id")
    except (PathTraversalError, ValueError) as e:
        raise ScopeValidationError(str(e)) from e


def validate_policy_id(policy_id: str) -> str:
    """
    Validate policy_id.
    
    Args:
        policy_id: Policy ID to validate
        
    Returns:
        Validated policy_id
        
    Raises:
        PolicyIdValidationError: If validation fails
    """
    try:
        return validate_id(policy_id, "policy_id")
    except (PathTraversalError, ValueError) as e:
        raise PolicyIdValidationError(str(e)) from e


def validate_version(version: str) -> str:
    """
    Validate policy version string.
    
    Args:
        version: Version string (e.g., "v1.0")
        
    Returns:
        Validated version string
        
    Raises:
        PolicyIdValidationError: If validation fails
    """
    try:
        return validate_id(version, "version")
    except (PathTraversalError, ValueError) as e:
        raise PolicyIdValidationError(str(e)) from e


# ==========================================
# Scope ID Generation
# ==========================================

def generate_scope_id() -> str:
    """
    Generate a new analysis_scope_id.
    
    Format: run_{YYYYMMDD_HHMMSS}
    
    Returns:
        Generated scope ID
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"run_{timestamp}"


# ==========================================
# Path Resolution (Single Source of Truth)
# ==========================================

def resolve_stats_dir(
    base_dir: str,
    policy: Optional['BaseConditionPolicy'],
    analysis_scope_id: Optional[str],
) -> Path:
    """
    Resolve the statistics data directory.
    
    This is the SINGLE SOURCE OF TRUTH for path resolution.
    
    Rules:
      1. Policy is None → base_dir (Legacy Mode)
         - Legacy mode では scope_id は無視され、直下のファイルが使われる
         
      2. Policy exists → base_dir/policies/{id}/{ver}/{scope} (Policy Mode)
         - scope_id が None/空 の場合は ScopeValidationError
    
    Directory Layout (Policy Mode):
      data/statistics/
      └── policies/
          └── {policy_id}/
              └── {version}/
                  └── {analysis_scope_id}/
                      ├── w2_records.jsonl
                      ├── w2_conditions.jsonl
                      └── w3_candidates/
    
    Args:
        base_dir: Base directory for statistics data
        policy: BaseConditionPolicy instance (or None for Legacy)
        analysis_scope_id: Analysis scope identifier
        
    Returns:
        Path to the statistics directory
        
    Raises:
        ScopeValidationError: If scope validation fails in Policy mode
        PolicyIdValidationError: If policy ID/version validation fails
    """
    root = Path(base_dir)
    
    # Legacy Mode
    if policy is None:
        return root
    
    # Policy Mode: Validate all IDs
    policy_id = validate_policy_id(policy.policy_id)
    version = validate_version(policy.version)
    scope = validate_scope_id(analysis_scope_id, policy_mode=True)
    
    # Build path: base/policies/{policy_id}/{version}/{scope}
    return root / POLICIES_DIR / policy_id / version / scope


def resolve_w2_paths(
    base_dir: str,
    policy: Optional['BaseConditionPolicy'],
    analysis_scope_id: Optional[str],
) -> Dict[str, str]:
    """
    Resolve W2 file paths.
    
    Returns:
        Dict with 'records_path' and 'conditions_path'
    """
    stats_dir = resolve_stats_dir(base_dir, policy, analysis_scope_id)
    
    if policy is None:
        # Legacy: use default paths in base_dir
        return {
            'records_path': str(stats_dir / "w2_conditional_stats.jsonl"),
            'conditions_path': str(stats_dir / "w2_conditions.jsonl"),
        }
    else:
        # Policy mode: use subdirectory
        return {
            'records_path': str(stats_dir / "w2_records.jsonl"),
            'conditions_path': str(stats_dir / "w2_conditions.jsonl"),
        }


def resolve_w3_output_dir(
    base_dir: str,
    policy: Optional['BaseConditionPolicy'],
    analysis_scope_id: Optional[str],
) -> str:
    """
    Resolve W3 output directory.
    
    Returns:
        Path string for W3 candidates directory
    """
    stats_dir = resolve_stats_dir(base_dir, policy, analysis_scope_id)
    return str(stats_dir / "w3_candidates")


# ==========================================
# Policy Signature Hash (GPT P1-4)
# ==========================================

def compute_policy_signature_hash(policy: Optional['BaseConditionPolicy']) -> Optional[str]:
    """
    Compute a signature hash for a policy.
    
    This hash uniquely identifies the policy configuration.
    Used for W6 metadata to ensure reproducibility.
    
    Args:
        policy: BaseConditionPolicy instance (or None)
        
    Returns:
        64-char hex hash or None if no policy
    """
    if policy is None:
        return None
    
    # Build canonical representation
    policy_data = {
        'policy_id': policy.policy_id,
        'version': policy.version,
    }
    
    # Add target_keys if StandardConditionPolicy
    if hasattr(policy, 'target_keys'):
        policy_data['target_keys'] = sorted(policy.target_keys)
    
    # Canonical JSON
    canonical = json.dumps(
        policy_data,
        sort_keys=True,
        separators=(',', ':'),
        ensure_ascii=True,
    ).encode('utf-8')
    
    return hashlib.sha256(canonical).hexdigest()


# ==========================================
# ExecutionContext (GPT P1-5)
# ==========================================

class ExecutionContext:
    """
    Execution context for a statistics run.
    
    Centralizes metadata that needs to flow through W2/W3/W6.
    Avoids having W6 depend on W2 internals.
    
    GPT Audit P1-5: Metadata should be passed through Engine, not extracted
    from W2Aggregator by W6Exporter.
    """
    
    def __init__(
        self,
        analysis_scope_id: str,
        policy: Optional['BaseConditionPolicy'] = None,
        signature_source: str = "legacy",
    ):
        """
        Initialize execution context.
        
        Args:
            analysis_scope_id: Analysis scope identifier
            policy: Policy instance (or None for legacy)
            signature_source: "policy" or "legacy"
        """
        # Validate scope in policy mode
        if policy is not None:
            self.analysis_scope_id = validate_scope_id(analysis_scope_id, policy_mode=True)
        else:
            self.analysis_scope_id = analysis_scope_id or ""
        
        self.policy = policy
        self.signature_source = signature_source if policy else "legacy"
    
    @property
    def policy_id(self) -> Optional[str]:
        """Get policy ID or None."""
        return self.policy.policy_id if self.policy else None
    
    @property
    def policy_version(self) -> Optional[str]:
        """Get policy version or None."""
        return self.policy.version if self.policy else None
    
    @property
    def policy_signature_hash(self) -> Optional[str]:
        """Get policy signature hash or None."""
        return compute_policy_signature_hash(self.policy)
    
    def to_metadata_dict(self) -> Dict[str, Any]:
        """
        Export as metadata dictionary for W6.
        
        Returns:
            Dict with all relevant metadata fields
        """
        return {
            'analysis_scope_id': self.analysis_scope_id,
            'policy_id': self.policy_id,
            'policy_version': self.policy_version,
            'policy_signature_hash': self.policy_signature_hash,
            'signature_source': self.signature_source,
        }
    
    def __repr__(self) -> str:
        return (
            f"ExecutionContext("
            f"scope={self.analysis_scope_id!r}, "
            f"policy={self.policy_id!r}, "
            f"source={self.signature_source!r})"
        )


# ==========================================
# Test
# ==========================================

if __name__ == "__main__":
    print("Migration Phase 3: Statistics Utils Test")
    print("=" * 60)
    
    # Test 1: ID Validation
    print("\n[Test 1] ID Validation")
    
    # Valid IDs
    valid_ids = ["my_scope", "run_20260125", "v1.0", "test-policy"]
    for id_val in valid_ids:
        result = validate_id(id_val, "test")
        print(f"  ✅ '{id_val}' -> '{result}'")
    
    # Invalid IDs
    print("\n  Invalid IDs:")
    invalid_ids = [
        ("../hack", "path traversal"),
        ("my/scope", "contains slash"),
        ("legacy", "reserved word"),
        ("unknown_scope", "reserved word"),
        ("", "empty"),
        ("a" * 200, "too long"),
    ]
    for id_val, reason in invalid_ids:
        try:
            validate_id(id_val, "test")
            print(f"  ❌ '{id_val}' should have failed ({reason})")
        except (PathTraversalError, ValueError) as e:
            print(f"  ✅ '{id_val}' rejected ({reason})")
    
    # Test 2: Scope Validation in Policy Mode
    print("\n[Test 2] Scope Validation (Policy Mode)")
    
    try:
        validate_scope_id(None, policy_mode=True)
        print("  ❌ Should have raised ScopeValidationError")
    except ScopeValidationError as e:
        print(f"  ✅ None rejected: {str(e)[:50]}...")
    
    try:
        validate_scope_id("", policy_mode=True)
        print("  ❌ Should have raised ScopeValidationError")
    except ScopeValidationError as e:
        print(f"  ✅ Empty rejected: {str(e)[:50]}...")
    
    result = validate_scope_id("my_scope", policy_mode=True)
    print(f"  ✅ Valid scope: '{result}'")
    
    # Test 3: Legacy Mode (scope ignored)
    print("\n[Test 3] Legacy Mode")
    
    result = validate_scope_id(None, policy_mode=False)
    print(f"  ✅ None in legacy: '{result}' (empty)")
    
    result = validate_scope_id("ignored", policy_mode=False)
    print(f"  ✅ Value in legacy: '{result}' (empty)")
    
    # Test 4: Path Resolution
    print("\n[Test 4] Path Resolution")
    
    # Mock policy
    class MockPolicy:
        policy_id = "test_policy"
        version = "v1.0"
        target_keys = ["a", "b"]
    
    # Legacy mode
    path = resolve_stats_dir("data/stats", None, None)
    print(f"  Legacy path: {path}")
    
    # Policy mode
    policy = MockPolicy()
    path = resolve_stats_dir("data/stats", policy, "run_20260125")
    print(f"  Policy path: {path}")
    
    # Policy mode without scope (should fail)
    try:
        resolve_stats_dir("data/stats", policy, None)
        print("  ❌ Should have raised ScopeValidationError")
    except ScopeValidationError:
        print("  ✅ Policy mode without scope rejected")
    
    # Test 5: Policy Signature Hash
    print("\n[Test 5] Policy Signature Hash")
    
    hash1 = compute_policy_signature_hash(MockPolicy())
    print(f"  Hash: {hash1[:32]}...")
    
    hash2 = compute_policy_signature_hash(MockPolicy())
    assert hash1 == hash2, "Hash should be deterministic"
    print("  ✅ Deterministic")
    
    # Test 6: ExecutionContext
    print("\n[Test 6] ExecutionContext")
    
    ctx = ExecutionContext(
        analysis_scope_id="run_20260125",
        policy=MockPolicy(),
        signature_source="policy",
    )
    print(f"  Context: {ctx}")
    print(f"  Metadata: {ctx.to_metadata_dict()}")
    
    # Test 7: Scope ID Generation
    print("\n[Test 7] Scope ID Generation")
    
    scope = generate_scope_id()
    print(f"  Generated: {scope}")
    validate_scope_id(scope, policy_mode=True)
    print("  ✅ Valid format")
    
    print("\n" + "=" * 60)
    print("All tests passed! ✅")
