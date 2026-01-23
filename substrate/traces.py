"""
ESDE Substrate Layer: Trace Normalization and Validation
=========================================================

This module defines the type constraints and normalization rules for traces.

Philosophy: "Describe, but do not decide."

GPT Audit Compliance:
  - GPT追記1: traces型制約の明文化
  - GPT追記2: Canonical JSON仕様固定

Invariants:
  INV-SUB-002: No Semantic Transform
  INV-SUB-003: Machine-Observable
  INV-SUB-004: No Inference

Spec: Substrate Layer v0.1.0
"""

import math
import re
from typing import Any, Dict, List, Optional, Tuple, Union

# ==========================================
# Constants (GPT追記1: 型制約の明文化)
# ==========================================

# Version for trace normalization logic
TRACE_NORMALIZER_VERSION = "v0.1.0"

# Float precision for normalization (GPT追記2)
FLOAT_PRECISION = 9  # 9 decimal places

# Integer constraints
INT_MIN = -(2**31)   # -2147483648
INT_MAX = 2**31 - 1  # 2147483647

# String constraints
STRING_MAX_LENGTH = 4096  # Max characters per string value

# ==========================================
# Forbidden Namespace Prefixes (GPT追記1)
# ==========================================
# These namespaces imply semantic interpretation and are BANNED
# INV-SUB-002: No Semantic Transform

FORBIDDEN_NAMESPACE_PREFIXES = frozenset([
    "meaning:",    # Semantic interpretation
    "category:",   # Classification
    "intent:",     # Intent inference
    "quality:",    # Quality judgment
    "importance:", # Importance ranking
    "sentiment:",  # Sentiment analysis
    "topic:",      # Topic classification
    "type:",       # Type classification (use meta: instead)
])

# ==========================================
# Forbidden Key Names (P1-SUB-002: GPT Audit)
# ==========================================
# These key NAMES (after namespace:) imply semantic interpretation
# Even with allowed namespaces, these names are banned
# INV-SUB-002: No Semantic Transform

FORBIDDEN_KEY_NAMES = frozenset([
    # Classification/Type interpretations
    "source_type",      # Use legacy:source_type only for migration
    "content_type",     # Ambiguous - use meta:content_type for MIME type only
    "document_type",
    "media_type",
    
    # Boolean interpretations (is_X patterns)
    "is_short",
    "is_long",
    "is_important",
    "is_relevant",
    "is_spam",
    "is_news",
    "is_dialog",
    "is_formal",
    "is_informal",
    "is_positive",
    "is_negative",
    
    # Quality/Importance judgments
    "quality_score",
    "importance_score",
    "relevance_score",
    "confidence_score",
    "trust_score",
    
    # Semantic labels
    "label",
    "tag",
    "class",
    "classification",
])

# Exception: These key names are allowed in specific namespaces
ALLOWED_KEY_NAME_EXCEPTIONS = {
    # legacy: namespace can use source_type for migration
    "legacy:source_type": True,
    "legacy:language_profile": True,
    # meta: namespace can use content_type for MIME type
    "meta:content_type": True,
}

# ==========================================
# Forbidden Key Names (P1-SUB-002)
# ==========================================
# These key names imply semantic interpretation regardless of namespace
# They are "interpretation words" that should not appear in trace keys
# INV-SUB-002: No Semantic Transform

FORBIDDEN_KEY_NAMES = frozenset([
    # Type/Category interpretation
    "source_type",
    "content_type_semantic",  # (meta:content_type for MIME is OK)
    "document_type",
    "media_type_semantic",
    
    # Boolean interpretations (use raw facts instead)
    "is_short",
    "is_long", 
    "is_formal",
    "is_informal",
    "is_news",
    "is_social",
    "is_spam",
    "is_valid",
    "is_important",
    "is_relevant",
    
    # Quality/Sentiment interpretations
    "quality_score",
    "sentiment_score",
    "importance_score",
    "relevance_score",
    "readability_score",
    
    # Classification results
    "classified_as",
    "labeled_as",
    "detected_as",
    "inferred_type",
])

# ==========================================
# Allowed Value Types
# ==========================================
# INV-SUB-003: Machine-Observable values only

ALLOWED_VALUE_TYPES = (str, int, float, bool, type(None))

# ==========================================
# Validation Functions
# ==========================================

def validate_trace_key(key: str) -> Tuple[bool, Optional[str]]:
    """
    Validate a trace key.
    
    Rules:
      - Must be string
      - Must contain exactly one ":"
      - Must match pattern "namespace:name"
      - Must not use forbidden namespace prefixes (INV-SUB-002)
      - Must not use forbidden key names unless in exception list (P1-SUB-002)
      - Namespace and name must be non-empty
      - Must contain only [a-z0-9_] characters
    
    Args:
        key: The trace key to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(key, str):
        return False, f"Key must be string, got {type(key).__name__}"
    
    if ":" not in key:
        return False, f"Key '{key}' must contain ':' separator (format: namespace:name)"
    
    parts = key.split(":")
    if len(parts) != 2:
        return False, f"Key '{key}' must have exactly one ':' (got {len(parts)-1})"
    
    namespace, name = parts
    
    if not namespace:
        return False, f"Key '{key}' has empty namespace"
    
    if not name:
        return False, f"Key '{key}' has empty name"
    
    # Check for forbidden namespaces (INV-SUB-002)
    for forbidden in FORBIDDEN_NAMESPACE_PREFIXES:
        if key.startswith(forbidden):
            return False, f"Key '{key}' uses forbidden namespace prefix '{forbidden}' (INV-SUB-002)"
    
    # Check for forbidden key names (P1-SUB-002)
    # Allow exceptions for specific namespace:name combinations
    if name in FORBIDDEN_KEY_NAMES:
        if key not in ALLOWED_KEY_NAME_EXCEPTIONS:
            return False, f"Key '{key}' uses forbidden interpretation word '{name}' (INV-SUB-002, P1-SUB-002)"
    
    # Check character set
    key_pattern = re.compile(r'^[a-z][a-z0-9_]*:[a-z][a-z0-9_]*$')
    if not key_pattern.match(key):
        return False, f"Key '{key}' must match pattern [a-z][a-z0-9_]*:[a-z][a-z0-9_]*"
    
    return True, None


def validate_trace_value(value: Any) -> Tuple[bool, Optional[str]]:
    """
    Validate a trace value.
    
    Rules (GPT追記1):
      - Must be one of: str, int, float, bool, None
      - No List or Dict (prevents nesting)
      - Float: no NaN, no Inf
      - Int: within INT_MIN to INT_MAX
      - String: max STRING_MAX_LENGTH characters
    
    Args:
        value: The trace value to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check type
    if not isinstance(value, ALLOWED_VALUE_TYPES):
        return False, f"Value type {type(value).__name__} not allowed. Use: str, int, float, bool, None"
    
    # Float constraints
    if isinstance(value, float):
        if math.isnan(value):
            return False, "Float value NaN is not allowed"
        if math.isinf(value):
            return False, "Float value Inf/-Inf is not allowed"
    
    # Int constraints (exclude bool which is subclass of int)
    if isinstance(value, int) and not isinstance(value, bool):
        if value < INT_MIN or value > INT_MAX:
            return False, f"Integer {value} out of range [{INT_MIN}, {INT_MAX}]"
    
    # String constraints
    if isinstance(value, str):
        if len(value) > STRING_MAX_LENGTH:
            return False, f"String length {len(value)} exceeds max {STRING_MAX_LENGTH}"
    
    return True, None


def validate_traces(traces: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate entire traces dict.
    
    Args:
        traces: The traces dict to validate
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    if not isinstance(traces, dict):
        return False, [f"Traces must be dict, got {type(traces).__name__}"]
    
    errors = []
    
    for key, value in traces.items():
        # Validate key
        key_valid, key_error = validate_trace_key(key)
        if not key_valid:
            errors.append(key_error)
            continue
        
        # Validate value
        value_valid, value_error = validate_trace_value(value)
        if not value_valid:
            errors.append(f"Key '{key}': {value_error}")
    
    return len(errors) == 0, errors


# ==========================================
# Normalization Functions (GPT追記2)
# ==========================================

def normalize_float(value: float) -> float:
    """
    Normalize float to fixed precision.
    
    GPT追記2: 浮動小数点の文字列表現を固定
    
    Args:
        value: Float value to normalize
        
    Returns:
        Normalized float with FLOAT_PRECISION decimal places
    """
    if math.isnan(value) or math.isinf(value):
        raise ValueError(f"Cannot normalize NaN/Inf: {value}")
    
    return round(value, FLOAT_PRECISION)


def normalize_string(value: str) -> str:
    """
    Normalize string for canonical representation.
    
    GPT追記2: 文字列正規化ルール
    
    Rules:
      - No leading/trailing whitespace normalization (preserve raw)
      - No newline normalization (preserve raw)
      - Just truncate if too long
    
    Rationale:
      Substrateは「観測値そのもの」を記録する (INV-SUB-002)
      文字列の正規化は「解釈」に近いため、最小限に留める
    
    Args:
        value: String value to normalize
        
    Returns:
        Normalized string (truncated if necessary)
    """
    if len(value) > STRING_MAX_LENGTH:
        return value[:STRING_MAX_LENGTH]
    return value


def normalize_value(value: Any) -> Any:
    """
    Normalize a single trace value.
    
    Args:
        value: Value to normalize
        
    Returns:
        Normalized value
    """
    if value is None:
        return None
    
    if isinstance(value, bool):
        return value
    
    if isinstance(value, int):
        # Clamp to valid range
        return max(INT_MIN, min(INT_MAX, value))
    
    if isinstance(value, float):
        return normalize_float(value)
    
    if isinstance(value, str):
        return normalize_string(value)
    
    # Should not reach here if validate_trace_value passed
    raise TypeError(f"Cannot normalize type {type(value).__name__}")


def normalize_traces(traces: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize traces dict for canonical representation.
    
    This function:
      1. Validates all keys and values
      2. Normalizes values (float precision, string length)
      3. Returns sorted dict (by key)
    
    Args:
        traces: Raw traces dict
        
    Returns:
        Normalized traces dict (sorted by key)
        
    Raises:
        ValueError: If validation fails
    """
    # Validate first
    is_valid, errors = validate_traces(traces)
    if not is_valid:
        raise ValueError(f"Trace validation failed: {errors}")
    
    # Normalize and sort
    normalized = {}
    for key in sorted(traces.keys()):
        normalized[key] = normalize_value(traces[key])
    
    return normalized


# ==========================================
# Utility Functions
# ==========================================

def extract_namespace(key: str) -> Optional[str]:
    """
    Extract namespace from trace key.
    
    Args:
        key: Trace key (e.g., "html:tag_count")
        
    Returns:
        Namespace (e.g., "html") or None if invalid
    """
    if ":" not in key:
        return None
    return key.split(":")[0]


def group_by_namespace(traces: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Group traces by namespace.
    
    Args:
        traces: Flat traces dict
        
    Returns:
        Nested dict: {namespace: {name: value}}
    """
    grouped: Dict[str, Dict[str, Any]] = {}
    
    for key, value in traces.items():
        if ":" not in key:
            continue
        namespace, name = key.split(":", 1)
        if namespace not in grouped:
            grouped[namespace] = {}
        grouped[namespace][name] = value
    
    return grouped


# ==========================================
# Test
# ==========================================

if __name__ == "__main__":
    print("Substrate Traces Module Test")
    print("=" * 60)
    
    # Test 1: Valid key validation
    print("\n[Test 1] Valid key validation")
    valid_keys = [
        "html:tag_count",
        "meta:domain",
        "text:char_count",
        "time:hour_of_day",
        "legacy:source_type",
    ]
    for key in valid_keys:
        is_valid, error = validate_trace_key(key)
        status = "✅" if is_valid else "❌"
        print(f"  {status} '{key}': {error or 'OK'}")
    
    # Test 2: Invalid key validation
    print("\n[Test 2] Invalid key validation")
    invalid_keys = [
        ("no_colon", "no colon"),
        ("meaning:test", "forbidden namespace"),
        ("category:news", "forbidden namespace"),
        ("HTML:TAG", "uppercase"),
        (":empty_namespace", "empty namespace"),
        ("empty_name:", "empty name"),
    ]
    for key, reason in invalid_keys:
        is_valid, error = validate_trace_key(key)
        status = "✅" if not is_valid else "❌"
        print(f"  {status} '{key}' ({reason}): {error or 'Should have failed'}")
    
    # Test 3: Value validation
    print("\n[Test 3] Value validation")
    values = [
        ("string", "hello", True),
        ("int", 42, True),
        ("float", 3.14159, True),
        ("bool", True, True),
        ("None", None, True),
        ("list", [1, 2, 3], False),
        ("dict", {"a": 1}, False),
        ("NaN", float('nan'), False),
        ("Inf", float('inf'), False),
    ]
    for name, value, expected in values:
        is_valid, error = validate_trace_value(value)
        status = "✅" if is_valid == expected else "❌"
        result = "valid" if is_valid else f"invalid: {error}"
        print(f"  {status} {name}: {result}")
    
    # Test 4: Float normalization
    print("\n[Test 4] Float normalization")
    floats = [
        (3.14159265358979, 3.141592654),
        (0.1 + 0.2, 0.3),  # Famous float issue
        (1e-10, 1e-10),
    ]
    for original, _ in floats:
        normalized = normalize_float(original)
        print(f"  {original} -> {normalized}")
    
    # Test 5: Full traces normalization
    print("\n[Test 5] Full traces normalization")
    raw_traces = {
        "text:char_count": 1234,
        "html:has_h1": True,
        "meta:domain": "example.com",
        "text:avg_word_len": 4.567891234567,
    }
    normalized = normalize_traces(raw_traces)
    print(f"  Input: {raw_traces}")
    print(f"  Output: {normalized}")
    assert list(normalized.keys()) == sorted(normalized.keys()), "Keys should be sorted"
    print("  ✅ Keys sorted correctly")
    
    # Test 6: Forbidden namespace rejection
    print("\n[Test 6] Forbidden namespace rejection")
    try:
        bad_traces = {"meaning:sentiment": "positive"}
        normalize_traces(bad_traces)
        print("  ❌ Should have raised ValueError")
    except ValueError as e:
        print(f"  ✅ Correctly rejected: {e}")
    
    # Test 7: Forbidden key name rejection (P1-SUB-002)
    print("\n[Test 7] Forbidden key name rejection (P1-SUB-002)")
    forbidden_keys = [
        ("data:source_type", "forbidden interpretation word"),
        ("text:is_short", "forbidden interpretation word"),
        ("data:quality_score", "forbidden interpretation word"),
    ]
    for key, reason in forbidden_keys:
        is_valid, error = validate_trace_key(key)
        status = "✅" if not is_valid else "❌"
        print(f"  {status} '{key}' ({reason}): {'rejected' if not is_valid else 'SHOULD FAIL'}")
    
    # Test 8: Allowed exceptions (P1-SUB-002)
    print("\n[Test 8] Allowed exceptions (P1-SUB-002)")
    allowed_exceptions = [
        ("legacy:source_type", "migration exception"),
        ("meta:content_type", "MIME type exception"),
    ]
    for key, reason in allowed_exceptions:
        is_valid, error = validate_trace_key(key)
        status = "✅" if is_valid else "❌"
        print(f"  {status} '{key}' ({reason}): {'allowed' if is_valid else error}")
    
    print("\n" + "=" * 60)
    print("All tests completed!")
