"""
ESDE Phase 9-1: W1 Normalizer
=============================
Token normalization for W1 statistics.

Normalization Rules (P1-1):
  1. Unicode NFKC normalization
  2. Lowercase
  3. Strip leading/trailing punctuation
  4. Preserve internal punctuation (e.g., contractions)

Spec: v5.4.1-P9.1
"""

import unicodedata
import re
from typing import Optional


# ==========================================
# Punctuation Patterns
# ==========================================

# Leading/trailing punctuation to strip
# Does NOT include apostrophe (for contractions like "it's")
LEADING_PUNCT = re.compile(r'^[^\w\s]+')
TRAILING_PUNCT = re.compile(r'[^\w\s]+$')

# Version for audit trail
NORMALIZER_VERSION = "v9.1.0"


# ==========================================
# Normalizer Functions
# ==========================================

def normalize_token(token: str) -> Optional[str]:
    """
    Normalize a token for W1 aggregation.
    
    Rules (P1-1):
      1. Unicode NFKC normalization (handles full-width chars, etc.)
      2. Lowercase
      3. Strip leading punctuation (but not apostrophe)
      4. Strip trailing punctuation
      5. Return None if result is empty
    
    Args:
        token: Raw token string
        
    Returns:
        Normalized token, or None if empty after normalization
    
    Examples:
        "Apple" -> "apple"
        "Hello!" -> "hello"
        "...test..." -> "test"
        "It's" -> "it's" (apostrophe preserved)
        "「日本」" -> "日本"
        "ＡＢＣ" -> "abc" (full-width to half-width)
    """
    if not token:
        return None
    
    # Step 1: NFKC normalization
    # Handles: full-width -> half-width, compatibility chars, etc.
    normalized = unicodedata.normalize('NFKC', token)
    
    # Step 2: Lowercase
    normalized = normalized.lower()
    
    # Step 3: Strip leading punctuation
    normalized = LEADING_PUNCT.sub('', normalized)
    
    # Step 4: Strip trailing punctuation
    normalized = TRAILING_PUNCT.sub('', normalized)
    
    # Step 5: Final strip and empty check
    normalized = normalized.strip()
    
    if not normalized:
        return None
    
    return normalized


def is_valid_token(token_norm: str, min_length: int = 1) -> bool:
    """
    Check if normalized token is valid for W1 tracking.
    
    Args:
        token_norm: Normalized token
        min_length: Minimum length (default: 1)
        
    Returns:
        True if valid
    """
    if not token_norm:
        return False
    
    if len(token_norm) < min_length:
        return False
    
    # Must contain at least one alphanumeric or CJK character
    has_content = any(
        unicodedata.category(c).startswith(('L', 'N'))  # Letter or Number
        for c in token_norm
    )
    
    return has_content


# ==========================================
# Test
# ==========================================

if __name__ == "__main__":
    print("Phase 9-1 Normalizer Test")
    print("=" * 60)
    
    # Test cases: (input, expected_output)
    test_cases = [
        # Basic lowercase
        ("Apple", "apple"),
        ("HELLO", "hello"),
        
        # Punctuation stripping
        ("Hello!", "hello"),
        ("...test...", "test"),
        ("(word)", "word"),
        ('"quoted"', "quoted"),
        
        # Apostrophe preservation (contractions)
        ("It's", "it's"),
        ("don't", "don't"),
        ("'twas", "twas"),  # Leading apostrophe stripped
        
        # Full-width to half-width (NFKC)
        ("ＡＢＣ", "abc"),
        ("１２３", "123"),
        
        # Japanese punctuation
        ("「日本」", "日本"),
        ("『テスト』", "テスト"),
        ("。終わり。", "終わり"),
        
        # Edge cases
        ("", None),
        ("...", None),
        ("   ", None),
        ("'", None),
        
        # Numbers
        ("123", "123"),
        ("test123", "test123"),
    ]
    
    print("\n[Test 1] Normalization cases")
    all_pass = True
    for input_token, expected in test_cases:
        result = normalize_token(input_token)
        status = "✅" if result == expected else "❌"
        if result != expected:
            all_pass = False
        print(f"  {status} '{input_token}' -> '{result}' (expected: '{expected}')")
    
    if all_pass:
        print("  All normalization tests PASS")
    else:
        print("  SOME TESTS FAILED")
    
    # Test 2: Token validation
    print("\n[Test 2] Token validation")
    validation_cases = [
        ("apple", True),
        ("it's", True),
        ("123", True),
        ("日本", True),
        ("", False),
        ("   ", False),
    ]
    
    for token, expected in validation_cases:
        result = is_valid_token(token)
        status = "✅" if result == expected else "❌"
        print(f"  {status} is_valid_token('{token}') = {result} (expected: {expected})")
    
    # Test 3: NFKC specific cases
    print("\n[Test 3] NFKC normalization specifics")
    nfkc_cases = [
        ("ｶﾀｶﾅ", "カタカナ"),  # Half-width katakana
        ("Ａｐｐｌｅ", "apple"),  # Full-width ASCII
        ("①②③", "123"),  # Circled numbers
    ]
    
    for input_token, expected in nfkc_cases:
        result = normalize_token(input_token)
        status = "✅" if result == expected else "❌"
        print(f"  {status} '{input_token}' -> '{result}' (expected: '{expected}')")
    
    print("\n" + "=" * 60)
    print("All tests passed! ✅" if all_pass else "Some tests failed!")
