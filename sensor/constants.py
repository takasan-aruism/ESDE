"""
ESDE Sensor - Constants
Canonical definitions for operators and thresholds.

This module is the single source of truth for operator definitions.
Both canonical and legacy validators must reference this module.
"""

# ==========================================
# Operator Definitions (v0.3 Complete)
# ==========================================
VALID_OPERATORS = {
    '×',    # 結合: Connection (A × B)
    '▷',    # 作用: Action (A ▷ B)
    '→',    # 遷移: Transition (A → B)
    '⊕',    # 並置: Juxtaposition (A ⊕ B)
    '|',    # 条件: Condition (A | B)
    '◯',    # 対象: Target (A × ◯)
    '↺',    # 再帰: Recursion (A ↺ A)
    '〈',    # 階層開始: Hierarchy open
    '〉',    # 階層終了: Hierarchy close
    '≡',    # 等価: Equivalence (A ≡ B)
    '≃',    # 実用等価: Practical Equivalence (A ≃ B)
    '¬',    # 否定: Negation (¬A)
    '⇒',    # 創発: Emergence (A ⇒ B)
    '⇒+',   # 創造的創発: Creative Emergence
    '-|>',  # 破壊的創発: Destructive Emergence
}

# Bracket pairs for syntax validation
BRACKET_PAIRS = {
    '〈': '〉',
}

# ==========================================
# Validation Thresholds
# ==========================================
# Reserved for future use (e.g., confidence thresholds)
