#!/usr/bin/env python3
"""
ESDE Phase 9: Condition Provider
=================================

Pluggable condition extraction for W2 aggregation.

Design Philosophy:
  - Conditions come from INTERNAL structure, not external metadata
  - Each provider extracts ONE axis (no combination explosion)
  - Aggregation unit is configurable (token/sentence/section)

Available Providers:
  - SectionConditionProvider: section_id as condition
  - PassiveConditionProvider: is_passive_sentence (0/1)
  - ParenthesesConditionProvider: inside_parentheses (0/1)
  - QuoteConditionProvider: is_in_quote (0/1)
  - ProperNounConditionProvider: sentence contains PROPN (0/1)

Usage:
    provider = SectionConditionProvider()
    condition_id = provider.get_condition(token_feature, context)

Spec: Phase 9 Condition Provider v1.0
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


# ==========================================
# Aggregation Context
# ==========================================

@dataclass
class AggregationContext:
    """
    Context for condition extraction.
    
    Provides access to surrounding tokens for sentence/section level decisions.
    """
    # Current token info
    token_idx: int
    sentence_idx: int
    section_idx: int
    section_name: str
    
    # Sentence-level info (for sentence_level aggregation)
    sentence_tokens: List[Any]  # List of TokenFeature in same sentence
    sentence_has_passive: bool
    sentence_has_propn: bool
    
    # Article-level info
    article_id: str
    total_sections: int


# ==========================================
# Base Condition Provider
# ==========================================

class BaseConditionProvider(ABC):
    """
    Abstract base class for condition extraction.
    
    Each provider extracts ONE condition axis from token features.
    """
    
    @property
    @abstractmethod
    def provider_id(self) -> str:
        """Unique identifier for this provider (e.g., 'section_v1')."""
        pass
    
    @property
    @abstractmethod
    def axis_name(self) -> str:
        """Human-readable axis name (e.g., 'section')."""
        pass
    
    @property
    def aggregation_unit(self) -> str:
        """
        Aggregation unit: 'token', 'sentence', or 'section'.
        Default is 'sentence' (recommended for Phase 9).
        """
        return "sentence"
    
    @abstractmethod
    def get_condition_id(self, token_feature: Any, context: AggregationContext) -> str:
        """
        Extract condition ID from token and context.
        
        Args:
            token_feature: TokenFeature from W1
            context: AggregationContext with surrounding info
            
        Returns:
            Condition ID string (e.g., "Lead", "passive:1")
        """
        pass
    
    def get_condition_factors(self, condition_id: str) -> Dict[str, Any]:
        """
        Get human-readable factors for a condition ID.
        
        Args:
            condition_id: The condition ID string
            
        Returns:
            Dict with factor names and values
        """
        return {self.axis_name: condition_id}


# ==========================================
# Section Condition Provider
# ==========================================

class SectionConditionProvider(BaseConditionProvider):
    """
    Uses section_id (section name) as condition.
    
    This captures "where in the article" the token appears.
    
    Expected conditions:
      - "Lead" (introduction)
      - "Early life" 
      - "Military campaigns"
      - "Death"
      - "Legacy"
      - etc.
    """
    
    @property
    def provider_id(self) -> str:
        return "section_v1"
    
    @property
    def axis_name(self) -> str:
        return "section"
    
    @property
    def aggregation_unit(self) -> str:
        return "section"  # Section-level makes sense for this axis
    
    def get_condition_id(self, token_feature: Any, context: AggregationContext) -> str:
        """Return section name as condition."""
        # Normalize section name (lowercase, trim)
        section = context.section_name.strip().lower()
        
        # Handle empty section name
        if not section:
            section = "unknown"
        
        return section
    
    def get_condition_factors(self, condition_id: str) -> Dict[str, Any]:
        return {
            "section_name": condition_id,
            "axis": "section",
        }


# ==========================================
# Passive Voice Condition Provider
# ==========================================

class PassiveConditionProvider(BaseConditionProvider):
    """
    Uses is_passive_sentence as condition (0/1).
    
    This captures "narrative mode" - whether the sentence uses
    passive voice construction.
    
    Expected conditions:
      - "passive:0" (active voice)
      - "passive:1" (passive voice)
    """
    
    @property
    def provider_id(self) -> str:
        return "passive_v1"
    
    @property
    def axis_name(self) -> str:
        return "passive"
    
    @property
    def aggregation_unit(self) -> str:
        return "sentence"
    
    def get_condition_id(self, token_feature: Any, context: AggregationContext) -> str:
        """Return passive status as condition."""
        is_passive = 1 if context.sentence_has_passive else 0
        return f"passive:{is_passive}"
    
    def get_condition_factors(self, condition_id: str) -> Dict[str, Any]:
        value = int(condition_id.split(":")[1])
        return {
            "is_passive": bool(value),
            "axis": "passive",
        }


# ==========================================
# Parentheses Condition Provider
# ==========================================

class ParenthesesConditionProvider(BaseConditionProvider):
    """
    Uses inside_parentheses as condition (0/1).
    
    This captures "information type" - main content vs supplementary.
    
    Expected conditions:
      - "paren:0" (main text)
      - "paren:1" (inside parentheses - supplementary)
    """
    
    # Feature index for inside_parentheses in TokenFeature.vector
    PAREN_INDEX = 8
    
    @property
    def provider_id(self) -> str:
        return "paren_v1"
    
    @property
    def axis_name(self) -> str:
        return "parentheses"
    
    @property
    def aggregation_unit(self) -> str:
        return "token"  # Token-level for this axis
    
    def get_condition_id(self, token_feature: Any, context: AggregationContext) -> str:
        """Return parentheses status as condition."""
        is_paren = 1 if token_feature.vector[self.PAREN_INDEX] == 1.0 else 0
        return f"paren:{is_paren}"
    
    def get_condition_factors(self, condition_id: str) -> Dict[str, Any]:
        value = int(condition_id.split(":")[1])
        return {
            "inside_parentheses": bool(value),
            "axis": "parentheses",
        }


# ==========================================
# Quote Condition Provider
# ==========================================

class QuoteConditionProvider(BaseConditionProvider):
    """
    Uses is_in_quote as condition (0/1).
    
    This captures "voice type" - narration vs direct speech/citation.
    
    Expected conditions:
      - "quote:0" (narration)
      - "quote:1" (inside quote)
    """
    
    # Feature index for is_in_quote in TokenFeature.vector
    QUOTE_INDEX = 9
    
    @property
    def provider_id(self) -> str:
        return "quote_v1"
    
    @property
    def axis_name(self) -> str:
        return "quote"
    
    @property
    def aggregation_unit(self) -> str:
        return "token"
    
    def get_condition_id(self, token_feature: Any, context: AggregationContext) -> str:
        """Return quote status as condition."""
        is_quote = 1 if token_feature.vector[self.QUOTE_INDEX] == 1.0 else 0
        return f"quote:{is_quote}"
    
    def get_condition_factors(self, condition_id: str) -> Dict[str, Any]:
        value = int(condition_id.split(":")[1])
        return {
            "is_in_quote": bool(value),
            "axis": "quote",
        }


# ==========================================
# Proper Noun Presence Condition Provider
# ==========================================

class ProperNounConditionProvider(BaseConditionProvider):
    """
    Uses "sentence contains proper noun" as condition (0/1).
    
    This captures "content type" - descriptive vs entity-focused.
    
    Expected conditions:
      - "propn:0" (no proper nouns in sentence)
      - "propn:1" (has proper nouns)
    """
    
    @property
    def provider_id(self) -> str:
        return "propn_v1"
    
    @property
    def axis_name(self) -> str:
        return "proper_noun"
    
    @property
    def aggregation_unit(self) -> str:
        return "sentence"
    
    def get_condition_id(self, token_feature: Any, context: AggregationContext) -> str:
        """Return proper noun presence as condition."""
        has_propn = 1 if context.sentence_has_propn else 0
        return f"propn:{has_propn}"
    
    def get_condition_factors(self, condition_id: str) -> Dict[str, Any]:
        value = int(condition_id.split(":")[1])
        return {
            "has_proper_noun": bool(value),
            "axis": "proper_noun",
        }


# ==========================================
# Section × Passive Combined Provider
# ==========================================

class SectionPassiveConditionProvider(BaseConditionProvider):
    """
    Combines section and passive voice as condition.
    
    This creates finer-grained conditions (up to 12 = 6 sections × 2 passive states).
    
    Expected conditions:
      - "lead:passive:0"
      - "lead:passive:1"
      - "military:passive:0"
      - "military:passive:1"
      - etc.
    """
    
    @property
    def provider_id(self) -> str:
        return "section_passive_v1"
    
    @property
    def axis_name(self) -> str:
        return "section_passive"
    
    @property
    def aggregation_unit(self) -> str:
        return "sentence"
    
    def get_condition_id(self, token_feature: Any, context: AggregationContext) -> str:
        """Return combined section + passive condition."""
        # Section (normalized)
        section = context.section_name.strip().lower()
        if not section:
            section = "unknown"
        
        # Passive
        is_passive = 1 if context.sentence_has_passive else 0
        
        return f"{section}:passive:{is_passive}"
    
    def get_condition_factors(self, condition_id: str) -> Dict[str, Any]:
        parts = condition_id.split(":")
        section = parts[0]
        is_passive = int(parts[2]) if len(parts) > 2 else 0
        
        return {
            "section_name": section,
            "is_passive": bool(is_passive),
            "axis": "section_passive",
        }


# ==========================================
# Provider Registry
# ==========================================

CONDITION_PROVIDERS = {
    "section": SectionConditionProvider,
    "passive": PassiveConditionProvider,
    "paren": ParenthesesConditionProvider,
    "quote": QuoteConditionProvider,
    "propn": ProperNounConditionProvider,
    "section_passive": SectionPassiveConditionProvider,
}


def get_condition_provider(axis: str) -> BaseConditionProvider:
    """
    Get condition provider by axis name.
    
    Args:
        axis: One of 'section', 'passive', 'paren', 'quote', 'propn'
        
    Returns:
        Instantiated provider
        
    Raises:
        ValueError: If axis not found
    """
    if axis not in CONDITION_PROVIDERS:
        available = ", ".join(CONDITION_PROVIDERS.keys())
        raise ValueError(f"Unknown axis: {axis}. Available: {available}")
    
    return CONDITION_PROVIDERS[axis]()


# ==========================================
# Test
# ==========================================

if __name__ == "__main__":
    print("Condition Provider Test")
    print("=" * 60)
    
    # Test each provider
    for axis, provider_cls in CONDITION_PROVIDERS.items():
        provider = provider_cls()
        print(f"\n[{axis}]")
        print(f"  provider_id: {provider.provider_id}")
        print(f"  axis_name: {provider.axis_name}")
        print(f"  aggregation_unit: {provider.aggregation_unit}")
    
    print("\n" + "=" * 60)
    print("All providers loaded successfully!")
