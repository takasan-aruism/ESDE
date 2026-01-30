#!/usr/bin/env python3
"""
ESDE Phase 9: Internal Condition Providers
==========================================

Provides condition_id from internal text structure instead of external metadata.

Design Philosophy:
  - External condition (source_meta) dies when data is homogeneous
  - Internal condition (section/passive/paren) survives because it varies within articles
  - This is the key insight: conditions emerge from data, not labels

Condition Providers:
  - SectionConditionProvider: section_id as condition
  - PassiveConditionProvider: is_passive_sentence as condition
  - ParenConditionProvider: inside_parentheses as condition
  - QuoteConditionProvider: is_in_quote as condition
  - PropnConditionProvider: proper_noun_presence as condition

Usage:
    from statistics.features.condition_providers import SectionConditionProvider
    
    provider = SectionConditionProvider()
    condition_id = provider.get_condition(sentence_context)

Spec: Phase 9 Internal Conditions v1.0
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import hashlib
import json


# ==========================================
# Condition Context (Input to Provider)
# ==========================================

@dataclass
class SentenceContext:
    """
    Context for a sentence, used by ConditionProvider.
    
    This represents a single "observation unit" for W2 statistics.
    All conditions are derived from this context.
    """
    # Sentence identification
    sentence_idx: int
    section_idx: int
    section_name: str
    
    # Sentence-level features (aggregated from tokens)
    token_count: int
    has_passive: bool           # Any is_passive_participle in sentence
    has_paren: bool             # Any inside_parentheses in sentence
    has_quote: bool             # Any is_in_quote in sentence
    has_proper_noun: bool       # Any PROPN in sentence
    
    # Token list for W2 counting
    tokens: List[str]           # Normalized tokens in this sentence
    
    # Optional: raw features for advanced providers
    token_features: Optional[List[Dict[str, float]]] = None


# ==========================================
# Base Condition Provider
# ==========================================

class BaseConditionProvider(ABC):
    """
    Abstract base class for condition providers.
    
    A ConditionProvider maps a SentenceContext to a condition_id.
    This condition_id is used by W2 for statistics grouping.
    """
    
    @property
    @abstractmethod
    def provider_id(self) -> str:
        """Unique identifier for this provider (e.g., 'section_v1')."""
        pass
    
    @property
    @abstractmethod
    def condition_axis(self) -> str:
        """Name of the condition axis (e.g., 'section', 'passive')."""
        pass
    
    @abstractmethod
    def get_condition_id(self, ctx: SentenceContext) -> str:
        """
        Get condition_id for a sentence context.
        
        Args:
            ctx: SentenceContext with sentence-level information
            
        Returns:
            condition_id string (e.g., "section:Lead", "passive:1")
        """
        pass
    
    def get_condition_signature(self, condition_id: str) -> str:
        """
        Get deterministic signature for condition_id.
        
        Default: SHA256 hash of canonical JSON.
        Override if needed.
        """
        canonical = json.dumps(
            {"provider": self.provider_id, "condition": condition_id},
            sort_keys=True,
            separators=(',', ':'),
        )
        return hashlib.sha256(canonical.encode('utf-8')).hexdigest()
    
    def get_condition_factors(self, condition_id: str) -> Dict[str, Any]:
        """
        Get human-readable factors for condition_id.
        
        Used for logging and debugging (not for signature).
        """
        return {
            "provider": self.provider_id,
            "axis": self.condition_axis,
            "condition_id": condition_id,
        }


# ==========================================
# Section Condition Provider
# ==========================================

class SectionConditionProvider(BaseConditionProvider):
    """
    Uses section_id (section_name) as condition.
    
    This captures "where in the article" the text appears.
    Expected to show differences like:
      - Lead: introductory language
      - Military: action verbs, battle terminology
      - Legacy: evaluative language, passive voice
    """
    
    def __init__(self, normalize_sections: bool = True):
        """
        Args:
            normalize_sections: If True, normalize section names
                               (lowercase, strip, collapse whitespace)
        """
        self.normalize_sections = normalize_sections
    
    @property
    def provider_id(self) -> str:
        return "section_v1"
    
    @property
    def condition_axis(self) -> str:
        return "section"
    
    def get_condition_id(self, ctx: SentenceContext) -> str:
        section = ctx.section_name
        
        if self.normalize_sections:
            section = self._normalize(section)
        
        return f"section:{section}"
    
    def _normalize(self, section: str) -> str:
        """Normalize section name for consistency."""
        import re
        # Lowercase
        s = section.lower()
        # Strip
        s = s.strip()
        # Collapse whitespace
        s = re.sub(r'\s+', '_', s)
        # Remove special characters
        s = re.sub(r'[^\w_]', '', s)
        return s or "unknown"


# ==========================================
# Passive Condition Provider
# ==========================================

class PassiveConditionProvider(BaseConditionProvider):
    """
    Uses is_passive_sentence (0/1) as condition.
    
    This captures "voice mode" of the sentence.
    Expected to show differences like:
      - passive:1: "was born", "was killed", "is regarded as"
      - passive:0: active verbs, direct actions
    """
    
    @property
    def provider_id(self) -> str:
        return "passive_v1"
    
    @property
    def condition_axis(self) -> str:
        return "passive"
    
    def get_condition_id(self, ctx: SentenceContext) -> str:
        flag = 1 if ctx.has_passive else 0
        return f"passive:{flag}"


# ==========================================
# Parentheses Condition Provider
# ==========================================

class ParenConditionProvider(BaseConditionProvider):
    """
    Uses inside_parentheses presence as condition.
    
    This captures "supplementary vs main" content.
    Expected to show differences like:
      - paren:1: dates, readings, explanations
      - paren:0: main narrative
    """
    
    @property
    def provider_id(self) -> str:
        return "paren_v1"
    
    @property
    def condition_axis(self) -> str:
        return "paren"
    
    def get_condition_id(self, ctx: SentenceContext) -> str:
        flag = 1 if ctx.has_paren else 0
        return f"paren:{flag}"


# ==========================================
# Quote Condition Provider
# ==========================================

class QuoteConditionProvider(BaseConditionProvider):
    """
    Uses is_in_quote presence as condition.
    
    This captures "quoted vs narrative" content.
    Expected to show differences like:
      - quote:1: titles, names, direct speech
      - quote:0: narrative prose
    """
    
    @property
    def provider_id(self) -> str:
        return "quote_v1"
    
    @property
    def condition_axis(self) -> str:
        return "quote"
    
    def get_condition_id(self, ctx: SentenceContext) -> str:
        flag = 1 if ctx.has_quote else 0
        return f"quote:{flag}"


# ==========================================
# Proper Noun Condition Provider
# ==========================================

class PropnConditionProvider(BaseConditionProvider):
    """
    Uses proper_noun_presence as condition.
    
    This captures "entity-rich vs general" sentences.
    Expected to show differences like:
      - propn:1: sentences about specific people/places
      - propn:0: general descriptions
    """
    
    @property
    def provider_id(self) -> str:
        return "propn_v1"
    
    @property
    def condition_axis(self) -> str:
        return "propn"
    
    def get_condition_id(self, ctx: SentenceContext) -> str:
        flag = 1 if ctx.has_proper_noun else 0
        return f"propn:{flag}"


# ==========================================
# Compound Condition Provider (Future)
# ==========================================

class CompoundConditionProvider(BaseConditionProvider):
    """
    Combines multiple providers (for future use).
    
    WARNING: This causes combinatorial explosion.
    Only use when you understand the tradeoffs.
    
    Example:
        compound = CompoundConditionProvider([
            SectionConditionProvider(),
            PassiveConditionProvider(),
        ])
        # condition_id = "section:lead+passive:1"
    """
    
    def __init__(self, providers: List[BaseConditionProvider]):
        self.providers = providers
        self._provider_id = "+".join(p.provider_id for p in providers)
        self._axis = "+".join(p.condition_axis for p in providers)
    
    @property
    def provider_id(self) -> str:
        return f"compound:{self._provider_id}"
    
    @property
    def condition_axis(self) -> str:
        return self._axis
    
    def get_condition_id(self, ctx: SentenceContext) -> str:
        parts = [p.get_condition_id(ctx) for p in self.providers]
        return "+".join(parts)


# ==========================================
# Factory
# ==========================================

def get_condition_provider(axis: str) -> BaseConditionProvider:
    """
    Factory function to get condition provider by axis name.
    
    Args:
        axis: 'section', 'passive', 'paren', 'quote', 'propn'
        
    Returns:
        ConditionProvider instance
    """
    providers = {
        "section": SectionConditionProvider,
        "passive": PassiveConditionProvider,
        "paren": ParenConditionProvider,
        "quote": QuoteConditionProvider,
        "propn": PropnConditionProvider,
    }
    
    if axis not in providers:
        raise ValueError(f"Unknown axis: {axis}. Available: {list(providers.keys())}")
    
    return providers[axis]()


# ==========================================
# Export
# ==========================================

__all__ = [
    # Context
    "SentenceContext",
    
    # Base
    "BaseConditionProvider",
    
    # Providers
    "SectionConditionProvider",
    "PassiveConditionProvider",
    "ParenConditionProvider",
    "QuoteConditionProvider",
    "PropnConditionProvider",
    "CompoundConditionProvider",
    
    # Factory
    "get_condition_provider",
]
