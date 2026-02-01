"""
ESDE Phase 9: Lens Definitions
================================

A "Lens" is a (ConditionProvider, FeatureMode) pair.

Design source:
  - Gemini: 3 lenses (Structure / Semantic / Hybrid)
  - GPT audit: Policy and Feature must be separate → Lens = tuple

Feature Modes:
  - "token": Token frequency counting (existing W3 S-Score pipeline)
  - "vector": 20-dimensional feature vector averaging (new W3 vector pipeline)

Spec: Phase 9 Lens v1.1 (added threshold_floor per lens)
"""

from typing import Dict, Any


# ==========================================
# Lens Definitions
# ==========================================

LENSES: Dict[str, Dict[str, Any]] = {
    # 1. Structure Lens (existing behavior)
    # "How Wikipedia structures articles" (template topology)
    "structure": {
        "condition": "section",      # SectionConditionProvider
        "feature_mode": "token",     # Token frequency → S-Score → Resonance
        "threshold_floor": 0.85,     # Structural sims are high (0.85-0.95 range)
        "description": "Structural template analysis (Hub vs Narrative vs Institutional)",
    },
    
    # 2. Semantic Lens (new)
    # "What the article is about" (subject matter)
    "semantic": {
        "condition": "document",     # DocumentConditionProvider (1 article = 1 condition)
        "feature_mode": "vector",    # 20-dim vector mean → Δ profile → Cosine
        "threshold_floor": 0.0,      # z-score cosine ranges -1 to +1; let data decide
        "description": "Semantic subject analysis (War vs Philosophy vs City)",
    },
    
    # 3. Hybrid Lens (new)
    # "How meaning varies across sections within articles"
    "hybrid": {
        "condition": "section",      # SectionConditionProvider
        "feature_mode": "vector",    # 20-dim vector mean per section
        "threshold_floor": 0.0,      # z-score cosine; let data decide
        "description": "Semantic bias within structural sections",
    },
}


def get_lens(name: str) -> Dict[str, Any]:
    """Get lens definition by name."""
    if name not in LENSES:
        available = ", ".join(LENSES.keys())
        raise ValueError(f"Unknown lens: '{name}'. Available: {available}")
    return LENSES[name]


def list_lenses() -> Dict[str, str]:
    """Get lens names and descriptions."""
    return {name: lens["description"] for name, lens in LENSES.items()}
