"""
ESDE Phase 8-8: Generation Strategies
=====================================

Feedback Modulatorが選択する生成戦略の定義。

Spec v8.8 + GPT監査v8.8.1:
- Crystallized (R > 0.9) → Disruptive
- Volatile (R < 0.3) → Stabilizing  
- Neutral (その他) → Neutral
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class StrategyMode(Enum):
    """戦略モード"""
    NEUTRAL = "neutral"
    DISRUPTIVE = "disruptive"
    STABILIZING = "stabilizing"


@dataclass
class GenerationStrategy:
    """
    生成戦略。
    
    LLMの生成パラメータを決定する。
    """
    mode: StrategyMode
    temperature: float
    prompt_suffix: Optional[str] = None
    reason: str = ""
    
    def to_dict(self) -> dict:
        return {
            "mode": self.mode.value,
            "temperature": self.temperature,
            "prompt_suffix": self.prompt_suffix,
            "reason": self.reason
        }


# ==========================================
# 定義済み戦略 (Spec v8.8)
# ==========================================

STRATEGY_NEUTRAL = GenerationStrategy(
    mode=StrategyMode.NEUTRAL,
    temperature=0.1,
    prompt_suffix=None,
    reason="Normal operation"
)

STRATEGY_DISRUPTIVE = GenerationStrategy(
    mode=StrategyMode.DISRUPTIVE,
    temperature=0.7,
    prompt_suffix="Current definition is too rigid. Doubt it. Find contradictions or alternative interpretations.",
    reason="Concept crystallized - attempting disruption"
)

STRATEGY_STABILIZING = GenerationStrategy(
    mode=StrategyMode.STABILIZING,
    temperature=0.0,
    prompt_suffix="Definition is volatile. Consolidate. Find commonality and stable patterns.",
    reason="Concept volatile - attempting stabilization"
)

STRATEGY_UNKNOWN = GenerationStrategy(
    mode=StrategyMode.NEUTRAL,
    temperature=0.1,
    prompt_suffix=None,
    reason="Unknown atom (first observation) - using neutral"
)
