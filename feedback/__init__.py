"""
ESDE Feedback Package
=====================
Phase 8-8: Live Integration & Introspective Feedback

L2 Indexの指標に基づいて生成戦略を動的に決定する。
"""

from .strategies import (
    StrategyMode,
    GenerationStrategy,
    STRATEGY_NEUTRAL,
    STRATEGY_DISRUPTIVE,
    STRATEGY_STABILIZING,
    STRATEGY_UNKNOWN,
)

from .modulator import (
    Modulator,
    decide_strategy,
    get_target_atom_from_candidates,
    RIGIDITY_HIGH,
    RIGIDITY_LOW,
    DEFAULT_WINDOW,
    ALERT_RIGIDITY_THRESHOLD,
    ALERT_MIN_OBSERVATIONS,
)


__all__ = [
    # Classes
    "StrategyMode",
    "GenerationStrategy",
    "Modulator",
    
    # Predefined Strategies
    "STRATEGY_NEUTRAL",
    "STRATEGY_DISRUPTIVE", 
    "STRATEGY_STABILIZING",
    "STRATEGY_UNKNOWN",
    
    # Functions
    "decide_strategy",
    "get_target_atom_from_candidates",
    
    # Constants
    "RIGIDITY_HIGH",
    "RIGIDITY_LOW",
    "DEFAULT_WINDOW",
    "ALERT_RIGIDITY_THRESHOLD",
    "ALERT_MIN_OBSERVATIONS",
]

__version__ = "8.8.0"
