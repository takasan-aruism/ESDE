"""
ESDE Phase 8-8: Feedback Modulator
==================================

L2 Indexから得られた指標に基づき、LLMの生成パラメータを動的に決定する。

Spec v8.8 + GPT監査v8.8.1:
- Target Atom = Sensor top-1
- 未知Atom（Index未登録）は必ずNeutral
- 閾値は定数: RIGIDITY_HIGH=0.9, RIGIDITY_LOW=0.3
- ALERT条件: R >= 0.98 かつ N_total >= 10
"""

import logging
from typing import Optional, List, Dict, Any

from .strategies import (
    GenerationStrategy,
    StrategyMode,
    STRATEGY_NEUTRAL,
    STRATEGY_DISRUPTIVE,
    STRATEGY_STABILIZING,
    STRATEGY_UNKNOWN,
)


# ==========================================
# 監査可能な定数 (GPT Audit v8.8.1)
# ==========================================
RIGIDITY_HIGH = 0.9      # これ以上 → Disruptive
RIGIDITY_LOW = 0.3       # これ以下 → Stabilizing
DEFAULT_WINDOW = 1000    # Index参照のウィンドウサイズ

# Alert条件
ALERT_RIGIDITY_THRESHOLD = 0.98
ALERT_MIN_OBSERVATIONS = 10


# ==========================================
# Logger
# ==========================================
logger = logging.getLogger("esde.feedback.modulator")


# ==========================================
# Modulator
# ==========================================
class Modulator:
    """
    Feedback Modulator - L2指標に基づく戦略決定。
    """
    
    def __init__(
        self,
        rigidity_high: float = RIGIDITY_HIGH,
        rigidity_low: float = RIGIDITY_LOW,
        default_window: int = DEFAULT_WINDOW
    ):
        """
        Args:
            rigidity_high: Disruptive閾値
            rigidity_low: Stabilizing閾値
            default_window: Indexクエリのウィンドウサイズ
        """
        self.rigidity_high = rigidity_high
        self.rigidity_low = rigidity_low
        self.default_window = default_window
    
    def decide_strategy(
        self,
        atom_id: str,
        index: Any,  # SemanticIndex or QueryAPI
        window: int = None
    ) -> GenerationStrategy:
        """
        AtomのRigidityに基づいて生成戦略を決定する。
        
        Args:
            atom_id: Target Atom ID
            index: SemanticIndex または QueryAPI
            window: オプションのウィンドウサイズ
            
        Returns:
            GenerationStrategy
        """
        window = window or self.default_window
        
        # Rigidity取得
        rigidity = self._get_rigidity(atom_id, index, window)
        
        # 未知Atom（Index未登録）
        if rigidity is None:
            logger.debug(f"[Modulator] {atom_id}: unknown atom → NEUTRAL")
            return STRATEGY_UNKNOWN
        
        # 戦略決定
        if rigidity > self.rigidity_high:
            logger.info(f"[Modulator] {atom_id}: R={rigidity:.3f} > {self.rigidity_high} → DISRUPTIVE")
            return GenerationStrategy(
                mode=StrategyMode.DISRUPTIVE,
                temperature=STRATEGY_DISRUPTIVE.temperature,
                prompt_suffix=STRATEGY_DISRUPTIVE.prompt_suffix,
                reason=f"R={rigidity:.3f} > {self.rigidity_high}"
            )
        
        if rigidity < self.rigidity_low:
            logger.info(f"[Modulator] {atom_id}: R={rigidity:.3f} < {self.rigidity_low} → STABILIZING")
            return GenerationStrategy(
                mode=StrategyMode.STABILIZING,
                temperature=STRATEGY_STABILIZING.temperature,
                prompt_suffix=STRATEGY_STABILIZING.prompt_suffix,
                reason=f"R={rigidity:.3f} < {self.rigidity_low}"
            )
        
        logger.debug(f"[Modulator] {atom_id}: R={rigidity:.3f} → NEUTRAL")
        return GenerationStrategy(
            mode=StrategyMode.NEUTRAL,
            temperature=STRATEGY_NEUTRAL.temperature,
            prompt_suffix=None,
            reason=f"R={rigidity:.3f} in [{self.rigidity_low}, {self.rigidity_high}]"
        )
    
    def _get_rigidity(
        self,
        atom_id: str,
        index: Any,
        window: int
    ) -> Optional[float]:
        """
        IndexからRigidityを取得する。
        
        QueryAPIとSemanticIndex両方に対応。
        """
        # QueryAPIの場合
        if hasattr(index, 'get_rigidity'):
            return index.get_rigidity(atom_id, window=window)
        
        # SemanticIndexの場合（直接アクセス）
        if hasattr(index, 'get_atom_stats'):
            from index.rigidity import calculate_rigidity
            stats = index.get_atom_stats(atom_id)
            if stats is None:
                return None
            return calculate_rigidity(stats)
        
        return None
    
    def check_alert(
        self,
        atom_id: str,
        index: Any,
        window: int = None
    ) -> Optional[Dict[str, Any]]:
        """
        Alert条件をチェックする。
        
        GPT監査v8.8.1: R >= 0.98 かつ N_total >= 10
        
        Returns:
            Alert情報のdict、またはNone
        """
        window = window or self.default_window
        
        # Stats取得
        stats = self._get_atom_stats(atom_id, index)
        if stats is None:
            return None
        
        rigidity = self._get_rigidity(atom_id, index, window)
        if rigidity is None:
            return None
        
        n_total = stats.count_total if hasattr(stats, 'count_total') else 0
        
        # Alert条件チェック
        if rigidity >= ALERT_RIGIDITY_THRESHOLD and n_total >= ALERT_MIN_OBSERVATIONS:
            alert = {
                "type": "CONCEPT_CRYSTALLIZED",
                "atom_id": atom_id,
                "rigidity": rigidity,
                "observations": n_total,
                "message": f"[ALERT] CONCEPT_CRYSTALLIZED: {atom_id} (R={rigidity:.3f}, N={n_total})"
            }
            logger.warning(alert["message"])
            return alert
        
        return None
    
    def _get_atom_stats(self, atom_id: str, index: Any):
        """AtomStatsを取得"""
        # QueryAPIの場合
        if hasattr(index, 'index'):
            return index.index.get_atom_stats(atom_id)
        
        # SemanticIndexの場合
        if hasattr(index, 'get_atom_stats'):
            return index.get_atom_stats(atom_id)
        
        return None


# ==========================================
# ヘルパー関数
# ==========================================
def decide_strategy(
    atom_id: str,
    index: Any,
    window: int = DEFAULT_WINDOW
) -> GenerationStrategy:
    """
    関数インターフェース（Spec v8.8準拠）
    """
    modulator = Modulator()
    return modulator.decide_strategy(atom_id, index, window)


def get_target_atom_from_candidates(candidates: List[Dict]) -> Optional[str]:
    """
    Sensorの候補リストからTarget Atom（top-1）を取得する。
    
    GPT監査v8.8.1: Target Atom = Sensor top-1
    
    Args:
        candidates: Sensor V2の候補リスト
        
    Returns:
        top-1のconcept_id、またはNone
    """
    if not candidates:
        return None
    
    # candidates[0]がtop-1（Sensor V2はランク順で返す）
    return candidates[0].get("concept_id")
