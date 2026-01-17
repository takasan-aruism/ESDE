"""
ESDE Phase 8-7: Rigidity Signals
================================

硬直度（Rigidity）の計算。

Spec v8.7.1:
  R = N_mode / N_total
  
  - N_total: そのAtomが観測された総回数
  - N_mode: そのAtomを含む最頻出formula_signatureの出現回数

解釈:
  - R → 1.0: 硬直（真理化）- 常に同じパターンで出現
  - R → 0.0: 揺らぎ（探索中）- 多様なパターンで出現
"""

from typing import Dict, Optional, Tuple
from .semantic_index import SemanticIndex, AtomStats


def calculate_rigidity(stats: AtomStats) -> float:
    """
    AtomStatsから硬直度を計算する。
    
    R = N_mode / N_total
    
    Args:
        stats: Atomの統計情報
        
    Returns:
        硬直度 [0.0, 1.0]
    """
    n_total = stats.count_total
    
    if n_total == 0:
        return 0.0
    
    # 最頻出formulaのカウントを取得
    if not stats.formula_counts:
        return 0.0
    
    n_mode = stats.formula_counts.most_common(1)[0][1]
    
    return n_mode / n_total


def calculate_rigidity_windowed(stats: AtomStats, window: int = 1000) -> float:
    """
    直近windowイベント内での硬直度を計算する。
    
    注意: 現在の実装ではformula_countsは全期間の累積。
    真のwindowed計算にはformula_countの時系列保持が必要だが、
    メモリ効率のため簡易実装（全期間ベース）を使用。
    
    将来的にはwindow対応の統計構造を追加可能。
    
    Args:
        stats: Atomの統計情報
        window: ウィンドウサイズ（現時点では参考値）
        
    Returns:
        硬直度 [0.0, 1.0]
    """
    # 現時点では全期間ベースで計算
    # TODO: window対応のformula_counts実装
    return calculate_rigidity(stats)


def get_rigidity_for_atom(index: SemanticIndex, atom_id: str, window: int = 1000) -> Optional[float]:
    """
    特定AtomのRigidityを取得する。
    
    Args:
        index: SemanticIndex
        atom_id: Atom ID
        window: ウィンドウサイズ
        
    Returns:
        硬直度、またはAtomが存在しない場合None
    """
    stats = index.get_atom_stats(atom_id)
    if stats is None:
        return None
    
    return calculate_rigidity_windowed(stats, window)


def get_mode_formula(stats: AtomStats) -> Optional[Tuple[str, int]]:
    """
    最頻出のformula_signatureを取得する。
    
    Args:
        stats: Atomの統計情報
        
    Returns:
        (formula_signature, count) または None
    """
    if not stats.formula_counts:
        return None
    
    return stats.formula_counts.most_common(1)[0]


def classify_rigidity(r: float) -> str:
    """
    硬直度を分類する。
    
    Args:
        r: 硬直度 [0.0, 1.0]
        
    Returns:
        分類ラベル
    """
    if r >= 0.9:
        return "crystallized"  # 結晶化（完全に固定）
    elif r >= 0.7:
        return "rigid"         # 硬直
    elif r >= 0.4:
        return "stable"        # 安定
    elif r >= 0.2:
        return "fluid"         # 流動的
    else:
        return "volatile"      # 揮発的（探索中）


def get_all_rigidities(index: SemanticIndex, window: int = 1000) -> Dict[str, float]:
    """
    全AtomのRigidityを取得する。
    
    Args:
        index: SemanticIndex
        window: ウィンドウサイズ
        
    Returns:
        {atom_id: rigidity} のdict
    """
    result = {}
    for atom_id in index.get_all_atoms():
        r = get_rigidity_for_atom(index, atom_id, window)
        if r is not None:
            result[atom_id] = r
    return result
