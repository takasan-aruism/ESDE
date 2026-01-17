"""
ESDE Phase 8-7: Projector
=========================

L1 (Immutable Ledger) から L2 (Semantic Index) への投影。

Spec v8.7.1 準拠:
- rebuild(): 全件走査して構築
- on_event(): 逐次更新
- Parity Consistency: rebuild結果 == 逐次更新結果

GPT監査追記:
- formula_signature: formula → atoms/operators連結 → __unknown__
"""

from typing import Dict, List, Optional, Iterator, Any
from .semantic_index import SemanticIndex


def extract_formula_signature(molecule: Dict) -> str:
    """
    Moleculeからformula_signatureを抽出する。
    
    優先順位 (GPT監査 v8.7.1):
    1. molecule.formula がある → そのまま使用
    2. atoms/operators がある → canonical連結
    3. どちらもない → "__unknown__"
    
    Args:
        molecule: Ledger entryのdata.molecule
        
    Returns:
        formula_signature文字列
    """
    if not molecule:
        return "__unknown__"
    
    # 1. formulaがあればそのまま使用
    formula = molecule.get("formula")
    if formula and isinstance(formula, str) and formula.strip():
        return formula.strip()
    
    # 2. atoms/operatorsから生成
    # active_atomsからatom IDを抽出
    active_atoms = molecule.get("active_atoms", [])
    atom_ids = []
    for aa in active_atoms:
        atom = aa.get("atom") if isinstance(aa, dict) else None
        if atom:
            atom_ids.append(atom)
    
    operators = molecule.get("operators", [])
    
    if atom_ids:
        # atoms + operators を連結
        atoms_str = "|".join(sorted(atom_ids))
        ops_str = "|".join(operators) if operators else "NONE"
        return f"{atoms_str}::{ops_str}"
    
    # 3. どちらもない
    return "__unknown__"


def extract_atoms_from_molecule(molecule: Dict) -> List[str]:
    """
    Moleculeからatom IDリストを抽出する。
    
    Args:
        molecule: Ledger entryのdata.molecule
        
    Returns:
        atom IDのリスト
    """
    if not molecule:
        return []
    
    active_atoms = molecule.get("active_atoms", [])
    atom_ids = []
    
    for aa in active_atoms:
        atom = aa.get("atom") if isinstance(aa, dict) else None
        if atom:
            atom_ids.append(atom)
    
    return atom_ids


class Projector:
    """
    L1 → L2 投影器
    
    Ledgerからentryを受け取り、SemanticIndexを更新する。
    """
    
    def __init__(self, index: SemanticIndex):
        """
        Args:
            index: 更新対象のSemanticIndex
        """
        self.index = index
    
    def on_event(self, entry: Dict) -> bool:
        """
        単一イベントを処理してIndexを更新する。
        
        Args:
            entry: Ledger entry dict
            
        Returns:
            処理したかどうか（molecule.observe以外はスキップ）
        """
        # molecule.observe のみ処理
        event_type = entry.get("event_type", "")
        if event_type != "molecule.observe":
            return False
        
        seq = entry.get("seq", -1)
        direction = entry.get("direction", "=>")
        
        data = entry.get("data", {})
        molecule = data.get("molecule", {})
        
        # Atom抽出
        atoms = extract_atoms_from_molecule(molecule)
        if not atoms:
            return False
        
        # formula_signature抽出
        formula_signature = extract_formula_signature(molecule)
        
        # Index更新
        self.index.update(
            seq=seq,
            atoms=atoms,
            direction=direction,
            formula_signature=formula_signature
        )
        
        return True
    
    def rebuild(self, entries: Iterator[Dict]) -> int:
        """
        全件を走査してIndexを再構築する。
        
        Args:
            entries: Ledger entryのイテレータ
            
        Returns:
            処理したイベント数
        """
        # まずクリア
        self.index.clear()
        
        count = 0
        for entry in entries:
            if self.on_event(entry):
                count += 1
        
        return count
    
    def rebuild_from_ledger(self, ledger: Any) -> int:
        """
        PersistentLedgerから再構築する。
        
        Args:
            ledger: PersistentLedger instance
            
        Returns:
            処理したイベント数
        """
        return self.rebuild(ledger.iter_entries())


def create_index_from_ledger(ledger: Any, seq_history_limit: int = 10000) -> SemanticIndex:
    """
    Ledgerから新しいSemanticIndexを構築するファクトリ関数。
    
    Args:
        ledger: PersistentLedger instance
        seq_history_limit: 直近seq保持数
        
    Returns:
        構築されたSemanticIndex
    """
    index = SemanticIndex(seq_history_limit=seq_history_limit)
    projector = Projector(index)
    projector.rebuild_from_ledger(ledger)
    return index
