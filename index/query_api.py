"""
ESDE Phase 8-7: Query API
=========================

L2 Semantic Indexへの外部クエリインターフェース。

Spec v8.7.1 (GPT監査追記):
- get_frequency(atom_id, window=1000)
- get_rigidity(atom_id, window=1000)
- get_recent_directions(limit=1000)
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

from .semantic_index import SemanticIndex
from .projector import Projector, create_index_from_ledger
from .rigidity import (
    get_rigidity_for_atom,
    get_all_rigidities,
    get_mode_formula,
    classify_rigidity,
)


@dataclass
class AtomInfo:
    """Atom情報の応答構造"""
    atom_id: str
    frequency: int
    rigidity: float
    rigidity_class: str
    mode_formula: Optional[str]
    mode_count: Optional[int]
    last_seen_seq: int


@dataclass
class DirectionBalance:
    """方向性バランスの応答構造"""
    total: int
    creative: int      # =>+
    destructive: int   # -|>
    neutral: int       # =>
    r_creative: float  # creative ratio
    r_destructive: float


class QueryAPI:
    """
    L2 Semantic Index へのクエリAPI。
    
    Usage:
        api = QueryAPI(ledger)
        api.get_rigidity("EMO.love")
        api.get_frequency("EMO.love", window=500)
        api.get_recent_directions(limit=100)
    """
    
    def __init__(
        self,
        ledger: Any = None,
        index: SemanticIndex = None,
        seq_history_limit: int = 10000
    ):
        """
        Args:
            ledger: PersistentLedger instance（indexがNoneの場合に使用）
            index: 既存のSemanticIndex（指定すればledgerは不要）
            seq_history_limit: 直近seq保持数
        """
        if index is not None:
            self.index = index
        elif ledger is not None:
            self.index = create_index_from_ledger(ledger, seq_history_limit)
        else:
            self.index = SemanticIndex(seq_history_limit)
        
        self.projector = Projector(self.index)
        self._ledger = ledger
    
    # ==========================================
    # Core Query Methods
    # ==========================================
    
    def get_frequency(self, atom_id: str, window: int = 1000) -> int:
        """
        Atomの出現頻度を取得する。
        
        Args:
            atom_id: Atom ID
            window: ウィンドウサイズ（直近N件）
            
        Returns:
            出現回数（windowが全件より大きい場合は全件カウント）
        """
        stats = self.index.get_atom_stats(atom_id)
        if stats is None:
            return 0
        
        # window内のカウント
        if window >= len(stats.recent_seqs):
            # 全件
            return stats.count_total
        else:
            # 直近window件（recent_seqsはdequeなので後ろからwindow件）
            return min(window, len(stats.recent_seqs))
    
    def get_rigidity(self, atom_id: str, window: int = 1000) -> Optional[float]:
        """
        Atomの硬直度を取得する。
        
        Args:
            atom_id: Atom ID
            window: ウィンドウサイズ
            
        Returns:
            硬直度 [0.0, 1.0]、Atomが存在しない場合None
        """
        return get_rigidity_for_atom(self.index, atom_id, window)
    
    def get_recent_directions(self, limit: int = 1000) -> DirectionBalance:
        """
        直近イベントの方向性バランスを取得する。
        
        Args:
            limit: 取得するイベント数
            
        Returns:
            DirectionBalance
        """
        creative = 0
        destructive = 0
        neutral = 0
        
        # 各方向のstatsから直近をカウント
        for direction, stats in self.index.direction_stats.items():
            # 直近limit件に含まれるかを判定
            count = min(limit, len(stats.recent_seqs))
            
            if direction == "=>+":
                creative = count
            elif direction == "-|>":
                destructive = count
            elif direction == "=>":
                neutral = count
        
        total = creative + destructive + neutral
        
        return DirectionBalance(
            total=total,
            creative=creative,
            destructive=destructive,
            neutral=neutral,
            r_creative=creative / total if total > 0 else 0.0,
            r_destructive=destructive / total if total > 0 else 0.0
        )
    
    # ==========================================
    # Extended Query Methods
    # ==========================================
    
    def get_atom_info(self, atom_id: str, window: int = 1000) -> Optional[AtomInfo]:
        """
        Atomの詳細情報を取得する。
        
        Args:
            atom_id: Atom ID
            window: ウィンドウサイズ
            
        Returns:
            AtomInfo、またはAtomが存在しない場合None
        """
        stats = self.index.get_atom_stats(atom_id)
        if stats is None:
            return None
        
        rigidity = get_rigidity_for_atom(self.index, atom_id, window)
        mode = get_mode_formula(stats)
        
        return AtomInfo(
            atom_id=atom_id,
            frequency=stats.count_total,
            rigidity=rigidity or 0.0,
            rigidity_class=classify_rigidity(rigidity or 0.0),
            mode_formula=mode[0] if mode else None,
            mode_count=mode[1] if mode else None,
            last_seen_seq=stats.last_seen_seq
        )
    
    def get_cooccurrence(self, atom_a: str, atom_b: str) -> int:
        """
        2つのAtomの共起回数を取得する。
        
        Args:
            atom_a: Atom ID A
            atom_b: Atom ID B
            
        Returns:
            共起回数
        """
        # ペアはソート済みで格納されている
        pair = tuple(sorted([atom_a, atom_b]))
        return self.index.cooccurrence.get(pair, 0)
    
    def get_top_cooccurrences(self, atom_id: str, limit: int = 10) -> List[Tuple[str, int]]:
        """
        特定Atomとの共起が多いAtomを取得する。
        
        Args:
            atom_id: 基準Atom ID
            limit: 取得数
            
        Returns:
            [(atom_id, count), ...] のリスト
        """
        results = []
        
        for pair, count in self.index.cooccurrence.items():
            if atom_id in pair:
                other = pair[0] if pair[1] == atom_id else pair[1]
                results.append((other, count))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]
    
    def get_all_rigidities(self, window: int = 1000) -> Dict[str, float]:
        """
        全AtomのRigidityを取得する。
        
        Args:
            window: ウィンドウサイズ
            
        Returns:
            {atom_id: rigidity} のdict
        """
        return get_all_rigidities(self.index, window)
    
    # ==========================================
    # Index Management
    # ==========================================
    
    def rebuild(self) -> int:
        """
        Ledgerから全件再構築する。
        
        Returns:
            処理したイベント数
        """
        if self._ledger is None:
            raise RuntimeError("No ledger attached")
        
        return self.projector.rebuild_from_ledger(self._ledger)
    
    def on_event(self, entry: dict) -> bool:
        """
        新規イベントでIndexを更新する。
        
        Args:
            entry: Ledger entry
            
        Returns:
            処理したかどうか
        """
        return self.projector.on_event(entry)
    
    def status(self) -> dict:
        """
        Index状態を取得する。
        """
        snapshot = self.index.snapshot()
        return {
            **snapshot,
            "seq_history_limit": self.index.seq_history_limit,
            "has_ledger": self._ledger is not None
        }
