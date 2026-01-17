"""
ESDE Phase 8-7: Semantic Index
==============================

L2: Semantic Index (理解の射影)

L1 (Immutable Ledger) から投影されたインメモリ構造。
検索・硬直度計算・傾向分析のためのキャッシュ。

Spec v8.7.1 準拠:
- L2はL1の射影（独自情報を持たない）
- 100%再構築可能
- 方式B: count_total + last_seen_seq保持、seq listは直近のみ
"""

from collections import defaultdict, deque, Counter
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Deque


# 直近保持するseqの最大数
DEFAULT_SEQ_HISTORY_LIMIT = 10000


@dataclass
class AtomStats:
    """Atom単位の統計情報"""
    count_total: int = 0
    last_seen_seq: int = -1
    recent_seqs: Deque[int] = field(default_factory=lambda: deque(maxlen=DEFAULT_SEQ_HISTORY_LIMIT))
    formula_counts: Counter = field(default_factory=Counter)
    
    def record(self, seq: int, formula_signature: str) -> None:
        """観測を記録"""
        self.count_total += 1
        self.last_seen_seq = seq
        self.recent_seqs.append(seq)
        self.formula_counts[formula_signature] += 1


@dataclass  
class DirectionStats:
    """方向性の統計情報"""
    count_total: int = 0
    recent_seqs: Deque[int] = field(default_factory=lambda: deque(maxlen=DEFAULT_SEQ_HISTORY_LIMIT))
    
    def record(self, seq: int) -> None:
        self.count_total += 1
        self.recent_seqs.append(seq)


class SemanticIndex:
    """
    L2: Semantic Index
    
    L1 Ledgerから投影されたインメモリ構造。
    rebuild()で全件再構築、on_event()で逐次更新。
    """
    
    def __init__(self, seq_history_limit: int = DEFAULT_SEQ_HISTORY_LIMIT):
        """
        Args:
            seq_history_limit: 各Atomが保持する直近seqの最大数
        """
        self.seq_history_limit = seq_history_limit
        
        # Atom統計 { atom_id: AtomStats }
        self.atom_stats: Dict[str, AtomStats] = defaultdict(
            lambda: AtomStats(recent_seqs=deque(maxlen=self.seq_history_limit))
        )
        
        # Direction統計 { direction: DirectionStats }
        self.direction_stats: Dict[str, DirectionStats] = defaultdict(
            lambda: DirectionStats(recent_seqs=deque(maxlen=self.seq_history_limit))
        )
        
        # 共起カウント { (atom_a, atom_b): count } （ソート済みペア）
        self.cooccurrence: Counter = Counter()
        
        # メタ情報
        self.last_seq: int = -1
        self.total_events: int = 0
    
    def clear(self) -> None:
        """全データをクリア（rebuild前に使用）"""
        self.atom_stats.clear()
        self.direction_stats.clear()
        self.cooccurrence.clear()
        self.last_seq = -1
        self.total_events = 0
    
    def update(
        self,
        seq: int,
        atoms: List[str],
        direction: str,
        formula_signature: str
    ) -> None:
        """
        単一イベントでIndexを更新する。
        
        Args:
            seq: イベントのシーケンス番号
            atoms: 含まれるAtom IDのリスト
            direction: 方向性 (=>, =>+, -|>)
            formula_signature: 正規化されたformula文字列
        """
        self.last_seq = seq
        self.total_events += 1
        
        # Atom統計を更新
        for atom_id in atoms:
            stats = self.atom_stats[atom_id]
            # dequeのmaxlenを設定し直す（defaultdictで生成された場合の対応）
            if stats.recent_seqs.maxlen != self.seq_history_limit:
                stats.recent_seqs = deque(stats.recent_seqs, maxlen=self.seq_history_limit)
            stats.record(seq, formula_signature)
        
        # Direction統計を更新
        dir_stats = self.direction_stats[direction]
        if dir_stats.recent_seqs.maxlen != self.seq_history_limit:
            dir_stats.recent_seqs = deque(dir_stats.recent_seqs, maxlen=self.seq_history_limit)
        dir_stats.record(seq)
        
        # 共起を更新（2つ以上のAtomがある場合）
        if len(atoms) >= 2:
            sorted_atoms = sorted(set(atoms))
            for i in range(len(sorted_atoms)):
                for j in range(i + 1, len(sorted_atoms)):
                    pair = (sorted_atoms[i], sorted_atoms[j])
                    self.cooccurrence[pair] += 1
    
    def get_atom_stats(self, atom_id: str) -> Optional[AtomStats]:
        """Atom統計を取得"""
        if atom_id in self.atom_stats:
            return self.atom_stats[atom_id]
        return None
    
    def get_direction_stats(self, direction: str) -> Optional[DirectionStats]:
        """Direction統計を取得"""
        if direction in self.direction_stats:
            return self.direction_stats[direction]
        return None
    
    def get_all_atoms(self) -> Set[str]:
        """記録されている全Atom IDを取得"""
        return set(self.atom_stats.keys())
    
    def snapshot(self) -> dict:
        """現在の状態のスナップショットを取得（デバッグ・テスト用）"""
        return {
            "last_seq": self.last_seq,
            "total_events": self.total_events,
            "atom_count": len(self.atom_stats),
            "direction_counts": {
                d: s.count_total for d, s in self.direction_stats.items()
            },
            "cooccurrence_pairs": len(self.cooccurrence)
        }
    
    def __repr__(self) -> str:
        return (
            f"SemanticIndex(last_seq={self.last_seq}, "
            f"atoms={len(self.atom_stats)}, "
            f"events={self.total_events})"
        )
