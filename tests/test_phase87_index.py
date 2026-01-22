"""
ESDE Phase 8-7: Index Tests
===========================

Spec v8.7.1 監査テスト:
- Parity Test: rebuild結果 == 逐次更新結果
- Rigidity Check: 特定パターンで期待通りのR値
- No Side Effect: Index構築でL1が変更されない
"""

import os
import sys
import json
import tempfile
import hashlib
from pathlib import Path
from copy import deepcopy

# テスト対象をインポート
sys.path.insert(0, str(Path(__file__).parent))

from ledger import PersistentLedger
from index import (
    SemanticIndex,
    Projector,
    QueryAPI,
    create_index_from_ledger,
    extract_formula_signature,
    calculate_rigidity,
    classify_rigidity,
)


class TestResult:
    """テスト結果"""
    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.error = None
    
    def __str__(self):
        status = "✅ PASS" if self.passed else f"❌ FAIL: {self.error}"
        return f"{self.name}: {status}"


def run_test(name: str, test_func) -> TestResult:
    """テストを実行"""
    result = TestResult(name)
    try:
        test_func()
        result.passed = True
    except AssertionError as e:
        result.error = str(e)
    except Exception as e:
        result.error = f"{type(e).__name__}: {e}"
    return result


def make_test_molecule(atom_id: str, formula: str = None):
    """テスト用の意味分子を作成"""
    return {
        "active_atoms": [
            {"atom": atom_id, "axis": "resonance", "level": "continuation"}
        ],
        "formula": formula or f"{atom_id} ▷ ◯",
        "operators": ["▷"],
        "coordinate": {"axis": "interconnection", "level": "catalytic"}
    }


def make_multi_atom_molecule(atom_ids: list, formula: str = None):
    """複数Atomを含む意味分子を作成"""
    return {
        "active_atoms": [
            {"atom": aid, "axis": "resonance", "level": "continuation"}
            for aid in atom_ids
        ],
        "formula": formula or " × ".join(atom_ids),
        "operators": ["×"],
        "coordinate": {"axis": "interconnection", "level": "catalytic"}
    }


# ============================================================
# 基本テスト
# ============================================================

def test_index_creation():
    """SemanticIndexが正しく作成される"""
    index = SemanticIndex()
    assert index.last_seq == -1
    assert index.total_events == 0
    assert len(index.atom_stats) == 0


def test_index_update():
    """Index更新が正しく動作する"""
    index = SemanticIndex()
    
    index.update(
        seq=1,
        atoms=["EMO.love"],
        direction="=>+",
        formula_signature="EMO.love ▷ ◯"
    )
    
    assert index.last_seq == 1
    assert index.total_events == 1
    assert "EMO.love" in index.atom_stats
    
    stats = index.atom_stats["EMO.love"]
    assert stats.count_total == 1
    assert stats.last_seen_seq == 1


def test_formula_signature_extraction():
    """formula_signatureの抽出が正しく動作する"""
    # 1. formulaがある場合
    mol1 = {"formula": "EMO.love ▷ ◯"}
    assert extract_formula_signature(mol1) == "EMO.love ▷ ◯"
    
    # 2. formulaがなくatoms/operatorsがある場合
    mol2 = {
        "active_atoms": [{"atom": "EMO.love"}, {"atom": "ACT.create"}],
        "operators": ["×"]
    }
    sig2 = extract_formula_signature(mol2)
    assert "ACT.create" in sig2
    assert "EMO.love" in sig2
    assert "::" in sig2
    
    # 3. どちらもない場合
    mol3 = {}
    assert extract_formula_signature(mol3) == "__unknown__"
    
    # 4. Noneの場合
    assert extract_formula_signature(None) == "__unknown__"


# ============================================================
# Parity Test (最重要)
# ============================================================

def test_parity_rebuild_vs_incremental():
    """
    Parity Test: rebuild結果 == 逐次更新結果
    
    「起動時に全件Rebuildした状態」と「逐次Updateした状態」が
    完全に一致することを検証する。
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.jsonl"
        
        # Ledgerを作成してデータを追加
        ledger = PersistentLedger(str(path))
        
        molecules = [
            make_test_molecule("EMO.love", "EMO.love ▷ ◯"),
            make_test_molecule("EMO.love", "EMO.love ▷ ◯"),  # 同じformula
            make_test_molecule("EMO.love", "EMO.love → ◯"),  # 違うformula
            make_test_molecule("ACT.create", "ACT.create ▷ ◯"),
            make_multi_atom_molecule(["EMO.love", "ACT.create"], "EMO.love × ACT.create"),
        ]
        
        directions = ["=>+", "=>+", "=>", "-|>", "=>+"]
        
        entries = []
        for mol, direction in zip(molecules, directions):
            entry = ledger.observe_molecule(
                source_text="test",
                molecule=mol,
                direction=direction
            )
            entries.append(entry)
        
        # 方法1: Rebuildで構築
        index_rebuild = SemanticIndex()
        projector_rebuild = Projector(index_rebuild)
        projector_rebuild.rebuild_from_ledger(ledger)
        
        # 方法2: 逐次更新で構築
        index_incremental = SemanticIndex()
        projector_incremental = Projector(index_incremental)
        for entry in entries:
            projector_incremental.on_event(entry)
        
        # 一致検証
        # 1. 基本メタ情報
        assert index_rebuild.last_seq == index_incremental.last_seq, \
            f"last_seq mismatch: {index_rebuild.last_seq} vs {index_incremental.last_seq}"
        assert index_rebuild.total_events == index_incremental.total_events, \
            f"total_events mismatch"
        
        # 2. Atom統計
        assert set(index_rebuild.atom_stats.keys()) == set(index_incremental.atom_stats.keys()), \
            "atom_stats keys mismatch"
        
        for atom_id in index_rebuild.atom_stats.keys():
            stats_r = index_rebuild.atom_stats[atom_id]
            stats_i = index_incremental.atom_stats[atom_id]
            
            assert stats_r.count_total == stats_i.count_total, \
                f"{atom_id} count_total mismatch"
            assert stats_r.last_seen_seq == stats_i.last_seen_seq, \
                f"{atom_id} last_seen_seq mismatch"
            assert dict(stats_r.formula_counts) == dict(stats_i.formula_counts), \
                f"{atom_id} formula_counts mismatch"
        
        # 3. Direction統計
        for direction in ["=>", "=>+", "-|>"]:
            stats_r = index_rebuild.direction_stats.get(direction)
            stats_i = index_incremental.direction_stats.get(direction)
            
            if stats_r and stats_i:
                assert stats_r.count_total == stats_i.count_total, \
                    f"direction {direction} count_total mismatch"
        
        # 4. 共起
        assert dict(index_rebuild.cooccurrence) == dict(index_incremental.cooccurrence), \
            "cooccurrence mismatch"


# ============================================================
# Rigidity Test
# ============================================================

def test_rigidity_constant_formula():
    """常に同じformulaで出現するAtomはR=1.0"""
    index = SemanticIndex()
    
    # 10回同じformulaで観測
    for i in range(10):
        index.update(
            seq=i,
            atoms=["EMO.love"],
            direction="=>+",
            formula_signature="EMO.love ▷ ◯"  # 常に同じ
        )
    
    stats = index.atom_stats["EMO.love"]
    r = calculate_rigidity(stats)
    
    assert r == 1.0, f"Expected R=1.0 for constant formula, got {r}"
    assert classify_rigidity(r) == "crystallized"


def test_rigidity_varying_formula():
    """異なるformulaで出現するAtomはR<1.0"""
    index = SemanticIndex()
    
    # 5種類のformulaで観測（各1回）
    formulas = [
        "EMO.love ▷ ◯",
        "EMO.love → ◯",
        "EMO.love × ACT.create",
        "EMO.love ⊕ VAL.truth",
        "EMO.love | COND",
    ]
    
    for i, formula in enumerate(formulas):
        index.update(
            seq=i,
            atoms=["EMO.love"],
            direction="=>+",
            formula_signature=formula
        )
    
    stats = index.atom_stats["EMO.love"]
    r = calculate_rigidity(stats)
    
    # N_mode=1, N_total=5 → R=0.2
    assert r == 0.2, f"Expected R=0.2 for uniform distribution, got {r}"
    assert classify_rigidity(r) == "fluid"


def test_rigidity_mixed():
    """混合パターン: 7回同じ + 3回異なる = R=0.7"""
    index = SemanticIndex()
    
    # 7回同じformula
    for i in range(7):
        index.update(
            seq=i,
            atoms=["EMO.love"],
            direction="=>+",
            formula_signature="EMO.love ▷ ◯"
        )
    
    # 3回異なるformula
    for i, formula in enumerate(["A", "B", "C"]):
        index.update(
            seq=7 + i,
            atoms=["EMO.love"],
            direction="=>+",
            formula_signature=formula
        )
    
    stats = index.atom_stats["EMO.love"]
    r = calculate_rigidity(stats)
    
    # N_mode=7, N_total=10 → R=0.7
    assert r == 0.7, f"Expected R=0.7, got {r}"
    assert classify_rigidity(r) == "rigid"


# ============================================================
# No Side Effect Test
# ============================================================

def test_no_side_effect_on_ledger():
    """Index構築がL1 (JSONL) を変更しない"""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.jsonl"
        
        # Ledgerを作成
        ledger = PersistentLedger(str(path))
        
        for i in range(5):
            ledger.observe_molecule(
                source_text=f"test {i}",
                molecule=make_test_molecule("EMO.love"),
                direction="=>+"
            )
        
        # ファイルのハッシュを記録
        with open(path, 'rb') as f:
            hash_before = hashlib.sha256(f.read()).hexdigest()
        
        # Indexを構築
        index = create_index_from_ledger(ledger)
        
        # 再度ハッシュを確認
        with open(path, 'rb') as f:
            hash_after = hashlib.sha256(f.read()).hexdigest()
        
        assert hash_before == hash_after, "Ledger file was modified during Index build!"


# ============================================================
# QueryAPI Test
# ============================================================

def test_query_api_basic():
    """QueryAPIの基本動作"""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.jsonl"
        
        ledger = PersistentLedger(str(path))
        
        # データ追加
        for i in range(10):
            ledger.observe_molecule(
                source_text=f"test {i}",
                molecule=make_test_molecule("EMO.love", "EMO.love ▷ ◯"),
                direction="=>+" if i % 2 == 0 else "=>"
            )
        
        # API作成
        api = QueryAPI(ledger)
        
        # get_frequency
        freq = api.get_frequency("EMO.love")
        assert freq == 10, f"Expected frequency=10, got {freq}"
        
        # get_rigidity
        r = api.get_rigidity("EMO.love")
        assert r == 1.0, f"Expected rigidity=1.0, got {r}"
        
        # get_recent_directions
        balance = api.get_recent_directions()
        assert balance.total == 10
        assert balance.creative == 5  # =>+
        assert balance.neutral == 5   # =>


def test_query_api_atom_info():
    """QueryAPI.get_atom_infoの動作"""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.jsonl"
        
        ledger = PersistentLedger(str(path))
        
        ledger.observe_molecule(
            source_text="test",
            molecule=make_test_molecule("EMO.love", "EMO.love ▷ ◯"),
            direction="=>+"
        )
        
        api = QueryAPI(ledger)
        
        info = api.get_atom_info("EMO.love")
        assert info is not None
        assert info.atom_id == "EMO.love"
        assert info.frequency == 1
        assert info.rigidity == 1.0
        assert info.rigidity_class == "crystallized"
        assert info.mode_formula == "EMO.love ▷ ◯"


def test_query_api_cooccurrence():
    """QueryAPI.get_cooccurrenceの動作"""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.jsonl"
        
        ledger = PersistentLedger(str(path))
        
        # 共起するmolecule
        for _ in range(3):
            ledger.observe_molecule(
                source_text="test",
                molecule=make_multi_atom_molecule(["EMO.love", "ACT.create"]),
                direction="=>+"
            )
        
        api = QueryAPI(ledger)
        
        cooc = api.get_cooccurrence("EMO.love", "ACT.create")
        assert cooc == 3, f"Expected cooccurrence=3, got {cooc}"
        
        # 逆順でも同じ結果
        cooc_rev = api.get_cooccurrence("ACT.create", "EMO.love")
        assert cooc_rev == 3


def test_query_api_window():
    """window引数が機能する"""
    index = SemanticIndex(seq_history_limit=100)
    
    # 大量のイベント
    for i in range(50):
        index.update(
            seq=i,
            atoms=["EMO.love"],
            direction="=>+",
            formula_signature=f"formula_{i % 5}"  # 5種類のformula
        )
    
    api = QueryAPI(index=index)
    
    # 全件
    freq_all = api.get_frequency("EMO.love", window=1000)
    assert freq_all == 50
    
    # window=10
    freq_10 = api.get_frequency("EMO.love", window=10)
    assert freq_10 <= 10


# ============================================================
# Direction Balance Test
# ============================================================

def test_direction_balance():
    """方向性バランスの計算"""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.jsonl"
        
        ledger = PersistentLedger(str(path))
        
        # =>+ : 6回, => : 3回, -|> : 1回
        directions = ["=>+"] * 6 + ["=>"] * 3 + ["-|>"] * 1
        
        for i, d in enumerate(directions):
            ledger.observe_molecule(
                source_text=f"test {i}",
                molecule=make_test_molecule("EMO.love"),
                direction=d
            )
        
        api = QueryAPI(ledger)
        balance = api.get_recent_directions()
        
        assert balance.total == 10
        assert balance.creative == 6
        assert balance.neutral == 3
        assert balance.destructive == 1
        assert abs(balance.r_creative - 0.6) < 0.01
        assert abs(balance.r_destructive - 0.1) < 0.01


# ============================================================
# メイン
# ============================================================

def main():
    """全テストを実行"""
    tests = [
        # 基本テスト
        ("Index Creation", test_index_creation),
        ("Index Update", test_index_update),
        ("Formula Signature Extraction", test_formula_signature_extraction),
        
        # Parity Test (最重要)
        ("Parity: Rebuild vs Incremental", test_parity_rebuild_vs_incremental),
        
        # Rigidity Test
        ("Rigidity: Constant Formula (R=1.0)", test_rigidity_constant_formula),
        ("Rigidity: Varying Formula (R<1.0)", test_rigidity_varying_formula),
        ("Rigidity: Mixed Pattern (R=0.7)", test_rigidity_mixed),
        
        # No Side Effect Test
        ("No Side Effect on Ledger", test_no_side_effect_on_ledger),
        
        # QueryAPI Test
        ("QueryAPI Basic", test_query_api_basic),
        ("QueryAPI AtomInfo", test_query_api_atom_info),
        ("QueryAPI Cooccurrence", test_query_api_cooccurrence),
        ("QueryAPI Window", test_query_api_window),
        
        # Direction Balance Test
        ("Direction Balance", test_direction_balance),
    ]
    
    print("=" * 60)
    print("ESDE Phase 8-7: Index Test Suite")
    print("=" * 60)
    print()
    
    results = []
    for name, func in tests:
        result = run_test(name, func)
        results.append(result)
        print(result)
    
    print()
    print("=" * 60)
    
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    
    print(f"Results: {passed}/{total} passed")
    
    if passed == total:
        print("✅ ALL TESTS PASSED - Phase 8-7 Ready")
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
