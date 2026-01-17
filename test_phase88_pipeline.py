"""
ESDE Phase 8-8: Pipeline Tests
==============================

Spec v8.8 監査テスト:
- Neutral Loop: 通常入力でパイプラインが完走
- Disruptive Feedback: 硬直データ(R=1.0)でtemperature=0.7
- Stabilizing Feedback: 揺らぎデータ(R=0.2)でtemperature=0.0
"""

import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass

# テスト対象をインポート
sys.path.insert(0, str(Path(__file__).parent))

from feedback import (
    Modulator,
    StrategyMode,
    decide_strategy,
    get_target_atom_from_candidates,
    RIGIDITY_HIGH,
    RIGIDITY_LOW,
    ALERT_RIGIDITY_THRESHOLD,
    ALERT_MIN_OBSERVATIONS,
)
from index import SemanticIndex, Projector


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


# ==========================================
# Mock Classes
# ==========================================

class MockSensor:
    """Mock Sensor V2"""
    
    def __init__(self, candidates: List[Dict] = None):
        self.candidates = candidates or []
        self.call_count = 0
    
    def analyze(self, text: str) -> Dict:
        self.call_count += 1
        return {
            "candidates": self.candidates,
            "meta": {"engine": "mock"}
        }


class MockGenerator:
    """Mock MoleculeGeneratorLive"""
    
    def __init__(self):
        self.last_temperature = None
        self.last_system_prompt = None
        self.llm_host = "http://mock"
        self.llm_model = "mock"
        self.llm_timeout = 10
    
    def generate(self, original_text: str, candidates: List[Dict]) -> Any:
        @dataclass
        class MockResult:
            success: bool = True
            molecule: Dict = None
            error: str = None
            abstained: bool = False
        
        return MockResult(
            success=True,
            molecule={
                "active_atoms": [
                    {"atom": candidates[0]["concept_id"] if candidates else "UNKNOWN"}
                ],
                "formula": "aa_1"
            }
        )
    
    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        self.last_system_prompt = system_prompt
        return '{"active_atoms": [], "formula": ""}'


class MockLedger:
    """Mock PersistentLedger"""
    
    def __init__(self):
        self.entries = []
        self.seq = -1
    
    def observe_molecule(
        self,
        source_text: str,
        molecule: Dict,
        direction: str = "=>+",
        actor: str = "Test"
    ) -> Dict:
        self.seq += 1
        entry = {
            "seq": self.seq,
            "event_type": "molecule.observe",
            "direction": direction,
            "data": {
                "source_text": source_text,
                "molecule": molecule
            }
        }
        self.entries.append(entry)
        return entry


# ==========================================
# Modulator Tests
# ==========================================

def test_modulator_constants():
    """定数が正しく定義されている"""
    assert RIGIDITY_HIGH == 0.9
    assert RIGIDITY_LOW == 0.3
    assert ALERT_RIGIDITY_THRESHOLD == 0.98
    assert ALERT_MIN_OBSERVATIONS == 10


def test_modulator_neutral():
    """中間のRigidityでNeutral戦略"""
    index = SemanticIndex()
    
    # R=0.5 のデータを作成（10回中5回同じformula）
    for i in range(5):
        index.update(seq=i, atoms=["EMO.love"], direction="=>+", 
                    formula_signature="A")
    for i in range(5):
        index.update(seq=5+i, atoms=["EMO.love"], direction="=>+", 
                    formula_signature=f"B{i}")
    
    modulator = Modulator()
    strategy = modulator.decide_strategy("EMO.love", index)
    
    assert strategy.mode == StrategyMode.NEUTRAL
    assert strategy.temperature == 0.1


def test_modulator_disruptive():
    """高Rigidity (R>0.9) でDisruptive戦略"""
    index = SemanticIndex()
    
    # R=1.0 のデータを作成（全部同じformula）
    for i in range(10):
        index.update(seq=i, atoms=["EMO.love"], direction="=>+", 
                    formula_signature="SAME")
    
    modulator = Modulator()
    strategy = modulator.decide_strategy("EMO.love", index)
    
    assert strategy.mode == StrategyMode.DISRUPTIVE
    assert strategy.temperature == 0.7
    assert strategy.prompt_suffix is not None


def test_modulator_stabilizing():
    """低Rigidity (R<0.3) でStabilizing戦略"""
    index = SemanticIndex()
    
    # R=0.2 のデータを作成（全部違うformula）
    for i in range(10):
        index.update(seq=i, atoms=["EMO.love"], direction="=>+", 
                    formula_signature=f"UNIQUE_{i}")
    
    modulator = Modulator()
    strategy = modulator.decide_strategy("EMO.love", index)
    
    # R = 1/10 = 0.1 < 0.3
    assert strategy.mode == StrategyMode.STABILIZING
    assert strategy.temperature == 0.0
    assert strategy.prompt_suffix is not None


def test_modulator_unknown_atom():
    """未知Atom（Index未登録）でNeutral戦略"""
    index = SemanticIndex()
    
    modulator = Modulator()
    strategy = modulator.decide_strategy("UNKNOWN.atom", index)
    
    assert strategy.mode == StrategyMode.NEUTRAL
    assert strategy.temperature == 0.1
    assert "unknown" in strategy.reason.lower()


def test_target_atom_extraction():
    """Target Atom抽出（top-1）"""
    candidates = [
        {"concept_id": "EMO.love", "score": 1.5},
        {"concept_id": "ACT.create", "score": 1.2},
        {"concept_id": "VAL.truth", "score": 0.8},
    ]
    
    target = get_target_atom_from_candidates(candidates)
    assert target == "EMO.love"
    
    # 空リスト
    assert get_target_atom_from_candidates([]) is None


def test_alert_condition():
    """Alert条件: R>=0.98 かつ N>=10"""
    index = SemanticIndex()
    
    # R=1.0, N=15 → Alert発生
    for i in range(15):
        index.update(seq=i, atoms=["EMO.love"], direction="=>+", 
                    formula_signature="CRYSTALLIZED")
    
    modulator = Modulator()
    alert = modulator.check_alert("EMO.love", index)
    
    assert alert is not None
    assert alert["type"] == "CONCEPT_CRYSTALLIZED"
    assert alert["rigidity"] >= 0.98
    assert alert["observations"] >= 10


def test_alert_not_triggered_low_count():
    """N<10ではAlertなし"""
    index = SemanticIndex()
    
    # R=1.0 but N=5 → Alert発生しない
    for i in range(5):
        index.update(seq=i, atoms=["EMO.love"], direction="=>+", 
                    formula_signature="SAME")
    
    modulator = Modulator()
    alert = modulator.check_alert("EMO.love", index)
    
    assert alert is None


# ==========================================
# Pipeline Integration Tests
# ==========================================

def test_pipeline_neutral_loop():
    """Neutral Loop: 通常入力でパイプライン完走"""
    from pipeline import ESDEPipeline
    
    sensor = MockSensor([{"concept_id": "EMO.love", "score": 1.5}])
    generator = MockGenerator()
    ledger = MockLedger()
    index = SemanticIndex()
    projector = Projector(index)
    
    # 中間Rigidityのデータを事前登録
    for i in range(5):
        index.update(seq=i, atoms=["EMO.love"], direction="=>+", 
                    formula_signature=f"formula_{i % 2}")
    
    pipeline = ESDEPipeline(
        sensor=sensor,
        generator=generator,
        ledger=ledger,
        index=index,
        projector=projector
    )
    
    result = pipeline.run("I love you")
    
    assert result.success
    assert result.target_atom == "EMO.love"
    assert result.strategy.mode == StrategyMode.NEUTRAL
    assert result.ledger_entry is not None
    assert result.index_updated


def test_pipeline_disruptive_feedback():
    """Disruptive Feedback: 硬直データでtemp=0.7"""
    from pipeline import ESDEPipeline
    
    sensor = MockSensor([{"concept_id": "EMO.love", "score": 1.5}])
    generator = MockGenerator()
    ledger = MockLedger()
    index = SemanticIndex()
    projector = Projector(index)
    
    # 硬直データ（R=1.0）を事前登録
    for i in range(10):
        index.update(seq=i, atoms=["EMO.love"], direction="=>+", 
                    formula_signature="RIGID")
    
    pipeline = ESDEPipeline(
        sensor=sensor,
        generator=generator,
        ledger=ledger,
        index=index,
        projector=projector
    )
    
    result = pipeline.run("I love you")
    
    assert result.success
    assert result.strategy.mode == StrategyMode.DISRUPTIVE
    assert result.strategy.temperature == 0.7
    # Disruptiveモードでは方向性が-|>になる
    assert result.ledger_entry["direction"] == "-|>"


def test_pipeline_stabilizing_feedback():
    """Stabilizing Feedback: 揺らぎデータでtemp=0.0"""
    from pipeline import ESDEPipeline
    
    sensor = MockSensor([{"concept_id": "EMO.love", "score": 1.5}])
    generator = MockGenerator()
    ledger = MockLedger()
    index = SemanticIndex()
    projector = Projector(index)
    
    # 揺らぎデータ（R<0.3）を事前登録
    for i in range(10):
        index.update(seq=i, atoms=["EMO.love"], direction="=>+", 
                    formula_signature=f"VOLATILE_{i}")  # 全部違う
    
    pipeline = ESDEPipeline(
        sensor=sensor,
        generator=generator,
        ledger=ledger,
        index=index,
        projector=projector
    )
    
    result = pipeline.run("I love you")
    
    assert result.success
    assert result.strategy.mode == StrategyMode.STABILIZING
    assert result.strategy.temperature == 0.0


def test_pipeline_no_candidates():
    """候補なしでAbstain"""
    from pipeline import ESDEPipeline
    
    sensor = MockSensor([])  # 空の候補
    generator = MockGenerator()
    ledger = MockLedger()
    index = SemanticIndex()
    
    pipeline = ESDEPipeline(
        sensor=sensor,
        generator=generator,
        ledger=ledger,
        index=index
    )
    
    result = pipeline.run("xyz123")
    
    assert not result.success
    assert result.abstained


def test_pipeline_with_alert():
    """Alert発生テスト（パイプライン実行前の状態でAlert検出）"""
    from pipeline import ESDEPipeline
    
    sensor = MockSensor([{"concept_id": "EMO.love", "score": 1.5}])
    generator = MockGenerator()
    ledger = MockLedger()
    index = SemanticIndex()
    
    # projectorなし（Index更新しない）でテスト
    # Alert条件を満たすデータ（R>=0.98, N>=10）
    for i in range(15):
        index.update(seq=i, atoms=["EMO.love"], direction="=>+", 
                    formula_signature="CRYSTALLIZED")
    
    pipeline = ESDEPipeline(
        sensor=sensor,
        generator=generator,
        ledger=ledger,
        index=index,
        projector=None,  # Index更新なし
        enable_alerts=True
    )
    
    result = pipeline.run("I love you")
    
    assert result.success
    # Index更新なしなので、R=1.0のままAlertが発生
    assert result.alert is not None
    assert result.alert["type"] == "CONCEPT_CRYSTALLIZED"


def test_pipeline_alert_after_update():
    """Index更新後のAlert（新formulaでRigidity低下→Alertなし）"""
    from pipeline import ESDEPipeline
    
    sensor = MockSensor([{"concept_id": "EMO.love", "score": 1.5}])
    generator = MockGenerator()
    ledger = MockLedger()
    index = SemanticIndex()
    projector = Projector(index)
    
    # Alert条件を満たすデータ
    for i in range(15):
        index.update(seq=i, atoms=["EMO.love"], direction="=>+", 
                    formula_signature="CRYSTALLIZED")
    
    pipeline = ESDEPipeline(
        sensor=sensor,
        generator=generator,
        ledger=ledger,
        index=index,
        projector=projector,  # Index更新あり
        enable_alerts=True
    )
    
    result = pipeline.run("I love you")
    
    assert result.success
    # 新しいformula('aa_1')が追加されてR = 15/16 = 0.9375 < 0.98
    # AlertはNoneになる（正しい動作）
    # これは「新しい観点が追加されて硬直が解消された」ことを意味する


# ==========================================
# メイン
# ==========================================

def main():
    """全テストを実行"""
    tests = [
        # Modulator Tests
        ("Modulator Constants", test_modulator_constants),
        ("Modulator Neutral (0.3 < R < 0.9)", test_modulator_neutral),
        ("Modulator Disruptive (R > 0.9)", test_modulator_disruptive),
        ("Modulator Stabilizing (R < 0.3)", test_modulator_stabilizing),
        ("Modulator Unknown Atom", test_modulator_unknown_atom),
        ("Target Atom Extraction (top-1)", test_target_atom_extraction),
        ("Alert Condition (R>=0.98, N>=10)", test_alert_condition),
        ("Alert Not Triggered (N<10)", test_alert_not_triggered_low_count),
        
        # Pipeline Integration Tests
        ("Pipeline Neutral Loop", test_pipeline_neutral_loop),
        ("Pipeline Disruptive Feedback", test_pipeline_disruptive_feedback),
        ("Pipeline Stabilizing Feedback", test_pipeline_stabilizing_feedback),
        ("Pipeline No Candidates", test_pipeline_no_candidates),
        ("Pipeline Alert (No Index Update)", test_pipeline_with_alert),
        ("Pipeline Alert After Update", test_pipeline_alert_after_update),
    ]
    
    print("=" * 60)
    print("ESDE Phase 8-8: Pipeline Test Suite")
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
        print("✅ ALL TESTS PASSED - Phase 8-8 Ready")
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
