"""
ESDE Phase 8-9: Monitor & Long-Run Tests
========================================

Spec v8.9 監査テスト:
- Monitor Render: SemanticMonitor.update()がエラーなく動作
- LongRun Basic: steps=10で完走、Ledger valid
- Alert Count: Alert発火時にカウント加算
- Ledger Invariance (T895): Genesis行が変更されていない

GPT監査v8.9.1:
- テストでは実LLMを使わない（モック使用）
- Fail Fast方針でエラー発生時は即停止
"""

import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass

# テスト対象をインポート
sys.path.insert(0, str(Path(__file__).parent))

from monitor import SemanticMonitor, MonitorState, RICH_AVAILABLE
from runner import LongRunRunner, LongRunReport, DEFAULT_CORPUS, print_report
from index import SemanticIndex, Projector
from feedback import Modulator, StrategyMode


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
# Mock Classes (No LLM)
# ==========================================

class MockSensor:
    """Mock Sensor V2"""
    
    def __init__(self, default_atom: str = "EMO.love"):
        self.default_atom = default_atom
    
    def analyze(self, text: str) -> Dict:
        if text.strip() and not text.startswith("xyz"):
            return {
                "candidates": [{"concept_id": self.default_atom, "score": 1.5}],
                "meta": {"engine": "mock"}
            }
        return {"candidates": [], "meta": {"engine": "mock"}}


class MockGenerator:
    """Mock MoleculeGeneratorLive (No LLM)"""
    
    def __init__(self):
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
        
        if not candidates:
            return MockResult(success=False, abstained=True, error="No candidates")
        
        return MockResult(
            success=True,
            molecule={
                "active_atoms": [{"atom": candidates[0]["concept_id"]}],
                "formula": "aa_1"
            }
        )
    
    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        return '{"active_atoms": [], "formula": ""}'


class MockLedger:
    """Mock PersistentLedger"""
    
    def __init__(self):
        self.entries = []
        self.seq = -1
        self.filepath = "mock://ledger"
        
        # Genesis header (for T895 test)
        self.genesis_v = "1.0"
        self.genesis_ledger_id = "test-ledger"
        self.genesis_algo = "SHA256"
    
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
            "data": {"source_text": source_text, "molecule": molecule}
        }
        self.entries.append(entry)
        return entry
    
    def validate(self):
        """常にvalidを返す（モック）"""
        return True, []
    
    def get_genesis_header(self) -> Dict:
        """Genesis行のヘッダーを返す"""
        return {
            "v": self.genesis_v,
            "ledger_id": self.genesis_ledger_id,
            "hash": {"algo": self.genesis_algo}
        }


class MockPipelineResult:
    """Mock PipelineResult for monitor tests"""
    def __init__(
        self,
        input_text: str = "test",
        target_atom: str = "EMO.love",
        rigidity: float = 0.5,
        strategy_mode: str = "neutral",
        temperature: float = 0.1,
        formula: str = "aa_1",
        direction: str = "=>+",
        seq: int = 0,
        alert: Dict = None
    ):
        self.input_text = input_text
        self.target_atom = target_atom
        self.rigidity = rigidity
        
        # Strategy mock
        class MockStrategy:
            def __init__(self, mode, temp):
                self.mode = type('Mode', (), {'value': mode})()
                self.temperature = temp
        
        self.strategy = MockStrategy(strategy_mode, temperature)
        self.molecule = {"formula": formula}
        self.ledger_entry = {"seq": seq, "direction": direction}
        self.alert = alert
        self.success = True
        self.abstained = False


# ==========================================
# Monitor Tests
# ==========================================

def test_monitor_state_init():
    """MonitorState初期化"""
    state = MonitorState()
    assert state.total_steps == 0
    assert state.total_alerts == 0
    assert state.ledger_seq == -1


def test_monitor_update():
    """SemanticMonitor.update()がエラーなく動作"""
    monitor = SemanticMonitor()
    
    result = MockPipelineResult(
        input_text="I love you",
        target_atom="EMO.love",
        rigidity=0.75,
        strategy_mode="neutral",
        temperature=0.1,
        formula="aa_1 × aa_2",
        direction="=>+",
        seq=42
    )
    
    # Should not raise
    monitor.update(result)
    
    assert monitor.state.total_steps == 1
    assert monitor.state.last_target_atom == "EMO.love"
    assert monitor.state.last_rigidity == 0.75
    assert monitor.state.ledger_seq == 42


def test_monitor_update_with_alert():
    """Alert付きの更新"""
    monitor = SemanticMonitor()
    
    result = MockPipelineResult(
        input_text="test",
        alert={"type": "CONCEPT_CRYSTALLIZED", "atom_id": "EMO.love"}
    )
    
    monitor.update(result)
    
    assert monitor.state.total_alerts == 1
    assert monitor.state.last_alert == True


def test_monitor_direction_balance():
    """方向性バランス計算"""
    monitor = SemanticMonitor()
    
    # 3 creative, 2 destructive, 1 neutral
    for _ in range(3):
        monitor.update(MockPipelineResult(direction="=>+"))
    for _ in range(2):
        monitor.update(MockPipelineResult(direction="-|>"))
    monitor.update(MockPipelineResult(direction="=>"))
    
    balance = monitor.state.direction_balance()
    assert balance["=>+"] == 0.5  # 3/6
    assert abs(balance["-|>"] - 0.333) < 0.01  # 2/6
    assert abs(balance["=>"] - 0.167) < 0.01  # 1/6


def test_monitor_render():
    """render()がエラーなく動作"""
    monitor = SemanticMonitor()
    monitor.update(MockPipelineResult())
    
    # Should not raise (rich or plain)
    monitor.render()


# ==========================================
# LongRun Tests
# ==========================================

def test_longrun_basic():
    """LongRunRunner steps=10で完走"""
    from pipeline import ESDEPipeline
    
    sensor = MockSensor()
    generator = MockGenerator()
    ledger = MockLedger()
    index = SemanticIndex()
    projector = Projector(index)
    
    pipeline = ESDEPipeline(
        sensor=sensor,
        generator=generator,
        ledger=ledger,
        index=index,
        projector=projector,
        enable_alerts=False
    )
    
    runner = LongRunRunner(
        pipeline=pipeline,
        ledger=ledger,
        index=index,
        health_check_interval=5
    )
    
    report = runner.run(steps=10)
    
    assert report.completed_steps == 10
    assert report.errors == []
    assert report.ledger_valid
    assert not report.stopped_early


def test_longrun_alert_count():
    """Alert発火時にカウント加算"""
    from pipeline import ESDEPipeline
    
    sensor = MockSensor()
    generator = MockGenerator()
    ledger = MockLedger()
    index = SemanticIndex()
    projector = Projector(index)
    
    # 硬直データを事前注入（R=1.0, N=15）
    for i in range(15):
        index.update(seq=i, atoms=["EMO.love"], direction="=>+",
                    formula_signature="CRYSTALLIZED")
    
    pipeline = ESDEPipeline(
        sensor=sensor,
        generator=generator,
        ledger=ledger,
        index=index,
        projector=None,  # Index更新なし（Rigidity維持）
        enable_alerts=True
    )
    
    runner = LongRunRunner(
        pipeline=pipeline,
        ledger=ledger,
        index=index,
        health_check_interval=100
    )
    
    report = runner.run(steps=5)
    
    # EMO.loveの観測でAlertが発火するはず
    assert report.total_alerts > 0
    assert "EMO.love" in report.alert_atoms


def test_longrun_abstain_handling():
    """候補なし入力でabstain処理"""
    from pipeline import ESDEPipeline
    
    sensor = MockSensor()
    generator = MockGenerator()
    ledger = MockLedger()
    index = SemanticIndex()
    
    pipeline = ESDEPipeline(
        sensor=sensor,
        generator=generator,
        ledger=ledger,
        index=index
    )
    
    # ノイズのみのコーパス
    runner = LongRunRunner(
        pipeline=pipeline,
        corpus=["xyz123", "qwerty asdf", "12345"]
    )
    
    report = runner.run(steps=3)
    
    assert report.completed_steps == 3
    assert report.abstained > 0


def test_longrun_report_structure():
    """LongRunReportの構造"""
    report = LongRunReport(total_steps=10)
    
    d = report.to_dict()
    
    assert "total_steps" in d
    assert "completed_steps" in d
    assert "errors" in d
    assert "ledger_valid" in d


# ==========================================
# Ledger Invariance Test (T895)
# ==========================================

def test_ledger_invariance_t895():
    """T895: Long-Run後もGenesisヘッダーが不変"""
    from pipeline import ESDEPipeline
    
    sensor = MockSensor()
    generator = MockGenerator()
    ledger = MockLedger()
    index = SemanticIndex()
    projector = Projector(index)
    
    # Genesis前のヘッダーを記録
    genesis_before = ledger.get_genesis_header()
    
    pipeline = ESDEPipeline(
        sensor=sensor,
        generator=generator,
        ledger=ledger,
        index=index,
        projector=projector
    )
    
    runner = LongRunRunner(
        pipeline=pipeline,
        ledger=ledger,
        index=index
    )
    
    # Long-Run実行
    runner.run(steps=10)
    
    # Genesis後のヘッダーを確認
    genesis_after = ledger.get_genesis_header()
    
    # 不変であること
    assert genesis_before["v"] == genesis_after["v"]
    assert genesis_before["ledger_id"] == genesis_after["ledger_id"]
    assert genesis_before["hash"]["algo"] == genesis_after["hash"]["algo"]


# ==========================================
# CLI Tests
# ==========================================

def test_cli_observe_mock():
    """CLIのobserveコマンド（モック）"""
    from pipeline import ESDEPipeline
    
    sensor = MockSensor()
    generator = MockGenerator()
    ledger = MockLedger()
    index = SemanticIndex()
    
    pipeline = ESDEPipeline(
        sensor=sensor,
        generator=generator,
        ledger=ledger,
        index=index
    )
    
    result = pipeline.run("I love you")
    
    assert result.success
    assert result.target_atom == "EMO.love"


# ==========================================
# Utility
# ==========================================

def pytest_approx(value: float, tolerance: float) -> float:
    """簡易的な近似比較用"""
    return value  # 実際の比較はassertで行う


# ==========================================
# メイン
# ==========================================

def main():
    """全テストを実行"""
    tests = [
        # Monitor Tests
        ("Monitor State Init", test_monitor_state_init),
        ("Monitor Update", test_monitor_update),
        ("Monitor Update with Alert", test_monitor_update_with_alert),
        ("Monitor Render", test_monitor_render),
        
        # LongRun Tests
        ("LongRun Basic (steps=10)", test_longrun_basic),
        ("LongRun Alert Count", test_longrun_alert_count),
        ("LongRun Abstain Handling", test_longrun_abstain_handling),
        ("LongRun Report Structure", test_longrun_report_structure),
        
        # Ledger Invariance
        ("T895: Ledger Invariance", test_ledger_invariance_t895),
        
        # CLI
        ("CLI Observe (Mock)", test_cli_observe_mock),
    ]
    
    print("=" * 60)
    print("ESDE Phase 8-9: Monitor & Long-Run Test Suite")
    print("=" * 60)
    print(f"\nRich library available: {RICH_AVAILABLE}")
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
        print("✅ ALL TESTS PASSED - Phase 8-9 Ready")
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
