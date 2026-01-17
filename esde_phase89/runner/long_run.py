"""
ESDE Phase 8-9: Long-Run Runner
===============================

システム統合検証のためのハーネス。
長時間稼働させてもLedger整合性やメモリ消費が維持されることを検証する。

Features:
- Input Cycle: 定義済みコーパスを無限ループで入力
- Health Check: 定期的にLedger.validate()を実行
- Report: 終了時にサマリーを出力

Error Handling Policy (GPT監査v8.9.1):
- Fail Fast: エラー発生時は即座に停止しreportを返す
"""

import time
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field


# ==========================================
# Default Corpus
# ==========================================
DEFAULT_CORPUS = [
    # Emotion
    "I love you",
    "I hate this situation",
    "She feels happy today",
    "He is deeply sad",
    "They are filled with joy",
    
    # Law / Rules
    "The law requires obedience",
    "Rules must be followed",
    "Justice demands equality",
    "Order maintains society",
    
    # Relationships
    "Friends help each other",
    "Family bonds are strong",
    "Trust builds connection",
    
    # Actions
    "Create something new",
    "Destroy the old barriers",
    "Transform your thinking",
    
    # Noise (should produce empty/abstain)
    "xyz123 qwerty",
    "asdfgh jklzxc",
    "12345 67890",
]


# ==========================================
# Long-Run Report
# ==========================================
@dataclass
class LongRunReport:
    """Long-Run実行結果レポート"""
    # Execution
    total_steps: int = 0
    completed_steps: int = 0
    start_time: str = ""
    end_time: str = ""
    duration_seconds: float = 0.0
    
    # Results
    successful_observations: int = 0
    abstained: int = 0
    
    # Alerts
    total_alerts: int = 0
    alert_atoms: List[str] = field(default_factory=list)
    
    # Health
    ledger_valid: bool = True
    validation_checks: int = 0
    
    # Errors (Fail Fast: 最初のエラーで停止)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    stopped_early: bool = False
    
    # Stats snapshot
    final_ledger_seq: int = -1
    final_index_size: int = 0
    direction_balance: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_steps": self.total_steps,
            "completed_steps": self.completed_steps,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_seconds": self.duration_seconds,
            "successful_observations": self.successful_observations,
            "abstained": self.abstained,
            "total_alerts": self.total_alerts,
            "alert_atoms": self.alert_atoms,
            "ledger_valid": self.ledger_valid,
            "validation_checks": self.validation_checks,
            "errors": self.errors,
            "stopped_early": self.stopped_early,
            "final_ledger_seq": self.final_ledger_seq,
            "final_index_size": self.final_index_size,
            "direction_balance": self.direction_balance,
        }


# ==========================================
# Logger
# ==========================================
logger = logging.getLogger("esde.runner.long_run")


# ==========================================
# Long-Run Runner
# ==========================================
class LongRunRunner:
    """
    Long-Run検証ランナー
    
    Usage:
        runner = LongRunRunner(pipeline, ledger, index)
        report = runner.run(steps=100)
    """
    
    def __init__(
        self,
        pipeline: Any,
        ledger: Any = None,
        index: Any = None,
        corpus: List[str] = None,
        health_check_interval: int = 10,
        monitor: Any = None
    ):
        """
        Args:
            pipeline: ESDEPipeline instance
            ledger: PersistentLedger instance (for validation)
            index: SemanticIndex instance (for stats)
            corpus: 入力テキストのリスト（デフォルト: DEFAULT_CORPUS）
            health_check_interval: validate()を実行する間隔
            monitor: SemanticMonitor instance (optional, for live display)
        """
        self.pipeline = pipeline
        self.ledger = ledger
        self.index = index
        self.corpus = corpus or DEFAULT_CORPUS
        self.health_check_interval = health_check_interval
        self.monitor = monitor
        
        self._corpus_index = 0
    
    def run(
        self,
        steps: int,
        input_mode: str = "cycle"
    ) -> LongRunReport:
        """
        Long-Runを実行する。
        
        Args:
            steps: 実行ステップ数
            input_mode: "cycle" (コーパスをループ) | "random" (未実装)
            
        Returns:
            LongRunReport
        """
        report = LongRunReport(
            total_steps=steps,
            start_time=datetime.now(timezone.utc).isoformat()
        )
        
        start = time.time()
        
        try:
            for step in range(steps):
                # Get next input
                text = self._get_next_input(input_mode)
                
                # Run pipeline
                try:
                    result = self.pipeline.run(text)
                    
                    # Update report
                    if result.success:
                        report.successful_observations += 1
                    if getattr(result, 'abstained', False):
                        report.abstained += 1
                    
                    # Alert tracking
                    alert = getattr(result, 'alert', None)
                    if alert:
                        report.total_alerts += 1
                        atom_id = alert.get('atom_id')
                        if atom_id and atom_id not in report.alert_atoms:
                            report.alert_atoms.append(atom_id)
                    
                    # Update monitor
                    if self.monitor:
                        self.monitor.update(result)
                        if self.index:
                            self.monitor.update_from_index(self.index)
                    
                except Exception as e:
                    # Fail Fast: エラー発生時は停止
                    error_info = {
                        "step": step,
                        "input": text[:50],
                        "error": str(e),
                        "type": type(e).__name__
                    }
                    report.errors.append(error_info)
                    report.stopped_early = True
                    logger.error(f"[LongRun] Error at step {step}: {e}")
                    break
                
                # Health Check
                if (step + 1) % self.health_check_interval == 0:
                    if not self._health_check(report, step):
                        report.stopped_early = True
                        break
                
                report.completed_steps = step + 1
            
            # Final validation
            self._health_check(report, steps)
            
        except KeyboardInterrupt:
            logger.info("[LongRun] Interrupted by user")
            report.stopped_early = True
        
        # Finalize report
        report.end_time = datetime.now(timezone.utc).isoformat()
        report.duration_seconds = time.time() - start
        
        # Stats snapshot
        if self.ledger and hasattr(self.ledger, 'seq'):
            report.final_ledger_seq = self.ledger.seq
        
        if self.index:
            if hasattr(self.index, 'atom_stats'):
                report.final_index_size = len(self.index.atom_stats)
            if hasattr(self.index, 'direction_stats'):
                ds = self.index.direction_stats
                total = ds.count_total if hasattr(ds, 'count_total') else 0
                if total > 0:
                    # Direction balance from monitor if available
                    if self.monitor:
                        report.direction_balance = self.monitor.state.direction_balance()
        
        return report
    
    def _get_next_input(self, mode: str) -> str:
        """次の入力テキストを取得"""
        if mode == "cycle":
            text = self.corpus[self._corpus_index % len(self.corpus)]
            self._corpus_index += 1
            return text
        else:
            # Default to cycle
            return self._get_next_input("cycle")
    
    def _health_check(self, report: LongRunReport, step: int) -> bool:
        """
        Ledgerの健全性をチェックする。
        
        Returns:
            True if healthy, False if validation failed
        """
        report.validation_checks += 1
        
        if self.ledger and hasattr(self.ledger, 'validate'):
            try:
                is_valid, errors = self.ledger.validate()
                if not is_valid:
                    report.ledger_valid = False
                    report.errors.append({
                        "step": step,
                        "type": "LedgerValidationError",
                        "error": f"Ledger invalid: {errors}"
                    })
                    logger.error(f"[LongRun] Ledger validation failed at step {step}")
                    return False
            except Exception as e:
                report.errors.append({
                    "step": step,
                    "type": "ValidationException",
                    "error": str(e)
                })
                logger.error(f"[LongRun] Validation exception at step {step}: {e}")
                return False
        
        return True


# ==========================================
# Utility Functions
# ==========================================
def print_report(report: LongRunReport) -> None:
    """レポートを整形して出力する"""
    print("\n" + "=" * 60)
    print("ESDE Long-Run Report")
    print("=" * 60)
    
    print(f"\n[Execution]")
    print(f"  Steps: {report.completed_steps}/{report.total_steps}")
    print(f"  Duration: {report.duration_seconds:.2f}s")
    print(f"  Stopped Early: {report.stopped_early}")
    
    print(f"\n[Results]")
    print(f"  Successful: {report.successful_observations}")
    print(f"  Abstained: {report.abstained}")
    
    print(f"\n[Alerts]")
    print(f"  Total: {report.total_alerts}")
    if report.alert_atoms:
        print(f"  Atoms: {', '.join(report.alert_atoms)}")
    
    print(f"\n[Health]")
    print(f"  Ledger Valid: {report.ledger_valid}")
    print(f"  Validation Checks: {report.validation_checks}")
    
    print(f"\n[Final State]")
    print(f"  Ledger Seq: {report.final_ledger_seq}")
    print(f"  Index Size: {report.final_index_size}")
    if report.direction_balance:
        print(f"  Direction Balance:")
        for d, ratio in report.direction_balance.items():
            print(f"    {d}: {ratio:.1%}")
    
    if report.errors:
        print(f"\n[Errors]")
        for err in report.errors:
            print(f"  Step {err.get('step')}: {err.get('type')} - {err.get('error')}")
    
    print("\n" + "=" * 60)
    
    if not report.errors and report.ledger_valid:
        print("✅ LONG-RUN PASSED")
    else:
        print("❌ LONG-RUN FAILED")
    
    print("=" * 60)


# ==========================================
# Test
# ==========================================
if __name__ == "__main__":
    print("LongRunRunner module loaded")
    print(f"Default corpus size: {len(DEFAULT_CORPUS)}")
