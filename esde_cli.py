#!/usr/bin/env python3
"""
ESDE CLI - Command Line Interface
=================================

Phase 8-9: Semantic Monitor & Continuous Existence

Usage:
    python esde_cli.py observe "I love you"
    python esde_cli.py monitor
    python esde_cli.py longrun --steps 100
    python esde_cli.py status
"""

import argparse
import sys
import logging
from typing import Optional


# ==========================================
# Logging Setup
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)
logger = logging.getLogger("esde.cli")


# ==========================================
# Commands
# ==========================================

def cmd_observe(args, pipeline) -> int:
    """単発観測コマンド"""
    text = args.text
    
    print(f"\n[Input] {text}")
    print("-" * 50)
    
    result = pipeline.run(text)
    
    # Display result
    print(f"Success: {result.success}")
    print(f"Target Atom: {result.target_atom}")
    print(f"Rigidity: {result.rigidity}")
    
    if result.strategy:
        print(f"Strategy: {result.strategy.mode.value} (T={result.strategy.temperature})")
    
    if result.molecule:
        print(f"Formula: {result.molecule.get('formula', '-')}")
    
    if result.ledger_entry:
        print(f"Ledger Seq: {result.ledger_entry.get('seq')}")
        print(f"Direction: {result.ledger_entry.get('direction')}")
    
    if result.alert:
        print(f"\n⚠ ALERT: {result.alert.get('message')}")
    
    if result.generation_error:
        print(f"\n[Error] {result.generation_error}")
    
    return 0 if result.success else 1


def cmd_monitor(args, pipeline, index) -> int:
    """TUIモニターコマンド"""
    from monitor import SemanticMonitor, RICH_AVAILABLE
    from runner import LongRunRunner, DEFAULT_CORPUS
    
    if not RICH_AVAILABLE:
        print("[Warning] 'rich' library not installed. Using plain text mode.")
        print("Install with: pip install rich")
    
    monitor = SemanticMonitor()
    
    # Create runner with monitor
    runner = LongRunRunner(
        pipeline=pipeline,
        index=index,
        monitor=monitor,
        health_check_interval=10
    )
    
    steps = args.steps or 50  # Default 50 steps for monitor mode
    
    if RICH_AVAILABLE:
        from rich.live import Live
        
        print(f"Starting ESDE Monitor (steps={steps})...")
        print("Press Ctrl+C to stop\n")
        
        try:
            with Live(monitor.get_renderable(), refresh_per_second=2) as live:
                for i in range(steps):
                    text = DEFAULT_CORPUS[i % len(DEFAULT_CORPUS)]
                    result = pipeline.run(text)
                    monitor.update(result)
                    if index:
                        monitor.update_from_index(index)
                    live.update(monitor.get_renderable())
        except KeyboardInterrupt:
            print("\nMonitor stopped.")
    else:
        # Plain text mode
        for i in range(steps):
            text = DEFAULT_CORPUS[i % len(DEFAULT_CORPUS)]
            result = pipeline.run(text)
            monitor.update(result)
            if i % 5 == 0:
                monitor.render()
    
    return 0


def cmd_longrun(args, pipeline, ledger, index) -> int:
    """Long-Runテストコマンド"""
    from runner import LongRunRunner, print_report
    from monitor import SemanticMonitor
    
    steps = args.steps
    
    print(f"Starting ESDE Long-Run Test (steps={steps})...")
    print("-" * 50)
    
    monitor = SemanticMonitor() if args.verbose else None
    
    runner = LongRunRunner(
        pipeline=pipeline,
        ledger=ledger,
        index=index,
        monitor=monitor,
        health_check_interval=args.check_interval
    )
    
    report = runner.run(steps=steps)
    print_report(report)
    
    return 0 if (not report.errors and report.ledger_valid) else 1


def cmd_status(args, pipeline, ledger, index) -> int:
    """ステータス表示コマンド"""
    print("\n" + "=" * 50)
    print("ESDE Status")
    print("=" * 50)
    
    # Pipeline status
    if hasattr(pipeline, 'status'):
        status = pipeline.status()
        print(f"\n[Pipeline]")
        for key, value in status.items():
            print(f"  {key}: {value}")
    
    # Ledger status
    if ledger:
        print(f"\n[Ledger]")
        if hasattr(ledger, 'seq'):
            print(f"  Seq: {ledger.seq}")
        if hasattr(ledger, 'filepath'):
            print(f"  Path: {ledger.filepath}")
    
    # Index status
    if index:
        print(f"\n[Index]")
        if hasattr(index, 'total_events'):
            print(f"  Total Events: {index.total_events}")
        if hasattr(index, 'atom_stats'):
            print(f"  Atoms: {len(index.atom_stats)}")
    
    print("\n" + "=" * 50)
    return 0


# ==========================================
# Factory (Mock for standalone testing)
# ==========================================

def create_mock_pipeline():
    """テスト用のモックパイプラインを作成"""
    from index import SemanticIndex, Projector
    from feedback import Modulator
    
    class MockSensor:
        def analyze(self, text):
            # 簡易的にEMO.loveを返す
            if any(w in text.lower() for w in ['love', 'happy', 'joy', 'sad', 'hate']):
                return {"candidates": [{"concept_id": "EMO.love", "score": 1.5}]}
            elif any(w in text.lower() for w in ['law', 'rule', 'justice', 'order']):
                return {"candidates": [{"concept_id": "LAW.rule", "score": 1.2}]}
            elif any(w in text.lower() for w in ['create', 'destroy', 'transform']):
                return {"candidates": [{"concept_id": "ACT.create", "score": 1.0}]}
            return {"candidates": []}
    
    class MockGenerator:
        def __init__(self):
            self.llm_host = "http://mock"
            self.llm_model = "mock"
            self.llm_timeout = 10
        
        def generate(self, original_text, candidates):
            class Result:
                success = bool(candidates)
                molecule = {"formula": "aa_1", "active_atoms": [{"atom": candidates[0]["concept_id"]}]} if candidates else None
                error = None
                abstained = not candidates
            return Result()
        
        def _call_llm(self, *args):
            return "{}"
    
    class MockLedger:
        def __init__(self):
            self.entries = []
            self.seq = -1
            self.filepath = "mock://ledger"
        
        def observe_molecule(self, source_text, molecule, direction="=>+", actor="CLI"):
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
            return True, []
    
    # Import pipeline
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
        enable_alerts=True
    )
    
    return pipeline, ledger, index


# ==========================================
# Main
# ==========================================

def main():
    parser = argparse.ArgumentParser(
        description="ESDE Command Line Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python esde_cli.py observe "I love you"
  python esde_cli.py monitor --steps 100
  python esde_cli.py longrun --steps 50 --check-interval 5
  python esde_cli.py status
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # observe
    observe_parser = subparsers.add_parser("observe", help="Single observation")
    observe_parser.add_argument("text", help="Input text to observe")
    
    # monitor
    monitor_parser = subparsers.add_parser("monitor", help="TUI Monitor")
    monitor_parser.add_argument("--steps", type=int, default=50, help="Number of steps")
    
    # longrun
    longrun_parser = subparsers.add_parser("longrun", help="Long-run validation test")
    longrun_parser.add_argument("--steps", type=int, default=100, help="Number of steps")
    longrun_parser.add_argument("--check-interval", type=int, default=10, help="Health check interval")
    longrun_parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output with monitor")
    
    # status
    status_parser = subparsers.add_parser("status", help="Show system status")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 0
    
    # Create pipeline (mock for standalone testing)
    # In production, this would load actual components
    try:
        pipeline, ledger, index = create_mock_pipeline()
    except Exception as e:
        logger.error(f"Failed to create pipeline: {e}")
        return 1
    
    # Dispatch command
    if args.command == "observe":
        return cmd_observe(args, pipeline)
    elif args.command == "monitor":
        return cmd_monitor(args, pipeline, index)
    elif args.command == "longrun":
        return cmd_longrun(args, pipeline, ledger, index)
    elif args.command == "status":
        return cmd_status(args, pipeline, ledger, index)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
