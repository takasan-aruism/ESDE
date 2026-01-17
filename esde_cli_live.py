#!/usr/bin/env python3
"""
ESDE CLI (Live) - Command Line Interface for Real Environment
=============================================================

Phase 8-9: Semantic Monitor & Continuous Existence

実環境用CLI（実際のSensor V2、MoleculeGeneratorLive、PersistentLedgerを使用）

Usage:
    python esde_cli_live.py observe "I love you"
    python esde_cli_live.py monitor --steps 100
    python esde_cli_live.py longrun --steps 50
    python esde_cli_live.py status
"""

import argparse
import sys
import os
import json
import logging
from pathlib import Path
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
# File Paths (from Technical Specification)
# ==========================================
BASE_DIR = Path(__file__).parent

# Data files
SYNAPSE_FILE = BASE_DIR / "esde_synapses_v2_1.json"
GLOSSARY_FILE = BASE_DIR / "glossary_results.json"
LEDGER_FILE = BASE_DIR / "data" / "semantic_ledger.jsonl"

# Ensure data directory exists
(BASE_DIR / "data").mkdir(exist_ok=True)


# ==========================================
# Load Glossary
# ==========================================
def load_glossary(filepath: Path) -> dict:
    """Glossaryをロード"""
    if not filepath.exists():
        logger.warning(f"Glossary not found: {filepath}")
        return {}
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, dict):
        if "glossary" in data:
            glossary = data["glossary"]
            if isinstance(glossary, list):
                return {item["concept_id"]: item for item in glossary 
                       if isinstance(item, dict) and "concept_id" in item}
            return glossary
        return data
    return {}


# ==========================================
# Create Live Pipeline
# ==========================================
def create_live_pipeline(use_mock_llm: bool = False):
    """
    実環境用パイプラインを作成
    
    Args:
        use_mock_llm: TrueならLLMをモックに置き換え（テスト用）
    """
    # Import packages
    from index import SemanticIndex, Projector
    from feedback import Modulator
    from pipeline import ESDEPipeline
    
    # Sensor V2
    from esde_sensor_v2_modular import ESDESensorV2
    
    sensor = ESDESensorV2(
        synapse_file=str(SYNAPSE_FILE),
        glossary_file=str(GLOSSARY_FILE)
    )
    
    # Glossary for Generator
    glossary = load_glossary(GLOSSARY_FILE)
    
    # MoleculeGenerator
    if use_mock_llm:
        # Mock Generator（LLMなし）
        from sensor.molecule_generator_live import MockMoleculeGeneratorLive
        generator = MockMoleculeGeneratorLive(glossary=glossary)
        logger.info("[CLI] Using Mock Generator (no LLM)")
    else:
        # Live Generator（実LLM）
        from sensor.molecule_generator_live import MoleculeGeneratorLive
        generator = MoleculeGeneratorLive(glossary=glossary)
        logger.info(f"[CLI] Using Live Generator (LLM: {generator.llm_host})")
    
    # PersistentLedger
    from ledger import PersistentLedger
    ledger = PersistentLedger(path=str(LEDGER_FILE))
    logger.info(f"[CLI] Ledger: {LEDGER_FILE} (seq={ledger.seq})")
    
    # SemanticIndex + Projector
    index = SemanticIndex()
    projector = Projector(index)
    
    # Rebuild index from ledger
    if ledger.seq >= 0:
        logger.info(f"[CLI] Rebuilding index from ledger ({ledger.seq + 1} entries)...")
        projector.rebuild_from_ledger(ledger)
        logger.info(f"[CLI] Index rebuilt: {len(index.atom_stats)} atoms")
    
    # Pipeline
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
# Commands
# ==========================================

def cmd_observe(args) -> int:
    """単発観測コマンド"""
    text = args.text
    
    print(f"\n[Input] {text}")
    print("-" * 50)
    
    try:
        pipeline, ledger, index = create_live_pipeline(use_mock_llm=args.mock)
        result = pipeline.run(text)
        
        # Display result
        print(f"Success: {result.success}")
        print(f"Target Atom: {result.target_atom}")
        print(f"Rigidity: {result.rigidity}")
        
        if result.strategy:
            print(f"Strategy: {result.strategy.mode.value} (T={result.strategy.temperature})")
        
        if result.molecule:
            print(f"Formula: {result.molecule.get('formula', '-')}")
            atoms = result.molecule.get('active_atoms', [])
            if atoms:
                print(f"Active Atoms: {[a.get('atom') for a in atoms]}")
        
        if result.ledger_entry:
            print(f"Ledger Seq: {result.ledger_entry.get('seq')}")
            print(f"Direction: {result.ledger_entry.get('direction')}")
        
        if result.alert:
            print(f"\n⚠ ALERT: {result.alert.get('message')}")
        
        if result.generation_error:
            print(f"\n[Error] {result.generation_error}")
        
        return 0 if result.success else 1
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


def cmd_monitor(args) -> int:
    """TUIモニターコマンド"""
    from monitor import SemanticMonitor, RICH_AVAILABLE
    from runner import LongRunRunner, DEFAULT_CORPUS
    
    if not RICH_AVAILABLE:
        print("[Warning] 'rich' library not installed. Using plain text mode.")
        print("Install with: pip install rich")
    
    try:
        pipeline, ledger, index = create_live_pipeline(use_mock_llm=args.mock)
        
        monitor = SemanticMonitor()
        monitor.update_from_index(index)
        
        steps = args.steps
        corpus = args.corpus.split(',') if args.corpus else DEFAULT_CORPUS
        
        if RICH_AVAILABLE and not args.plain:
            from rich.live import Live
            
            print(f"Starting ESDE Monitor (steps={steps})...")
            print("Press Ctrl+C to stop\n")
            
            try:
                with Live(monitor.get_renderable(), refresh_per_second=1) as live:
                    for i in range(steps):
                        text = corpus[i % len(corpus)]
                        result = pipeline.run(text)
                        monitor.update(result)
                        monitor.update_from_index(index)
                        live.update(monitor.get_renderable())
            except KeyboardInterrupt:
                print("\nMonitor stopped.")
        else:
            # Plain text mode
            for i in range(steps):
                text = corpus[i % len(corpus)]
                result = pipeline.run(text)
                monitor.update(result)
                if i % 5 == 0 or i == steps - 1:
                    monitor.render()
        
        return 0
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


def cmd_longrun(args) -> int:
    """Long-Runテストコマンド"""
    from runner import LongRunRunner, print_report
    from monitor import SemanticMonitor
    
    steps = args.steps
    
    print(f"Starting ESDE Long-Run Test (steps={steps})...")
    print("-" * 50)
    
    try:
        pipeline, ledger, index = create_live_pipeline(use_mock_llm=args.mock)
        
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
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


def cmd_status(args) -> int:
    """ステータス表示コマンド"""
    print("\n" + "=" * 50)
    print("ESDE Status")
    print("=" * 50)
    
    # File Status
    print(f"\n[Files]")
    print(f"  Synapses: {SYNAPSE_FILE} ({'✓' if SYNAPSE_FILE.exists() else '✗'})")
    print(f"  Glossary: {GLOSSARY_FILE} ({'✓' if GLOSSARY_FILE.exists() else '✗'})")
    print(f"  Ledger:   {LEDGER_FILE} ({'✓' if LEDGER_FILE.exists() else '✗'})")
    
    # Ledger Status
    if LEDGER_FILE.exists():
        try:
            from ledger import PersistentLedger
            ledger = PersistentLedger(path=str(LEDGER_FILE))
            print(f"\n[Ledger]")
            print(f"  Seq: {ledger.seq}")
            print(f"  Ledger ID: {ledger.LEDGER_ID}")
            
            # Validate
            report = ledger.validate()
            print(f"  Valid: {report.valid}")
            if report.errors:
                print(f"  Errors: {report.errors[:3]}")
        except Exception as e:
            print(f"  Error loading ledger: {e}")
    
    # Index Status (rebuild from ledger)
    if LEDGER_FILE.exists():
        try:
            from index import SemanticIndex, Projector
            from ledger import PersistentLedger
            
            ledger = PersistentLedger(path=str(LEDGER_FILE))
            index = SemanticIndex()
            projector = Projector(index)
            projector.rebuild_from_ledger(ledger)
            
            print(f"\n[Index (L2)]")
            print(f"  Total Events: {index.total_events}")
            print(f"  Atoms: {len(index.atom_stats)}")
            
            # Top atoms by count
            if index.atom_stats:
                top_atoms = sorted(
                    index.atom_stats.items(),
                    key=lambda x: x[1].count_total,
                    reverse=True
                )[:5]
                print(f"  Top Atoms:")
                for atom_id, stats in top_atoms:
                    print(f"    {atom_id}: N={stats.count_total}")
        except Exception as e:
            print(f"  Error loading index: {e}")
    
    print("\n" + "=" * 50)
    return 0


# ==========================================
# Main
# ==========================================

def main():
    parser = argparse.ArgumentParser(
        description="ESDE Command Line Interface (Live Environment)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python esde_cli_live.py observe "I love you"
  python esde_cli_live.py observe "I love you" --mock  # No LLM
  python esde_cli_live.py monitor --steps 100
  python esde_cli_live.py longrun --steps 50
  python esde_cli_live.py status
        """
    )
    
    # Global options
    parser.add_argument("--mock", action="store_true", 
                       help="Use mock LLM (no real LLM calls)")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # observe
    observe_parser = subparsers.add_parser("observe", help="Single observation")
    observe_parser.add_argument("text", help="Input text to observe")
    observe_parser.add_argument("--mock", action="store_true",
                               help="Use mock LLM")
    
    # monitor
    monitor_parser = subparsers.add_parser("monitor", help="TUI Monitor")
    monitor_parser.add_argument("--steps", type=int, default=50, 
                               help="Number of steps")
    monitor_parser.add_argument("--corpus", type=str, default=None,
                               help="Comma-separated input texts")
    monitor_parser.add_argument("--plain", action="store_true",
                               help="Force plain text mode")
    monitor_parser.add_argument("--mock", action="store_true",
                               help="Use mock LLM")
    
    # longrun
    longrun_parser = subparsers.add_parser("longrun", help="Long-run test")
    longrun_parser.add_argument("--steps", type=int, default=100, 
                               help="Number of steps")
    longrun_parser.add_argument("--check-interval", type=int, default=10, 
                               help="Health check interval")
    longrun_parser.add_argument("--verbose", "-v", action="store_true", 
                               help="Verbose output with monitor")
    longrun_parser.add_argument("--mock", action="store_true",
                               help="Use mock LLM")
    
    # status
    status_parser = subparsers.add_parser("status", help="Show system status")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 0
    
    # Dispatch command
    if args.command == "observe":
        return cmd_observe(args)
    elif args.command == "monitor":
        return cmd_monitor(args)
    elif args.command == "longrun":
        return cmd_longrun(args)
    elif args.command == "status":
        return cmd_status(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
