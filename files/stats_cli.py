#!/usr/bin/env python3
"""
ESDE Statistics CLI - Command Line Interface for W2/W3 Pipeline
===============================================================

Migration Phase 3: Statistics processing with scope isolation.

Usage:
    # Legacy mode (no policy)
    python stats_cli.py w2 --input data/articles/ --output data/stats/
    
    # Policy mode with auto-generated scope
    python stats_cli.py w2 --input data/articles/ --policy standard
    
    # Policy mode with explicit scope
    python stats_cli.py w2 --input data/articles/ --policy standard --scope run_20260125
    
    # Run full pipeline (W2 + W3)
    python stats_cli.py pipeline --input data/articles/ --policy standard
    
    # Show resolved paths
    python stats_cli.py paths --policy standard --scope my_scope

Commands:
    w2        Run W2 aggregation
    w3        Run W3 calculation
    pipeline  Run W2 + W3 pipeline
    paths     Show resolved paths for configuration
    status    Show current statistics status

Spec: Migration Phase 3 v0.3.1
"""

import argparse
import sys
import os
import json
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone


# ==========================================
# Logging Setup
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)
logger = logging.getLogger("esde.stats_cli")


# ==========================================
# Constants
# ==========================================

CLI_VERSION = "0.3.1"
DEFAULT_BASE_DIR = "data/stats"
DEFAULT_W1_PATH = "data/stats/w1_global_stats.json"


# ==========================================
# Policy Factory
# ==========================================

def create_policy(policy_name: str, target_keys: Optional[List[str]] = None):
    """
    Create a policy by name.
    
    Args:
        policy_name: Policy name (e.g., "standard", "legacy_migration")
        target_keys: Optional list of target keys
        
    Returns:
        BaseConditionPolicy instance
    """
    from statistics.policies import StandardConditionPolicy
    
    # Default target keys for common policies
    DEFAULT_KEYS = {
        "standard": ["legacy:source_type", "legacy:language_profile"],
        "legacy_migration": ["legacy:source_type", "legacy:language_profile"],
        "source_only": ["legacy:source_type"],
        "language_only": ["legacy:language_profile"],
    }
    
    keys = target_keys or DEFAULT_KEYS.get(policy_name, DEFAULT_KEYS["standard"])
    
    return StandardConditionPolicy(
        policy_id=policy_name,
        target_keys=keys,
        version="v1.0",
    )


# ==========================================
# Commands
# ==========================================

def cmd_w2(args) -> int:
    """Run W2 aggregation."""
    from statistics.runner import StatisticsPipelineRunner
    
    print(f"\n{'=' * 60}")
    print("ESDE Statistics CLI - W2 Aggregation")
    print(f"{'=' * 60}")
    
    # Setup
    policy = None
    registry = None
    
    if args.policy:
        policy = create_policy(args.policy)
        print(f"Policy: {policy.policy_id} (v{policy.version})")
        
        # Load registry if available
        try:
            from substrate import SubstrateRegistry
            registry_path = args.registry or "data/substrate/registry.jsonl"
            if os.path.exists(registry_path):
                registry = SubstrateRegistry(registry_path=registry_path)
                print(f"Registry: {registry_path}")
        except ImportError:
            print("Warning: substrate package not available, using legacy mode")
    
    # Create runner
    runner = StatisticsPipelineRunner(
        registry=registry,
        policy=policy,
        analysis_scope_id=args.scope,
        base_dir=args.base_dir or DEFAULT_BASE_DIR,
        auto_generate_scope=not args.no_auto_scope,
    )
    
    print(f"Scope: {runner.analysis_scope_id or 'None (Legacy mode)'}")
    print(f"Mode: {'Policy' if runner.is_policy_mode else 'Legacy'}")
    
    # Show paths
    paths = runner.get_resolved_paths()
    print(f"\nOutput paths:")
    print(f"  Records:    {paths['w2_records_path']}")
    print(f"  Conditions: {paths['w2_conditions_path']}")
    
    # Load articles
    print(f"\nLoading articles from: {args.input}")
    articles = load_articles(args.input)
    
    if not articles:
        print("Error: No articles found")
        return 1
    
    print(f"Found {len(articles)} articles")
    
    # Run W2
    print("\nRunning W2 aggregation...")
    result = runner.run_w2(articles, save=not args.dry_run)
    
    # Report
    print(f"\n{'=' * 60}")
    print("W2 Aggregation Result")
    print(f"{'=' * 60}")
    print(f"Success: {result.success}")
    print(f"Articles processed: {result.articles_processed}")
    print(f"Total records: {result.total_records}")
    print(f"Total conditions: {result.total_conditions}")
    
    if result.errors:
        print(f"\nErrors:")
        for err in result.errors:
            print(f"  - {err}")
    
    if args.dry_run:
        print("\n[Dry run - no files saved]")
    else:
        print(f"\nSaved to:")
        print(f"  {result.records_path}")
        print(f"  {result.conditions_path}")
    
    return 0 if result.success else 1


def cmd_w3(args) -> int:
    """Run W3 calculation."""
    from statistics.runner import StatisticsPipelineRunner
    from statistics.schema import W1GlobalStats
    from statistics.schema_w2 import W2GlobalStats
    
    print(f"\n{'=' * 60}")
    print("ESDE Statistics CLI - W3 Calculation")
    print(f"{'=' * 60}")
    
    # Load W1 stats
    w1_path = args.w1_path or DEFAULT_W1_PATH
    print(f"Loading W1 stats from: {w1_path}")
    
    if not os.path.exists(w1_path):
        print(f"Error: W1 stats not found at {w1_path}")
        return 1
    
    w1_stats = load_w1_stats(w1_path)
    if not w1_stats:
        print("Error: Failed to load W1 stats")
        return 1
    
    # Setup policy
    policy = None
    if args.policy:
        policy = create_policy(args.policy)
        print(f"Policy: {policy.policy_id}")
    
    # Create runner
    runner = StatisticsPipelineRunner(
        policy=policy,
        analysis_scope_id=args.scope,
        base_dir=args.base_dir or DEFAULT_BASE_DIR,
        auto_generate_scope=not args.no_auto_scope,
    )
    
    print(f"Scope: {runner.analysis_scope_id or 'None (Legacy mode)'}")
    
    # Load W2 stats
    paths = runner.get_resolved_paths()
    w2_records_path = args.w2_records or paths['w2_records_path']
    w2_conditions_path = args.w2_conditions or paths['w2_conditions_path']
    
    print(f"\nLoading W2 stats:")
    print(f"  Records:    {w2_records_path}")
    print(f"  Conditions: {w2_conditions_path}")
    
    w2_stats = load_w2_stats(w2_records_path, w2_conditions_path)
    if not w2_stats:
        print("Error: Failed to load W2 stats")
        return 1
    
    # Run W3
    print(f"\nRunning W3 calculation...")
    result = runner.run_w3(w1_stats, w2_stats, save=not args.dry_run)
    
    # Report
    print(f"\n{'=' * 60}")
    print("W3 Calculation Result")
    print(f"{'=' * 60}")
    print(f"Success: {result.success}")
    print(f"Conditions calculated: {result.conditions_calculated}")
    
    if result.errors:
        print(f"\nErrors:")
        for err in result.errors:
            print(f"  - {err}")
    
    if args.dry_run:
        print("\n[Dry run - no files saved]")
    else:
        print(f"\nSaved to: {result.output_dir}")
    
    return 0 if result.success else 1


def cmd_pipeline(args) -> int:
    """Run full W2 + W3 pipeline."""
    from statistics.runner import StatisticsPipelineRunner
    from statistics.schema import W1GlobalStats
    
    print(f"\n{'=' * 60}")
    print("ESDE Statistics CLI - Full Pipeline (W2 + W3)")
    print(f"{'=' * 60}")
    
    # Setup
    policy = None
    registry = None
    
    if args.policy:
        policy = create_policy(args.policy)
        print(f"Policy: {policy.policy_id} (v{policy.version})")
        
        try:
            from substrate import SubstrateRegistry
            registry_path = args.registry or "data/substrate/registry.jsonl"
            if os.path.exists(registry_path):
                registry = SubstrateRegistry(registry_path=registry_path)
        except ImportError:
            pass
    
    # Create runner
    runner = StatisticsPipelineRunner(
        registry=registry,
        policy=policy,
        analysis_scope_id=args.scope,
        base_dir=args.base_dir or DEFAULT_BASE_DIR,
        auto_generate_scope=not args.no_auto_scope,
    )
    
    print(f"Scope: {runner.analysis_scope_id}")
    print(f"Mode: {'Policy' if runner.is_policy_mode else 'Legacy'}")
    
    # Show paths
    paths = runner.get_resolved_paths()
    print(f"\nResolved paths:")
    for key, path in paths.items():
        print(f"  {key}: {path}")
    
    # Load articles
    print(f"\n[Step 1] Loading articles from: {args.input}")
    articles = load_articles(args.input)
    
    if not articles:
        print("Error: No articles found")
        return 1
    
    print(f"Found {len(articles)} articles")
    
    # Run W2
    print(f"\n[Step 2] Running W2 aggregation...")
    w2_result = runner.run_w2(articles, save=not args.dry_run)
    
    if not w2_result.success:
        print(f"Error: W2 failed - {w2_result.errors}")
        return 1
    
    print(f"  Records: {w2_result.total_records}")
    print(f"  Conditions: {w2_result.total_conditions}")
    
    # Load W1 stats
    w1_path = args.w1_path or DEFAULT_W1_PATH
    print(f"\n[Step 3] Loading W1 stats from: {w1_path}")
    
    w1_stats = load_w1_stats(w1_path)
    if not w1_stats:
        print("Error: Failed to load W1 stats")
        return 1
    
    # Run W3
    print(f"\n[Step 4] Running W3 calculation...")
    w3_result = runner.run_w3(w1_stats, save=not args.dry_run)
    
    if not w3_result.success:
        print(f"Error: W3 failed - {w3_result.errors}")
        return 1
    
    print(f"  Conditions calculated: {w3_result.conditions_calculated}")
    
    # Final report
    print(f"\n{'=' * 60}")
    print("Pipeline Complete")
    print(f"{'=' * 60}")
    
    summary = runner.get_summary()
    print(f"\nExecution Context (for W6):")
    ctx = summary["execution_context"]
    for key, value in ctx.items():
        if value is not None:
            print(f"  {key}: {value[:40] if isinstance(value, str) and len(value) > 40 else value}")
    
    if args.dry_run:
        print("\n[Dry run - no files saved]")
    
    return 0


def cmd_paths(args) -> int:
    """Show resolved paths for configuration."""
    from statistics.runner import StatisticsPipelineRunner
    
    print(f"\n{'=' * 60}")
    print("ESDE Statistics CLI - Path Resolution")
    print(f"{'=' * 60}")
    
    policy = None
    if args.policy:
        policy = create_policy(args.policy)
        print(f"Policy: {policy.policy_id}")
    
    runner = StatisticsPipelineRunner(
        policy=policy,
        analysis_scope_id=args.scope,
        base_dir=args.base_dir or DEFAULT_BASE_DIR,
        auto_generate_scope=not args.no_auto_scope,
    )
    
    print(f"Scope: {runner.analysis_scope_id or 'None (Legacy mode)'}")
    print(f"Mode: {'Policy' if runner.is_policy_mode else 'Legacy'}")
    
    print(f"\nResolved paths:")
    paths = runner.get_resolved_paths()
    for key, path in paths.items():
        print(f"  {key}:")
        print(f"    {path}")
    
    print(f"\nW6 Metadata:")
    metadata = runner.get_w6_metadata()
    for key, value in metadata.items():
        print(f"  {key}: {value}")
    
    return 0


def cmd_status(args) -> int:
    """Show current statistics status."""
    print(f"\n{'=' * 60}")
    print("ESDE Statistics CLI - Status")
    print(f"{'=' * 60}")
    
    base_dir = args.base_dir or DEFAULT_BASE_DIR
    
    print(f"\nBase directory: {base_dir}")
    
    # Check legacy files
    legacy_files = [
        "w1_global_stats.json",
        "w2_conditional_stats.jsonl",
        "w2_conditions.jsonl",
    ]
    
    print(f"\nLegacy files:")
    for f in legacy_files:
        path = os.path.join(base_dir, f)
        if os.path.exists(path):
            size = os.path.getsize(path)
            print(f"  ✅ {f} ({size:,} bytes)")
        else:
            print(f"  ❌ {f} (not found)")
    
    # Check policies directory
    policies_dir = os.path.join(base_dir, "policies")
    if os.path.exists(policies_dir):
        print(f"\nPolicy directories:")
        for policy_id in os.listdir(policies_dir):
            policy_path = os.path.join(policies_dir, policy_id)
            if os.path.isdir(policy_path):
                versions = os.listdir(policy_path)
                for version in versions:
                    version_path = os.path.join(policy_path, version)
                    if os.path.isdir(version_path):
                        scopes = os.listdir(version_path)
                        print(f"  {policy_id}/{version}/")
                        for scope in scopes[:5]:  # Show first 5
                            scope_path = os.path.join(version_path, scope)
                            if os.path.isdir(scope_path):
                                files = os.listdir(scope_path)
                                print(f"    └── {scope}/ ({len(files)} files)")
                        if len(scopes) > 5:
                            print(f"    └── ... and {len(scopes) - 5} more")
    else:
        print(f"\nNo policy directories found")
    
    return 0


# ==========================================
# Data Loaders (Stubs - implement based on actual data format)
# ==========================================

def load_articles(input_path: str) -> List:
    """
    Load articles from input path.
    
    Args:
        input_path: Path to articles (directory or file)
        
    Returns:
        List of ArticleRecord-like objects
    """
    # This is a stub - implement based on actual data format
    try:
        from integration import ContentGateway
        
        if os.path.isdir(input_path):
            # Load from directory
            gateway = ContentGateway()
            articles = []
            for filename in os.listdir(input_path):
                if filename.endswith('.json'):
                    filepath = os.path.join(input_path, filename)
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        # Convert to ArticleRecord
                        article = gateway.ingest_from_dict(data)
                        articles.append(article)
            return articles
        else:
            # Load from single file
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    gateway = ContentGateway()
                    return [gateway.ingest_from_dict(d) for d in data]
                else:
                    gateway = ContentGateway()
                    return [gateway.ingest_from_dict(data)]
    except Exception as e:
        logger.warning(f"Failed to load articles: {e}")
        return []


def load_w1_stats(path: str):
    """Load W1 stats from file."""
    try:
        from statistics.schema import W1GlobalStats
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return W1GlobalStats.from_dict(data)
    except Exception as e:
        logger.warning(f"Failed to load W1 stats: {e}")
        return None


def load_w2_stats(records_path: str, conditions_path: str):
    """Load W2 stats from files."""
    try:
        from statistics.schema_w2 import W2GlobalStats, W2Record, ConditionEntry
        
        stats = W2GlobalStats()
        
        # Load conditions
        if os.path.exists(conditions_path):
            with open(conditions_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entry = ConditionEntry.from_jsonl(line)
                        stats.conditions[entry.signature] = entry
            stats.total_conditions = len(stats.conditions)
        
        # Load records
        if os.path.exists(records_path):
            with open(records_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        record = W2Record.from_jsonl(line)
                        stats.records[record.record_id] = record
            stats.total_records = len(stats.records)
        
        return stats
    except Exception as e:
        logger.warning(f"Failed to load W2 stats: {e}")
        return None


# ==========================================
# Main
# ==========================================

def main():
    parser = argparse.ArgumentParser(
        description="ESDE Statistics CLI - Migration Phase 3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Legacy mode
  python stats_cli.py w2 --input data/articles/

  # Policy mode with auto-generated scope
  python stats_cli.py w2 --input data/articles/ --policy standard

  # Policy mode with explicit scope
  python stats_cli.py w2 --input data/articles/ --policy standard --scope run_20260125

  # Full pipeline
  python stats_cli.py pipeline --input data/articles/ --policy standard

  # Show paths
  python stats_cli.py paths --policy standard --scope my_scope
        """
    )
    
    parser.add_argument("--version", action="version", version=f"%(prog)s {CLI_VERSION}")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Common arguments
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--policy", type=str, help="Policy name (e.g., 'standard')")
    common.add_argument("--scope", type=str, help="Analysis scope ID")
    common.add_argument("--base-dir", type=str, help=f"Base directory (default: {DEFAULT_BASE_DIR})")
    common.add_argument("--no-auto-scope", action="store_true", 
                       help="Don't auto-generate scope ID in policy mode")
    common.add_argument("--dry-run", action="store_true", help="Don't save files")
    
    # w2 command
    w2_parser = subparsers.add_parser("w2", parents=[common], help="Run W2 aggregation")
    w2_parser.add_argument("--input", required=True, help="Input articles path")
    w2_parser.add_argument("--registry", type=str, help="Substrate registry path")
    
    # w3 command
    w3_parser = subparsers.add_parser("w3", parents=[common], help="Run W3 calculation")
    w3_parser.add_argument("--w1-path", type=str, help=f"W1 stats path (default: {DEFAULT_W1_PATH})")
    w3_parser.add_argument("--w2-records", type=str, help="W2 records path (auto-resolved)")
    w3_parser.add_argument("--w2-conditions", type=str, help="W2 conditions path (auto-resolved)")
    
    # pipeline command
    pipeline_parser = subparsers.add_parser("pipeline", parents=[common], 
                                            help="Run W2 + W3 pipeline")
    pipeline_parser.add_argument("--input", required=True, help="Input articles path")
    pipeline_parser.add_argument("--registry", type=str, help="Substrate registry path")
    pipeline_parser.add_argument("--w1-path", type=str, help=f"W1 stats path (default: {DEFAULT_W1_PATH})")
    
    # paths command
    paths_parser = subparsers.add_parser("paths", parents=[common], 
                                         help="Show resolved paths")
    
    # status command
    status_parser = subparsers.add_parser("status", help="Show statistics status")
    status_parser.add_argument("--base-dir", type=str, help=f"Base directory (default: {DEFAULT_BASE_DIR})")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 0
    
    # Dispatch
    if args.command == "w2":
        return cmd_w2(args)
    elif args.command == "w3":
        return cmd_w3(args)
    elif args.command == "pipeline":
        return cmd_pipeline(args)
    elif args.command == "paths":
        return cmd_paths(args)
    elif args.command == "status":
        return cmd_status(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
