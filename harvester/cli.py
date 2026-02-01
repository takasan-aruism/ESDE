"""
ESDE Harvester: CLI
====================

Command-line interface for data collection and caching.

Usage:
  # Harvest a dataset (fetch from Wikipedia + cache locally)
  python -m harvester.cli harvest --dataset mixed
  python -m harvester.cli harvest --dataset warlords
  
  # List cached datasets
  python -m harvester.cli list
  
  # Show dataset details
  python -m harvester.cli info --dataset mixed
  
  # Re-harvest (force refresh)
  python -m harvester.cli harvest --dataset mixed --force
"""

import argparse
import sys
import time

from .datasets import get_dataset, make_article_id, DATASETS
from .fetcher import fetch_wikipedia
from .storage import (
    save_artifact,
    save_distilled,
    compute_traces,
    save_manifest,
    load_manifest,
    load_dataset,
    list_datasets,
    ARTIFACTS_DIR,
    DATASETS_DIR,
)


# ==========================================
# Harvest Command
# ==========================================

def cmd_harvest(args):
    """Fetch dataset from Wikipedia and cache locally."""
    dataset_name = args.dataset
    force = args.force
    
    # Check if already cached
    if not force:
        manifest = load_manifest(dataset_name)
        if manifest:
            count = manifest.get("article_count", 0)
            created = manifest.get("created_at", "unknown")
            print(f"\n[!] Dataset '{dataset_name}' already cached ({count} articles, {created})")
            print(f"    Use --force to re-harvest.")
            return
    
    # Get dataset definition
    try:
        dataset_def = get_dataset(dataset_name)
    except ValueError as e:
        print(f"\n[Error] {e}")
        return
    
    print()
    print("=" * 60)
    print(f" ESDE Harvester: {dataset_name}")
    print("=" * 60)
    
    articles_meta = {}
    total_articles = sum(len(titles) for titles in dataset_def.values())
    fetched = 0
    failed = 0
    
    for prefix, titles in dataset_def.items():
        print(f"\n[Group: {prefix}] ({len(titles)} articles)")
        print("-" * 40)
        
        for title in titles:
            article_id = make_article_id(prefix, title)
            print(f"  Fetching: {title}...", end=" ", flush=True)
            
            # Fetch
            result = fetch_wikipedia(title)
            
            if not result.success:
                print(f"FAILED ({result.error_message})")
                failed += 1
                continue
            
            # Layer A: Save artifact (raw JSON)
            artifact_path = save_artifact(article_id, result)
            
            # Layer B: Save distilled text
            save_distilled(article_id, result.extract_text, dataset_name)
            
            # Compute Substrate-compatible traces
            traces = compute_traces(article_id, result, artifact_path)
            
            articles_meta[article_id] = {
                "title": title,
                "prefix": prefix,
                "text_file": f"{article_id}.txt",
                "artifact_file": f"../../artifacts/{article_id}.json",
                "traces": traces,
            }
            
            chars = result.content_length
            sections = result.section_count
            print(f"OK ({chars:,} chars, {sections} sections)")
            fetched += 1
            
            # Rate limiting (be nice to Wikipedia)
            time.sleep(0.5)
    
    # Save manifest
    manifest_path = save_manifest(dataset_name, articles_meta)
    
    # Summary
    print()
    print("=" * 60)
    print(f" Harvest Complete: {dataset_name}")
    print("=" * 60)
    print(f"  Fetched:  {fetched}/{total_articles}")
    print(f"  Failed:   {failed}")
    print(f"  Manifest: {manifest_path}")
    print(f"  Artifacts: {ARTIFACTS_DIR}/")
    print(f"  Texts:     {DATASETS_DIR}/{dataset_name}/")
    print()
    print(f"  Now run pipeline:")
    print(f"    python -m statistics.pipeline.run_full_pipeline --dataset {dataset_name}")


# ==========================================
# List Command
# ==========================================

def cmd_list(args):
    """List cached datasets."""
    datasets = list_datasets()
    
    print()
    print("=" * 60)
    print(" Cached Datasets")
    print("=" * 60)
    
    if not datasets:
        print("  (none)")
        print()
        print("  Available for harvest:")
        for name in DATASETS:
            print(f"    python -m harvester.cli harvest --dataset {name}")
        return
    
    print(f"  {'Name':<15} {'Articles':>8}   Created")
    print("  " + "-" * 50)
    
    for ds in datasets:
        print(f"  {ds['name']:<15} {ds['article_count']:>8}   {ds['created_at']}")
    
    print()
    print("  Available for harvest:")
    cached_names = {ds['name'] for ds in datasets}
    for name in DATASETS:
        status = "✅ cached" if name in cached_names else "⬜ not cached"
        print(f"    {name}: {status}")


# ==========================================
# Info Command
# ==========================================

def cmd_info(args):
    """Show dataset details."""
    dataset_name = args.dataset
    manifest = load_manifest(dataset_name)
    
    if not manifest:
        print(f"\n[!] Dataset '{dataset_name}' not found.")
        print(f"    Run: python -m harvester.cli harvest --dataset {dataset_name}")
        return
    
    print()
    print("=" * 60)
    print(f" Dataset: {dataset_name}")
    print("=" * 60)
    print(f"  Created: {manifest.get('created_at', 'unknown')}")
    print(f"  Articles: {manifest.get('article_count', 0)}")
    print()
    
    articles = manifest.get("articles", {})
    
    # Group by prefix
    groups = {}
    for aid, meta in articles.items():
        prefix = meta.get("prefix", "unknown")
        if prefix not in groups:
            groups[prefix] = []
        groups[prefix].append((aid, meta))
    
    for prefix, items in sorted(groups.items()):
        print(f"  [{prefix}] ({len(items)} articles)")
        for aid, meta in items:
            traces = meta.get("traces", {})
            chars = traces.get("text:char_count", 0)
            sections = traces.get("wiki:section_count", 0)
            print(f"    {aid:<35} {chars:>8,} chars  {sections:>3} sections")
    
    print()
    print(f"  Substrate traces per article:")
    if articles:
        sample_traces = list(articles.values())[0].get("traces", {})
        for key in sorted(sample_traces.keys()):
            print(f"    {key}")


# ==========================================
# Main
# ==========================================

def main():
    parser = argparse.ArgumentParser(
        description="ESDE Harvester: Data Collection & Caching"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command")
    
    # harvest
    p_harvest = subparsers.add_parser("harvest", help="Fetch dataset from Wikipedia")
    p_harvest.add_argument("--dataset", required=True, choices=list(DATASETS.keys()),
                           help="Dataset to harvest")
    p_harvest.add_argument("--force", action="store_true",
                           help="Force re-harvest (overwrite cache)")
    p_harvest.set_defaults(func=cmd_harvest)
    
    # list
    p_list = subparsers.add_parser("list", help="List cached datasets")
    p_list.set_defaults(func=cmd_list)
    
    # info
    p_info = subparsers.add_parser("info", help="Show dataset details")
    p_info.add_argument("--dataset", required=True,
                        help="Dataset name")
    p_info.set_defaults(func=cmd_info)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    args.func(args)


if __name__ == "__main__":
    main()
