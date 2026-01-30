#!/usr/bin/env python3
"""
ESDE Phase 9 Sengoku Experiment
================================
戦国武将Wikipedia記事のPhase 9統計処理実験

フロー:
  W0 (ContentGateway) → ArticleRecord作成
  W1 (GlobalStats)    → 全体トークン統計
  W2 (CondStats)      → 条件別統計
  W3 (AxisCandidates) → S-Score計算
  W4 (Projection)     → 共鳴ベクトル
  W5 (Condensation)   → 島クラスタリング
  W6 (Observation)    → 証拠抽出・出力

Usage:
    python experiment_sengoku_phase9.py --mode wiring   # 1件配線テスト
    python experiment_sengoku_phase9.py --mode sky      # 10件星空テスト

Requirements:
    - ESDE repository in PYTHONPATH
    - No external dependencies (stdlib only for Wikipedia fetch)

Spec: Sengoku Experiment v0.1.0
"""

import sys
import os
import re
import json
import argparse
from pathlib import Path
from datetime import datetime, timezone
from urllib.request import Request, urlopen
from urllib.parse import quote
from typing import List, Dict, Any, Optional
import uuid

# ==========================================
# Configuration
# ==========================================

# 戦国武将URL（10件）
SENGOKU_URLS = {
    "織田信長": "https://ja.wikipedia.org/wiki/織田信長",
    "豊臣秀吉": "https://ja.wikipedia.org/wiki/豊臣秀吉",
    "徳川家康": "https://ja.wikipedia.org/wiki/徳川家康",
    "明智光秀": "https://ja.wikipedia.org/wiki/明智光秀",
    "武田信玄": "https://ja.wikipedia.org/wiki/武田信玄",
    "上杉謙信": "https://ja.wikipedia.org/wiki/上杉謙信",
    "伊達政宗": "https://ja.wikipedia.org/wiki/伊達政宗",
    "真田幸村": "https://ja.wikipedia.org/wiki/真田信繁",  # 正式名
    "石田三成": "https://ja.wikipedia.org/wiki/石田三成",
    "毛利元就": "https://ja.wikipedia.org/wiki/毛利元就",
}

# Output directory
OUTPUT_BASE = Path("data/experiments/sengoku_proto_v1")

# W5 Parameters
W5_THRESHOLD = 0.70
W5_MIN_ISLAND_SIZE = 3


# ==========================================
# Wikipedia Fetcher (stdlib only)
# ==========================================

def fetch_wikipedia_text(url: str, max_chars: int = 50000) -> str:
    """
    Fetch Wikipedia article text using stdlib only.
    
    Args:
        url: Wikipedia article URL
        max_chars: Maximum characters to extract
        
    Returns:
        Plain text content
    """
    print(f"  Fetching: {url}")
    
    # URL encode Japanese characters
    # Split URL into base and path, encode path only
    if '/wiki/' in url:
        base, path = url.split('/wiki/', 1)
        encoded_url = f"{base}/wiki/{quote(path, safe='')}"
    else:
        encoded_url = url
    
    print(f"  Encoded:  {encoded_url}")
    
    headers = {
        'User-Agent': 'ESDE-SengokuExperiment/0.1 (research; github.com/takasan-aruism/ESDE)'
    }
    
    req = Request(encoded_url, headers=headers)
    
    try:
        with urlopen(req, timeout=30) as response:
            html = response.read().decode('utf-8')
    except Exception as e:
        print(f"    Error fetching: {e}")
        return ""
    
    # Ultra-simple HTML→text extraction
    # Remove script and style
    text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # Remove footnote markers [1], [2], etc.
    text = re.sub(r'\[\d+\]', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Truncate if too long
    if len(text) > max_chars:
        text = text[:max_chars]
    
    print(f"    Extracted: {len(text)} chars")
    return text


# ==========================================
# ArticleRecord Creation (Manual)
# ==========================================

def create_article_record_manual(
    text: str,
    source_url: str,
    source_meta: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Create ArticleRecord-like dict manually.
    
    This bypasses ContentGateway for simplicity.
    Compatible with W1/W2 Aggregator expectations.
    """
    article_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    
    # Create observation event
    observation = {
        "observation_id": str(uuid.uuid4()),
        "source_id": article_id,
        "segment_index": 0,
        "segment_span": (0, len(text)),
        "segment_text": text,  # Cache for convenience
        "timestamp": now,
        "context_meta": {},
    }
    
    return {
        "article_id": article_id,
        "source_url": source_url,
        "ingestion_time": now,
        "source_meta": source_meta,
        "raw_text": text,
        "segments": [(0, len(text))],
        "observations": [observation],
        "substrate_ref": None,  # Legacy mode
    }


class ArticleRecordWrapper:
    """
    Wrapper to make dict behave like ArticleRecord dataclass.
    
    W1/W2 Aggregators expect attribute access.
    """
    def __init__(self, data: Dict[str, Any]):
        self._data = data
        
        # Convert observations to objects with attributes
        self.observations = []
        for obs in data.get("observations", []):
            self.observations.append(ObservationWrapper(obs))
    
    def __getattr__(self, name):
        if name.startswith('_') or name == 'observations':
            raise AttributeError(name)
        return self._data.get(name)


class ObservationWrapper:
    """Wrapper for observation dict."""
    def __init__(self, data: Dict[str, Any]):
        self._data = data
    
    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        return self._data.get(name)


# ==========================================
# Pipeline Execution
# ==========================================

def run_pipeline(
    articles: List[ArticleRecordWrapper],
    output_dir: Path,
    scope_id: str,
) -> Dict[str, Any]:
    """
    Run full W1→W6 pipeline.
    
    Args:
        articles: List of ArticleRecordWrapper
        output_dir: Output directory
        scope_id: Analysis scope identifier
        
    Returns:
        Pipeline result summary
    """
    results = {
        "scope_id": scope_id,
        "article_count": len(articles),
        "stages": {},
    }
    
    # Ensure output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ==========================================
    # Stage 1: W1 - Global Statistics
    # ==========================================
    print("\n" + "=" * 60)
    print("[W1] Global Statistics")
    print("=" * 60)
    
    try:
        from statistics import W1Aggregator
        
        w1_path = output_dir / "w1_global_stats.json"
        w1_agg = W1Aggregator(storage_path=str(w1_path))
        
        for article in articles:
            summary = w1_agg.process_article(article)
            print(f"  Processed: {article.article_id[:8]}... ({summary['valid_tokens']} tokens)")
        
        w1_agg.save()
        w1_stats = w1_agg.stats
        
        results["stages"]["w1"] = {
            "success": True,
            "total_tokens": w1_stats.total_tokens_processed,
            "unique_tokens": len(w1_stats.records),
            "output_path": str(w1_path),
        }
        print(f"  ✅ W1 Complete: {len(w1_stats.records)} unique tokens")
        
    except Exception as e:
        print(f"  ❌ W1 Error: {e}")
        results["stages"]["w1"] = {"success": False, "error": str(e)}
        return results
    
    # ==========================================
    # Stage 2: W2 - Conditional Statistics
    # ==========================================
    print("\n" + "=" * 60)
    print("[W2] Conditional Statistics")
    print("=" * 60)
    
    try:
        from statistics import W2Aggregator
        
        w2_records_path = output_dir / "w2_records.jsonl"
        w2_conditions_path = output_dir / "w2_conditions.jsonl"
        
        w2_agg = W2Aggregator(
            records_path=str(w2_records_path),
            conditions_path=str(w2_conditions_path),
        )
        
        for article in articles:
            summary = w2_agg.process_article(article)
            print(f"  Processed: {article.article_id[:8]}... (cond: {summary['condition_signature'][:16]}...)")
        
        w2_agg.save()
        w2_stats = w2_agg.stats
        
        results["stages"]["w2"] = {
            "success": True,
            "total_records": w2_stats.total_records,
            "total_conditions": w2_stats.total_conditions,
            "records_path": str(w2_records_path),
            "conditions_path": str(w2_conditions_path),
        }
        print(f"  ✅ W2 Complete: {w2_stats.total_records} records, {w2_stats.total_conditions} conditions")
        
    except Exception as e:
        print(f"  ❌ W2 Error: {e}")
        import traceback
        traceback.print_exc()
        results["stages"]["w2"] = {"success": False, "error": str(e)}
        return results
    
    # ==========================================
    # Stage 3: W3 - Axis Candidates (S-Score)
    # ==========================================
    print("\n" + "=" * 60)
    print("[W3] Axis Candidates (S-Score)")
    print("=" * 60)
    
    try:
        from statistics import W3Calculator
        
        w3_output_dir = output_dir / "w3_candidates"
        w3_output_dir.mkdir(exist_ok=True)
        
        calculator = W3Calculator(
            w1_stats=w1_stats,
            w2_stats=w2_stats,
        )
        
        w3_records = calculator.calculate_all()
        
        # Save W3 records
        for w3_record in w3_records:
            w3_path = w3_output_dir / f"{w3_record.condition_signature[:16]}.json"
            with open(w3_path, 'w', encoding='utf-8') as f:
                json.dump(w3_record.to_dict() if hasattr(w3_record, 'to_dict') else vars(w3_record), f, indent=2, ensure_ascii=False, default=str)
        
        results["stages"]["w3"] = {
            "success": True,
            "conditions_calculated": len(w3_records),
            "output_dir": str(w3_output_dir),
        }
        print(f"  ✅ W3 Complete: {len(w3_records)} condition(s) calculated")
        
        if w3_records:
            top_record = w3_records[0]
            if hasattr(top_record, 'positive_candidates') and top_record.positive_candidates:
                top_token = top_record.positive_candidates[0]
                print(f"  Top positive: '{top_token.token_norm}' (S={top_token.s_score:.4f})")
        
    except Exception as e:
        print(f"  ❌ W3 Error: {e}")
        import traceback
        traceback.print_exc()
        results["stages"]["w3"] = {"success": False, "error": str(e)}
        return results
    
    # ==========================================
    # Stage 4: W4 - Structural Projection
    # ==========================================
    print("\n" + "=" * 60)
    print("[W4] Structural Projection (Resonance)")
    print("=" * 60)
    
    try:
        from statistics import W4Projector
        
        w4_output_dir = output_dir / "w4_projections"
        w4_output_dir.mkdir(exist_ok=True)
        
        # Create projector and load W3 records
        projector = W4Projector(output_dir=str(w4_output_dir))
        projector.load_w3_records(w3_records)
        
        print(f"  Loaded {len(projector.get_loaded_conditions())} conditions from W3")
        
        w4_records = []
        for article in articles:
            w4_record = projector.project(article)
            w4_records.append(w4_record)
            
            # Save W4 record
            projector.save(w4_record)
            
            vec_dims = len(w4_record.resonance_vector) if hasattr(w4_record, 'resonance_vector') else 0
            print(f"  Projected: {article.article_id[:8]}... ({vec_dims} dims)")
        
        results["stages"]["w4"] = {
            "success": True,
            "articles_projected": len(w4_records),
            "output_dir": str(w4_output_dir),
        }
        print(f"  ✅ W4 Complete: {len(w4_records)} articles projected")
        
    except Exception as e:
        print(f"  ❌ W4 Error: {e}")
        import traceback
        traceback.print_exc()
        results["stages"]["w4"] = {"success": False, "error": str(e)}
        return results
    
    # ==========================================
    # Stage 5: W5 - Structural Condensation
    # ==========================================
    print("\n" + "=" * 60)
    print("[W5] Structural Condensation (Islands)")
    print("=" * 60)
    
    try:
        from statistics import W5Condensator
        
        condensator = W5Condensator(
            threshold=W5_THRESHOLD,
            min_island_size=W5_MIN_ISLAND_SIZE,
        )
        
        w5_structure = condensator.condense(w4_records)
        
        # Save W5 structure
        w5_path = output_dir / "w5_structure.json"
        with open(w5_path, 'w', encoding='utf-8') as f:
            json.dump(w5_structure.to_dict() if hasattr(w5_structure, 'to_dict') else vars(w5_structure), f, indent=2, ensure_ascii=False, default=str)
        
        results["stages"]["w5"] = {
            "success": True,
            "island_count": w5_structure.island_count,
            "noise_count": w5_structure.noise_count,
            "output_path": str(w5_path),
        }
        print(f"  ✅ W5 Complete: {w5_structure.island_count} islands, {w5_structure.noise_count} noise")
        
        if w5_structure.island_count == 0:
            print("  ⚠️ No islands formed - need more articles or lower threshold")
        
    except Exception as e:
        print(f"  ❌ W5 Error: {e}")
        import traceback
        traceback.print_exc()
        results["stages"]["w5"] = {"success": False, "error": str(e)}
        return results
    
    # ==========================================
    # Stage 6: W6 - Structural Observation
    # ==========================================
    print("\n" + "=" * 60)
    print("[W6] Structural Observation (Evidence)")
    print("=" * 60)
    
    try:
        from discovery import W6Analyzer, W6Exporter
        
        analyzer = W6Analyzer(
            scope_id=scope_id,
            tokenizer_version="hybrid_v1",
            normalizer_version="v9.1.0",
        )
        
        # W6 requires ArticleRecord objects for raw_text access
        # Convert our wrappers back for W6
        observatory = analyzer.analyze(
            w5_structure=w5_structure,
            w4_records=w4_records,
            w3_records=w3_records,
            article_records=articles,
        )
        
        # Export
        w6_output_dir = output_dir / "w6_observations"
        w6_output_dir.mkdir(exist_ok=True)
        
        exporter = W6Exporter(output_dir=str(w6_output_dir))
        export_result = exporter.export(observatory, formats=["json", "md"])
        
        results["stages"]["w6"] = {
            "success": True,
            "observation_id": observatory.observation_id,
            "islands": len(observatory.islands),
            "topology_pairs": len(observatory.topology_pairs),
            "output_dir": str(w6_output_dir),
        }
        print(f"  ✅ W6 Complete: {len(observatory.islands)} island details")
        
        # Print evidence summary
        for island in observatory.islands[:3]:  # Top 3
            print(f"\n  Island {island.island_id[:12]}... (size={island.size})")
            if hasattr(island, 'evidence_tokens') and island.evidence_tokens:
                for token in island.evidence_tokens[:5]:
                    print(f"    - '{token.token_norm}' (score={token.evidence_score:.6f})")
        
    except Exception as e:
        print(f"  ❌ W6 Error: {e}")
        import traceback
        traceback.print_exc()
        results["stages"]["w6"] = {"success": False, "error": str(e)}
    
    return results


# ==========================================
# Main Entry Points
# ==========================================

def run_wiring_test():
    """
    配線テスト: 織田信長1件のみ
    
    目的: 全ステージが繋がることを確認
    """
    print("\n" + "#" * 70)
    print("# ESDE Sengoku Experiment: Wiring Test (1 article)")
    print("#" * 70)
    
    scope_id = f"sengoku_wiring_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = OUTPUT_BASE / scope_id
    
    # Fetch 織田信長 only
    name = "織田信長"
    url = SENGOKU_URLS[name]
    
    print(f"\n[Fetch] {name}")
    text = fetch_wikipedia_text(url)
    
    if not text:
        print("❌ Failed to fetch article")
        return 1
    
    # Create ArticleRecord
    article_data = create_article_record_manual(
        text=text,
        source_url=url,
        source_meta={
            "source_type": "wiki",
            "language_profile": "ja",
            "domain": "ja.wikipedia.org",
            "subject": name,
        }
    )
    articles = [ArticleRecordWrapper(article_data)]
    
    # Run pipeline
    results = run_pipeline(articles, output_dir, scope_id)
    
    # Summary
    print("\n" + "=" * 70)
    print("WIRING TEST SUMMARY")
    print("=" * 70)
    
    all_success = True
    for stage, data in results.get("stages", {}).items():
        status = "✅" if data.get("success") else "❌"
        print(f"  {stage}: {status}")
        if not data.get("success"):
            all_success = False
    
    print(f"\n  Output: {output_dir}")
    
    if all_success:
        print("\n✅ WIRING TEST PASSED - All stages connected!")
        return 0
    else:
        print("\n❌ WIRING TEST FAILED - Check errors above")
        return 1


def run_sky_test():
    """
    星空テスト: 戦国武将10件
    
    目的: 島クラスタリングが実際に動作することを確認
    """
    print("\n" + "#" * 70)
    print("# ESDE Sengoku Experiment: Sky Test (10 articles)")
    print("#" * 70)
    
    scope_id = f"sengoku_sky_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = OUTPUT_BASE / scope_id
    
    articles = []
    
    for name, url in SENGOKU_URLS.items():
        print(f"\n[Fetch] {name}")
        text = fetch_wikipedia_text(url)
        
        if not text:
            print(f"  ⚠️ Skipping {name} - fetch failed")
            continue
        
        article_data = create_article_record_manual(
            text=text,
            source_url=url,
            source_meta={
                "source_type": "wiki",
                "language_profile": "ja",
                "domain": "ja.wikipedia.org",
                "subject": name,
            }
        )
        articles.append(ArticleRecordWrapper(article_data))
    
    print(f"\n[Total] {len(articles)} articles fetched")
    
    if len(articles) < 3:
        print("❌ Not enough articles for clustering")
        return 1
    
    # Run pipeline
    results = run_pipeline(articles, output_dir, scope_id)
    
    # Summary
    print("\n" + "=" * 70)
    print("SKY TEST SUMMARY")
    print("=" * 70)
    
    all_success = True
    for stage, data in results.get("stages", {}).items():
        status = "✅" if data.get("success") else "❌"
        detail = ""
        if stage == "w5":
            detail = f" ({data.get('island_count', 0)} islands)"
        elif stage == "w1":
            detail = f" ({data.get('unique_tokens', 0)} tokens)"
        print(f"  {stage}: {status}{detail}")
        if not data.get("success"):
            all_success = False
    
    print(f"\n  Output: {output_dir}")
    
    # Save full results
    results_path = output_dir / "pipeline_results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    if all_success:
        w5_data = results.get("stages", {}).get("w5", {})
        if w5_data.get("island_count", 0) > 0:
            print("\n🌟 SKY TEST PASSED - Islands formed!")
        else:
            print("\n⚠️ SKY TEST COMPLETE - No islands (try lower threshold)")
        return 0
    else:
        print("\n❌ SKY TEST FAILED - Check errors above")
        return 1


# ==========================================
# CLI
# ==========================================

def main():
    parser = argparse.ArgumentParser(
        description="ESDE Phase 9 Sengoku Experiment"
    )
    parser.add_argument(
        "--mode",
        choices=["wiring", "sky"],
        default="wiring",
        help="Test mode: wiring (1 article) or sky (10 articles)"
    )
    
    args = parser.parse_args()
    
    if args.mode == "wiring":
        return run_wiring_test()
    else:
        return run_sky_test()


if __name__ == "__main__":
    sys.exit(main())
