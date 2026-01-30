#!/usr/bin/env python3
"""
ESDE Phase 9: Integrated Pipeline Test
=======================================

Tests the full pipeline: New W1 → New W2 → Legacy W3 → Results

This script demonstrates:
  - Internal condition extraction (section_id, passive, etc.)
  - S-Score calculation with meaningful axes
  - Recovery from "1-condition death" problem

Usage:
    cd /path/to/esde
    python -m statistics.pipeline.test_integrated_pipeline
    python -m statistics.pipeline.test_integrated_pipeline --axis passive
    python -m statistics.pipeline.test_integrated_pipeline --article "Tokugawa Ieyasu"

Spec: Phase 9 Integrated Pipeline v1.0
"""

import sys
import json
import re
import argparse
import urllib.request
import urllib.parse
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

# Relative imports
from ..features import FeatureExtractor, TokenFeature
from .condition_provider import (
    get_condition_provider,
    AggregationContext,
    CONDITION_PROVIDERS,
)
from .w2_aggregator import W2Aggregator
from .w3_calculator import W3Calculator
from .w2_adapter import (
    convert_to_legacy_format,
    get_conversion_summary,
)


# ==========================================
# Utility Functions (Wikipedia API)
# ==========================================

def fetch_wikipedia_article(title: str, lang: str = "en") -> Optional[str]:
    """Fetch article text from Wikipedia API."""
    base_url = f"https://{lang}.wikipedia.org/w/api.php"
    
    params = {
        "action": "query",
        "titles": title,
        "prop": "extracts",
        "explaintext": "true",
        "format": "json",
    }
    
    url = f"{base_url}?{urllib.parse.urlencode(params)}"
    
    try:
        print(f"[Wikipedia] Fetching: {title}")
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "ESDE/1.0 (research project)"}
        )
        
        with urllib.request.urlopen(request, timeout=30) as response:
            data = json.loads(response.read().decode('utf-8'))
        
        pages = data.get("query", {}).get("pages", {})
        
        for page_id, page_data in pages.items():
            if page_id == "-1":
                print(f"[Wikipedia] Article not found: {title}")
                return None
            
            text = page_data.get("extract", "")
            print(f"[Wikipedia] Fetched {len(text):,} chars")
            return text
        
        return None
    except Exception as e:
        print(f"[Wikipedia] Error: {e}")
        return None


def split_into_sections(text: str) -> List[Dict[str, Any]]:
    """Split Wikipedia text into sections by == headers ==."""
    sections = []
    
    # Pattern for section headers
    header_pattern = re.compile(r'^(={2,})\s*(.+?)\s*\1\s*$', re.MULTILINE)
    
    # Find all headers
    matches = list(header_pattern.finditer(text))
    
    if not matches:
        # No headers found - entire text is lead section
        return [{"title": "Lead", "level": 0, "content": text}]
    
    # Lead section (before first header)
    if matches[0].start() > 0:
        lead_content = text[:matches[0].start()].strip()
        if lead_content:
            sections.append({
                "title": "Lead",
                "level": 0,
                "content": lead_content,
            })
    
    # Process each section
    for i, match in enumerate(matches):
        header_level = len(match.group(1)) - 1  # == is level 1
        header_title = match.group(2).strip()
        
        # Content starts after this header
        content_start = match.end()
        
        # Content ends at next header or end of text
        if i + 1 < len(matches):
            content_end = matches[i + 1].start()
        else:
            content_end = len(text)
        
        content = text[content_start:content_end].strip()
        
        sections.append({
            "title": header_title,
            "level": header_level,
            "content": content,
        })
    
    return sections


# ==========================================
# Pipeline Functions
# ==========================================

def run_w1_extraction(
    article_title: str,
) -> Tuple[List[TokenFeature], List[Dict], str]:
    """
    Run W1 feature extraction on a Wikipedia article.
    
    Returns:
        (features, sections, raw_text)
    """
    print(f"\n[W1] Fetching article: {article_title}")
    
    # Fetch article
    raw_text = fetch_wikipedia_article(article_title)
    if not raw_text:
        raise ValueError(f"Could not fetch article: {article_title}")
    
    print(f"  Raw text length: {len(raw_text):,} chars")
    
    # Split into sections
    sections = split_into_sections(raw_text)
    print(f"  Sections found: {len(sections)}")
    
    # Extract features
    extractor = FeatureExtractor()
    
    all_features = []
    for sec_idx, section in enumerate(sections):
        section_features = extractor.extract(
            section['content'],
            section_idx=sec_idx,
        )
        
        # Attach section info to features
        for feat in section_features:
            feat.section_idx = sec_idx
            feat.section_name = section['title']
        
        all_features.extend(section_features)
    
    print(f"  Total features: {len(all_features):,}")
    
    return all_features, sections, raw_text


def run_w2_aggregation(
    article_id: str,
    features: List[TokenFeature],
    sections: List[Dict],
    axis: str,
) -> Any:
    """
    Run W2 aggregation with specified condition axis.
    
    Args:
        article_id: Unique article identifier
        features: List of TokenFeature from W1
        sections: List of section dicts
        axis: Condition axis ('section', 'passive', 'paren', 'quote', 'propn')
        
    Returns:
        W2Stats
    """
    print(f"\n[W2] Aggregating by axis: {axis}")
    
    aggregator = W2Aggregator(axis=axis)
    result = aggregator.process_article(article_id, features, sections)
    
    print(f"  Tokens processed: {result['tokens_processed']:,}")
    print(f"  Conditions seen: {len(result['conditions_seen'])}")
    
    summary = aggregator.get_summary()
    print(f"\n  Top conditions by token count:")
    for cond in summary['conditions'][:10]:
        print(f"    {cond['condition_id']}: {cond['total_tokens']:,} tokens")
    
    return aggregator.get_stats()


def run_w3_calculation(
    w1_stats: Any,
    w2_stats: Any,
    top_k: int = 50,
) -> Any:
    """
    Run W3 S-Score calculation.
    
    Args:
        w1_stats: Legacy W1GlobalStats
        w2_stats: Legacy W2GlobalStats
        top_k: Number of top candidates per condition
        
    Returns:
        W3Result
    """
    print(f"\n[W3] Calculating S-Scores (top_k={top_k})")
    
    # Import legacy W3 calculator with adapter
    from statistics.pipeline.w3_calculator import W3Calculator as NewW3Calculator
    
    # Convert legacy stats to format our W3 expects
    # (Our W3 uses W2Stats directly, so we need to adapt)
    
    # Actually, let's use our new W3 calculator which takes W2Stats
    # But wait, we converted to legacy format for compatibility...
    
    # For now, use our new W3 calculator
    # We need to create a W2Stats-like object from the new stats
    
    # Actually the conversion was to legacy format for the existing W3
    # Let me check what we have...
    
    # The w1_stats and w2_stats here are the legacy-formatted ones
    # We need to use them with a calculator that expects that format
    
    # For this test, let's use our new W3 calculator directly with the
    # pre-conversion W2Stats. We'll pass the original stats through.
    
    print("  Note: Using integrated W3 calculator")
    
    return None  # We'll fix this in the next iteration


def analyze_results(
    w2_stats: Any,
    axis: str,
) -> Dict[str, Any]:
    """
    Analyze W2 statistics to show meaningful patterns.
    
    This demonstrates that internal conditions produce useful differentiation.
    """
    print(f"\n[Analysis] Condition Differentiation ({axis} axis)")
    print("=" * 60)
    
    # Get all conditions
    conditions = list(w2_stats.conditions.items())
    
    if len(conditions) < 2:
        print("  ⚠️ Only 1 condition found - no differentiation possible")
        return {"success": False, "reason": "single_condition"}
    
    print(f"  ✅ {len(conditions)} conditions found - differentiation possible!")
    
    # Find tokens that differ between conditions
    # Compare first two conditions
    cond1_id, cond1 = conditions[0]
    cond2_id, cond2 = conditions[1]
    
    # Get tokens specific to each condition
    cond1_tokens = set(cond1.token_counts.keys())
    cond2_tokens = set(cond2.token_counts.keys())
    
    only_cond1 = cond1_tokens - cond2_tokens
    only_cond2 = cond2_tokens - cond1_tokens
    shared = cond1_tokens & cond2_tokens
    
    print(f"\n  Comparing: '{cond1_id}' vs '{cond2_id}'")
    print(f"    Tokens only in {cond1_id}: {len(only_cond1)}")
    print(f"    Tokens only in {cond2_id}: {len(only_cond2)}")
    print(f"    Shared tokens: {len(shared)}")
    
    # Show top unique tokens
    if only_cond1:
        top_cond1 = sorted(
            [(t, cond1.token_counts[t]) for t in only_cond1],
            key=lambda x: -x[1]
        )[:5]
        print(f"\n  Top tokens unique to '{cond1_id}':")
        for token, count in top_cond1:
            print(f"    {token}: {count}")
    
    if only_cond2:
        top_cond2 = sorted(
            [(t, cond2.token_counts[t]) for t in only_cond2],
            key=lambda x: -x[1]
        )[:5]
        print(f"\n  Top tokens unique to '{cond2_id}':")
        for token, count in top_cond2:
            print(f"    {token}: {count}")
    
    # Show tokens with different proportions
    if shared:
        print(f"\n  Tokens with different proportions:")
        
        total1 = cond1.total_tokens
        total2 = cond2.total_tokens
        
        proportions = []
        for token in shared:
            p1 = cond1.token_counts[token] / total1 if total1 > 0 else 0
            p2 = cond2.token_counts[token] / total2 if total2 > 0 else 0
            ratio = p1 / p2 if p2 > 0 else float('inf')
            proportions.append((token, p1, p2, ratio, cond1.token_counts[token], cond2.token_counts[token]))
        
        # Sort by ratio (highest first = most specific to cond1)
        proportions.sort(key=lambda x: -x[3])
        
        print(f"\n  Most specific to '{cond1_id}':")
        for token, p1, p2, ratio, c1, c2 in proportions[:5]:
            if ratio > 1.5:  # At least 50% more common
                print(f"    {token}: {c1} vs {c2} (ratio: {ratio:.2f}x)")
        
        print(f"\n  Most specific to '{cond2_id}':")
        proportions.sort(key=lambda x: x[3])  # Lowest ratio = most specific to cond2
        for token, p1, p2, ratio, c1, c2 in proportions[:5]:
            if ratio < 0.67:  # At least 50% less common
                print(f"    {token}: {c1} vs {c2} (ratio: {ratio:.2f}x)")
    
    return {
        "success": True,
        "condition_count": len(conditions),
        "comparison": {
            "cond1": cond1_id,
            "cond2": cond2_id,
            "unique_to_cond1": len(only_cond1),
            "unique_to_cond2": len(only_cond2),
            "shared": len(shared),
        },
    }


def run_full_pipeline(
    article_title: str = "Oda Nobunaga",
    axis: str = "section",
) -> Dict[str, Any]:
    """
    Run the full integrated pipeline.
    
    Args:
        article_title: Wikipedia article to analyze
        axis: Condition axis to use
        
    Returns:
        Pipeline results summary
    """
    print("\n" + "=" * 70)
    print(f"ESDE Phase 9: Integrated Pipeline Test")
    print(f"Article: {article_title}")
    print(f"Axis: {axis}")
    print("=" * 70)
    
    # Step 1: W1 Feature Extraction
    features, sections, raw_text = run_w1_extraction(article_title)
    
    # Step 2: W2 Aggregation (with internal conditions)
    article_id = article_title.lower().replace(" ", "_")
    w2_stats = run_w2_aggregation(article_id, features, sections, axis)
    
    # Step 3: Analysis (show differentiation)
    analysis = analyze_results(w2_stats, axis)
    
    # Step 4: Convert to legacy format (for W3)
    print("\n[Adapter] Converting to legacy format")
    w1_legacy, w2_legacy = convert_to_legacy_format(w2_stats)
    summary = get_conversion_summary(w1_legacy, w2_legacy)
    print(f"  W1: {summary['w1']['unique_tokens']} unique tokens")
    print(f"  W2: {summary['w2']['total_conditions']} conditions, {summary['w2']['total_records']} records")
    
    # Step 5: W3 S-Score Calculation
    print("\n[W3] Calculating S-Scores")
    from statistics.pipeline.w3_calculator import W3Calculator
    
    calculator = W3Calculator(w2_stats, top_k=30, min_count=3)
    w3_result = calculator.calculate_all()
    
    print(f"  Calculated {len(w3_result.conditions)} condition candidates")
    
    # Show S-Score results
    print("\n[Results] S-Score Top Candidates by Condition")
    print("-" * 60)
    
    w3_summary = calculator.get_summary(w3_result)
    for cond in w3_summary['conditions'][:5]:  # Top 5 conditions
        print(f"\n  [{cond['condition_id']}]")
        print(f"    Positive (specific to this condition):")
        for token, score in cond['top_positive'][:5]:
            print(f"      {token}: S={score:+.6f}")
        print(f"    Negative (less common in this condition):")
        for token, score in cond['top_negative'][:3]:
            print(f"      {token}: S={score:+.6f}")
    
    # Summary
    print("\n" + "=" * 70)
    print("PIPELINE SUMMARY")
    print("=" * 70)
    print(f"  Article: {article_title}")
    print(f"  Axis: {axis}")
    print(f"  Conditions: {len(w2_stats.conditions)} ({'✅ Multiple' if len(w2_stats.conditions) > 1 else '⚠️ Single'})")
    print(f"  Total tokens: {w2_stats.global_total:,}")
    print(f"  Unique tokens: {len(w2_stats.global_counts):,}")
    
    if analysis['success']:
        print(f"\n  🎉 SUCCESS: Internal conditions produce meaningful differentiation!")
        print(f"     The '{axis}' axis creates {analysis['condition_count']} distinct conditions")
    else:
        print(f"\n  ⚠️ WARNING: {analysis.get('reason', 'Unknown issue')}")
    
    return {
        "article": article_title,
        "axis": axis,
        "w1_features": len(features),
        "w2_conditions": len(w2_stats.conditions),
        "w3_candidates": len(w3_result.conditions),
        "success": analysis['success'],
    }


# ==========================================
# Main
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ESDE Phase 9 Integrated Pipeline Test")
    parser.add_argument("--article", default="Oda Nobunaga", help="Wikipedia article title")
    parser.add_argument("--axis", default="section", choices=list(CONDITION_PROVIDERS.keys()),
                        help="Condition axis to use")
    
    args = parser.parse_args()
    
    result = run_full_pipeline(args.article, args.axis)
    
    print("\n[Exit]")
    if result['success']:
        print("  Pipeline completed successfully ✅")
        sys.exit(0)
    else:
        print("  Pipeline completed with warnings ⚠️")
        sys.exit(1)
