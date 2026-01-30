#!/usr/bin/env python3
"""
ESDE Phase 9: Stage 2 - Real Data Test (v2)
============================================

Tests feature extraction on actual Wikipedia article.
Fixed based on GPT audit feedback.

Fixes in v2:
  - Section parsing and section_index_norm
  - Wiki markup cleanup
  - Proper noun unique count fix
  - N-gram noise filtering
  - Clearer dictionary coverage metrics

Usage:
    python tests/test_stage2_real_data.py
    python tests/test_stage2_real_data.py --article "Tokugawa Ieyasu"
    python tests/test_stage2_real_data.py --save-output

Spec: Phase 9 W1 Feature Extraction v1.0
"""

import sys
import time
import json
import re
import argparse
import urllib.request
import urllib.parse
from pathlib import Path
from collections import Counter
from typing import Dict, List, Any, Optional, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from statistics.features import (
    FeatureExtractor,
    NgramCollector,
    DictionaryProvider,
    FEATURE_NAMES,
    FEATURE_DIM,
    NULL_SCORE,
)


# ==========================================
# Wikipedia Fetcher
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
            print(f"[Wikipedia] Fetched {len(text):,} characters")
            return text
        
        return None
        
    except Exception as e:
        print(f"[Wikipedia] Error: {e}")
        return None


# ==========================================
# Text Preprocessing (Wiki Markup Cleanup)
# ==========================================

def clean_wiki_text(text: str) -> str:
    """
    Clean Wikipedia plain text of residual markup and noise.
    
    Removes:
      - Section markers (== Heading ==)
      - Multiple consecutive newlines
      - Unicode artifacts
    """
    # Remove section markers but preserve the heading text
    # Pattern: == Heading == or === Heading ===
    text = re.sub(r'^(=+)\s*(.+?)\s*\1\s*$', r'\2.', text, flags=re.MULTILINE)
    
    # Normalize whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)  # Max 2 newlines
    text = re.sub(r'[ \t]+', ' ', text)      # Multiple spaces to single
    
    # Remove common Wikipedia artifacts
    text = re.sub(r'\[\d+\]', '', text)      # Citation numbers [1], [2]
    text = re.sub(r'\(listen\)', '', text)   # Audio markers
    
    return text.strip()


def split_into_sections(text: str) -> List[Dict[str, Any]]:
    """
    Split Wikipedia text into sections based on headings.
    
    Returns list of {title, level, content} dicts.
    """
    # Pattern for section headings: == Title == or === Title ===
    heading_pattern = re.compile(r'^(=+)\s*(.+?)\s*\1\s*$', re.MULTILINE)
    
    sections = []
    matches = list(heading_pattern.finditer(text))
    
    # Add lead section (before first heading)
    if matches:
        lead_end = matches[0].start()
        lead_content = text[:lead_end].strip()
    else:
        lead_content = text.strip()
    
    if lead_content:
        sections.append({
            "title": "Lead",
            "level": 0,
            "content": clean_wiki_text(lead_content),
        })
    
    # Process each section
    skip_titles = {"references", "see also", "external links", "notes", 
                   "further reading", "bibliography", "sources"}
    
    for i, match in enumerate(matches):
        level = len(match.group(1)) - 1  # == is level 1, === is level 2
        title = match.group(2).strip()
        
        # Get content until next heading
        content_start = match.end()
        if i + 1 < len(matches):
            content_end = matches[i + 1].start()
        else:
            content_end = len(text)
        
        content = text[content_start:content_end].strip()
        
        # Skip reference sections and empty content
        if content and title.lower() not in skip_titles:
            sections.append({
                "title": title,
                "level": level,
                "content": clean_wiki_text(content),
            })
    
    return sections


# ==========================================
# Analysis Functions
# ==========================================

def analyze_feature_distribution(features: List[Any]) -> Dict[str, Any]:
    """Analyze distribution of each feature dimension."""
    if not features:
        return {}
    
    n = len(features)
    sums = [0.0] * FEATURE_DIM
    counts = [0] * FEATURE_DIM
    mins = [float('inf')] * FEATURE_DIM
    maxs = [float('-inf')] * FEATURE_DIM
    binary_counts = [0] * FEATURE_DIM
    
    for feat in features:
        for i, val in enumerate(feat.vector):
            if val != NULL_SCORE:
                sums[i] += val
                counts[i] += 1
                mins[i] = min(mins[i], val)
                maxs[i] = max(maxs[i], val)
            if val == 1.0:
                binary_counts[i] += 1
    
    stats = {}
    for i, name in enumerate(FEATURE_NAMES):
        if counts[i] > 0:
            mean = sums[i] / counts[i]
            null_rate = (n - counts[i]) / n
            stats[name] = {
                "mean": round(mean, 4),
                "min": round(mins[i], 4) if mins[i] != float('inf') else None,
                "max": round(maxs[i], 4) if maxs[i] != float('-inf') else None,
                "null_rate": round(null_rate, 4),
                "count_1": binary_counts[i],
            }
        else:
            stats[name] = {"mean": None, "null_rate": 1.0, "count_1": 0}
    
    return stats


def analyze_proper_nouns(features: List[Any]) -> Tuple[Counter, int, int]:
    """
    Extract and count proper nouns.
    
    Returns:
        (counter, total_count, unique_count)
    """
    proper_nouns = Counter()
    
    for feat in features:
        if feat.vector[14] == 1.0:  # is_proper_noun index
            # Keep original case for proper nouns
            proper_nouns[feat.token] += 1
    
    total = sum(proper_nouns.values())
    unique = len(proper_nouns)
    
    return proper_nouns, total, unique


def analyze_passive_patterns(features: List[Any]) -> Dict[str, Any]:
    """Analyze passive voice patterns."""
    be_aux_tokens = []
    passive_participles = []
    by_agents = []
    
    for feat in features:
        named = feat.to_named_dict()
        if named.get("is_be_aux", 0) > 0:
            be_aux_tokens.append(feat.token)
        if named.get("is_passive_participle", 0) > 0:
            passive_participles.append(feat.token)
        if named.get("is_by_agent", 0) > 0:
            by_agents.append(feat.token)
    
    return {
        "be_aux_count": len(be_aux_tokens),
        "passive_participle_count": len(passive_participles),
        "passive_participles": Counter(passive_participles).most_common(20),
        "by_agent_count": len(by_agents),
    }


def analyze_dictionary_coverage(features: List[Any]) -> Dict[str, Any]:
    """Analyze dictionary hit rates per dictionary."""
    dict_indices = {
        "concreteness": 10,
        "aoa": 11,
        "sensorimotor": 12,
        "valence": 13,
    }
    
    n = len(features)
    results = {}
    any_hit_count = 0
    
    for feat in features:
        has_any_hit = False
        for dict_name, idx in dict_indices.items():
            if feat.vector[idx] != NULL_SCORE:
                has_any_hit = True
                break
        if has_any_hit:
            any_hit_count += 1
    
    for dict_name, idx in dict_indices.items():
        hits = sum(1 for f in features if f.vector[idx] != NULL_SCORE)
        results[dict_name] = {
            "hits": hits,
            "total": n,
            "hit_rate": round(hits / n, 4) if n > 0 else 0,
        }
    
    # Add "any psycholinguistic hit" rate
    results["any_psycholinguistic"] = {
        "hits": any_hit_count,
        "total": n,
        "hit_rate": round(any_hit_count / n, 4) if n > 0 else 0,
    }
    
    return results


# ==========================================
# N-gram Analysis (with noise filtering)
# ==========================================

def analyze_ngrams_filtered(features: List[Any]) -> Dict[str, Any]:
    """
    Collect n-grams with noise filtering.
    
    Filters out:
      - Punctuation-only tokens
      - Newline/whitespace tokens
      - Markup artifacts
    """
    collector = NgramCollector()
    
    # Filter tokens for n-gram analysis
    def is_valid_token(token: str) -> bool:
        # Skip punctuation-only
        if re.match(r'^[\W_]+$', token):
            return False
        # Skip whitespace/newline
        if not token.strip():
            return False
        # Skip wiki markup artifacts
        if token in ('=', '==', '===', '\n', '\n\n'):
            return False
        return True
    
    # Group by sentence
    sentences: Dict[int, List[Tuple[str, float]]] = {}
    for feat in features:
        if is_valid_token(feat.token):
            sent_idx = feat.sentence_idx
            if sent_idx not in sentences:
                sentences[sent_idx] = []
            pos = feat.vector[0] if feat.vector else 0.5
            sentences[sent_idx].append((feat.token, pos))
    
    # Process sentences
    for sent_idx in sorted(sentences.keys()):
        items = sentences[sent_idx]
        tokens = [t for t, p in items]
        positions = [p for t, p in items]
        collector.process_sentence(tokens, positions)
    
    return {
        "collector": collector,
        "stats": collector.get_stats(),
    }


# ==========================================
# Main Test
# ==========================================

def run_stage2_test(
    article_title: str = "Oda Nobunaga",
    save_output: bool = False,
    output_dir: Optional[Path] = None,
):
    """Run Stage 2 real data test."""
    
    print("\n" + "#" * 70)
    print("# ESDE Phase 9: Stage 2 - Real Data Test (v2)")
    print("#" * 70)
    
    # ==========================================
    # Step 1: Fetch Wikipedia Article
    # ==========================================
    print("\n" + "=" * 60)
    print("[Step 1] Fetch Wikipedia Article")
    print("=" * 60)
    
    text = fetch_wikipedia_article(article_title)
    
    if not text:
        print("Failed to fetch article. Exiting.")
        return 1
    
    print(f"  Article: {article_title}")
    print(f"  Raw characters: {len(text):,}")
    
    # ==========================================
    # Step 2: Section Parsing & Cleanup
    # ==========================================
    print("\n" + "=" * 60)
    print("[Step 2] Section Parsing & Cleanup")
    print("=" * 60)
    
    sections = split_into_sections(text)
    
    print(f"  Sections found: {len(sections)}")
    print(f"\n  Section list:")
    for i, sec in enumerate(sections[:15]):  # Show first 15
        level_indent = "  " * sec['level']
        char_count = len(sec['content'])
        title_display = sec['title'][:35]
        print(f"    {i:2}. {level_indent}{title_display:35} ({char_count:,} chars)")
    
    if len(sections) > 15:
        print(f"    ... and {len(sections) - 15} more sections")
    
    # ==========================================
    # Step 3: Feature Extraction (Section-aware)
    # ==========================================
    print("\n" + "=" * 60)
    print("[Step 3] Feature Extraction (Section-aware)")
    print("=" * 60)
    
    extractor = FeatureExtractor(use_spacy=True)
    
    all_features = []
    total_sections = len(sections)
    
    start_time = time.time()
    
    for section_idx, section in enumerate(sections):
        section_features = extractor.extract_text(
            section['content'],
            section_idx=section_idx,
            total_sections=total_sections,
        )
        all_features.extend(section_features)
        
        if (section_idx + 1) % 10 == 0:
            print(f"    Processed {section_idx + 1}/{total_sections} sections...")
    
    elapsed = time.time() - start_time
    features = all_features
    
    print(f"\n  Tokens extracted: {len(features):,}")
    print(f"  Processing time: {elapsed:.2f} seconds")
    print(f"  Tokens/second: {len(features) / elapsed:,.0f}")
    
    stats = extractor.get_stats()
    print(f"  Sentences: {stats['sentences_processed']}")
    
    # ==========================================
    # Step 4: Dictionary Coverage
    # ==========================================
    print("\n" + "=" * 60)
    print("[Step 4] Dictionary Coverage")
    print("=" * 60)
    
    dict_coverage = analyze_dictionary_coverage(features)
    
    for dict_name, info in dict_coverage.items():
        rate_pct = info['hit_rate'] * 100
        label = f"{dict_name:20}" if dict_name != "any_psycholinguistic" else "ANY (at least one)   "
        print(f"  {label}: {info['hits']:,} / {info['total']:,} ({rate_pct:.1f}%)")
    
    # ==========================================
    # Step 5: Feature Distribution
    # ==========================================
    print("\n" + "=" * 60)
    print("[Step 5] Feature Distribution")
    print("=" * 60)
    
    feat_stats = analyze_feature_distribution(features)
    
    print("\n  Structural Features (mean / count=1):")
    for name in FEATURE_NAMES[:10]:
        s = feat_stats.get(name, {})
        mean = s.get('mean', 'N/A')
        count_1 = s.get('count_1', 0)
        if isinstance(mean, float):
            print(f"    {name:25}: mean={mean:.3f}, count_1={count_1}")
        else:
            print(f"    {name:25}: {mean}")
    
    print("\n  Psycholinguistic Features (mean / null_rate):")
    for name in FEATURE_NAMES[10:]:
        s = feat_stats.get(name, {})
        mean = s.get('mean', 'N/A')
        null_rate = s.get('null_rate', 1.0)
        if isinstance(mean, float):
            print(f"    {name:25}: mean={mean:.3f}, null={null_rate:.1%}")
        else:
            print(f"    {name:25}: null={null_rate:.1%}")
    
    # ==========================================
    # Step 6: Proper Noun Analysis
    # ==========================================
    print("\n" + "=" * 60)
    print("[Step 6] Proper Noun Analysis")
    print("=" * 60)
    
    proper_nouns, total_proper, unique_proper = analyze_proper_nouns(features)
    
    print(f"  Total proper noun tokens: {total_proper:,}")
    print(f"  Unique proper nouns: {unique_proper}")
    print(f"\n  Top 20 proper nouns:")
    
    for i, (noun, count) in enumerate(proper_nouns.most_common(20)):
        print(f"    {i+1:2}. {noun:20}: {count}")
    
    # ==========================================
    # Step 7: Passive Voice Analysis
    # ==========================================
    print("\n" + "=" * 60)
    print("[Step 7] Passive Voice Analysis")
    print("=" * 60)
    
    passive = analyze_passive_patterns(features)
    
    print(f"  Be-auxiliary count: {passive['be_aux_count']}")
    print(f"  Passive participles: {passive['passive_participle_count']}")
    print(f"  By-agent markers: {passive['by_agent_count']}")
    
    print(f"\n  Top passive participles:")
    for participle, count in passive['passive_participles'][:10]:
        print(f"    {participle:20}: {count}")
    
    # ==========================================
    # Step 8: N-gram Statistics (Filtered)
    # ==========================================
    print("\n" + "=" * 60)
    print("[Step 8] N-gram Statistics (Filtered)")
    print("=" * 60)
    
    ngram_result = analyze_ngrams_filtered(features)
    collector = ngram_result["collector"]
    ngram_stats = ngram_result["stats"]
    
    print(f"  Unique bigrams: {ngram_stats['unique_bigrams']:,}")
    print(f"  Unique trigrams: {ngram_stats['unique_trigrams']:,}")
    
    print(f"\n  Top 15 bigrams (content words):")
    bigrams = collector.get_bigram_stats(min_count=2, top_k=30)
    shown = 0
    for record in bigrams:
        # Skip boundary markers
        if '<BOS>' in record.tokens or '<EOS>' in record.tokens:
            continue
        tokens_str = " ".join(record.tokens)
        print(f"    '{tokens_str}': {record.count}")
        shown += 1
        if shown >= 15:
            break
    
    # ==========================================
    # Summary
    # ==========================================
    print("\n" + "=" * 60)
    print("[Summary]")
    print("=" * 60)
    
    summary = {
        "article": article_title,
        "raw_characters": len(text),
        "sections": len(sections),
        "tokens": len(features),
        "sentences": stats['sentences_processed'],
        "processing_time_sec": round(elapsed, 2),
        "tokens_per_sec": round(len(features) / elapsed, 0),
        "dictionary_coverage": dict_coverage,
        "proper_noun_total": total_proper,
        "proper_noun_unique": unique_proper,
        "passive_participle_count": passive['passive_participle_count'],
    }
    
    print(f"\n  Article: {summary['article']}")
    print(f"  Sections: {summary['sections']}")
    print(f"  Tokens: {summary['tokens']:,}")
    print(f"  Processing: {summary['processing_time_sec']}s ({summary['tokens_per_sec']:,.0f} tok/s)")
    print(f"  Proper nouns: {summary['proper_noun_total']:,} total, {summary['proper_noun_unique']} unique")
    print(f"  Passive participles: {summary['passive_participle_count']}")
    print(f"  Any dictionary hit: {dict_coverage['any_psycholinguistic']['hit_rate']:.1%}")
    
    # ==========================================
    # Save Output (Optional)
    # ==========================================
    if save_output:
        print("\n" + "=" * 60)
        print("[Saving Output]")
        print("=" * 60)
        
        if output_dir is None:
            output_dir = Path("data/stage2_output")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        safe_title = article_title.replace(' ', '_')
        
        # Save summary
        summary_path = output_dir / f"{safe_title}_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"  Saved: {summary_path}")
        
        # Save sample features
        sample_path = output_dir / f"{safe_title}_features_sample.jsonl"
        with open(sample_path, 'w', encoding='utf-8') as f:
            for feat in features[:200]:
                f.write(json.dumps(feat.to_dict(), ensure_ascii=False) + "\n")
        print(f"  Saved: {sample_path} (first 200 tokens)")
        
        # Save n-grams
        ngram_path = output_dir / f"{safe_title}_ngrams.jsonl"
        count = collector.save(str(ngram_path), min_count=2)
        print(f"  Saved: {ngram_path} ({count} entries)")
    
    print("\n" + "#" * 70)
    print("# Stage 2 Test Complete!")
    print("#" * 70)
    
    return 0


# ==========================================
# Entry Point
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Stage 2: Real Data Test")
    parser.add_argument("--article", default="Oda Nobunaga",
                        help="Wikipedia article title")
    parser.add_argument("--save-output", action="store_true",
                        help="Save output files")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory")
    
    args = parser.parse_args()
    
    return run_stage2_test(
        article_title=args.article,
        save_output=args.save_output,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    sys.exit(main())
