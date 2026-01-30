#!/usr/bin/env python3
"""
ESDE Phase 9: Complete Pipeline (W2 → W3 → W4 → W5 → W6)
=========================================================

Full integrated pipeline with 10 Sengoku warlords.

Usage:
    cd /path/to/esde
    
    # Fetch from Wikipedia API (10 warlords)
    python -m statistics.pipeline.run_full_pipeline --fetch --axis section
    
    # Use embedded sample data (3 warlords, fallback)
    python -m statistics.pipeline.run_full_pipeline --axis section

Output:
    ./output/
      ├── report.md     # Human-readable report
      └── analysis.json # Machine-readable data

Spec: Phase 9 Complete Pipeline v1.3
"""

import sys
import os
import re
import json
import argparse
import urllib.request
import urllib.parse
from typing import List, Dict, Any, Optional
from datetime import datetime

# Relative imports
from ..features import FeatureExtractor, TokenFeature
from .condition_provider import CONDITION_PROVIDERS
from .w2_aggregator import W2Aggregator
from .w3_calculator import W3Calculator
from .w4_projector import W4Projector, compute_pairwise_similarities
from .w5_w6_adapter import (
    convert_to_w4_records,
    SimpleCondensator,
    export_structure_markdown,
    export_structure_json,
)


# ==========================================
# 10 Sengoku Warlords
# ==========================================

SENGOKU_WARLORDS = [
    "Oda Nobunaga",
    "Toyotomi Hideyoshi",
    "Tokugawa Ieyasu",
    "Takeda Shingen",
    "Uesugi Kenshin",
    "Date Masamune",
    "Sanada Yukimura",
    "Mōri Motonari",
    "Hōjō Ujiyasu",
    "Akechi Mitsuhide",
]


# ==========================================
# Wikipedia API
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
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "ESDE/1.0 (research project)"}
        )
        
        with urllib.request.urlopen(request, timeout=30) as response:
            data = json.loads(response.read().decode('utf-8'))
        
        pages = data.get("query", {}).get("pages", {})
        
        for page_id, page_data in pages.items():
            if page_id == "-1":
                print(f"  [!] Article not found: {title}")
                return None
            
            text = page_data.get("extract", "")
            return text
        
        return None
    except Exception as e:
        print(f"  [!] Error fetching {title}: {e}")
        return None


def fetch_all_warlords() -> Dict[str, str]:
    """Fetch all 10 warlord articles from Wikipedia."""
    articles = {}
    
    print("\n[Fetching from Wikipedia]")
    for title in SENGOKU_WARLORDS:
        print(f"  Fetching: {title}...", end=" ", flush=True)
        text = fetch_wikipedia_article(title)
        if text:
            article_id = title.lower().replace(" ", "_").replace("ō", "o").replace("ū", "u")
            articles[article_id] = text
            print(f"OK ({len(text):,} chars)")
        else:
            print("FAILED")
    
    print(f"\n  Total: {len(articles)} articles fetched")
    return articles


# ==========================================
# Embedded Sample Data (Fallback - 3 warlords)
# ==========================================

SAMPLE_ARTICLES = {
    'oda_nobunaga': '''
== Lead ==
Oda Nobunaga was a powerful daimyo of Japan in the late 16th century.
He is regarded as one of the three great unifiers of Japan.
Nobunaga was known for his innovative military tactics and ruthless efficiency.
His ambition was to bring all of Japan under his rule.

== Early Life ==
Nobunaga was born in 1534 in Owari Province.
His father was Oda Nobuhide, a minor but ambitious daimyo.
Young Nobunaga was considered eccentric and was called the Fool of Owari.
He received education in both literary and military arts.

== Military Campaigns ==
Nobunaga launched numerous military campaigns to unify Japan.
He defeated the Imagawa clan at the Battle of Okehazama in 1560.
This victory established his reputation as a military genius.
His armies were known for effectively using firearms.
The use of guns revolutionized Japanese warfare under his command.

== Policies ==
Nobunaga implemented innovative economic policies.
He established free markets and reduced barriers to trade.
Castle towns flourished under his administration.
He suppressed Buddhist militant groups that opposed him.

== Death ==
Nobunaga was betrayed by his general Akechi Mitsuhide in 1582.
He died at Honno-ji temple during the surprise attack.
His death shocked the nation and led to a power struggle.
The circumstances of his death remain partially mysterious.

== Legacy ==
Nobunaga's legacy includes the unification process he started.
His economic policies promoted commerce and prosperity.
He is remembered as a brilliant but ruthless leader.
Many modern Japanese view him as a revolutionary figure.
''',

    'toyotomi_hideyoshi': '''
== Lead ==
Toyotomi Hideyoshi was a preeminent daimyo and imperial regent.
He completed the unification of Japan that Nobunaga had begun.
Hideyoshi rose from humble peasant origins to become ruler.
His life story is considered one of Japan's greatest rags-to-riches tales.

== Early Life ==
Hideyoshi was born around 1537 to a peasant family.
His father was a foot soldier with no samurai status.
He left home as a young man to seek his fortune.
Hideyoshi eventually entered the service of Oda Nobunaga.

== Military Campaigns ==
Hideyoshi proved himself as a capable general under Nobunaga.
After Nobunaga's death, he quickly moved to avenge his lord.
He defeated Akechi Mitsuhide at the Battle of Yamazaki.
His forces conquered Kyushu and Shikoku in major campaigns.
The invasions of Korea demonstrated his military ambitions.

== Policies ==
Hideyoshi conducted nationwide land surveys for taxation.
He instituted the sword hunt to disarm the peasantry.
The class separation edicts froze social mobility.
He promoted foreign trade while restricting Christianity.

== Death ==
Hideyoshi died of illness in 1598 at Fushimi Castle.
His death created a succession crisis for his young heir.
The power vacuum led to the rise of Tokugawa Ieyasu.
His dreams of continental conquest ended with his death.

== Legacy ==
Hideyoshi is remembered for completing Japan's unification.
His cultural contributions include promoting the tea ceremony.
The Momoyama period bears his artistic influence.
He remains one of Japan's most celebrated historical figures.
''',

    'tokugawa_ieyasu': '''
== Lead ==
Tokugawa Ieyasu founded the Tokugawa shogunate in 1603.
He was the first of fifteen Tokugawa shoguns who ruled Japan.
Ieyasu is considered the third of Japan's great unifiers.
His victory established over 250 years of peace and stability.

== Early Life ==
Ieyasu was born in 1543 as the son of a minor lord.
He spent much of his childhood as a hostage.
First held by the Imagawa clan, later by the Oda.
These experiences taught him patience and diplomacy.

== Military Campaigns ==
Ieyasu allied with Nobunaga and expanded his territories.
After Hideyoshi's death, he emerged as the strongest daimyo.
He won the decisive Battle of Sekigahara in 1600.
His victory gave him control over all of Japan.
The defeated western coalition was stripped of their lands.

== Policies ==
Ieyasu established the shogunate in Edo, modern Tokyo.
He implemented the sankin-kotai system for controlling daimyo.
Strict social hierarchy was maintained through laws.
Foreign trade was gradually restricted under his successors.

== Death ==
Ieyasu died in 1616 at Sunpu Castle at age 73.
He was deified as Tosho Daigongen after death.
His mausoleum at Nikko became a pilgrimage site.
The stable succession ensured the continuation of his regime.

== Legacy ==
Ieyasu's legacy is the Tokugawa peace that lasted centuries.
His patient strategy is often contrasted with Nobunaga's boldness.
The proverb says Nobunaga made the rice cake, Hideyoshi shaped it, Ieyasu ate it.
Modern Japan owes much to the administrative systems he established.
''',
}


# ==========================================
# Utility Functions
# ==========================================

def split_into_sections(text: str) -> List[Dict[str, Any]]:
    """Split Wikipedia-style text into sections."""
    sections = []
    
    # Pattern for section headers (== Title == or === Title ===)
    header_pattern = re.compile(r'^(={2,})\s*(.+?)\s*\1\s*$', re.MULTILINE)
    matches = list(header_pattern.finditer(text))
    
    # Lead section (before first header)
    if matches:
        lead_content = text[:matches[0].start()].strip()
        if lead_content:
            sections.append({
                'title': 'Lead',
                'level': 0,
                'content': lead_content,
            })
    else:
        # No headers found - entire text is content
        sections.append({
            'title': 'Lead',
            'level': 0,
            'content': text.strip(),
        })
        return sections
    
    # Process each section
    for i, m in enumerate(matches):
        level = len(m.group(1)) - 1
        title = m.group(2).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()
        
        # Skip empty sections and certain sections
        skip_titles = ['see also', 'references', 'external links', 'notes', 
                       'bibliography', 'further reading', 'sources']
        if not content or title.lower() in skip_titles:
            continue
        
        sections.append({
            'title': title,
            'level': level,
            'content': content,
        })
    
    return sections


def print_section(title: str):
    """Print section header."""
    print(f"\n{'=' * 70}")
    print(f" {title}")
    print('=' * 70)


# ==========================================
# Main Pipeline
# ==========================================

def run_pipeline(
    axis: str = "section",
    threshold: float = 0.9,
    min_island_size: int = 2,
    output_dir: str = "./output",
    use_fetch: bool = False,
) -> Dict[str, Any]:
    """
    Run the complete W2-W6 pipeline.
    
    Args:
        axis: Condition axis ('section', 'passive', 'section_passive', etc.)
        threshold: Similarity threshold for clustering
        min_island_size: Minimum island size
        output_dir: Directory for output files
        use_fetch: If True, fetch from Wikipedia API (10 warlords)
        
    Returns:
        Pipeline results summary
    """
    print_section("ESDE Phase 9: Complete Pipeline")
    print(f"Axis: {axis}")
    print(f"Threshold: {threshold}")
    print(f"Data source: {'Wikipedia API (10 warlords)' if use_fetch else 'Embedded sample (3 warlords)'}")
    
    # Ensure output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # ==========================================
    # Get Articles
    # ==========================================
    
    if use_fetch:
        articles = fetch_all_warlords()
        if len(articles) < 3:
            print("\n[!] Too few articles fetched, falling back to sample data")
            articles = SAMPLE_ARTICLES
    else:
        articles = SAMPLE_ARTICLES
        print(f"\n[Note] Using embedded sample data (3 warlords).")
        print(f"       Use --fetch to fetch 10 warlords from Wikipedia.")
    
    print(f"\nArticles to process: {len(articles)}")
    
    # ==========================================
    # Step 1: Feature Extraction (W1)
    # ==========================================
    
    print_section("Step 1: Feature Extraction (W1)")
    
    extractor = FeatureExtractor()
    all_features: Dict[str, List[TokenFeature]] = {}
    all_sections: Dict[str, List[Dict]] = {}
    
    for article_id, text in articles.items():
        sections = split_into_sections(text)
        all_sections[article_id] = sections
        
        features = []
        for sec_idx, sec in enumerate(sections):
            sec_features = extractor.extract_text(sec['content'], section_idx=sec_idx)
            for f in sec_features:
                f.section_idx = sec_idx
                f.section_name = sec['title']
            features.extend(sec_features)
        
        all_features[article_id] = features
        print(f"  {article_id}: {len(features)} tokens, {len(sections)} sections")
    
    total_tokens = sum(len(f) for f in all_features.values())
    print(f"\n  Total: {total_tokens:,} tokens from {len(articles)} articles")
    
    # ==========================================
    # Step 2: Conditional Statistics (W2)
    # ==========================================
    
    print_section(f"Step 2: Conditional Statistics (W2) - Axis: {axis}")
    
    aggregator = W2Aggregator(axis=axis)
    
    for article_id, features in all_features.items():
        sections = all_sections[article_id]
        aggregator.process_article(article_id, features, sections)
    
    w2_stats = aggregator.get_stats()
    
    print(f"  Total tokens processed: {w2_stats.global_total:,}")
    print(f"  Unique tokens: {len(w2_stats.global_counts):,}")
    print(f"  Conditions found: {len(w2_stats.conditions)}")
    print("\n  Condition breakdown:")
    for cid, cond in sorted(w2_stats.conditions.items(), key=lambda x: -x[1].total_tokens)[:15]:
        print(f"    {cid}: {cond.total_tokens} tokens")
    if len(w2_stats.conditions) > 15:
        print(f"    ... and {len(w2_stats.conditions) - 15} more")
    
    # ==========================================
    # Step 3: S-Score Calculation (W3)
    # ==========================================
    
    print_section("Step 3: S-Score Calculation (W3)")
    
    calculator = W3Calculator(w2_stats, top_k=30, min_count=2)
    w3_result = calculator.calculate_all()
    
    print(f"  Conditions analyzed: {len(w3_result.conditions)}")
    
    # Show top conditions
    shown = 0
    for cid, cond in sorted(w3_result.conditions.items()):
        if shown >= 8:
            print(f"\n  ... and {len(w3_result.conditions) - 8} more conditions")
            break
        print(f"\n  [{cid}]")
        if cond.positive_candidates:
            top_pos = cond.positive_candidates[:3]
            print(f"    + " + ", ".join([f"{c.token}({c.s_score:+.4f})" for c in top_pos]))
        if cond.negative_candidates:
            top_neg = cond.negative_candidates[:2]
            print(f"    - " + ", ".join([f"{c.token}({c.s_score:+.4f})" for c in top_neg]))
        shown += 1
    
    # ==========================================
    # Step 4: Article Projection (W4)
    # ==========================================
    
    print_section("Step 4: Article Projection (W4)")
    
    projector = W4Projector(w3_result)
    articles_list = [(aid, feat) for aid, feat in all_features.items()]
    w4_result = projector.project_all(articles_list)
    
    print(f"  Projected {len(w4_result.articles)} articles")
    
    for aid, av in w4_result.articles.items():
        print(f"\n  {aid}:")
        top_conditions = sorted(av.resonance_vector.items(), key=lambda x: -x[1])[:5]
        for cid, score in top_conditions:
            bar_len = min(int(abs(score) * 20), 40)
            bar = '█' * bar_len if score > 0 else '░' * bar_len
            print(f"    {cid:25s}: {score:+.4f} {bar}")
    
    # Pairwise similarities
    print("\n  Article Similarities:")
    sims = compute_pairwise_similarities(w4_result)
    
    # Show all if <= 15, otherwise top 15
    show_count = min(len(sims), 15)
    for a1, a2, sim in sims[:show_count]:
        print(f"    {a1} <-> {a2}: {sim:.4f}")
    if len(sims) > show_count:
        print(f"    ... and {len(sims) - show_count} more pairs")
    
    # ==========================================
    # Step 5: Clustering (W5)
    # ==========================================
    
    print_section("Step 5: Clustering (W5)")
    
    records = convert_to_w4_records(w4_result)
    condensator = SimpleCondensator(threshold=threshold, min_island_size=min_island_size)
    structure = condensator.condense(records)
    
    print(f"  Input articles: {structure.input_count}")
    print(f"  Islands formed: {structure.island_count}")
    print(f"  Noise articles: {structure.noise_count}")
    
    for i, island in enumerate(structure.islands, 1):
        print(f"\n  Island {i} ({island.size} members):")
        print(f"    Members: {', '.join(island.member_ids)}")
        print(f"    Cohesion: {island.cohesion_score:.4f}")
        print(f"    Top conditions:")
        for cid, score in sorted(island.representative_vector.items(), key=lambda x: -x[1])[:3]:
            print(f"      {cid}: {score:+.4f}")
    
    if structure.noise_ids:
        print(f"\n  Noise (unclustered): {', '.join(structure.noise_ids)}")
    
    # ==========================================
    # Step 6: Export (W6)
    # ==========================================
    
    print_section("Step 6: Export (W6)")
    
    # Markdown report
    md_path = os.path.join(output_dir, "report.md")
    export_structure_markdown(structure, w3_result, md_path)
    print(f"  Markdown: {md_path}")
    
    # JSON data
    json_path = os.path.join(output_dir, "analysis.json")
    export_structure_json(structure, w3_result, w4_result, json_path)
    print(f"  JSON: {json_path}")
    
    # ==========================================
    # Summary
    # ==========================================
    
    print_section("Pipeline Complete!")
    
    summary = {
        "axis": axis,
        "articles": len(articles),
        "total_tokens": total_tokens,
        "conditions": len(w2_stats.conditions),
        "islands": structure.island_count,
        "noise": structure.noise_count,
        "output_dir": output_dir,
    }
    
    print(f"""
  Configuration:
    Axis:           {axis}
    Threshold:      {threshold}
    Min Island:     {min_island_size}
    
  Results:
    Articles:       {summary['articles']}
    Total Tokens:   {summary['total_tokens']:,}
    Conditions:     {summary['conditions']}
    Islands:        {summary['islands']}
    Noise:          {summary['noise']}
    
  Output:
    {md_path}
    {json_path}
""")
    
    if summary['conditions'] > 1:
        print("  ✅ SUCCESS: Multiple conditions extracted from internal structure!")
    else:
        print("  ⚠️  Only 1 condition found.")
    
    return summary


# ==========================================
# Main
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ESDE Phase 9 Complete Pipeline")
    parser.add_argument("--axis", default="section", choices=list(CONDITION_PROVIDERS.keys()),
                        help="Condition axis")
    parser.add_argument("--threshold", type=float, default=0.9,
                        help="Clustering similarity threshold")
    parser.add_argument("--min-island", type=int, default=2,
                        help="Minimum island size")
    parser.add_argument("--output", default="./output",
                        help="Output directory")
    parser.add_argument("--fetch", action="store_true",
                        help="Fetch articles from Wikipedia API (10 warlords)")
    
    args = parser.parse_args()
    
    result = run_pipeline(
        axis=args.axis,
        threshold=args.threshold,
        min_island_size=args.min_island,
        output_dir=args.output,
        use_fetch=args.fetch,
    )
