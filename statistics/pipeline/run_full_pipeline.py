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
      ├── report.md           # Human-readable report
      ├── analysis.json       # Machine-readable data
      └── structure_stats.json # Structure statistics (sentence length, etc.)

Spec: Phase 9 Complete Pipeline v1.4
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
from ..features.structure_stats import (
    compute_structure_stats,
    compute_article_stats,
    compare_articles,
    StructureStats,
)
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
from .lens import get_lens, list_lenses, LENSES
from .w3_vector import W3VectorCalculator, export_vector_profiles
from .w4_vector import W4VectorProjector, export_vector_similarities, convert_to_w5_input
from .threshold import (
    summarize_distribution,
    relative_threshold_quantile,
    resolve_threshold,
    build_threshold_trace,
)
from .global_model import GlobalThresholdModel
from .edge_selector import create_edge_selector
from .chaining_metrics import compute_chaining_metrics
from .edge_policy import EdgePolicyResolver, export_sweep_csv
from .w1_cache import check_cache, save_cache, load_cache


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
# Mixed Dataset (Military Leaders / Scholars / Cities)
# ==========================================

MILITARY_LEADERS = [
    "Oda Nobunaga",           # Japan, 16th century
    "Cao Cao",                # China, 3rd century
    "Napoleon",               # France, 19th century
    "Hannibal",               # Carthage, 3rd century BC
    "Alexander the Great",    # Macedonia, 4th century BC
]

SCHOLARS = [
    "Albert Einstein",        # Physics
    "Abraham Maslow",         # Psychology
    "René Descartes",         # Philosophy
    "Jean-Henri Fabre",       # Biology (entomology)
    "Paracelsus",             # Toxicology/Medicine
]

CITIES = [
    "Tokyo",
    "London",
    "New York City",
    "Paris",
    "Berlin",
]

MIXED_DATASET = MILITARY_LEADERS + SCHOLARS + CITIES


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


def fetch_articles(titles: List[str], dataset_name: str = "articles") -> Dict[str, str]:
    """Fetch articles from Wikipedia."""
    articles = {}
    
    print(f"\n[Fetching {dataset_name} from Wikipedia]")
    for title in titles:
        print(f"  Fetching: {title}...", end=" ", flush=True)
        text = fetch_wikipedia_article(title)
        if text:
            # Normalize article ID
            article_id = title.lower()
            article_id = article_id.replace(" ", "_")
            article_id = article_id.replace("ō", "o").replace("ū", "u")
            article_id = article_id.replace("é", "e").replace("è", "e")
            articles[article_id] = text
            print(f"OK ({len(text):,} chars)")
        else:
            print("FAILED")
    
    print(f"\n  Total: {len(articles)} articles fetched")
    return articles


def fetch_all_warlords() -> Dict[str, str]:
    """Fetch all 10 warlord articles from Wikipedia."""
    return fetch_articles(SENGOKU_WARLORDS, "Sengoku Warlords")


def fetch_mixed_dataset() -> Dict[str, str]:
    """Fetch mixed dataset (military leaders, scholars, cities) from Wikipedia."""
    articles = {}
    
    # Fetch each category separately for better logging
    print("\n" + "=" * 60)
    print(" MIXED DATASET EXPERIMENT")
    print("=" * 60)
    
    # Military Leaders
    military = fetch_articles(MILITARY_LEADERS, "Military Leaders")
    for aid, text in military.items():
        articles[f"mil_{aid}"] = text
    
    # Scholars
    scholars = fetch_articles(SCHOLARS, "Scholars")
    for aid, text in scholars.items():
        articles[f"sch_{aid}"] = text
    
    # Cities
    cities = fetch_articles(CITIES, "Cities")
    for aid, text in cities.items():
        articles[f"city_{aid}"] = text
    
    print(f"\n  Total mixed dataset: {len(articles)} articles")
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
    dataset: str = "warlords",
    lens: str = None,
    threshold_mode: str = "fixed",
    threshold_q: float = 0.98,
    threshold_resolve: str = "safety_first",
    edge_filter: str = "none",
    knn_k: int = None,
    force_extract: bool = False,
) -> Dict[str, Any]:
    """
    Run the complete W2-W6 pipeline.
    
    Args:
        axis: Condition axis ('section', 'passive', 'document', etc.)
        threshold: Similarity threshold for clustering (used as abs_floor in adaptive modes)
        min_island_size: Minimum island size
        output_dir: Directory for output files
        use_fetch: If True, fetch from Wikipedia API
        dataset: 'warlords', 'mixed', or any Harvester-cached dataset name
        lens: If set ('structure'/'semantic'/'hybrid'), overrides axis with lens config
        threshold_mode: 'fixed' (legacy) or 'quantile' (adaptive)
        threshold_q: Quantile parameter for adaptive mode (default 0.98 = top 2%)
        threshold_resolve: How to combine abs+rel ('safety_first')
        edge_filter: 'none' (single-linkage) or 'mutual_knn' (anti-chaining)
        knn_k: k parameter for mutual-kNN. None = auto (ceil(log2(N)))
        
    Returns:
        Pipeline results summary
    """
    # Resolve lens → (axis, feature_mode)
    threshold_fallback = threshold  # CLI value as fallback when global is insufficient
    if lens:
        lens_config = get_lens(lens)
        axis = lens_config["condition"]
        feature_mode = lens_config["feature_mode"]
        lens_desc = lens_config["description"]
        # Lens-specific fallback (used when global model has insufficient data)
        if threshold == 0.9:  # default not changed by user
            threshold_fallback = lens_config.get("threshold_floor", 0.9)
    else:
        feature_mode = "token"
        lens_desc = None
    
    # Load global threshold model
    global_model = GlobalThresholdModel("./data/threshold")
    lens_key = lens or "legacy"
    global_n = global_model.get_count(lens_key, feature_mode)
    global_sufficient = global_model.is_sufficient(lens_key, feature_mode)
    
    print_section("ESDE Phase 9: Complete Pipeline")
    if lens:
        print(f"Lens: {lens} ({lens_desc})")
    print(f"Axis: {axis}")
    print(f"Feature Mode: {feature_mode}")
    print(f"Threshold Mode: {threshold_mode}")
    if threshold_mode == "quantile":
        print(f"Global Model: {global_n} pairs ({'sufficient' if global_sufficient else 'insufficient → fallback'})")
        if not global_sufficient:
            print(f"Fallback Floor: {threshold_fallback}")
        print(f"Quantile: {threshold_q}")
    else:
        print(f"Threshold: {threshold}")
    if edge_filter != "none":
        policy_resolver = None
        if knn_k == "auto":
            policy_resolver = EdgePolicyResolver()
            selector = None  # Will use policy resolver instead
            print(f"Edge Filter: {edge_filter} (k=auto sweep)")
        else:
            selector = create_edge_selector(edge_filter, k=knn_k)
            k_desc = f"k={knn_k}" if knn_k else "k=ceil(log2(N))"
            print(f"Edge Filter: {edge_filter} ({k_desc})")
    else:
        selector = None
        policy_resolver = None
        print(f"Edge Filter: none (single-linkage)")
    print(f"Dataset: {dataset}")
    
    # Ensure output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # ==========================================
    # Get Articles
    # ==========================================
    
    if use_fetch:
        if dataset == "mixed":
            print(f"Data source: Wikipedia API (Mixed Dataset) [LIVE]")
            articles = fetch_mixed_dataset()
        else:
            print(f"Data source: Wikipedia API (Sengoku Warlords) [LIVE]")
            articles = fetch_all_warlords()
        
        if len(articles) < 3:
            print("\n[!] Too few articles fetched, falling back to sample data")
            articles = SAMPLE_ARTICLES
    else:
        # Try loading from Harvester cache first
        try:
            from harvester.storage import load_dataset
            articles = load_dataset(dataset)
            print(f"Data source: Harvester cache ({dataset})")
        except (ImportError, FileNotFoundError):
            articles = SAMPLE_ARTICLES
            print(f"\n[Note] Using embedded sample data (3 warlords).")
            print(f"       To use cached data:")
            print(f"         1. python -m harvester.cli harvest --dataset {dataset}")
            print(f"         2. python -m statistics.pipeline.run_full_pipeline --dataset {dataset}")
            print(f"       Or use --fetch for live Wikipedia access.")
    
    print(f"\nArticles to process: {len(articles)}")
    
    # ==========================================
    # Step 1: Feature Extraction (W1) — with cache
    # ==========================================
    
    print_section("Step 1: Feature Extraction (W1)")
    
    cache_valid, cache_reason = check_cache(dataset, articles)
    
    if cache_valid and not force_extract:
        # Cache hit — skip spaCy entirely
        print(f"  Cache: HIT ({cache_reason})")
        all_features, all_sections = load_cache(dataset, list(articles.keys()))
        
        for article_id in articles:
            features = all_features[article_id]
            sections = all_sections[article_id]
            print(f"  {article_id}: {len(features)} tokens, {len(sections)} sections (cached)")
    else:
        # Cache miss — run spaCy extraction
        if force_extract:
            print(f"  Cache: FORCED re-extraction")
        else:
            print(f"  Cache: MISS ({cache_reason})")
        
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
        
        # Save to cache
        save_cache(dataset, articles, all_features, all_sections)
    
    total_tokens = sum(len(f) for f in all_features.values())
    print(f"\n  Total: {total_tokens:,} tokens from {len(articles)} articles")
    
    # ==========================================
    # Step 1.5: Structure Statistics
    # ==========================================
    
    print_section("Step 1.5: Structure Statistics")
    
    structure_stats = compute_article_stats(all_features)
    
    print(compare_articles(structure_stats))
    
    # Export structure stats
    structure_path = os.path.join(output_dir, "structure_stats.json")
    with open(structure_path, 'w', encoding='utf-8') as f:
        export_data = {
            "generated_at": datetime.now().isoformat(),
            "articles": {aid: stats.to_dict() for aid, stats in structure_stats.items()}
        }
        json.dump(export_data, f, indent=2, ensure_ascii=False)
    print(f"\n  Exported: {structure_path}")
    
    # ==========================================
    # Step 2: Conditional Statistics (W2)
    # ==========================================
    
    print_section(f"Step 2: Conditional Statistics (W2) - Axis: {axis}, Mode: {feature_mode}")
    
    aggregator = W2Aggregator(axis=axis, feature_mode=feature_mode)
    
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
    # Step 3-5: Branching by Feature Mode
    # ==========================================
    
    # Initialize variables that may only be set in one branch
    policy_trace = None
    
    if feature_mode == "token":
        # ==========================================
        # TOKEN PATH (existing logic, unchanged)
        # ==========================================
        
        # Step 3: S-Score Calculation (W3)
        print_section("Step 3: S-Score Calculation (W3) [token mode]")
        
        calculator = W3Calculator(w2_stats, top_k=30, min_count=2)
        w3_result = calculator.calculate_all()
        
        print(f"  Conditions analyzed: {len(w3_result.conditions)}")
        
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
        
        # Step 4: Article Projection (W4)
        print_section("Step 4: Article Projection (W4) [token mode]")
        
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
        
        show_count = min(len(sims), 15)
        for a1, a2, sim in sims[:show_count]:
            print(f"    {a1} <-> {a2}: {sim:.4f}")
        if len(sims) > show_count:
            print(f"    ... and {len(sims) - show_count} more pairs")
        
        # Step 4.5: Threshold Resolution (token mode)
        all_sims_token = [sim for _, _, sim in sims]
        dist_summary_token = summarize_distribution(all_sims_token)
        
        threshold_trace = None
        if threshold_mode == "quantile" and all_sims_token:
            # Relative: this experiment's distribution
            t_rel = relative_threshold_quantile(all_sims_token, threshold_q)
            
            # Absolute: global model (all historical data) or fallback
            t_abs, abs_info = global_model.get_threshold(
                lens_key, feature_mode, q=threshold_q, fallback=threshold_fallback,
            )
            
            t_resolved = resolve_threshold(t_abs, t_rel, threshold_resolve)
            threshold_trace = build_threshold_trace(
                t_abs=t_abs,
                t_rel=t_rel,
                t_resolved=t_resolved,
                mode=threshold_mode,
                resolve_strategy=threshold_resolve,
                dist_summary=dist_summary_token,
                quantile_q=threshold_q,
                lens=lens,
                axis=axis,
                feature_mode=feature_mode,
                abs_info=abs_info,
            )
            effective_threshold = t_resolved
            
            print(f"\n  Threshold Resolution:")
            print(f"    Mode:       {threshold_mode}")
            print(f"    Abs (t_abs): {t_abs:.4f}  [{abs_info['source']}]")
            print(f"    Rel (q={threshold_q}): {t_rel:.4f}  [this run]")
            print(f"    Resolved:   {effective_threshold:.4f}  [max(abs, rel)]")
            
            # Append current similarities to global model (after resolution)
            global_model.append(lens_key, feature_mode, all_sims_token, dataset=dataset, axis=axis)
            print(f"    Global model updated: +{len(all_sims_token)} pairs → {global_model.get_count(lens_key, feature_mode)} total")
        else:
            effective_threshold = threshold
            threshold_trace = build_threshold_trace(
                t_abs=threshold,
                t_rel=None,
                t_resolved=threshold,
                mode="fixed",
                resolve_strategy="none",
                dist_summary=dist_summary_token,
                lens=lens,
                axis=axis,
                feature_mode=feature_mode,
            )
        
        # Step 5: Clustering (W5)
        print_section("Step 5: Clustering (W5) [token mode]")
        
        records = convert_to_w4_records(w4_result)
        condensator = SimpleCondensator(threshold=effective_threshold, min_island_size=min_island_size)
        structure = condensator.condense(records)
        
        # Chaining metrics (token mode uses sims from W4)
        all_sims_tuples = [(a, b, s) for a, b, s in sims]
        island_dicts = [
            {"member_ids": island.member_ids, "size": island.size}
            for island in structure.islands
        ]
        chaining = compute_chaining_metrics(
            islands=island_dicts,
            noise_ids=structure.noise_ids,
            input_count=structure.input_count,
            all_sim_pairs=all_sims_tuples,
            threshold=effective_threshold,
        )
        
        print(f"  Input articles: {structure.input_count}")
        print(f"  Islands formed: {structure.island_count}")
        print(f"  Noise articles: {structure.noise_count}")
        
        # Chaining diagnostics
        print(f"\n  Chaining Metrics:")
        print(f"    Giant component ratio: {chaining['giant_component_ratio']:.4f}")
        print(f"    Edge sparsity: {chaining['edge_sparsity']:.4f}")
        if chaining['mean_intra_similarity'] is not None:
            print(f"    Mean intra-similarity: {chaining['mean_intra_similarity']:.4f}")
        
        for i, island in enumerate(structure.islands, 1):
            print(f"\n  Island {i} ({island.size} members):")
            print(f"    Members: {', '.join(island.member_ids)}")
            print(f"    Cohesion: {island.cohesion_score:.4f}")
            print(f"    Top conditions:")
            for cid, score in sorted(island.representative_vector.items(), key=lambda x: -x[1])[:3]:
                print(f"      {cid}: {score:+.4f}")
        
        if structure.noise_ids:
            print(f"\n  Noise (unclustered): {', '.join(structure.noise_ids)}")
        
        # Step 6: Export (W6)
        print_section("Step 6: Export (W6) [token mode]")
        
        md_path = os.path.join(output_dir, "report.md")
        export_structure_markdown(structure, w3_result, md_path)
        print(f"  Markdown: {md_path}")
        
        json_path = os.path.join(output_dir, "analysis.json")
        export_structure_json(structure, w3_result, w4_result, json_path)
        print(f"  JSON: {json_path}")
    
    else:
        # ==========================================
        # VECTOR PATH (Semantic / Hybrid Lens)
        # ==========================================
        
        # Fallback check: conditions < 2 → skip W3/W5
        condition_count = len(w2_stats.conditions)
        if condition_count < 2:
            print_section("Step 3-5: SKIPPED (only 1 condition)")
            print(f"  Semantic Lens with 1 article → no comparison possible.")
            print(f"  Output: document vector profile only.")
            
            # Export single doc vector
            if w2_stats.global_vector_sum is not None:
                global_mean = w2_stats.get_global_mean_vector()
                json_path = os.path.join(output_dir, "analysis.json")
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        "lens": lens or "vector",
                        "feature_mode": "vector",
                        "status": "single_condition_fallback",
                        "global_mean_vector": global_mean,
                    }, f, indent=2)
                print(f"  JSON: {json_path}")
            
            structure = None
        else:
            # Step 3: Vector Profile (W3)
            print_section("Step 3: Vector Profile (W3) [vector mode]")
            
            w3v_calculator = W3VectorCalculator(top_n=5)
            w3v_result = w3v_calculator.calculate(w2_stats)
            
            print(f"  Conditions profiled: {w3v_result.condition_count}")
            print(f"  Total tokens (vectorized): {w3v_result.total_tokens:,}")
            
            # Show top profiles
            shown = 0
            for cid, profile in sorted(w3v_result.profiles.items()):
                if shown >= 8:
                    print(f"\n  ... and {w3v_result.condition_count - 8} more")
                    break
                print(f"\n  [{cid}] ({profile.token_count:,} tokens)")
                if profile.top_positive:
                    dims = ", ".join([f"{d['name']}(z={d['z_score']:+.2f})" for d in profile.top_positive[:3]])
                    print(f"    ↑ {dims}")
                if profile.top_negative:
                    dims = ", ".join([f"{d['name']}(z={d['z_score']:+.2f})" for d in profile.top_negative[:3]])
                    print(f"    ↓ {dims}")
                shown += 1
            
            # Step 4: Vector Similarity (W4)
            print_section("Step 4: Vector Similarity (W4) [vector mode]")
            
            w4v_projector = W4VectorProjector()
            w4v_result = w4v_projector.project(w3v_result)
            
            print(f"  Pairs computed: {len(w4v_result.similarities)}")
            
            print("\n  Top similarities:")
            for pair in w4v_result.similarities[:10]:
                print(f"    {pair['a']} <-> {pair['b']}: {pair['similarity']:.4f}")
            
            if len(w4v_result.similarities) > 10:
                print(f"\n  Bottom similarities:")
                for pair in w4v_result.similarities[-5:]:
                    print(f"    {pair['a']} <-> {pair['b']}: {pair['similarity']:.4f}")
            
            # Step 4.5: Threshold Resolution
            # Observe the similarity distribution and resolve threshold
            all_sims = [pair["similarity"] for pair in w4v_result.similarities]
            dist_summary = summarize_distribution(all_sims)
            
            threshold_trace = None
            if threshold_mode == "quantile" and all_sims:
                # Relative: this experiment's distribution
                t_rel = relative_threshold_quantile(all_sims, threshold_q)
                
                # Absolute: global model (all historical data) or fallback
                t_abs, abs_info = global_model.get_threshold(
                    lens_key, feature_mode, q=threshold_q, fallback=threshold_fallback,
                )
                
                t_resolved = resolve_threshold(t_abs, t_rel, threshold_resolve)
                threshold_trace = build_threshold_trace(
                    t_abs=t_abs,
                    t_rel=t_rel,
                    t_resolved=t_resolved,
                    mode=threshold_mode,
                    resolve_strategy=threshold_resolve,
                    dist_summary=dist_summary,
                    quantile_q=threshold_q,
                    lens=lens,
                    axis=axis,
                    feature_mode=feature_mode,
                    abs_info=abs_info,
                )
                # Use resolved threshold for W5
                effective_threshold = t_resolved
                
                print(f"\n  Threshold Resolution:")
                print(f"    Mode:       {threshold_mode}")
                print(f"    Abs (t_abs): {t_abs:.4f}  [{abs_info['source']}]")
                print(f"    Rel (q={threshold_q}): {t_rel:.4f}  [this run]")
                print(f"    Resolved:   {effective_threshold:.4f}  [max(abs, rel)]")
                print(f"    Distribution: min={dist_summary['min']:.4f}, "
                      f"mean={dist_summary['mean']:.4f}, "
                      f"max={dist_summary['max']:.4f}, "
                      f"std={dist_summary['std']:.4f}")
                
                # Append current similarities to global model (after resolution)
                global_model.append(lens_key, feature_mode, all_sims, dataset=dataset, axis=axis)
                print(f"    Global model updated: +{len(all_sims)} pairs → {global_model.get_count(lens_key, feature_mode)} total")
            else:
                effective_threshold = threshold
                threshold_trace = build_threshold_trace(
                    t_abs=threshold,
                    t_rel=None,
                    t_resolved=threshold,
                    mode="fixed",
                    resolve_strategy="none",
                    dist_summary=dist_summary,
                    lens=lens,
                    axis=axis,
                    feature_mode=feature_mode,
                )
            
            # Step 5: Clustering (W5)
            # Convert vector results to W5-compatible format
            print_section("Step 5: Clustering (W5) [vector mode]")
            
            vectors_dict, sim_pairs = convert_to_w5_input(w4v_result)
            
            # Convert W4 similarity dicts to tuple format for edge selector
            all_sim_tuples = [
                (p["a"], p["b"], p["similarity"])
                for p in w4v_result.similarities
            ]
            
            # Edge selection (Mutual-kNN or pass-through)
            edge_filter_trace = None
            policy_trace = None
            sweep_results = None
            
            if policy_resolver is not None:
                # AUTO MODE: k-sweep with policy-based selection
                node_ids = list(vectors_dict.keys())
                node_vectors = {
                    cid: {f"dim_{i}": v for i, v in enumerate(vec)}
                    for cid, vec in vectors_dict.items()
                }
                
                print(f"  Edge Policy: k-sweep auto-selection")
                print(f"    Policy: giant_ratio ≤ {policy_resolver.max_giant_ratio}, "
                      f"mean_intra ≥ {policy_resolver.min_mean_intra}")
                print(f"    Candidates: {policy_resolver.candidates}")
                
                policy_result = policy_resolver.resolve(
                    all_sim_tuples=all_sim_tuples,
                    threshold=effective_threshold,
                    node_ids=node_ids,
                    node_vectors=node_vectors,
                    min_island_size=min_island_size,
                )
                
                structure = policy_result.structure
                chaining = policy_result.chaining
                edge_filter_trace = policy_result.edge_filter_trace
                policy_trace = policy_result.policy_trace
                sweep_results = policy_result.sweep
                
                # Print sweep table
                print(f"\n    {'k':>3}  {'edges':>6}  {'islands':>7}  {'noise':>5}  "
                      f"{'largest':>7}  {'gcr':>6}  {'mean_intra':>10}  {'ok':>3}")
                print(f"    {'---':>3}  {'------':>6}  {'-------':>7}  {'-----':>5}  "
                      f"{'-------':>7}  {'------':>6}  {'----------':>10}  {'---':>3}")
                for row in sweep_results:
                    mis = f"{row.mean_intra_sim:.4f}" if row.mean_intra_sim is not None else "   n/a"
                    marker = " ←" if row.k == policy_result.k_chosen else ""
                    ok = "✓" if row.satisfies_policy else ""
                    print(f"    {row.k:>3}  {row.n_edges:>6}  {row.n_islands:>7}  {row.n_noise:>5}  "
                          f"{row.largest_island:>7}  {row.giant_ratio:>6.4f}  {mis:>10}  {ok:>3}{marker}")
                
                print(f"\n    → k={policy_result.k_chosen} selected ({policy_result.policy_trace['selection_reason']})")
                
                # Export sweep CSV
                csv_path = os.path.join(output_dir, "k_sweep.csv")
                export_sweep_csv(sweep_results, csv_path)
                print(f"    Sweep CSV: {csv_path}")
                
            elif selector is not None:
                # FIXED k MODE
                filtered_edges, edge_filter_trace = selector.select(
                    all_sim_tuples, effective_threshold,
                )
                
                print(f"  Edge Filter: {edge_filter}")
                print(f"    k = {edge_filter_trace['k']} ({'auto' if edge_filter_trace['k_auto'] else 'fixed'})")
                print(f"    Above threshold: {edge_filter_trace['n_above_threshold']}")
                print(f"    After mutual-kNN: {edge_filter_trace['n_mutual_knn']}")
                print(f"    Reduction: {edge_filter_trace['reduction_ratio']:.1%}")
                
                # Build node vectors for centroid computation
                node_ids = list(vectors_dict.keys())
                node_vectors = {
                    cid: {f"dim_{i}": v for i, v in enumerate(vec)}
                    for cid, vec in vectors_dict.items()
                }
                
                condensator = SimpleCondensator(threshold=effective_threshold, min_island_size=min_island_size)
                structure = condensator.condense_from_edges(
                    node_ids=node_ids,
                    node_vectors=node_vectors,
                    edges=filtered_edges,
                )
            else:
                # No edge filter — original single-linkage behavior
                from .w5_w6_adapter import W4RecordCompat
                records = []
                for cid, vec in vectors_dict.items():
                    dim_dict = {f"dim_{i}": v for i, v in enumerate(vec)}
                    records.append(W4RecordCompat(
                        article_id=cid,
                        w4_analysis_id=f"vec_{cid}",
                        resonance_vector=dim_dict,
                        used_w3={"mode": "vector"},
                        token_count=w4v_result.records[cid].token_count,
                    ))
                
                condensator = SimpleCondensator(threshold=effective_threshold, min_island_size=min_island_size)
                structure = condensator.condense(records)
            
            # Chaining metrics (computed for non-policy paths; policy resolver already has them)
            if policy_resolver is None:
                island_dicts = [
                    {"member_ids": island.member_ids, "size": island.size}
                    for island in structure.islands
                ]
                chaining = compute_chaining_metrics(
                    islands=island_dicts,
                    noise_ids=structure.noise_ids,
                    input_count=structure.input_count,
                    all_sim_pairs=all_sim_tuples,
                    threshold=effective_threshold,
                )
            
            print(f"\n  Threshold used: {effective_threshold:.4f}")
            print(f"  Input conditions: {structure.input_count}")
            print(f"  Islands formed: {structure.island_count}")
            print(f"  Noise: {structure.noise_count}")
            
            # Chaining diagnostics
            print(f"\n  Chaining Metrics:")
            print(f"    Giant component ratio: {chaining['giant_component_ratio']:.4f}")
            print(f"    Largest island: {chaining['largest_island_size']}")
            print(f"    Edge sparsity: {chaining['edge_sparsity']:.4f}")
            if chaining['mean_intra_similarity'] is not None:
                print(f"    Mean intra-similarity: {chaining['mean_intra_similarity']:.4f}")
            if chaining['chaining_detected']:
                print(f"    ⚠ Chaining detected — consider --edge-filter mutual_knn")
            
            for i, island in enumerate(structure.islands, 1):
                print(f"\n  Island {i} ({island.size} members):")
                if island.size <= 30:
                    print(f"    Members: {', '.join(island.member_ids)}")
                else:
                    preview = ', '.join(island.member_ids[:10])
                    print(f"    Members: {preview}, ... and {island.size - 10} more")
                print(f"    Cohesion: {island.cohesion_score:.4f}")
            
            if structure.noise_ids:
                if len(structure.noise_ids) <= 30:
                    print(f"\n  Noise (unclustered): {', '.join(structure.noise_ids)}")
                else:
                    preview = ', '.join(structure.noise_ids[:10])
                    print(f"\n  Noise (unclustered): {preview}, ... and {len(structure.noise_ids) - 10} more")
            
            # Step 6: Export (W6)
            print_section("Step 6: Export (W6) [vector mode]")
            
            # Vector-specific JSON export
            json_path = os.path.join(output_dir, "analysis.json")
            export_data = {
                "lens": lens or "vector",
                "feature_mode": "vector",
                "axis": axis,
                "threshold": effective_threshold,
                "threshold_trace": threshold_trace,
                "edge_filter_trace": edge_filter_trace,
                "edge_policy_trace": policy_trace,
                "chaining_metrics": chaining,
                "w3_vector": export_vector_profiles(w3v_result),
                "w4_vector": export_vector_similarities(w4v_result),
                "w5_clustering": {
                    "input_count": structure.input_count,
                    "island_count": structure.island_count,
                    "noise_count": structure.noise_count,
                    "islands": [
                        {
                            "members": island.member_ids,
                            "size": island.size,
                            "cohesion": island.cohesion_score,
                        }
                        for island in structure.islands
                    ],
                    "noise": structure.noise_ids,
                },
            }
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            print(f"  JSON: {json_path}")
            
            # Markdown report
            md_path = os.path.join(output_dir, "report.md")
            _export_vector_report(
                structure, w3v_result, w4v_result,
                lens or "vector", axis, effective_threshold, md_path,
                threshold_trace=threshold_trace,
                edge_filter_trace=edge_filter_trace,
                chaining_metrics=chaining,
                policy_trace=policy_trace,
            )
            print(f"  Markdown: {md_path}")
    
    # ==========================================
    # Summary
    # ==========================================
    
    print_section("Pipeline Complete!")
    
    summary = {
        "lens": lens or "none",
        "axis": axis,
        "feature_mode": feature_mode,
        "articles": len(articles),
        "total_tokens": total_tokens,
        "conditions": len(w2_stats.conditions),
        "islands": structure.island_count if structure else 0,
        "noise": structure.noise_count if structure else 0,
        "output_dir": output_dir,
    }
    
    # Determine effective threshold for display
    eff_t = effective_threshold if 'effective_threshold' in dir() else threshold
    
    # Edge filter display
    if policy_trace is not None:
        edge_display = f"mutual_knn (k={policy_trace['k_chosen']}, auto-sweep)"
    elif edge_filter != "none":
        edge_display = edge_filter
    else:
        edge_display = "none"
    
    print(f"""
  Configuration:
    Lens:           {lens or 'none (legacy)'}
    Axis:           {axis}
    Feature Mode:   {feature_mode}
    Threshold Mode: {threshold_mode}
    Threshold:      {eff_t:.4f}
    Edge Filter:    {edge_display}
    Min Island:     {min_island_size}
    
  Results:
    Articles:       {summary['articles']}
    Total Tokens:   {summary['total_tokens']:,}
    Conditions:     {summary['conditions']}
    Islands:        {summary['islands']}
    Noise:          {summary['noise']}
    
  Output:
    {output_dir}/
""")
    
    if summary['conditions'] > 1:
        print("  ✅ SUCCESS: Multiple conditions extracted!")
    else:
        print("  ⚠️  Only 1 condition found.")
    
    return summary


# ==========================================
# Vector Report Generator
# ==========================================

def _export_vector_report(structure, w3v_result, w4v_result, lens_name, axis, threshold, path, threshold_trace=None, edge_filter_trace=None, chaining_metrics=None, policy_trace=None):
    """Generate Markdown report for vector-mode pipeline."""
    from .w3_vector import VECTOR_DIM_NAMES
    
    lines = [
        f"# ESDE Phase 9: Vector Analysis Report",
        f"",
        f"- **Lens:** {lens_name}",
        f"- **Axis:** {axis}",
        f"- **Feature Mode:** vector (20-dim)",
        f"- **Threshold:** {threshold:.4f}",
        f"- **Conditions:** {w3v_result.condition_count}",
        f"",
    ]
    
    # Threshold trace section
    if threshold_trace and threshold_trace.get("mode") != "fixed":
        abs_source = threshold_trace.get("abs_source", {})
        source_label = abs_source.get("source", "unknown")
        
        lines.extend([
            f"## Threshold Resolution",
            f"",
            f"- **Mode:** {threshold_trace['mode']}",
            f"- **Absolute (t_abs):** {threshold_trace['t_abs']:.4f} — source: {source_label}",
        ])
        if source_label == "global":
            lines.append(f"  - Global pairs: {abs_source.get('n_global', '?')}")
        elif source_label == "fallback":
            lines.append(f"  - Reason: {abs_source.get('reason', 'unknown')}")
        if threshold_trace.get('t_rel') is not None:
            lines.append(f"- **Relative (q={threshold_trace.get('quantile_q', '?')}):** {threshold_trace['t_rel']:.4f}")
        lines.append(f"- **Resolved:** {threshold_trace['t_resolved']:.4f}")
        dist = threshold_trace.get('run_distribution', {})
        if dist.get('count'):
            lines.append(f"- **Run distribution:** min={dist['min']:.4f}, mean={dist['mean']:.4f}, max={dist['max']:.4f}, std={dist['std']:.4f}")
        lines.append("")
    
    lines.extend([
        f"## Global Mean Vector",
        f"",
    ])
    
    for i, (val, name) in enumerate(zip(w3v_result.global_mean, VECTOR_DIM_NAMES)):
        lines.append(f"- {name}: {val:.4f}")
    
    lines.append("")
    lines.append("## Condition Profiles (top deviations)")
    lines.append("")
    
    for cid, profile in sorted(w3v_result.profiles.items()):
        lines.append(f"### {cid} ({profile.token_count:,} tokens)")
        if profile.top_positive:
            lines.append("**Above average:**")
            for d in profile.top_positive:
                lines.append(f"- {d['name']}: z={d['z_score']:+.2f} (Δ={d['delta']:+.4f})")
        if profile.top_negative:
            lines.append("**Below average:**")
            for d in profile.top_negative:
                lines.append(f"- {d['name']}: z={d['z_score']:+.2f} (Δ={d['delta']:+.4f})")
        lines.append("")
    
    lines.append("## Clustering")
    lines.append("")
    lines.append(f"- Islands: {structure.island_count}")
    lines.append(f"- Noise: {structure.noise_count}")
    lines.append("")
    
    # Edge filter info
    if edge_filter_trace and edge_filter_trace.get("method") != "none":
        lines.extend([
            "### Edge Filter",
            "",
            f"- **Method:** {edge_filter_trace['method']}",
            f"- **k:** {edge_filter_trace['k']} ({'auto' if edge_filter_trace.get('k_auto') else 'fixed'})",
            f"- **Above threshold:** {edge_filter_trace['n_above_threshold']}",
            f"- **After filter:** {edge_filter_trace['n_mutual_knn']}",
            f"- **Reduction:** {edge_filter_trace['reduction_ratio']:.1%}",
            "",
        ])
    
    # Edge Policy (k-sweep) in report
    if policy_trace:
        lines.extend([
            "### Edge Policy (k-sweep)",
            "",
            f"- **Selection:** k={policy_trace['k_chosen']} ({policy_trace['selection_reason']})",
            f"- **Policy:** giant_ratio ≤ {policy_trace['policy']['max_giant_ratio']}, "
            f"mean_intra ≥ {policy_trace['policy']['min_mean_intra']}",
            f"- **Candidates swept:** {len(policy_trace['candidates_swept'])}",
            f"- **Satisfying:** {policy_trace['n_satisfying']}",
            "",
            "| k | islands | noise | largest | gcr | mean_intra | ok |",
            "|--:|--------:|------:|--------:|----:|-----------:|:--:|",
        ])
        for s in policy_trace["sweep_summary"]:
            mis = f"{s['mean_intra']:.4f}" if s["mean_intra"] is not None else "n/a"
            ok = "✓" if s["satisfies"] else ""
            marker = " **←**" if s["k"] == policy_trace["k_chosen"] else ""
            lines.append(
                f"| {s['k']} | {s['islands']} | {s['noise']} | {s['largest']} "
                f"| {s['gcr']:.4f} | {mis} | {ok}{marker} |"
            )
        lines.append("")
    
    # Chaining metrics
    if chaining_metrics:
        lines.extend([
            "### Chaining Diagnostics",
            "",
            f"- **Giant component ratio:** {chaining_metrics['giant_component_ratio']:.4f}",
            f"- **Largest island:** {chaining_metrics['largest_island_size']}",
            f"- **Edge sparsity:** {chaining_metrics['edge_sparsity']:.4f}",
        ])
        if chaining_metrics.get('mean_intra_similarity') is not None:
            lines.append(f"- **Mean intra-similarity:** {chaining_metrics['mean_intra_similarity']:.4f}")
        if chaining_metrics.get('chaining_detected'):
            lines.append(f"- ⚠ **Chaining detected**")
        lines.append("")
    
    for i, island in enumerate(structure.islands, 1):
        lines.append(f"### Island {i} (cohesion: {island.cohesion_score:.4f})")
        for mid in island.member_ids:
            lines.append(f"- {mid}")
        lines.append("")
    
    if structure.noise_ids:
        lines.append("### Noise (unclustered)")
        for nid in structure.noise_ids:
            lines.append(f"- {nid}")
        lines.append("")
    
    with open(path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))


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
                        help="Fetch articles from Wikipedia API")
    parser.add_argument("--dataset", default="warlords",
                        help="Dataset name. Built-in: 'warlords', 'mixed'. Or any Harvester-cached dataset.")
    parser.add_argument("--lens", default=None, choices=list(LENSES.keys()),
                        help="Lens preset (overrides --axis): structure, semantic, hybrid")
    parser.add_argument("--threshold-mode", default="fixed", choices=["fixed", "quantile"],
                        help="Threshold mode: fixed (legacy) or quantile (adaptive)")
    parser.add_argument("--threshold-q", type=float, default=0.98,
                        help="Quantile for adaptive threshold (default: 0.98 = top 2%%)")
    parser.add_argument("--threshold-resolve", default="safety_first", choices=["safety_first"],
                        help="How to combine absolute and relative thresholds")
    parser.add_argument("--edge-filter", default="none", choices=["none", "mutual_knn"],
                        help="Edge filter for W5: none (single-linkage) or mutual_knn (anti-chaining)")
    parser.add_argument("--knn-k", default=None,
                        help="k for mutual-kNN. 'auto' = sweep & auto-select. Integer = fixed k. Default: ceil(log2(N))")
    parser.add_argument("--force-extract", action="store_true",
                        help="Force W1 feature re-extraction (ignore cache)")
    
    args = parser.parse_args()
    
    # Parse knn_k: None, "auto", or integer
    knn_k_raw = args.knn_k
    if knn_k_raw is None:
        knn_k = None
    elif knn_k_raw.lower() == "auto":
        knn_k = "auto"
    else:
        try:
            knn_k = int(knn_k_raw)
        except ValueError:
            parser.error(f"--knn-k must be 'auto' or an integer, got: {knn_k_raw}")
    
    result = run_pipeline(
        axis=args.axis,
        threshold=args.threshold,
        min_island_size=args.min_island,
        output_dir=args.output,
        use_fetch=args.fetch,
        dataset=args.dataset,
        lens=args.lens,
        threshold_mode=args.threshold_mode,
        threshold_q=args.threshold_q,
        threshold_resolve=args.threshold_resolve,
        edge_filter=args.edge_filter,
        knn_k=knn_k,
        force_extract=args.force_extract,
    )
