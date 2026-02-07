#!/usr/bin/env python3
"""
ESDE Relation Pipeline - Real Data Runner
==========================================

Runs the relation extraction pipeline on harvester datasets
and generates a diagnostic report for Design Brief update.

Usage:
  # Run on cached mixed dataset (15 articles)
  python -m integration.relations.run_relations --dataset mixed

  # Run on warlords dataset (10 articles)
  python -m integration.relations.run_relations --dataset warlords

  # Run on single article (live fetch)
  python -m integration.relations.run_relations --article "Oda Nobunaga"

  # Specify synapse file
  python -m integration.relations.run_relations --dataset mixed --synapse esde_synapses_v3.json

  # With Synapse patch overlay (v5.6.0+)
  python -m integration.relations.run_relations --dataset mixed --synapse-patches patches/synapse_v3.1.json

Output:
  output/relations/{dataset}/
    ├── {article_id}_edges.jsonl       # Raw edges per article
    ├── {article_id}_graph.json        # Entity graph per article
    ├── diagnostic_report.json         # Machine-readable diagnostics
    └── diagnostic_report.md           # Human-readable report

Spec: Phase 8 Integration - Observation C
"""

import sys
import re
import json
import time
import argparse
import urllib.request
import urllib.parse
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Any, Optional, Tuple

# Path setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from integration.relations.parser_adapter import ParserAdapter, ExtractionResult
from integration.relations.relation_logger import (
    RelationLogger,
    SynapseGrounder,
    aggregate_entity_graph,
    aggregate_section_profile,
    LIGHT_VERB_STOPLIST,
    POS_GUARD_BLOCKED_CATEGORIES,
)


# ==========================================
# Section Parsing (from test_stage2_real_data.py)
# ==========================================

def clean_wiki_text(text: str) -> str:
    """Clean Wikipedia plain text of residual markup."""
    text = re.sub(r'^(=+)\s*(.+?)\s*\1\s*$', r'\2.', text, flags=re.MULTILINE)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\(listen\)', '', text)
    return text.strip()


def split_into_sections(text: str) -> List[Dict[str, Any]]:
    """Split Wikipedia text into sections based on headings."""
    heading_pattern = re.compile(r'^(=+)\s*(.+?)\s*\1\s*$', re.MULTILINE)
    sections = []
    matches = list(heading_pattern.finditer(text))

    if matches:
        lead_content = text[:matches[0].start()].strip()
    else:
        lead_content = text.strip()

    if lead_content:
        sections.append({
            "title": "Lead",
            "level": 0,
            "content": clean_wiki_text(lead_content),
        })

    skip_titles = {
        "references", "see also", "external links", "notes",
        "further reading", "bibliography", "sources",
    }

    for i, match in enumerate(matches):
        level = len(match.group(1)) - 1
        title = match.group(2).strip()
        content_start = match.end()
        content_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[content_start:content_end].strip()

        if content and title.lower() not in skip_titles:
            sections.append({
                "title": title,
                "level": level,
                "content": clean_wiki_text(content),
            })

    return sections


def make_section_id(title: str) -> str:
    """Normalize section title to ID."""
    return re.sub(r'[^a-z0-9]+', '_', title.lower()).strip('_')


# ==========================================
# Data Loading
# ==========================================

def load_from_harvester(dataset_name: str) -> Dict[str, str]:
    """Load dataset from harvester cache."""
    try:
        from harvester.storage import load_dataset
        articles = load_dataset(dataset_name)
        print(f"  Source: Harvester cache ({dataset_name})")
        return articles
    except (ImportError, FileNotFoundError) as e:
        print(f"  Harvester not available: {e}")
        print(f"  Run: python -m harvester.cli harvest --dataset {dataset_name}")
        return {}


def fetch_single_article(title: str) -> Dict[str, str]:
    """Fetch a single article from Wikipedia."""
    base_url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query", "titles": title,
        "prop": "extracts", "explaintext": "true", "format": "json",
    }
    url = f"{base_url}?{urllib.parse.urlencode(params)}"

    try:
        request = urllib.request.Request(
            url, headers={"User-Agent": "ESDE/1.0 (research project)"}
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            data = json.loads(response.read().decode('utf-8'))

        pages = data.get("query", {}).get("pages", {})
        for page_id, page_data in pages.items():
            if page_id == "-1":
                return {}
            text = page_data.get("extract", "")
            article_id = re.sub(r'[^a-z0-9]+', '_', title.lower()).strip('_')
            return {article_id: text}
        return {}
    except Exception as e:
        print(f"  Fetch error: {e}")
        return {}


# ==========================================
# Single Article Processing
# ==========================================

def process_article(
    article_id: str,
    text: str,
    adapter: ParserAdapter,
    grounder: SynapseGrounder,
    output_dir: Path,
) -> Dict[str, Any]:
    """
    Process one article through the relation pipeline.

    Returns per-article diagnostic dict.
    """
    sections = split_into_sections(text)
    logger = RelationLogger(grounder)

    section_details = []

    for sec in sections:
        sec_id = make_section_id(sec["title"])
        result = adapter.extract(sec["content"], section_name=sec_id)
        edges = logger.process_section(result, sec_id, article_id)

        section_details.append({
            "section": sec_id,
            "sentences": result.sentences_processed,
            "triples": len(result.triples),
            "edges": len(edges),
        })

    all_edges = logger.get_edges()
    stats = logger.get_stats()

    # Write per-article files
    edges_path = output_dir / f"{article_id}_edges.jsonl"
    with open(edges_path, "w", encoding="utf-8") as f:
        for e in all_edges:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    graph = aggregate_entity_graph(all_edges)
    graph_path = output_dir / f"{article_id}_graph.json"
    with open(graph_path, "w", encoding="utf-8") as f:
        json.dump(graph, f, indent=2, ensure_ascii=False)

    # Collect diagnostics
    verb_atoms = Counter()
    verb_scores = {}
    ungrounded_verbs = Counter()
    lightverb_count = 0
    suspicious_groundings = []

    for e in all_edges:
        vl = e["verb_lemma"]
        atom = e["atom"]
        status = e.get("grounding_status", "")

        if status == "UNGROUNDED_LIGHTVERB":
            lightverb_count += 1
        elif atom == "UNGROUNDED":
            ungrounded_verbs[vl] += 1
        else:
            verb_atoms[f"{vl} → {atom}"] += 1
            cands = e.get("atom_candidates", [])
            if cands:
                top = cands[0]
                score = top.get("raw_score", 0)
                verb_scores.setdefault(vl, []).append(score)

                # Suspicious: score < 0.50 after threshold
                cat = atom.split(".")[0] if "." in atom else ""
                if score < 0.50:
                    suspicious_groundings.append({
                        "verb": vl,
                        "atom": atom,
                        "score": round(score, 4),
                        "category": cat,
                        "sentence": e.get("text_ref", "")[:80],
                    })

    # Top entities
    entity_degrees = {}
    for name, data in graph.get("nodes", {}).items():
        entity_degrees[name] = data.get("degree", 0)
    top_entities = sorted(entity_degrees.items(), key=lambda x: -x[1])[:10]

    return {
        "article_id": article_id,
        "sections": len(sections),
        "total_sentences": sum(s["sentences"] for s in section_details),
        "total_triples": stats["triples_processed"],
        "total_edges": len(all_edges),
        "grounded": stats["grounded"],
        "ungrounded": stats["ungrounded"],
        "lightverb": stats.get("lightverb", 0),
        "grounding_rate": stats["grounding_rate"],
        "unique_verbs": len(set(e["verb_lemma"] for e in all_edges)),
        "top_verb_atoms": verb_atoms.most_common(15),
        "ungrounded_verbs": ungrounded_verbs.most_common(15),
        "suspicious_groundings": suspicious_groundings[:20],
        "top_entities": top_entities,
        "section_details": section_details,
        "filter_log": stats.get("filter_log", {}),
    }


# ==========================================
# Diagnostic Report Generation
# ==========================================

def generate_diagnostic_report(
    article_diagnostics: List[Dict],
    output_dir: Path,
    dataset_name: str,
) -> Dict[str, Any]:
    """
    Aggregate diagnostics across all articles and generate report.
    """
    # ── Global aggregation ──
    total_articles = len(article_diagnostics)
    total_triples = sum(d["total_triples"] for d in article_diagnostics)
    total_grounded = sum(d["grounded"] for d in article_diagnostics)
    total_ungrounded = sum(d["ungrounded"] for d in article_diagnostics)
    total_lightverb = sum(d.get("lightverb", 0) for d in article_diagnostics)
    groundable = total_grounded + total_ungrounded
    global_rate = total_grounded / max(groundable, 1)

    # Aggregate verb→atom mappings
    global_verb_atoms = Counter()
    global_ungrounded = Counter()
    global_suspicious = []
    all_verb_scores = defaultdict(list)

    for d in article_diagnostics:
        for mapping, count in d["top_verb_atoms"]:
            global_verb_atoms[mapping] += count
        for verb, count in d["ungrounded_verbs"]:
            global_ungrounded[verb] += count
        global_suspicious.extend(d["suspicious_groundings"])
        # collect scores per verb across articles - need to reparse
        # (we have suspicious_groundings which have scores)

    # Deduplicate suspicious: group by verb→atom
    suspicious_grouped = defaultdict(list)
    for s in global_suspicious:
        key = f"{s['verb']} → {s['atom']}"
        suspicious_grouped[key].append(s)

    # Top suspicious patterns (by frequency)
    suspicious_ranked = sorted(
        suspicious_grouped.items(),
        key=lambda x: -len(x[1])
    )

    # ── Symptom classification ──
    symptoms = []

    # Symptom 1: High-frequency verb consistently misgrounded
    for mapping, instances in suspicious_ranked[:10]:
        if len(instances) >= 3:
            symptoms.append({
                "type": "CONSISTENT_MISGROUND",
                "severity": "HIGH" if len(instances) >= 5 else "MEDIUM",
                "description": f"'{mapping}' appears {len(instances)}x with low score or unexpected category",
                "verb": instances[0]["verb"],
                "atoms": list(set(i["atom"] for i in instances)),
                "count": len(instances),
                "examples": [i["sentence"] for i in instances[:3]],
                "score_range": [
                    min(i["score"] for i in instances),
                    max(i["score"] for i in instances),
                ],
            })

    # Symptom 2: Frequently ungrounded verbs (Synapse coverage gap)
    for verb, count in global_ungrounded.most_common(10):
        if count >= 2:
            symptoms.append({
                "type": "SYNAPSE_COVERAGE_GAP",
                "severity": "HIGH" if count >= 5 else "MEDIUM",
                "description": f"Verb '{verb}' ungrounded {count}x across corpus",
                "verb": verb,
                "count": count,
            })

    # Symptom 3: Noun-category atoms assigned to verbs (category mismatch)
    noun_cats = {"NAT", "MAT", "PRP", "SPA"}
    cat_mismatches = [s for s in global_suspicious if s["category"] in noun_cats]
    if cat_mismatches:
        mismatch_verbs = Counter(s["verb"] for s in cat_mismatches)
        for verb, count in mismatch_verbs.most_common(5):
            examples = [s for s in cat_mismatches if s["verb"] == verb]
            atoms = list(set(s["atom"] for s in examples))
            symptoms.append({
                "type": "CATEGORY_MISMATCH",
                "severity": "HIGH",
                "description": (
                    f"Verb '{verb}' grounded to noun-category atoms {atoms} "
                    f"({count}x). Synapse verb/noun confusion."
                ),
                "verb": verb,
                "atoms": atoms,
                "count": count,
            })

    # Symptom 4: Low global grounding rate
    if global_rate < 0.7:
        symptoms.append({
            "type": "LOW_GROUNDING_RATE",
            "severity": "HIGH",
            "description": f"Global grounding rate {global_rate:.1%} < 70% threshold",
        })

    # Collect filter log from first article that has one
    filter_log = {}
    for d in article_diagnostics:
        fl = d.get("filter_log", {})
        if fl:
            filter_log = fl
            break

    # ── Build report ──
    report = {
        "meta": {
            "dataset": dataset_name,
            "articles": total_articles,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "version": "0.2.0",
        },
        "summary": {
            "total_triples": total_triples,
            "grounded": total_grounded,
            "ungrounded": total_ungrounded,
            "lightverb": total_lightverb,
            "grounding_rate": round(global_rate, 4),
            "grounding_rate_note": "excludes lightverb (intentionally skipped)",
        },
        "filters": {
            "light_verb_stoplist": sorted(LIGHT_VERB_STOPLIST) if 'LIGHT_VERB_STOPLIST' in dir() else [],
            "pos_guard_blocked": sorted(POS_GUARD_BLOCKED_CATEGORIES) if 'POS_GUARD_BLOCKED_CATEGORIES' in dir() else [],
            "min_score": filter_log.get("min_score", "N/A"),
            "pos_guard_dropped": filter_log.get("pos_guard_dropped", 0),
            "threshold_dropped": filter_log.get("threshold_dropped", 0),
            "threshold_drop_top": filter_log.get("threshold_drop_top", [])[:15],
        },
        "per_article": [
            {
                "article": d["article_id"],
                "sections": d["sections"],
                "triples": d["total_triples"],
                "grounding_rate": d["grounding_rate"],
                "lightverb": d.get("lightverb", 0),
                "unique_verbs": d["unique_verbs"],
                "top_entities": d["top_entities"][:5],
            }
            for d in article_diagnostics
        ],
        "top_verb_atom_mappings": global_verb_atoms.most_common(30),
        "top_ungrounded_verbs": global_ungrounded.most_common(20),
        "symptoms": symptoms,
        "suspicious_groundings_sample": [
            {
                "mapping": k,
                "count": len(v),
                "examples": [
                    {"score": i["score"], "sentence": i["sentence"]}
                    for i in v[:3]
                ],
            }
            for k, v in suspicious_ranked[:20]
        ],
    }

    # Write JSON
    json_path = output_dir / "diagnostic_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Write Markdown
    md_path = output_dir / "diagnostic_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(render_markdown_report(report))

    print(f"\n  Diagnostic report: {json_path}")
    print(f"  Diagnostic report: {md_path}")

    return report


# ==========================================
# Markdown Renderer
# ==========================================

def render_markdown_report(report: Dict) -> str:
    """Render diagnostic report as Markdown."""
    lines = []
    meta = report["meta"]
    summary = report["summary"]

    lines.append(f"# ESDE Relation Pipeline — Diagnostic Report")
    lines.append(f"")
    lines.append(f"**Dataset**: {meta['dataset']}  ")
    lines.append(f"**Articles**: {meta['articles']}  ")
    lines.append(f"**Generated**: {meta['generated_at']}  ")
    lines.append(f"")

    # Summary
    lines.append(f"## 1. Summary")
    lines.append(f"")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Total SVO triples | {summary['total_triples']} |")
    lines.append(f"| Grounded | {summary['grounded']} |")
    lines.append(f"| Ungrounded | {summary['ungrounded']} |")
    lines.append(f"| Light verb (skipped) | {summary.get('lightverb', 0)} |")
    lines.append(f"| Grounding rate (excl. lightverb) | {summary['grounding_rate']:.1%} |")
    lines.append(f"")

    # Filter stats
    filters = report.get("filters", {})
    if filters:
        lines.append(f"## 1.5 Filter Statistics (v0.2.0)")
        lines.append(f"")
        lines.append(f"| Filter | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| POS Guard candidates dropped | {filters.get('pos_guard_dropped', 0)} |")
        lines.append(f"| Score threshold candidates dropped | {filters.get('threshold_dropped', 0)} |")
        lines.append(f"| Min score threshold | {filters.get('min_score', 'N/A')} |")
        lines.append(f"| Light verb stoplist | {', '.join(filters.get('light_verb_stoplist', [])[:8])}... |")
        lines.append(f"")
        
        drop_top = filters.get("threshold_drop_top", [])
        if drop_top:
            lines.append(f"**Top threshold-dropped candidates** (verbs that almost made it):")
            lines.append(f"")
            lines.append(f"| Verb | Atom | Score |")
            lines.append(f"|------|------|-------|")
            for verb, atom, score in drop_top[:10]:
                lines.append(f"| {verb} | {atom} | {score} |")
            lines.append(f"")

    # Per-article
    lines.append(f"## 2. Per-Article Breakdown")
    lines.append(f"")
    lines.append(f"| Article | Sections | Triples | Grounding | LightVerb | Verbs |")
    lines.append(f"|---------|----------|---------|-----------|-----------|-------|")
    for a in report["per_article"]:
        lines.append(
            f"| {a['article']} | {a['sections']} | {a['triples']} "
            f"| {a['grounding_rate']:.0%} | {a.get('lightverb', 0)} | {a['unique_verbs']} |"
        )
    lines.append(f"")

    # Symptoms
    lines.append(f"## 3. Symptoms (処方箋のための観測)")
    lines.append(f"")
    if not report["symptoms"]:
        lines.append(f"*No significant symptoms detected.*")
    else:
        for i, sym in enumerate(report["symptoms"], 1):
            sev = sym["severity"]
            icon = "🔴" if sev == "HIGH" else "🟡"
            lines.append(f"### {icon} Symptom {i}: {sym['type']}")
            lines.append(f"")
            lines.append(f"**Severity**: {sev}  ")
            lines.append(f"**Description**: {sym['description']}  ")
            if "examples" in sym:
                lines.append(f"")
                lines.append(f"Examples:")
                for ex in sym["examples"][:3]:
                    lines.append(f"- `{ex}`")
            if "score_range" in sym:
                lo, hi = sym["score_range"]
                lines.append(f"- Score range: {lo:.3f} – {hi:.3f}")
            lines.append(f"")

    # Top mappings
    lines.append(f"## 4. Most Frequent verb → Atom Mappings")
    lines.append(f"")
    lines.append(f"| Mapping | Count |")
    lines.append(f"|---------|-------|")
    for mapping, count in report["top_verb_atom_mappings"][:20]:
        lines.append(f"| {mapping} | {count} |")
    lines.append(f"")

    # Ungrounded verbs
    lines.append(f"## 5. Top Ungrounded Verbs")
    lines.append(f"")
    lines.append(f"| Verb | Count |")
    lines.append(f"|------|-------|")
    for verb, count in report["top_ungrounded_verbs"][:15]:
        lines.append(f"| {verb} | {count} |")
    lines.append(f"")

    # Suspicious groundings
    lines.append(f"## 6. Suspicious Groundings (Low Score / Category Mismatch)")
    lines.append(f"")
    for item in report["suspicious_groundings_sample"][:15]:
        lines.append(f"**{item['mapping']}** ({item['count']}x)")
        for ex in item["examples"][:2]:
            lines.append(f"- score={ex['score']:.3f}: `{ex['sentence']}`")
        lines.append(f"")

    return "\n".join(lines)


# ==========================================
# Main
# ==========================================

def main():
    parser = argparse.ArgumentParser(
        description="ESDE Relation Pipeline - Real Data Runner"
    )
    parser.add_argument("--dataset", type=str, default=None,
                        help="Harvester dataset name (mixed, warlords)")
    parser.add_argument("--article", type=str, default=None,
                        help="Single Wikipedia article title (live fetch)")
    parser.add_argument("--synapse", type=str, default="esde_synapses_v3.json",
                        help="Path to Synapse file")
    parser.add_argument("--min-score", type=float, default=0.45,
                        help="Minimum grounding score threshold (default: 0.45)")
    parser.add_argument("--output", type=str, default="output/relations",
                        help="Output directory")
    parser.add_argument("--max-articles", type=int, default=None,
                        help="Limit number of articles to process")
    # ── v5.6.0: Synapse patch overlay support ──
    parser.add_argument("--synapse-patches", type=str, nargs="*", default=None,
                        help="Synapse patch files to overlay (e.g. patches/synapse_v3.1.json)")
    args = parser.parse_args()

    if not args.dataset and not args.article:
        parser.error("Either --dataset or --article is required")

    print("=" * 70)
    print("ESDE Relation Pipeline — Real Data Runner")
    print("=" * 70)

    # Load articles
    if args.article:
        print(f"\n[1] Fetching: {args.article}")
        articles = fetch_single_article(args.article)
        dataset_name = "single"
    else:
        print(f"\n[1] Loading dataset: {args.dataset}")
        articles = load_from_harvester(args.dataset)
        dataset_name = args.dataset

    if not articles:
        print("  No articles loaded. Exiting.")
        return 1

    if args.max_articles:
        keys = list(articles.keys())[:args.max_articles]
        articles = {k: articles[k] for k in keys}

    print(f"  Articles: {len(articles)}")

    # Initialize pipeline
    print(f"\n[2] Initializing pipeline")
    adapter = ParserAdapter()
    synapse_path = Path(args.synapse)
    # ── v5.6.0: SynapseStore overlay when patches specified ──
    if args.synapse_patches:
        from synapse.store import SynapseStore
        store = SynapseStore()
        store.load(str(synapse_path), patches=args.synapse_patches)
        grounder = SynapseGrounder(synapse_data=store.get_synapse_dict(), min_score=args.min_score)
    elif synapse_path.exists():
        grounder = SynapseGrounder.from_file(str(synapse_path), min_score=args.min_score)
    else:
        print(f"  Warning: Synapse file not found at {synapse_path}, running raw mode")
        grounder = SynapseGrounder(min_score=args.min_score)

    # Output directory
    out_dir = Path(args.output) / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Output: {out_dir}")

    # Process articles
    print(f"\n[3] Processing articles")
    print("-" * 70)

    all_diagnostics = []
    t_start = time.time()

    for i, (article_id, text) in enumerate(articles.items(), 1):
        t0 = time.time()
        print(f"  [{i}/{len(articles)}] {article_id}...", end=" ", flush=True)

        diag = process_article(article_id, text, adapter, grounder, out_dir)
        all_diagnostics.append(diag)

        elapsed = time.time() - t0
        rate = f"{diag['grounding_rate']:.0%}"
        lv = diag.get('lightverb', 0)
        print(
            f"{diag['sections']} sec, "
            f"{diag['total_triples']} triples, "
            f"grounding={rate}, "
            f"lightverb={lv}, "
            f"{elapsed:.1f}s"
        )

    total_time = time.time() - t_start
    print("-" * 70)
    print(f"  Total: {total_time:.1f}s for {len(articles)} articles")

    # Generate diagnostic report
    print(f"\n[4] Generating diagnostic report")
    report = generate_diagnostic_report(all_diagnostics, out_dir, dataset_name)

    # Quick summary
    symptoms = report["symptoms"]
    print(f"\n{'=' * 70}")
    print(f"SYMPTOMS DETECTED: {len(symptoms)}")
    print(f"{'=' * 70}")
    for sym in symptoms:
        sev = sym["severity"]
        icon = "🔴" if sev == "HIGH" else "🟡"
        print(f"  {icon} [{sym['type']}] {sym['description']}")

    grounding = report["summary"]["grounding_rate"]
    lightverb = report["summary"].get("lightverb", 0)
    filters = report.get("filters", {})
    print(f"\n  Global grounding rate: {grounding:.1%} (excl. {lightverb} lightverb)")
    print(f"  Total triples: {report['summary']['total_triples']}")
    print(f"  POS Guard dropped: {filters.get('pos_guard_dropped', 0)} candidates")
    print(f"  Threshold dropped: {filters.get('threshold_dropped', 0)} candidates (min_score={filters.get('min_score', 'N/A')})")
    print(f"\n  Report: {out_dir}/diagnostic_report.md")

    return 0


if __name__ == "__main__":
    sys.exit(main())