#!/usr/bin/env python3
"""
ESDE Relation Pipeline - Integration Test
==========================================

Tests the full relation extraction pipeline:
  1. ParserAdapter: spaCy SVO extraction
  2. RelationLogger: Synapse grounding + edge output
  3. Aggregation: entity_graph + section_profile

Usage:
    python -m pytest tests/test_relations.py -v
    python tests/test_relations.py                    # Direct run
    python tests/test_relations.py --synapse PATH     # With Synapse grounding

Default synapse location: esde/esde_synapses_v3.json
    
Requires:
    pip install spacy
    python -m spacy download en_core_web_sm
"""

import sys
import json
import argparse
import tempfile
from pathlib import Path

# Add project root to path — handles both:
#   python tests/test_relations.py        (cwd = esde/)
#   python -m pytest tests/test_relations.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Debug: verify the import path
_relations_pkg = PROJECT_ROOT / "integration" / "relations"
if not _relations_pkg.exists():
    print(f"ERROR: Expected package at {_relations_pkg}")
    print(f"  PROJECT_ROOT = {PROJECT_ROOT}")
    print(f"  Contents of integration/:")
    _int_dir = PROJECT_ROOT / "integration"
    if _int_dir.exists():
        for p in sorted(_int_dir.iterdir()):
            print(f"    {p.name}")
    else:
        print(f"    (directory does not exist)")
    sys.exit(1)

_init_file = _relations_pkg / "__init__.py"
if not _init_file.exists():
    print(f"ERROR: Missing {_init_file}")
    print(f"  The integration/relations/__init__.py file must exist.")
    print(f"  Files in integration/relations/:")
    for p in sorted(_relations_pkg.iterdir()):
        print(f"    {p.name}")
    sys.exit(1)

from integration.relations.parser_adapter import ParserAdapter, SVOTriple, ExtractionResult
from integration.relations.relation_logger import (
    RelationLogger,
    SynapseGrounder,
    aggregate_entity_graph,
    aggregate_section_profile,
    run_relation_pipeline,
)


# ==========================================
# Test Data: Sengoku Period
# ==========================================

SENGOKU_SECTIONS = [
    {
        "title": "early_life",
        "content": (
            "Oda Nobunaga was born in 1534 in Owari Province. "
            "His father Nobuhide controlled a small domain. "
            "Nobunaga inherited the leadership after his father died."
        ),
    },
    {
        "title": "battle_of_okehazama",
        "content": (
            "In 1560, Nobunaga defeated Imagawa Yoshimoto at Okehazama. "
            "Yoshimoto had assembled a massive army of 25,000 soldiers. "
            "Nobunaga launched a surprise attack during a thunderstorm. "
            "The victory shocked the other daimyo across Japan."
        ),
    },
    {
        "title": "alliance_and_betrayal",
        "content": (
            "Nobunaga and Tokugawa formed a powerful alliance. "
            "Together they conquered Mino and Omi provinces. "
            "However, Akechi Mitsuhide betrayed Nobunaga at Honnoji. "
            "Nobunaga was killed in the attack. "
            "Tokugawa did not betray the alliance."
        ),
    },
    {
        "title": "legacy",
        "content": (
            "Hideyoshi seized control of the government after Nobunaga's death. "
            "He unified Japan through military conquest and diplomacy. "
            "The Azai and Asakura clans were destroyed by Nobunaga's campaigns."
        ),
    },
]


# ==========================================
# Test 1: ParserAdapter
# ==========================================

def test_parser_adapter():
    """Test SVO extraction from known sentences."""
    print("\n" + "=" * 60)
    print("[Test 1] ParserAdapter - SVO Extraction")
    print("=" * 60)
    
    adapter = ParserAdapter()
    
    # Test cases: (input, expected_min_triples, expected_patterns)
    test_cases = [
        (
            "Nobunaga attacked the Azai clan.",
            1,
            [("Nobunaga", "attack", "Azai")],  # subject contains, lemma, object contains
        ),
        (
            "The Azai were defeated by Nobunaga.",
            1,
            [("Nobunaga", "defeat", "Azai")],  # Passive: agent=Nobunaga, patient=Azai
        ),
        (
            "Tokugawa did not betray the alliance.",
            1,
            [("Tokugawa", "betray", "alliance")],  # Negation
        ),
        (
            "Nobunaga and Tokugawa formed an alliance.",
            1,  # At least 1 (could be 2 with conjunction expansion)
            [("Nobunaga", "form", "alliance")],
        ),
        (
            "The empire collapsed.",
            0,  # Intransitive, no object → no triple
            [],
        ),
    ]
    
    all_passed = True
    for text, min_triples, patterns in test_cases:
        result = adapter.extract(text)
        
        count = len(result.triples)
        status = "✅" if count >= min_triples else "❌"
        if count < min_triples:
            all_passed = False
        
        print(f"\n  {status} \"{text}\"")
        print(f"     Triples: {count} (expected ≥ {min_triples})")
        
        for t in result.triples:
            neg = " [NEG]" if t.negated else ""
            pas = " [PASSIVE]" if t.passive else ""
            print(f"     → {t.subject} ▷ {t.verb_lemma}{neg}{pas} ▷ {t.object}")
        
        # Check patterns
        for subj_pat, lemma_pat, obj_pat in patterns:
            matched = any(
                subj_pat.lower() in t.subject.lower()
                and t.verb_lemma == lemma_pat
                and obj_pat.lower() in t.object.lower()
                for t in result.triples
            )
            if not matched and count > 0:
                print(f"     ⚠️  Pattern not matched: {subj_pat} ▷ {lemma_pat} ▷ {obj_pat}")
    
    # Test negation detection
    neg_result = adapter.extract("Tokugawa did not betray the alliance.")
    if neg_result.triples:
        neg_ok = neg_result.triples[0].negated
        print(f"\n  {'✅' if neg_ok else '❌'} Negation detected: {neg_ok}")
        if not neg_ok:
            all_passed = False
    
    # Test passive detection
    pas_result = adapter.extract("The castle was destroyed by the army.")
    if pas_result.triples:
        pas_ok = pas_result.triples[0].passive
        print(f"  {'✅' if pas_ok else '❌'} Passive detected: {pas_ok}")
        if not pas_ok:
            all_passed = False
    
    stats = adapter.get_stats()
    print(f"\n  Stats: {stats}")
    
    return all_passed


# ==========================================
# Test 2: Batch Extraction
# ==========================================

def test_batch_extraction():
    """Test extraction across multiple sections."""
    print("\n" + "=" * 60)
    print("[Test 2] Batch Extraction - Sengoku Sections")
    print("=" * 60)
    
    adapter = ParserAdapter()
    results = adapter.extract_batch(SENGOKU_SECTIONS)
    
    total_triples = 0
    for section_name, result in results.items():
        count = len(result.triples)
        total_triples += count
        print(f"\n  Section: {section_name}")
        print(f"    Sentences: {result.sentences_processed}")
        print(f"    Triples: {count}")
        for t in result.triples:
            neg = " [NEG]" if t.negated else ""
            pas = " [PASSIVE]" if t.passive else ""
            print(f"      {t.subject} ▷ {t.verb_lemma}{neg}{pas} ▷ {t.object}")
    
    print(f"\n  Total triples across all sections: {total_triples}")
    
    # We expect at least some triples from this text
    ok = total_triples >= 5
    print(f"  {'✅' if ok else '❌'} Minimum triples threshold (≥5): {total_triples}")
    
    return ok


# ==========================================
# Test 3: Relation Logger (without Synapse)
# ==========================================

def test_relation_logger_raw():
    """Test RelationLogger in raw mode (no Synapse grounding)."""
    print("\n" + "=" * 60)
    print("[Test 3] RelationLogger - Raw Mode (no Synapse)")
    print("=" * 60)
    
    adapter = ParserAdapter()
    logger = RelationLogger()  # No grounder → raw mode
    
    for sec in SENGOKU_SECTIONS:
        result = adapter.extract(sec["content"])
        logger.process_section(result, sec["title"], "oda_nobunaga")
    
    edges = logger.get_edges()
    stats = logger.get_stats()
    
    print(f"\n  Total edges: {len(edges)}")
    print(f"  Grounding rate: {stats['grounding_rate']:.1%} (expected 0% in raw mode)")
    
    # All should be UNGROUNDED or UNGROUNDED_LIGHTVERB in raw mode
    ungrounded = sum(1 for e in edges if e["atom"] in ("UNGROUNDED", "UNGROUNDED_LIGHTVERB"))
    print(f"  UNGROUNDED (incl. lightverb): {ungrounded}/{len(edges)}")
    
    # Show sample edges
    for e in edges[:5]:
        print(f"    {e['source']} ▷ {e['atom']}({e['verb_lemma']}) ▷ {e['target']}")
    
    ok = len(edges) > 0 and stats["grounding_rate"] == 0.0
    print(f"\n  {'✅' if ok else '❌'} Raw mode working correctly")
    
    return ok


# ==========================================
# Test 4: Relation Logger (with Synapse)
# ==========================================

def test_relation_logger_grounded(synapse_path: str):
    """Test RelationLogger with actual Synapse data."""
    print("\n" + "=" * 60)
    print("[Test 4] RelationLogger - Grounded Mode")
    print("=" * 60)
    
    adapter = ParserAdapter()
    grounder = SynapseGrounder.from_file(synapse_path)
    logger = RelationLogger(grounder)
    
    for sec in SENGOKU_SECTIONS:
        result = adapter.extract(sec["content"])
        logger.process_section(result, sec["title"], "oda_nobunaga")
    
    edges = logger.get_edges()
    stats = logger.get_stats()
    
    print(f"\n  Total edges: {len(edges)}")
    print(f"  Grounding rate: {stats['grounding_rate']:.1%}")
    print(f"  Grounded: {stats['grounded']}, Ungrounded: {stats['ungrounded']}")
    
    # Show edges with grounding
    grounded_edges = [e for e in edges if e["atom"] not in ("UNGROUNDED", "UNGROUNDED_LIGHTVERB")]
    lightverb_edges = [e for e in edges if e.get("grounding_status") == "UNGROUNDED_LIGHTVERB"]
    print(f"\n  Sample grounded edges:")
    for e in grounded_edges[:8]:
        top_score = e["atom_candidates"][0]["raw_score"] if e["atom_candidates"] else 0
        print(f"    {e['source']} ▷ {e['atom']} (score={top_score:.3f}) ▷ {e['target']}")
        if len(e["atom_candidates"]) > 1:
            alt = e["atom_candidates"][1]
            print(f"      alt: {alt['concept_id']} (score={alt['raw_score']:.3f})")
    
    if lightverb_edges:
        print(f"\n  Light verb edges (atom suppressed, edge preserved): {len(lightverb_edges)}")
        for e in lightverb_edges[:3]:
            print(f"    {e['source']} ▷ LIGHTVERB({e['verb_lemma']}) ▷ {e['target']}")
    
    ungrounded_edges = [e for e in edges if e["atom"] == "UNGROUNDED"]
    if ungrounded_edges:
        print(f"\n  Ungrounded verbs:")
        seen = set()
        for e in ungrounded_edges:
            if e["verb_lemma"] not in seen:
                print(f"    {e['verb_lemma']}")
                seen.add(e["verb_lemma"])
    
    ok = len(edges) > 0 and stats["grounding_rate"] > 0
    print(f"\n  {'✅' if ok else '❌'} Grounding working (rate > 0%)")
    
    # Show filter stats
    flog = stats.get("filter_log", {})
    if flog:
        print(f"\n  Filter stats:")
        print(f"    POS Guard dropped: {flog.get('pos_guard_dropped', 0)} candidates")
        print(f"    Threshold dropped: {flog.get('threshold_dropped', 0)} candidates (min_score={flog.get('min_score', 'N/A')})")
        drop_top = flog.get("threshold_drop_top", [])
        if drop_top:
            print(f"    Top threshold-dropped:")
            for verb, atom, score in drop_top[:5]:
                print(f"      {verb} → {atom} (score={score})")
    
    return ok


# ==========================================
# Test 5: Aggregation
# ==========================================

def test_aggregation():
    """Test entity graph and section profile aggregation."""
    print("\n" + "=" * 60)
    print("[Test 5] Aggregation - Entity Graph + Section Profile")
    print("=" * 60)
    
    # Create sample edges
    sample_edges = [
        {"source": "Nobunaga", "target": "Azai", "atom": "ACT.attack",
         "section": "battle", "negated": False, "passive": False,
         "text_ref": "Nobunaga attacked the Azai"},
        {"source": "Nobunaga", "target": "Azai", "atom": "ACT.attack",
         "section": "battle", "negated": False, "passive": False,
         "text_ref": "Nobunaga defeated Azai forces"},
        {"source": "Nobunaga", "target": "Tokugawa", "atom": "REL.alliance",
         "section": "alliance", "negated": False, "passive": False,
         "text_ref": "Nobunaga formed alliance with Tokugawa"},
        {"source": "Mitsuhide", "target": "Nobunaga", "atom": "ACT.attack",
         "section": "betrayal", "negated": False, "passive": False,
         "text_ref": "Mitsuhide betrayed Nobunaga"},
    ]
    
    # Entity Graph
    graph = aggregate_entity_graph(sample_edges)
    
    print(f"\n  Entity Graph:")
    print(f"    Nodes: {graph['meta']['total_nodes']}")
    print(f"    Edges: {graph['meta']['total_edges']}")
    
    for name, data in graph["nodes"].items():
        print(f"    {name}: degree={data['degree']}, "
              f"source={data['as_source']}, target={data['as_target']}")
    
    # Verify Nobunaga has highest degree
    nobunaga = graph["nodes"].get("Nobunaga", {})
    ok_graph = nobunaga.get("degree", 0) >= 3
    print(f"  {'✅' if ok_graph else '❌'} Nobunaga highest degree: {nobunaga.get('degree', 0)}")
    
    # Section Profile
    profile = aggregate_section_profile(sample_edges)
    
    print(f"\n  Section Profiles:")
    for sec_name, data in profile.items():
        print(f"    {sec_name}: edges={data['edge_count']}, "
              f"entities={data['entity_count']}, "
              f"atoms={dict(data['predicate_atoms'])}")
    
    ok_profile = len(profile) == 3  # battle, alliance, betrayal
    print(f"  {'✅' if ok_profile else '❌'} Section count: {len(profile)} (expected 3)")
    
    return ok_graph and ok_profile


# ==========================================
# Test 6: JSONL Output
# ==========================================

def test_jsonl_output():
    """Test JSONL file output and readback."""
    print("\n" + "=" * 60)
    print("[Test 6] JSONL Output")
    print("=" * 60)
    
    adapter = ParserAdapter()
    logger = RelationLogger()
    
    for sec in SENGOKU_SECTIONS:
        result = adapter.extract(sec["content"])
        logger.process_section(result, sec["title"], "oda_nobunaga")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_edges.jsonl"
        count = logger.write_jsonl(str(path))
        
        print(f"  Written: {count} edges to {path.name}")
        
        # Read back and validate
        read_back = []
        with open(path, "r") as f:
            for line in f:
                edge = json.loads(line.strip())
                read_back.append(edge)
        
        print(f"  Read back: {len(read_back)} edges")
        
        # Verify structure
        if read_back:
            e = read_back[0]
            required_fields = {"source", "target", "atom", "operator",
                             "negated", "passive", "verb_lemma", "section",
                             "article", "sentence_idx", "text_ref"}
            present = set(e.keys())
            missing = required_fields - present
            
            ok = len(missing) == 0
            if missing:
                print(f"  ❌ Missing fields: {missing}")
            else:
                print(f"  ✅ All required fields present")
            
            print(f"  Sample: {e['source']} ▷ {e['atom']} ▷ {e['target']}")
        else:
            ok = False
            print(f"  ❌ No edges written")
    
    return ok


# ==========================================
# Test 7: Full Pipeline
# ==========================================

def test_full_pipeline(synapse_path: str = None):
    """Test the complete run_relation_pipeline function."""
    print("\n" + "=" * 60)
    print("[Test 7] Full Pipeline")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        summary = run_relation_pipeline(
            sections=SENGOKU_SECTIONS,
            article_id="oda_nobunaga",
            synapse_path=synapse_path,
            output_dir=tmpdir,
        )
        
        print(f"\n  Article: {summary['article_id']}")
        print(f"  Files:")
        for name, path in summary["files"].items():
            exists = Path(path).exists()
            size = Path(path).stat().st_size if exists else 0
            print(f"    {name}: {'✅' if exists else '❌'} ({size:,} bytes)")
        
        print(f"  Stats: {summary['stats']}")
        print(f"  Parser: {summary['parser_stats']}")
        
        # Verify all files exist
        all_exist = all(Path(p).exists() for p in summary["files"].values())
        
        # Load and verify relations.json
        rpath = Path(summary["files"]["relations_json"])
        if rpath.exists():
            with open(rpath) as f:
                relations = json.load(f)
            print(f"\n  relations.json: {len(relations['nodes'])} nodes, "
                  f"{len(relations['edges'])} edges")
        
        print(f"\n  {'✅' if all_exist else '❌'} All output files generated")
        
        return all_exist


# ==========================================
# Main
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="ESDE Relation Pipeline Tests")
    parser.add_argument("--synapse", type=str, default=None,
                       help="Path to esde_synapses_v3.json for grounded tests")
    args = parser.parse_args()
    
    print("#" * 60)
    print("# ESDE Relation Pipeline - Integration Test")
    print("#" * 60)
    
    results = {}
    
    # Test 1: Parser Adapter
    results["parser_adapter"] = test_parser_adapter()
    
    # Test 2: Batch Extraction
    results["batch_extraction"] = test_batch_extraction()
    
    # Test 3: Raw Logger
    results["logger_raw"] = test_relation_logger_raw()
    
    # Test 4: Grounded Logger (only if synapse provided)
    if args.synapse:
        results["logger_grounded"] = test_relation_logger_grounded(args.synapse)
    else:
        print("\n  ⏭️  Test 4 skipped (no --synapse path provided)")
    
    # Test 5: Aggregation
    results["aggregation"] = test_aggregation()
    
    # Test 6: JSONL Output
    results["jsonl_output"] = test_jsonl_output()
    
    # Test 7: Full Pipeline
    results["full_pipeline"] = test_full_pipeline(args.synapse)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}  {name}")
        if not passed:
            all_passed = False
    
    print(f"\n  Overall: {'✅ ALL PASSED' if all_passed else '❌ SOME FAILED'}")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())