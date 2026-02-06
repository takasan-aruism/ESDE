#!/usr/bin/env python3
"""
Synapse Expansion — Integration Smoke Test
=============================================
Validates that default_pipeline_runner correctly connects to
the real run_relations.py pipeline before running propose-synapse.

This catches integration bugs (import mismatches, schema gaps, etc.)
that mock-based unit tests cannot detect.

Usage:
  python tests/test_integration_smoke.py

Prerequisites:
  - Harvester cache for 'mixed' dataset
  - esde_synapses_v3.json
  - spaCy en_core_web_sm model

Expected: All 5 checks pass. If any fail, fix before running propose-synapse.

3AI: Gemini (design) → GPT (audit) → Claude (implementation)
"""

import json
import os
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def check(label: str, condition: bool, detail: str = ""):
    icon = "✅" if condition else "❌"
    msg = f"  {icon} {label}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    return condition


def main():
    print("=" * 60)
    print("Synapse Expansion — Integration Smoke Test")
    print("=" * 60)
    passed = 0
    total = 0

    # ── Check 1: Imports resolve ──
    total += 1
    try:
        from synapse.cli import default_pipeline_runner, generate_run_id
        from synapse.store import SynapseStore
        from synapse.diagnostic import DiagnosticResult
        from synapse.proposer import SynapseEdgeProposer
        passed += check("Import synapse package", True)
    except ImportError as e:
        check("Import synapse package", False, str(e))

    # ── Check 2: Lazy imports in default_pipeline_runner resolve ──
    total += 1
    try:
        from integration.relations.run_relations import (
            load_from_harvester,
            process_article,
            generate_diagnostic_report,
        )
        from integration.relations.parser_adapter import ParserAdapter
        from integration.relations.relation_logger import SynapseGrounder
        passed += check("Import run_relations pipeline", True)
    except ImportError as e:
        check("Import run_relations pipeline", False, str(e))

    # ── Check 3: Harvester cache exists ──
    total += 1
    try:
        articles = load_from_harvester("mixed")
        has_articles = articles is not None and len(articles) > 0
        passed += check(
            "Harvester 'mixed' cache",
            has_articles,
            f"{len(articles)} articles" if has_articles else "empty/missing",
        )
    except Exception as e:
        check("Harvester 'mixed' cache", False, str(e))

    # ── Check 4: SynapseStore loads and exports ──
    total += 1
    synapse_path = "esde_synapses_v3.json"
    try:
        store = SynapseStore()
        loaded = store.load(synapse_path)
        synapse_dict = store.get_synapse_dict()
        has_data = loaded and len(synapse_dict) > 0
        passed += check(
            "SynapseStore.load + get_synapse_dict",
            has_data,
            f"{len(synapse_dict)} synsets" if has_data else "failed",
        )
    except Exception as e:
        check("SynapseStore.load + get_synapse_dict", False, str(e))

    # ── Check 5: Single-article pipeline run ──
    total += 1
    try:
        # Pick the first article only (fast)
        first_id = list(articles.keys())[0]
        first_text = articles[first_id]

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            adapter = ParserAdapter()

            # Export store to temp file for grounder
            tmp_syn = os.path.join(tmpdir, "synapse_tmp.json")
            with open(tmp_syn, "w", encoding="utf-8") as f:
                json.dump({"synapses": synapse_dict}, f, ensure_ascii=False)

            grounder = SynapseGrounder.from_file(tmp_syn, min_score=0.45)

            diag = process_article(first_id, first_text, adapter, grounder, out_dir)

            # Verify per-article diagnostic has required fields
            has_fields = all(
                k in diag
                for k in ["grounded", "ungrounded", "grounding_rate", "total_triples"]
            )
            passed += check(
                f"Single article run ({first_id})",
                has_fields,
                f"{diag['total_triples']} triples, rate={diag['grounding_rate']:.0%}",
            )
    except Exception as e:
        check("Single article run", False, str(e))

    # ── Check 6: Full report generation + DiagnosticResult compatibility ──
    total += 1
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)

            # Reuse the single diag from check 5
            report = generate_diagnostic_report([diag], out_dir, "smoke_test")

            # Wrap in DiagnosticResult — this is the real integration point
            dr = DiagnosticResult(raw=report)

            # Verify accessors work
            rate_ok = isinstance(dr.grounding_rate, float)
            symptoms_ok = isinstance(dr.symptoms, list)
            gaps = dr.coverage_gaps
            gaps_have_verb = all("verb" in g for g in gaps)

            all_ok = rate_ok and symptoms_ok and gaps_have_verb
            detail_parts = [
                f"rate={dr.grounding_rate:.1%}",
                f"symptoms={len(dr.symptoms)}",
                f"gaps={len(gaps)}",
            ]

            # Check CONSISTENT_MISGROUND has verb field (the bug fix)
            misgrounds = dr.consistent_misgrounds
            if misgrounds:
                mg_has_verb = all("verb" in m for m in misgrounds)
                if not mg_has_verb:
                    all_ok = False
                    detail_parts.append("⚠ CONSISTENT_MISGROUND missing 'verb' field!")
                else:
                    detail_parts.append(f"misgrounds={len(misgrounds)}(verb✓)")

            passed += check(
                "DiagnosticResult compatibility",
                all_ok,
                ", ".join(detail_parts),
            )
    except Exception as e:
        check("DiagnosticResult compatibility", False, str(e))

    # ── Summary ──
    print()
    print(f"{'=' * 60}")
    print(f"Result: {passed}/{total} passed")
    if passed == total:
        print("🟢 All checks passed. Safe to run propose-synapse.")
    else:
        print("🔴 Fix failures before running propose-synapse.")
    print(f"{'=' * 60}")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
