#!/usr/bin/env python3
"""
ESDE Synapse Expansion CLI (Phase 3)
======================================
Two-phase workflow for safe Synapse expansion:

  1. propose-synapse:    Diagnose → Propose → Baseline
  2. evaluate-synapse-patch: Overlay → Re-diagnose → Diff → Verdict

Design Spec: v3.1 (Gemini) + GPT audit amendments
  - Run-ID scoped directories (GPT §1: collision-resistant)
  - Environment metadata in diagnostics (GPT §2)
  - Machine-judged FAIL conditions from DiagnosticResult (GPT §3)
  - No writes outside run-dir (GPT §4)
  - Baseline patch auto-inheritance in evaluate (GPT §5, v5.6.1)

Usage:
  python -m synapse.cli propose-synapse \\
      --dataset mixed \\
      --synapse esde_synapses_v3.json \\
      --dictionary esde_dictionary.json \\
      --synapse-patches patches/synapse_v3.1.json

  python -m synapse.cli evaluate-synapse-patch \\
      --run-dir proposals/run_20260206_143000_mixed_a1b2/ \\
      --dataset mixed

3AI: Gemini (design) → GPT (audit) → Claude (implementation)
"""

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .diagnostic import DiagnosticResult
from .proposer import SynapseEdgeProposer
from .schema import SynapsePatchEntry
from .store import SynapseStore

logger = logging.getLogger(__name__)

# ─── Constants ───────────────────────────────────────────────────────

PROPOSALS_BASE_DIR = "proposals"
DEFAULT_MIN_FREQ = 2


# ─── Protocol: Pipeline Runner ───────────────────────────────────────
# Dependency injection point for testing.
# In production, this calls the real run_relations pipeline.

PipelineRunnerFn = Callable[
    [str, "SynapseStore", float, Path],  # dataset, store, min_score, output_dir
    Dict[str, Any],  # raw diagnostic report dict
]


def default_pipeline_runner(
    dataset: str,
    synapse_store: SynapseStore,
    min_score: float,
    output_dir: Path,
) -> Dict[str, Any]:
    """
    Default pipeline runner: imports and calls run_relations.

    This integrates with the existing Relation Pipeline by:
    1. Writing a temp Synapse JSON from SynapseStore
    2. Invoking run_relations main logic
    3. Reading back the diagnostic_report.json
    """
    # Lazy import to avoid hard dependency in tests
    from integration.relations.run_relations import (
        load_from_harvester,
        process_article,
        generate_diagnostic_report,
        ParserAdapter,
        SynapseGrounder,
    )

    # Export SynapseStore to temp file for SynapseGrounder
    import tempfile
    tmp_synapse = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    )
    synapse_data = {"synapses": synapse_store.get_synapse_dict()}
    json.dump(synapse_data, tmp_synapse, ensure_ascii=False)
    tmp_synapse.close()

    try:
        articles = load_from_harvester(dataset)
        if not articles:
            raise RuntimeError(f"No articles found for dataset: {dataset}")

        adapter = ParserAdapter()
        grounder = SynapseGrounder.from_file(tmp_synapse.name, min_score=min_score)

        all_diagnostics = []
        for article_id, text in articles.items():
            diag = process_article(article_id, text, adapter, grounder, output_dir)
            all_diagnostics.append(diag)

        report = generate_diagnostic_report(all_diagnostics, output_dir, dataset)
        return report
    finally:
        os.unlink(tmp_synapse.name)


# ─── Run ID Generation (GPT audit §1) ────────────────────────────────

def generate_run_id(dataset: str) -> str:
    """
    Generate collision-resistant Run ID.
    Format: run_{YYYYMMDD_HHMMSS}_{dataset}_{rand4}
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    # Use hash of timestamp + pid for collision resistance
    rand_src = f"{ts}_{os.getpid()}_{time.time_ns()}"
    rand4 = hashlib.sha256(rand_src.encode()).hexdigest()[:4]
    return f"run_{ts}_{dataset}_{rand4}"


def get_code_version() -> str:
    """Get git SHA if available, else 'unknown'."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "unknown"


def get_dictionary_version(dictionary_path: str) -> str:
    """Read meta.version from dictionary JSON."""
    try:
        with open(dictionary_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return raw.get("meta", {}).get("version", "unknown")
    except Exception:
        return "unknown"


# ─── Diff Report Renderer ────────────────────────────────────────────

def render_diff_report(diff: Dict[str, Any], run_id: str) -> str:
    """Render diagnostic diff as Markdown."""
    lines = []
    metrics = diff["metrics"]
    gaps = diff["coverage_gaps"]
    reg = diff["regressions"]
    verdict = diff["verdict"]

    icon = {"PASS": "✅", "WARN": "🟡", "FAIL": "🔴"}[verdict]

    lines.append(f"# Synapse Expansion — Evaluation Report")
    lines.append(f"")
    lines.append(f"**Run ID**: {run_id}  ")
    lines.append(f"**Verdict**: {icon} **{verdict}** (exit code {diff['exit_code']})  ")
    lines.append(f"")

    if diff["fail_reasons"]:
        lines.append(f"## ⚠️ Fail Reasons")
        for reason in diff["fail_reasons"]:
            lines.append(f"- {reason}")
        lines.append(f"")

    lines.append(f"## 1. Metric Comparison")
    lines.append(f"")
    lines.append(f"| Metric | Before | After | Delta |")
    lines.append(f"|--------|--------|-------|-------|")
    lines.append(
        f"| Grounding Rate | {metrics['grounding_rate_before']:.1%} "
        f"| {metrics['grounding_rate_after']:.1%} "
        f"| {metrics['grounding_rate_delta']:+.1%} |"
    )
    lines.append(
        f"| Grounded count | — | — | {metrics['grounded_delta']:+d} |"
    )
    lines.append(
        f"| Ungrounded count | — | — | {metrics['ungrounded_delta']:+d} |"
    )
    lines.append(f"")

    lines.append(f"## 2. Coverage Gaps")
    lines.append(f"")
    lines.append(
        f"| Status | Count |"
    )
    lines.append(f"|--------|-------|")
    lines.append(f"| Before | {gaps['before_count']} |")
    lines.append(f"| After | {gaps['after_count']} |")
    lines.append(f"| Resolved | {len(gaps['resolved'])} |")
    lines.append(f"| New | {len(gaps['new'])} |")
    lines.append(f"")

    if gaps["resolved"]:
        lines.append(f"**Resolved gaps:** {', '.join(gaps['resolved'])}")
        lines.append(f"")

    if gaps["remaining"]:
        lines.append(f"**Remaining gaps:** {', '.join(gaps['remaining'])}")
        lines.append(f"")

    lines.append(f"## 3. Regression Check")
    lines.append(f"")
    lines.append(f"| Check | Result |")
    lines.append(f"|-------|--------|")
    lines.append(
        f"| Category mismatches | {reg['category_mismatches']} "
        f"{'🔴' if reg['category_mismatches'] > 0 else '✅'} |"
    )
    lines.append(
        f"| New consistent misgrounds | {len(reg['new_consistent_misgrounds'])} "
        f"{'🔴' if reg['new_consistent_misgrounds'] else '✅'} |"
    )
    lines.append(f"")

    if reg["new_consistent_misgrounds"]:
        lines.append(f"### New Misgrounds Detail")
        for m in reg["new_consistent_misgrounds"]:
            lines.append(f"- **{m.get('verb', '?')}**: {m.get('description', '')}")
        lines.append(f"")

    return "\n".join(lines)


# ─── Proposal Report Renderer ────────────────────────────────────────

def render_proposal_report(
    run_id: str,
    diagnostic: DiagnosticResult,
    proposals_by_lemma: Dict[str, List[SynapsePatchEntry]],
    total_proposals: int,
    min_freq: int,
) -> str:
    """Render proposal summary as Markdown for human review."""
    lines = []
    lines.append(f"# Synapse Expansion — Proposal Report")
    lines.append(f"")
    lines.append(f"**Run ID**: {run_id}  ")
    lines.append(f"**Grounding Rate**: {diagnostic.grounding_rate:.1%}  ")
    lines.append(f"**Coverage Gaps**: {len(diagnostic.coverage_gaps)}  ")
    lines.append(f"**Total Proposals**: {total_proposals}  ")
    lines.append(f"")

    lines.append(f"## Gap Verbs → Proposals")
    lines.append(f"")
    lines.append(f"| Verb | Corpus Count | Proposals |")
    lines.append(f"|------|-------------|-----------|")

    for verb, count in diagnostic.get_gap_verbs(min_freq=min_freq):
        n_proposals = len(proposals_by_lemma.get(verb, []))
        lines.append(f"| {verb} | {count} | {n_proposals} |")
    lines.append(f"")

    lines.append(f"## Top Proposals (by score)")
    lines.append(f"")
    lines.append(f"| edge_key | score | evidence |")
    lines.append(f"|----------|-------|----------|")

    all_proposals = []
    for ps in proposals_by_lemma.values():
        all_proposals.extend(ps)
    all_proposals.sort(key=lambda p: p.score, reverse=True)

    for p in all_proposals[:30]:
        lines.append(f"| {p.edge_key} | {p.score:.4f} | {p.metadata.get('evidence_count', 0)} |")
    lines.append(f"")

    lines.append(f"## Next Steps")
    lines.append(f"")
    lines.append(f"1. Review `patch_candidate.json` in this run directory")
    lines.append(f"2. Edit/approve proposals")
    lines.append(f"3. Run: `python -m synapse.cli evaluate-synapse-patch --run-dir {run_id}`")
    lines.append(f"")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
#  Command A: propose-synapse
# ═══════════════════════════════════════════════════════════════════════

def cmd_propose_synapse(
    dataset: str,
    synapse_path: str,
    dictionary_path: str,
    min_score: float = 0.45,
    min_freq: int = DEFAULT_MIN_FREQ,
    output_base: str = PROPOSALS_BASE_DIR,
    synapse_patches: Optional[List[str]] = None,       # ← v5.6.0: patch overlay
    pipeline_runner: Optional[PipelineRunnerFn] = None,
    proposer: Optional[SynapseEdgeProposer] = None,
) -> Tuple[int, str]:
    """
    Diagnose → Propose → Save baseline.

    Returns: (exit_code, run_dir_path)
    """
    runner = pipeline_runner or default_pipeline_runner

    # ── Step 0: Create run directory (GPT §1) ──
    run_id = generate_run_id(dataset)
    run_dir = Path(output_base) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"[CLI] Run directory: {run_dir}")

    # ── Step 1: Baseline diagnostic ──
    logger.info("[CLI] Step 1: Running baseline diagnostic...")
    store = SynapseStore()
    store.load(synapse_path, patches=synapse_patches or [])  # ← v5.6.0: overlay patches

    raw_report = runner(dataset, store, min_score, run_dir)

    # Inject env metadata (GPT §2)
    code_ver = get_code_version()
    dict_ver = get_dictionary_version(dictionary_path)
    diagnostic = DiagnosticResult.with_env_meta(
        raw=raw_report,
        synapse_base_path=synapse_path,
        patches_loaded=synapse_patches or [],              # ← v5.6.0: record patches
        dictionary_version=dict_ver,
        min_score=min_score,
        min_freq=min_freq,
        dataset=dataset,
        run_id=run_id,
        code_version=code_ver,
    )

    # Save baseline
    before_path = run_dir / "diagnostic_before.json"
    with open(before_path, "w", encoding="utf-8") as f:
        json.dump(diagnostic.raw, f, indent=2, ensure_ascii=False)
    logger.info(f"[CLI] Baseline saved: {before_path}")

    # ── Step 2: Extract gap verbs ──
    gap_verbs = diagnostic.get_gap_verbs(min_freq=min_freq)
    if not gap_verbs:
        logger.info("[CLI] No coverage gaps found. Nothing to propose.")
        # Still write empty report
        report_path = run_dir / "proposal_report.md"
        with open(report_path, "w") as f:
            f.write(f"# No coverage gaps detected\n\nRun ID: {run_id}\n")
        return 0, str(run_dir)

    logger.info(f"[CLI] Found {len(gap_verbs)} gap verbs (min_freq={min_freq})")

    # ── Step 3: Propose edges ──
    if proposer is None:
        proposer = SynapseEdgeProposer(
            dictionary_path=dictionary_path,
            log_dir=str(run_dir),
        )

    lemma_inputs = [
        {"lemma": verb, "evidence_count": count}
        for verb, count in gap_verbs
    ]
    proposals_by_lemma = proposer.propose_batch(lemma_inputs)

    # Flatten all proposals
    all_proposals = []
    for ps in proposals_by_lemma.values():
        all_proposals.extend(ps)

    # ── Step 4: Write outputs (all within run_dir — GPT §4) ──
    # Patch candidate
    patch_path = run_dir / "patch_candidate.json"
    proposer.export_patch_file(all_proposals, str(patch_path))

    # Rewrite trace (proposer writes to run_dir via log_dir)
    for verb, count in gap_verbs:
        if verb in proposals_by_lemma and proposals_by_lemma[verb]:
            proposer.write_trace(proposals_by_lemma[verb], verb)

    # Proposal report
    report_md = render_proposal_report(
        run_id=run_id,
        diagnostic=diagnostic,
        proposals_by_lemma=proposals_by_lemma,
        total_proposals=len(all_proposals),
        min_freq=min_freq,
    )
    report_path = run_dir / "proposal_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    logger.info(
        f"[CLI] Proposal complete: {len(all_proposals)} candidates "
        f"from {len(gap_verbs)} gap verbs → {run_dir}"
    )

    return 0, str(run_dir)


# ═══════════════════════════════════════════════════════════════════════
#  Command B: evaluate-synapse-patch
# ═══════════════════════════════════════════════════════════════════════

def cmd_evaluate_synapse_patch(
    run_dir: str,
    synapse_path: str,
    dictionary_path: str,
    dataset: str,
    patch_file: Optional[str] = None,
    min_score: float = 0.45,
    synapse_patches: Optional[List[str]] = None,       # ← v5.6.1: baseline patches
    pipeline_runner: Optional[PipelineRunnerFn] = None,
) -> int:
    """
    Overlay patch → Re-diagnose → Diff → Verdict.

    Returns: exit code (0=PASS, 1=WARN, 2=FAIL)

    GPT audit §4: NEVER writes to patches/ — only to run_dir.
    GPT audit §5 (v5.6.1): Baseline patch auto-inheritance.
      - If --synapse-patches not specified, auto-inherit from before's env_meta
      - If specified, validate consistency with before (WARN on mismatch)
    """
    runner = pipeline_runner or default_pipeline_runner
    run_path = Path(run_dir)

    # ── Step 0: Load baseline ──
    before_path = run_path / "diagnostic_before.json"
    if not before_path.exists():
        logger.error(f"[CLI] Baseline not found: {before_path}")
        return 2

    with open(before_path, "r", encoding="utf-8") as f:
        before_raw = json.load(f)
    before = DiagnosticResult(raw=before_raw)

    # ── Step 0.5: Resolve baseline patches (GPT §5, v5.6.1) ──────
    before_baseline_patches = before.env_meta.get("patches_loaded", [])

    if synapse_patches is None:
        # Auto-inherit from diagnostic_before.json
        baseline_patches = before_baseline_patches
        if baseline_patches:
            logger.info(
                f"[CLI] Auto-inherited {len(baseline_patches)} baseline patch(es) "
                f"from diagnostic_before.json"
            )
    else:
        # Explicit specification — validate consistency
        baseline_patches = synapse_patches
        if sorted(baseline_patches) != sorted(before_baseline_patches):
            logger.warning(
                f"[CLI] ⚠ Baseline patch MISMATCH:\n"
                f"       before used:  {before_baseline_patches}\n"
                f"       evaluate got: {baseline_patches}\n"
                f"       Diff reliability may be compromised."
            )

    # ── Step 1: Determine patch file ──
    if patch_file is None:
        patch_file = str(run_path / "patch_candidate.json")
    if not Path(patch_file).exists():
        logger.error(f"[CLI] Patch file not found: {patch_file}")
        return 2

    logger.info(f"[CLI] Evaluating patch: {patch_file}")
    if baseline_patches:
        logger.info(f"[CLI] Baseline patches: {baseline_patches}")

    # ── Step 2: Overlay baseline + candidate onto SynapseStore ────
    # (read-only, no patches/ write — GPT §4)
    store = SynapseStore()
    all_patches = list(baseline_patches) + [patch_file]    # ← v5.6.1
    store.load(synapse_path, patches=all_patches)

    # ── Step 3: Re-run diagnostic with patched store ──
    logger.info("[CLI] Running diagnostic with patch overlay...")
    after_raw = runner(dataset, store, min_score, run_path)

    # Inject env metadata
    code_ver = get_code_version()
    dict_ver = get_dictionary_version(dictionary_path)
    run_id = run_path.name
    after = DiagnosticResult.with_env_meta(
        raw=after_raw,
        synapse_base_path=synapse_path,
        patches_loaded=all_patches,                        # ← v5.6.1
        dictionary_version=dict_ver,
        min_score=min_score,
        min_freq=0,
        dataset=dataset,
        run_id=run_id,
        code_version=code_ver,
    )

    # Save after diagnostic
    after_path = run_path / "diagnostic_after.json"
    with open(after_path, "w", encoding="utf-8") as f:
        json.dump(after.raw, f, indent=2, ensure_ascii=False)

    # ── Step 4: Diff & Verdict ──
    diff = DiagnosticResult.diff(before, after)

    # Write diff report
    diff_json_path = run_path / "diagnostic_diff.json"
    with open(diff_json_path, "w", encoding="utf-8") as f:
        json.dump(diff, f, indent=2, ensure_ascii=False)

    diff_md = render_diff_report(diff, run_id)
    diff_md_path = run_path / "diagnostic_diff.md"
    with open(diff_md_path, "w", encoding="utf-8") as f:
        f.write(diff_md)

    # ── Report verdict ──
    verdict = diff["verdict"]
    exit_code = diff["exit_code"]
    icon = {"PASS": "✅", "WARN": "🟡", "FAIL": "🔴"}[verdict]

    logger.info(f"[CLI] {icon} Verdict: {verdict} (exit code {exit_code})")
    if diff["fail_reasons"]:
        for reason in diff["fail_reasons"]:
            logger.warning(f"[CLI]   FAIL reason: {reason}")

    return exit_code


# ═══════════════════════════════════════════════════════════════════════
#  CLI Entry Point
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="ESDE Synapse Expansion CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # ── propose-synapse ──
    p_propose = subparsers.add_parser(
        "propose-synapse",
        help="Diagnose coverage gaps and generate edge proposals",
    )
    p_propose.add_argument("--dataset", required=True, help="Harvester dataset name")
    p_propose.add_argument("--synapse", default="esde_synapses_v3.json", help="Synapse base file")
    p_propose.add_argument("--dictionary", default="esde_dictionary.json", help="ESDE dictionary")
    p_propose.add_argument("--min-score", type=float, default=0.45, help="Grounding min score")
    p_propose.add_argument("--min-freq", type=int, default=DEFAULT_MIN_FREQ, help="Min gap frequency")
    p_propose.add_argument("--output", default=PROPOSALS_BASE_DIR, help="Proposals base dir")
    # v5.6.0: Synapse patch overlay support
    p_propose.add_argument("--synapse-patches", type=str, nargs="*", default=None,
                           help="Synapse patch files to overlay on base (e.g. patches/synapse_v3.1.json)")

    # ── evaluate-synapse-patch ──
    p_eval = subparsers.add_parser(
        "evaluate-synapse-patch",
        help="Evaluate a patch by comparing before/after diagnostics",
    )
    p_eval.add_argument("--run-dir", required=True, help="Run directory to evaluate")
    p_eval.add_argument("--patch-file", default=None, help="Patch file (default: run-dir candidate)")
    p_eval.add_argument("--dataset", required=True, help="Harvester dataset name")
    p_eval.add_argument("--synapse", default="esde_synapses_v3.json", help="Synapse base file")
    p_eval.add_argument("--dictionary", default="esde_dictionary.json", help="ESDE dictionary")
    p_eval.add_argument("--min-score", type=float, default=0.45, help="Grounding min score")
    # v5.6.1: Baseline patches (auto-inherited from diagnostic_before.json if omitted)
    p_eval.add_argument("--synapse-patches", type=str, nargs="*", default=None,
                        help="Baseline patches (auto-inherited from diagnostic_before if omitted)")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )

    if args.command == "propose-synapse":
        exit_code, run_dir = cmd_propose_synapse(
            dataset=args.dataset,
            synapse_path=args.synapse,
            dictionary_path=args.dictionary,
            min_score=args.min_score,
            min_freq=args.min_freq,
            output_base=args.output,
            synapse_patches=args.synapse_patches,          # ← v5.6.0
        )
        print(f"\nRun directory: {run_dir}")
        sys.exit(exit_code)

    elif args.command == "evaluate-synapse-patch":
        exit_code = cmd_evaluate_synapse_patch(
            run_dir=args.run_dir,
            synapse_path=args.synapse,
            dictionary_path=args.dictionary,
            dataset=args.dataset,
            patch_file=args.patch_file,
            min_score=args.min_score,
            synapse_patches=args.synapse_patches,          # ← v5.6.1
        )
        sys.exit(exit_code)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
