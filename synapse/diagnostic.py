"""
ESDE Synapse - Diagnostic Result Model
========================================
Typed wrapper for the diagnostic report produced by run_relations.py.

Provides structured access to symptoms, coverage gaps, and metrics
for use by Phase 3 CLI (propose-synapse / evaluate-synapse-patch).

Design Spec v3.1: Evaluate commands need machine-readable access
to SYNAPSE_COVERAGE_GAP, CONSISTENT_MISGROUND, CATEGORY_MISMATCH.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class DiagnosticResult:
    """
    Structured wrapper for run_relations.py diagnostic report.

    The raw report dict is preserved as-is; this class adds
    typed accessors and comparison logic for Phase 3 CLI.
    """
    raw: Dict[str, Any]

    # ── Accessors ────────────────────────────────────────────────────

    @property
    def meta(self) -> Dict[str, Any]:
        return self.raw.get("meta", {})

    @property
    def summary(self) -> Dict[str, Any]:
        return self.raw.get("summary", {})

    @property
    def symptoms(self) -> List[Dict[str, Any]]:
        return self.raw.get("symptoms", [])

    @property
    def grounding_rate(self) -> float:
        return self.summary.get("grounding_rate", 0.0)

    @property
    def total_triples(self) -> int:
        return self.summary.get("total_triples", 0)

    @property
    def grounded(self) -> int:
        return self.summary.get("grounded", 0)

    @property
    def ungrounded(self) -> int:
        return self.summary.get("ungrounded", 0)

    # ── Symptom Queries ──────────────────────────────────────────────

    def get_symptoms_by_type(self, symptom_type: str) -> List[Dict[str, Any]]:
        """Get all symptoms of a specific type."""
        return [s for s in self.symptoms if s.get("type") == symptom_type]

    @property
    def coverage_gaps(self) -> List[Dict[str, Any]]:
        """SYNAPSE_COVERAGE_GAP symptoms."""
        return self.get_symptoms_by_type("SYNAPSE_COVERAGE_GAP")

    @property
    def consistent_misgrounds(self) -> List[Dict[str, Any]]:
        """CONSISTENT_MISGROUND symptoms."""
        return self.get_symptoms_by_type("CONSISTENT_MISGROUND")

    @property
    def category_mismatches(self) -> List[Dict[str, Any]]:
        """CATEGORY_MISMATCH symptoms."""
        return self.get_symptoms_by_type("CATEGORY_MISMATCH")

    def get_gap_verbs(self, min_freq: int = 2) -> List[Tuple[str, int]]:
        """
        Extract (verb, count) pairs from SYNAPSE_COVERAGE_GAP symptoms.
        Filtered by minimum frequency.
        """
        result = []
        for gap in self.coverage_gaps:
            verb = gap.get("verb", "")
            count = gap.get("count", 0)
            if count >= min_freq:
                result.append((verb, count))
        return sorted(result, key=lambda x: -x[1])

    # ── Environment Metadata (GPT audit §2) ──────────────────────────

    @property
    def env_meta(self) -> Dict[str, Any]:
        """Environment metadata for diff reliability."""
        return self.raw.get("env_meta", {})

    @classmethod
    def with_env_meta(
        cls,
        raw: Dict[str, Any],
        synapse_base_path: str = "",
        patches_loaded: List[str] = None,
        dictionary_version: str = "",
        min_score: float = 0.0,
        min_freq: int = 0,
        dataset: str = "",
        run_id: str = "",
        code_version: str = "unknown",
    ) -> "DiagnosticResult":
        """Create DiagnosticResult with environment metadata injected."""
        raw["env_meta"] = {
            "synapse_base_path": synapse_base_path,
            "patches_loaded": patches_loaded or [],
            "dictionary_version": dictionary_version,
            "min_score": min_score,
            "min_freq": min_freq,
            "dataset": dataset,
            "run_id": run_id,
            "code_version": code_version,
        }
        return cls(raw=raw)

    # ── Diff Logic (for evaluate-synapse-patch) ──────────────────────

    @staticmethod
    def diff(
        before: "DiagnosticResult",
        after: "DiagnosticResult",
    ) -> Dict[str, Any]:
        """
        Compare two diagnostic results and produce a diff.

        Returns:
            Dict with delta metrics, new/resolved symptoms,
            and audit gate verdict (PASS/WARN/FAIL).
        """
        # ── Metric deltas ──
        delta_rate = after.grounding_rate - before.grounding_rate
        delta_grounded = after.grounded - before.grounded
        delta_ungrounded = after.ungrounded - before.ungrounded

        # ── Coverage gap comparison ──
        before_gaps = {g.get("verb", ""): g for g in before.coverage_gaps}
        after_gaps = {g.get("verb", ""): g for g in after.coverage_gaps}
        resolved_gaps = sorted(set(before_gaps.keys()) - set(after_gaps.keys()))
        remaining_gaps = sorted(set(after_gaps.keys()))
        new_gaps = sorted(set(after_gaps.keys()) - set(before_gaps.keys()))

        # ── Category mismatch (FAIL condition) ──
        after_cat_mismatches = after.category_mismatches

        # ── New consistent misgrounds (FAIL condition, GPT audit §3) ──
        before_misground_keys = {
            (m.get("verb", ""), tuple(sorted(m.get("atoms", []))))
            for m in before.consistent_misgrounds
        }
        new_consistent_misgrounds = [
            m for m in after.consistent_misgrounds
            if (m.get("verb", ""), tuple(sorted(m.get("atoms", []))))
            not in before_misground_keys
        ]

        # ── Audit Gate (Design Spec v3.1 §3.2) ──
        #   Exit 0 (PASS): improvement, no regression
        #   Exit 1 (WARN): no improvement
        #   Exit 2 (FAIL): category mismatch or new consistent misgrounds
        has_cat_mismatch = len(after_cat_mismatches) > 0
        has_new_misground = len(new_consistent_misgrounds) >= 1
        has_improvement = len(resolved_gaps) > 0 or delta_rate > 0.001

        if has_cat_mismatch or has_new_misground:
            verdict = "FAIL"
            exit_code = 2
        elif not has_improvement:
            verdict = "WARN"
            exit_code = 1
        else:
            verdict = "PASS"
            exit_code = 0

        return {
            "verdict": verdict,
            "exit_code": exit_code,
            "metrics": {
                "grounding_rate_before": before.grounding_rate,
                "grounding_rate_after": after.grounding_rate,
                "grounding_rate_delta": round(delta_rate, 4),
                "grounded_delta": delta_grounded,
                "ungrounded_delta": delta_ungrounded,
            },
            "coverage_gaps": {
                "before_count": len(before_gaps),
                "after_count": len(after_gaps),
                "resolved": resolved_gaps,
                "remaining": remaining_gaps,
                "new": new_gaps,
            },
            "regressions": {
                "category_mismatches": len(after_cat_mismatches),
                "new_consistent_misgrounds": [
                    {
                        "type": m.get("type"),
                        "description": m.get("description"),
                        "verb": m.get("verb", ""),
                    }
                    for m in new_consistent_misgrounds
                ],
            },
            "fail_reasons": (
                (["CATEGORY_MISMATCH > 0"] if has_cat_mismatch else [])
                + ([f"New CONSISTENT_MISGROUND: {len(new_consistent_misgrounds)}"]
                   if has_new_misground else [])
            ),
        }
