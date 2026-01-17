#!/usr/bin/env python3
"""
ESDE Engine v5.3.4 - Phase 7B+: Unknown Queue Resolver (GPT Audit Compliant)

This resolver DOES NOT determine winners.
It evaluates all hypotheses independently and measures volatility.
"Unable to decide" is a valid and expected outcome.

v5.3.4 Key Changes (GPT Audit):
    1. State Key unified to aggregate_key = hash(token_norm, pos, route_set)
    2. Two-stage processing: seen vs finalized
    3. Observation model stored in state
    4. Re-evaluation conditions controlled by state
    5. Clear ledger/state separation

Usage:
    python resolve_unknown_queue_7bplus_v534.py [options]
"""

import json
import argparse
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Set, Tuple, Optional
from collections import defaultdict

from esde_engine.resolver import (
    QueueStateManager,  # Legacy, still used for queue record state
    EvidenceLedger,
    SearchCache,
    SearXNGProvider,
    # MockSearchProvider,   --- IGNORE ---　2026/01/09
    collect_evidence,  #仕様変更につきコメントアウト　2026/01/10
    evaluate_all_hypotheses,
    EvaluationReport,
    DEFAULT_LOW_VOLATILITY_THRESHOLD,
    DEFAULT_HIGH_VOLATILITY_THRESHOLD,
    COMPETE_TH
)
from esde_engine.config import VERSION

# =============================================================================
# Import or define AggregateStateManager
# =============================================================================

try:
    from esde_engine.resolver.aggregate_state import AggregateStateManager
except ImportError:
    # Inline definition if module not available
    from aggregate_state import AggregateStateManager


# =============================================================================
# Default Paths
# =============================================================================

QUEUE_FILE = "./data/unknown_queue.jsonl"
LEGACY_STATE_FILE = "./data/unknown_queue_state.json"
AGG_STATE_FILE = "./data/unknown_queue_state_7bplus.json"
LEDGER_FILE = "./data/evidence_ledger_7bplus.jsonl"
CACHE_DIR = "./data/cache"

# Phase 7B+ Aggregation settings
MAX_EXAMPLES_PER_AGGREGATE = 5

# Phase 7C Audit settings
AGGREGATED_QUEUE_FILE = "./data/unknown_queue_7bplus.jsonl"
AUDIT_LOG_FILE_7C = "./data/audit_log_7c.jsonl"
SPEC_VERSION_7C = "7C-1.0"

# Phase 7C' Triple Audit settings
AUDIT_VOTES_FILE_7CPRIME = "./data/audit_votes_7cprime.jsonl"
AUDIT_DRIFT_FILE_7CPRIME = "./data/audit_drift_7cprime.jsonl"
HUMAN_REVIEW_QUEUE_7CPRIME = "./data/human_review_queue_7cprime.jsonl"
QC_SAMPLES_FILE_7CPRIME = "./data/qc_samples_7cprime.jsonl"
QC_RESULTS_FILE_7CPRIME = "./data/qc_results_7cprime.jsonl"
QC_METRICS_FILE_7CPRIME = "./data/qc_metrics_7cprime.json"
SPEC_VERSION_7CPRIME = "7C'-1.0"

# Audit Profile IDs (fixed)
PROFILE_CONSERVATIVE = "AUDIT_CONSERVATIVE"
PROFILE_BALANCED = "AUDIT_BALANCED"
PROFILE_EXPLORATORY = "AUDIT_EXPLORATORY"


# =============================================================================
# Aggregate Key & Route Set (v5.3.4 Spec)
# =============================================================================

def get_route_set_from_record(record: Dict[str, Any]) -> Set[str]:
    """Extract route set from a queue record."""
    route = record.get("route")
    if route:
        return {route}
    
    routing_report = record.get("routing_report", {})
    decision = routing_report.get("decision", {})
    
    routes = set()
    if decision.get("winner"):
        routes.add(decision["winner"])
    
    for r in ["A", "B", "C", "D"]:
        score = decision.get(f"{r}_score", 0)
        if score >= COMPETE_TH:
            routes.add(r)
    
    return routes if routes else {"UNK"}


# =============================================================================
# Ledger Writer (Append-Only Audit Trail)
# =============================================================================

class AuditLedger:
    """
    Append-only audit ledger for 7B+ evaluations.
    
    Separation from state (v5.3.4):
        - ledger: Raw log of every evaluation (append-only, audit trail)
        - state: Aggregated current snapshot (can be updated)
    """
    
    def __init__(self, ledger_file: str):
        self.ledger_file = ledger_file
        Path(ledger_file).parent.mkdir(parents=True, exist_ok=True)
    
    def append(self, record: Dict[str, Any]) -> None:
        """Append a record to the ledger."""
        record["_logged_at"] = datetime.now(timezone.utc).isoformat()
        with open(self.ledger_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')


# =============================================================================
# Aggregated Output Record (v5.3.4)
# =============================================================================

class AggregatedRecord:
    """
    Aggregated multi-hypothesis record for output.
    
    This is the OUTPUT format - state is managed separately.
    """
    
    def __init__(self, token_norm: str, pos: str, route_set: Set[str]):
        self.token_norm = token_norm
        self.pos = pos or "UNK"
        self.route_set = route_set
        self.aggregate_key = AggregateStateManager.compute_aggregate_key(
            token_norm, self.pos, route_set
        )
        
        self.count = 0
        self.examples: List[Dict[str, Any]] = []
        
        self.scores_sum: Dict[str, float] = defaultdict(float)
        self.scores_max: Dict[str, float] = defaultdict(lambda: 0.0)
        
        self.volatility_sum = 0.0
        self.volatility_max = 0.0
        
        self.first_seen_run_id: str = None
        self.last_seen_run_id: str = None
        
        self.all_reasons: Dict[str, Set[str]] = defaultdict(set)
        self.all_signals: Dict[str, Set[str]] = defaultdict(set)
        self.status_votes: Dict[str, int] = defaultdict(int)
        self.all_competing_routes: Set[str] = set()
    
    def add_occurrence(self, 
                       report: EvaluationReport,
                       original_record: Dict[str, Any],
                       run_id: str) -> None:
        """Add a single occurrence to this aggregate."""
        self.count += 1
        
        if self.first_seen_run_id is None:
            self.first_seen_run_id = run_id
        self.last_seen_run_id = run_id
        
        if len(self.examples) < MAX_EXAMPLES_PER_AGGREGATE:
            self.examples.append({
                "input_text": original_record.get("input_text", ""),
                "context_window": original_record.get("context_window", [])[:3],
                "original_route": original_record.get("route"),
                "run_id": run_id
            })
        
        for route, hyp in report.hypotheses.items():
            self.scores_sum[route] += hyp.score
            self.scores_max[route] = max(self.scores_max[route], hyp.score)
            if hyp.reason:
                self.all_reasons[route].add(hyp.reason)
            for sig in hyp.signals:
                self.all_signals[route].add(sig)
        
        self.volatility_sum += report.global_volatility
        self.volatility_max = max(self.volatility_max, report.global_volatility)
        self.all_competing_routes.update(report.competing_routes)
        self.status_votes[report.status] += 1
    
    def get_final_status(self) -> str:
        """Determine final status based on votes and volatility."""
        if self.status_votes.get("quarantine", 0) > 0:
            return "quarantine"
        if self.volatility_max >= DEFAULT_HIGH_VOLATILITY_THRESHOLD:
            return "quarantine"
        avg_vol = self.volatility_sum / max(1, self.count)
        if avg_vol >= DEFAULT_LOW_VOLATILITY_THRESHOLD:
            return "defer"
        if self.status_votes.get("defer", 0) > 0:
            return "defer"
        return "candidate"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to output format."""
        scores_avg = {
            route: round(self.scores_sum[route] / max(1, self.count), 3)
            for route in ["A", "B", "C", "D"]
        }
        scores_max = {
            route: round(self.scores_max[route], 3)
            for route in ["A", "B", "C", "D"]
        }
        
        volatility_avg = round(self.volatility_sum / max(1, self.count), 3)
        competing_routes = [r for r in ["A", "B", "C", "D"] if self.scores_max[r] >= COMPETE_TH]
        
        return {
            "version": "7B+",
            "spec_version": "5.3.4",
            "engine_version": VERSION,
            
            "aggregate_key": self.aggregate_key,
            "token_norm": self.token_norm,
            "pos": self.pos,
            "route_set": sorted(self.route_set),
            
            "count": self.count,
            "examples": self.examples,
            
            "scores_avg": scores_avg,
            "scores_max": scores_max,
            
            "volatility_avg": volatility_avg,
            "volatility_max": round(self.volatility_max, 3),
            
            "first_seen_run_id": self.first_seen_run_id,
            "last_seen_run_id": self.last_seen_run_id,
            
            "hypotheses": {
                route: {
                    "score_avg": scores_avg[route],
                    "score_max": scores_max[route],
                    "reasons": sorted(self.all_reasons.get(route, set())),
                    "signals": sorted(self.all_signals.get(route, set()))
                }
                for route in ["A", "B", "C", "D"]
            },
            
            # EXPLICIT: No winner (v5.3.4 mandatory)
            "winner": None,
            "compete_th": COMPETE_TH,
            "competing_routes": competing_routes,
            "competing_count": len(competing_routes),
            
            "status": self.get_final_status(),
            "status_votes": dict(self.status_votes),
            
            "has_competing_hypotheses": len(competing_routes) >= 2,
            "is_volatile": self.volatility_max >= DEFAULT_HIGH_VOLATILITY_THRESHOLD,
            "needs_observation": self.get_final_status() in ("defer", "quarantine")
        }


# =============================================================================
# Queue Loading
# =============================================================================

def load_queue_records(queue_file: str) -> List[Dict[str, Any]]:
    """Load all records from unknown queue."""
    records = []
    
    if not Path(queue_file).exists():
        print(f"[7B+] Queue file not found: {queue_file}")
        return records
    
    try:
        with open(queue_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    except Exception as e:
        print(f"[7B+] Error loading queue: {e}")
    
    return records


# =============================================================================
# Candidate Filtering (v5.3.4)
# =============================================================================

def filter_candidates(records: List[Dict[str, Any]],
                      agg_state: AggregateStateManager,
                      include_finalized: bool = False) -> Dict[str, Any]:
    """
    Filter records for 7B+ processing using aggregate state.
    
    v5.3.4 Rules:
        - finalized=True → skip (unless include_finalized)
        - needs_observation=True → process
        - Re-evaluation conditions checked via should_process()
    """
    candidates = []
    skip_info = {
        "candidates_total": len(records),
        "eligible_pending": 0,
        "skipped_by_route": 0,
        "skipped_by_state": 0,
        "skipped_finalized": 0,
        "skipped_cooldown": 0,
        "skipped_observed": 0,
        "details": []
    }
    
    for record in records:
        token = record.get("token_norm", record.get("token", ""))
        pos = record.get("pos", "UNK")
        route_set = get_route_set_from_record(record)
        
        # Route eligibility check
        route = record.get("route")
        if not route:
            routing_report = record.get("routing_report", {})
            decision = routing_report.get("decision", {})
            route = decision.get("winner")
        
        is_eligible_route = (
            route in ("B", "C") or 
            record.get("routing_report", {}).get("decision", {}).get("action") == "abstain"
        )
        
        if not is_eligible_route:
            skip_info["skipped_by_route"] += 1
            continue
        
        # Check aggregate state (v5.3.4)
        agg_key = AggregateStateManager.compute_aggregate_key(token.lower(), pos, route_set)
        should_process, reason = agg_state.should_process(agg_key)
        
        if not should_process:
            if reason == "finalized":
                if include_finalized:
                    should_process = True
                else:
                    skip_info["skipped_finalized"] += 1
            elif "cooldown" in reason:
                skip_info["skipped_cooldown"] += 1
            elif reason == "already_observed":
                skip_info["skipped_observed"] += 1
            else:
                skip_info["skipped_by_state"] += 1
            
            if not should_process:
                skip_info["details"].append({
                    "token": token,
                    "aggregate_key": agg_key,
                    "reason": reason
                })
                continue
        
        skip_info["eligible_pending"] += 1
        candidates.append(record)
    
    return {
        "candidates": candidates,
        "skip_info": skip_info
    }


# =============================================================================
# Token Evaluation
# =============================================================================

def evaluate_token(record: Dict[str, Any],
                   search_provider,
                   cache: SearchCache,
                   min_sources: int,
                   max_sources: int) -> Dict[str, Any]:
    """Evaluate a token using 7B+ approach."""
    token = record.get("token_norm", record.get("token", ""))
    
    context = record.get("context", {})
    if not context:
        context = {
            "tokens_all": record.get("context_window", []),
            "tokens": record.get("context_window", [])
        }
    
    typo_candidates = record.get("typo_candidates", [])
    
    routing_report = record.get("routing_report", {})
    decision = routing_report.get("decision", {})
    queue_metrics = {
        "margin": decision.get("margin", 0),
        "entropy": decision.get("entropy", 0)
    }
    
    evidence = collect_evidence(
        token=token,
        context=context,
        search_provider=search_provider,
        typo_candidates=typo_candidates,
        route_winner=None,
        min_sources=min_sources,
        max_sources=max_sources,
        cache=cache
    )
    
    report = evaluate_all_hypotheses(
        token=token,
        evidence=evidence,
        typo_candidates=typo_candidates,
        queue_metrics=queue_metrics
    )
    
    return {
        "token": token,
        "report": report,
        "evidence_count": len(evidence.items),
        "original_record": record
    }


# =============================================================================
# Batch Processing (v5.3.4)
# =============================================================================

def process_batch(records: List[Dict[str, Any]],
                  agg_state: AggregateStateManager,
                  legacy_state: QueueStateManager,
                  audit_ledger: AuditLedger,
                  search_provider,
                  cache: SearchCache,
                  limit: int = 100,
                  min_sources: int = 3,
                  max_sources: int = 8,
                  dry_run: bool = False,
                  reprocess: bool = False,
                  run_id: str = None) -> Dict[str, Any]:
    """
    Process a batch of records using 7B+ approach with v5.3.4 state management.
    
    Key v5.3.4 changes:
        - Uses AggregateStateManager for aggregate-level state
        - Two-stage processing (seen/finalized)
        - Observation model stored in state
        - Ledger is append-only audit trail
    """
    if run_id is None:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    
    # Filter using aggregate state
    filter_result = filter_candidates(records, agg_state, include_finalized=reprocess)
    candidates = filter_result["candidates"]
    skip_info = filter_result["skip_info"]
    
    print(f"\n[7B+] Candidates breakdown:")
    print(f"  - Total records: {skip_info['candidates_total']}")
    print(f"  - Skipped by route: {skip_info['skipped_by_route']}")
    print(f"  - Skipped finalized: {skip_info['skipped_finalized']}")
    print(f"  - Skipped cooldown: {skip_info['skipped_cooldown']}")
    print(f"  - Skipped observed: {skip_info['skipped_observed']}")
    print(f"  - Eligible for processing: {skip_info['eligible_pending']}")
    
    if reprocess:
        print(f"[7B+] Reprocess mode enabled")
    
    # Results tracking
    results = {
        "processed": 0,
        "aggregated": 0,
        "quarantined": 0,
        "deferred": 0,
        "candidate": 0,
        "skipped_by_validation": 0,
        "by_volatility": {"low": 0, "medium": 0, "high": 0},
        "competing_hypotheses": 0,
        "avg_volatility": 0.0,
        "volatility_sum": 0.0,
        "skip_info": skip_info,
        "skipped_by_filter": 0
    }
    
    # Apply limit
    processing_list = candidates[:limit]
    results["skipped_by_filter"] = max(0, len(candidates) - limit)
    
    print(f"\n[7B+] Processing {len(processing_list)} of {len(candidates)} eligible records")
    print(f"[7B+] Run ID: {run_id}")
    
    # Aggregation structure (for output)
    output_aggregates: Dict[str, AggregatedRecord] = {}
    
    for record in processing_list:
        token = record.get("token_norm", record.get("token", ""))
        
        if not token:
            results["skipped_by_validation"] += 1
            continue
        
        # Evaluate
        eval_result = evaluate_token(
            record=record,
            search_provider=search_provider,
            cache=cache,
            min_sources=min_sources,
            max_sources=max_sources
        )
        
        report = eval_result["report"]
        results["processed"] += 1
        results["volatility_sum"] += report.global_volatility
        
        # Track volatility distribution
        if report.global_volatility < DEFAULT_LOW_VOLATILITY_THRESHOLD:
            results["by_volatility"]["low"] += 1
        elif report.global_volatility < DEFAULT_HIGH_VOLATILITY_THRESHOLD:
            results["by_volatility"]["medium"] += 1
        else:
            results["by_volatility"]["high"] += 1
        
        if report.competing_count >= 2:
            results["competing_hypotheses"] += 1
        
        # Compute aggregate key
        pos = record.get("pos", "UNK")
        route_set = get_route_set_from_record(record)
        agg_key = AggregateStateManager.compute_aggregate_key(token.lower(), pos, route_set)
        
        # Build output aggregate
        if agg_key not in output_aggregates:
            output_aggregates[agg_key] = AggregatedRecord(token.lower(), pos, route_set)
        output_aggregates[agg_key].add_occurrence(report, record, run_id)
        
        # Update aggregate state (v5.3.4)
        if not dry_run:
            evaluation_result = {
                "status": report.status,
                "global_volatility": report.global_volatility,
                "scores": {r: h.score for r, h in report.hypotheses.items()},
                "competing_routes": report.competing_routes
            }
            agg_state.upsert_observation(
                aggregate_key=agg_key,
                token_norm=token.lower(),
                pos=pos,
                route_set=route_set,
                evaluation_result=evaluation_result,
                run_id=run_id
            )
            
            # Write to audit ledger (append-only)
            ledger_record = {
                "run_id": run_id,
                "aggregate_key": agg_key,
                "token_norm": token.lower(),
                "evaluation": evaluation_result,
                "record_hash": record.get("hash", "")
            }
            audit_ledger.append(ledger_record)
        
        # Progress output
        status_char = "⊘" if report.status == "quarantine" else ("◐" if report.status == "defer" else "○")
        print(f"  {status_char} {token}: {report.format_log_line()} → {report.status}")
        
        for route in ["A", "B", "C", "D"]:
            if route in report.hypotheses:
                h = report.hypotheses[route]
                if h.score >= COMPETE_TH:
                    print(f"      {route}_reason={h.reason}")
    
    # Calculate averages
    if results["processed"] > 0:
        results["avg_volatility"] = round(results["volatility_sum"] / results["processed"], 3)
    del results["volatility_sum"]
    
    # Process output aggregates
    print(f"\n[7B+] Aggregation complete: {len(output_aggregates)} unique aggregate keys")
    results["aggregated"] = len(output_aggregates)
    
    output_records = []
    for agg_key, agg in output_aggregates.items():
        agg_dict = agg.to_dict()
        output_records.append(agg_dict)
        
        status = agg.get_final_status()
        if status == "quarantine":
            results["quarantined"] += 1
        elif status == "defer":
            results["deferred"] += 1
        else:
            results["candidate"] += 1
        
        if agg.count > 1:
            print(f"    ↳ Aggregated: {agg.token_norm} ({agg.count} occurrences) → {status}")
    
    # Write output file
    if not dry_run and output_records:
        output_file = Path(LEDGER_FILE).parent / "unknown_queue_7bplus.jsonl"
        with open(output_file, 'a', encoding='utf-8') as f:
            for rec in output_records:
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        print(f"\n[7B+] Wrote {len(output_records)} aggregated records to {output_file}")
    
    return results


# =============================================================================
# Phase 7C: Structure Audit (DO NOT modify state)
# =============================================================================

def audit_structure(aggregate: Dict[str, Any], state_entry: Dict[str, Any]) -> Dict[str, Any]:
    """
    Perform structural audit on an aggregate record.
    
    Checks only numerical/key consistency.
    NO semantic interpretation.
    NO state modification.
    
    Returns:
        {
            "struct_ok": bool,
            "checks": [{check_name, ok, reason}]
        }
    """
    checks = []
    
    # Check 1: winner == null
    winner = aggregate.get("winner")
    checks.append({
        "check_name": "winner_is_null",
        "ok": winner is None,
        "reason": f"winner={winner}" if winner is not None else "winner=null"
    })
    
    # Check 2: count == len(examples) (if examples exists)
    if "examples" in aggregate and "count" in aggregate:
        count = aggregate.get("count", 0)
        examples_len = len(aggregate.get("examples", []))
        # Note: examples may be capped at MAX_EXAMPLES_PER_AGGREGATE
        ok = count >= examples_len  # count should be >= examples (examples may be truncated)
        checks.append({
            "check_name": "count_examples_consistency",
            "ok": ok,
            "reason": f"count={count}, len(examples)={examples_len}"
        })
    
    # Check 3: competing_count == len(competing_routes) (if exists)
    if "competing_count" in aggregate and "competing_routes" in aggregate:
        competing_count = aggregate.get("competing_count", 0)
        competing_routes = aggregate.get("competing_routes", [])
        ok = competing_count == len(competing_routes)
        checks.append({
            "check_name": "competing_count_routes_match",
            "ok": ok,
            "reason": f"competing_count={competing_count}, len(competing_routes)={len(competing_routes)}"
        })
    
    # Check 4: competing_routes ⊆ {A, B, C, D}
    if "competing_routes" in aggregate:
        competing_routes = aggregate.get("competing_routes", [])
        valid_routes = {"A", "B", "C", "D"}
        invalid_routes = set(competing_routes) - valid_routes
        ok = len(invalid_routes) == 0
        checks.append({
            "check_name": "competing_routes_valid",
            "ok": ok,
            "reason": f"competing_routes={competing_routes}" + (f", invalid={list(invalid_routes)}" if invalid_routes else "")
        })
    
    # Check 5: has_competing_hypotheses == (competing_count >= 2) (if exists)
    if "has_competing_hypotheses" in aggregate and "competing_count" in aggregate:
        has_competing = aggregate.get("has_competing_hypotheses", False)
        competing_count = aggregate.get("competing_count", 0)
        expected = competing_count >= 2
        ok = has_competing == expected
        checks.append({
            "check_name": "has_competing_hypotheses_consistent",
            "ok": ok,
            "reason": f"has_competing_hypotheses={has_competing}, competing_count={competing_count}, expected={expected}"
        })
    
    # Check 6: status and needs_observation consistency with determine_status logic
    if "status" in aggregate:
        status = aggregate.get("status")
        needs_obs = aggregate.get("needs_observation", False)
        volatility_avg = aggregate.get("volatility_avg", 0)
        volatility_max = aggregate.get("volatility_max", 0)
        
        # Verify status is valid
        valid_statuses = {"candidate", "defer", "quarantine"}
        status_valid = status in valid_statuses
        
        # Verify needs_observation consistency
        # Per spec: needs_observation should be True if status in ("defer", "quarantine")
        expected_needs_obs = status in ("defer", "quarantine")
        needs_obs_consistent = needs_obs == expected_needs_obs
        
        checks.append({
            "check_name": "status_needs_observation_consistent",
            "ok": status_valid and needs_obs_consistent,
            "reason": f"status={status}, needs_observation={needs_obs}, expected_needs_obs={expected_needs_obs}"
        })
    
    # Overall result
    struct_ok = all(c["ok"] for c in checks)
    
    return {
        "struct_ok": struct_ok,
        "checks": checks
    }


def extract_hypothesis_snapshot(aggregate: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract hypothesis snapshot from aggregate.
    
    NO processing, NO normalization.
    Just read and return as-is.
    """
    snapshot = {}
    
    hypotheses = aggregate.get("hypotheses", {})
    
    for route in ["A", "B", "C", "D"]:
        if route in hypotheses:
            h = hypotheses[route]
            snapshot[route] = {
                "score_avg": h.get("score_avg"),
                "score_max": h.get("score_max"),
                "reasons": h.get("reasons", []),
                "signals": h.get("signals", [])
            }
        else:
            snapshot[route] = None
    
    # Also include top-level scores if present
    if "scores_avg" in aggregate:
        snapshot["_scores_avg"] = aggregate["scores_avg"]
    if "scores_max" in aggregate:
        snapshot["_scores_max"] = aggregate["scores_max"]
    
    return snapshot


def extract_volatility_snapshot(aggregate: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract volatility information from aggregate.
    
    NO processing, just read.
    """
    return {
        "volatility_avg": aggregate.get("volatility_avg"),
        "volatility_max": aggregate.get("volatility_max"),
        "competing_count": aggregate.get("competing_count"),
        "competing_routes": aggregate.get("competing_routes", []),
        "has_competing_hypotheses": aggregate.get("has_competing_hypotheses")
    }


def generate_narrative(aggregate: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate natural language explanation of volatility.
    
    ONLY describes facts from aggregate.
    NO interpretation, NO prediction, NO winner selection.
    
    Returns:
        {
            "narrative": str,
            "sayable": [str],
            "unsayable": [str],
            "recommendation": str
        }
    """
    sayable = []
    unsayable = []
    
    token = aggregate.get("token_norm", "unknown")
    status = aggregate.get("status", "unknown")
    competing_count = aggregate.get("competing_count", 0)
    competing_routes = aggregate.get("competing_routes", [])
    volatility_avg = aggregate.get("volatility_avg", 0)
    volatility_max = aggregate.get("volatility_max", 0)
    
    # Build sayable facts
    sayable.append(f"トークン '{token}' の評価結果")
    
    if competing_count >= 2:
        routes_str = "/".join(competing_routes)
        sayable.append(f"{routes_str} の複数仮説が同時に成立している")
    elif competing_count == 1:
        sayable.append(f"{competing_routes[0]} 仮説のみが閾値を超えている")
    else:
        sayable.append("閾値を超える仮説が存在しない")
    
    if volatility_avg >= 0.50:
        sayable.append("揺れが大きく、意味が収束していない状態")
    elif volatility_avg >= 0.25:
        sayable.append("中程度の揺れがあり、追加観測が必要")
    else:
        sayable.append("揺れは小さいが、確定ではない")
    
    sayable.append(f"ステータス: {status}")
    
    # Build unsayable (what we CANNOT say)
    unsayable.append("正解は〜である")
    unsayable.append("おそらく〜である")
    unsayable.append("最も可能性が高いのは〜")
    unsayable.append("〜と判断できる")
    
    # Build narrative
    narrative_parts = [
        f"トークン '{token}' について、",
    ]
    
    if competing_count >= 2:
        narrative_parts.append(f"{'/'.join(competing_routes)} の複数仮説が同時に成立している。")
    else:
        narrative_parts.append(f"仮説の競合は検出されていない。")
    
    narrative_parts.append(f"平均揺れ度 {volatility_avg:.3f}、最大揺れ度 {volatility_max:.3f}。")
    narrative_parts.append("現時点では意味を確定できない状態である。")
    
    narrative = " ".join(narrative_parts)
    
    # Recommendation (fixed rules per spec)
    if status == "candidate":
        recommendation = "human_review"
    elif status == "defer":
        needs_obs = aggregate.get("needs_observation", False)
        if needs_obs:
            recommendation = "observe"
        else:
            recommendation = "human_review"
    elif status == "quarantine":
        recommendation = "quarantine_review"
    else:
        recommendation = "unknown"
    
    return {
        "narrative": narrative,
        "sayable": sayable,
        "unsayable": unsayable,
        "recommendation": recommendation
    }


def run_7c_audit(aggregate: Dict[str, Any], state_entry: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run Phase 7C audit on a single aggregate.
    
    Returns complete audit log record.
    """
    now = datetime.now(timezone.utc)
    
    # Generate audit ID
    agg_key = aggregate.get("aggregate_key", "unknown")
    audit_id = f"7C_{agg_key}_{now.strftime('%Y%m%d%H%M%S')}"
    
    # Run structural audit
    struct_result = audit_structure(aggregate, state_entry)
    
    # Extract snapshots (read-only)
    hypothesis_snapshot = extract_hypothesis_snapshot(aggregate)
    volatility_snapshot = extract_volatility_snapshot(aggregate)
    
    # Generate narrative
    narrative_result = generate_narrative(aggregate)
    
    return {
        "audit_id": audit_id,
        "spec_version": SPEC_VERSION_7C,
        "engine_version": VERSION,
        "aggregate_key": agg_key,
        "token_norm": aggregate.get("token_norm"),
        "struct_ok": struct_result["struct_ok"],
        "struct_checks": struct_result["checks"],
        "hypothesis_snapshot": hypothesis_snapshot,
        "volatility_snapshot": volatility_snapshot,
        "narrative": narrative_result["narrative"],
        "sayable": narrative_result["sayable"],
        "unsayable": narrative_result["unsayable"],
        "recommendation": narrative_result["recommendation"],
        "timestamp": now.isoformat()
    }


def load_aggregated_queue(file_path: str) -> List[Dict[str, Any]]:
    """Load aggregated queue file (7B+ output)."""
    records = []
    
    if not Path(file_path).exists():
        print(f"[7C] Aggregated queue file not found: {file_path}")
        return records
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    except Exception as e:
        print(f"[7C] Error loading aggregated queue: {e}")
    
    return records


def append_audit_log(record: Dict[str, Any], file_path: str) -> None:
    """Append audit log record (append-only)."""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')


def run_7c_mode(state_file: str, aggregated_file: str, audit_log_file: str) -> Dict[str, Any]:
    """
    Run Phase 7C audit mode.
    
    Reads 7B+ aggregated output and state (read-only).
    Generates audit logs for each aggregate.
    
    NO state modification.
    NO 7B+ processing.
    """
    print("=" * 60)
    print(f"ESDE Engine v{VERSION} - Phase 7C Audit Mode")
    print("=" * 60)
    print()
    print("Phase 7C: Structural audit and narrative generation")
    print("NO winner determination. NO state modification.")
    print()
    
    # Load state (read-only)
    agg_state = AggregateStateManager(state_file)
    agg_state.load()
    print(f"[7C] Loaded aggregate state: {state_file}")
    
    # Load aggregated queue (7B+ output)
    aggregates = load_aggregated_queue(aggregated_file)
    print(f"[7C] Loaded {len(aggregates)} aggregates from: {aggregated_file}")
    
    if not aggregates:
        print("[7C] No aggregates to audit. Exiting.")
        return {"audited": 0, "struct_ok": 0, "struct_fail": 0}
    
    # Results tracking
    results = {
        "audited": 0,
        "struct_ok": 0,
        "struct_fail": 0
    }
    
    print(f"\n[7C] Running audit on {len(aggregates)} aggregates...")
    print()
    
    for aggregate in aggregates:
        agg_key = aggregate.get("aggregate_key", "unknown")
        token = aggregate.get("token_norm", "unknown")
        
        # Get state entry (read-only)
        state_entry = agg_state.get_entry(agg_key) or {}
        
        # Run audit
        audit_record = run_7c_audit(aggregate, state_entry)
        
        # Append to audit log
        append_audit_log(audit_record, audit_log_file)
        
        # Track results
        results["audited"] += 1
        if audit_record["struct_ok"]:
            results["struct_ok"] += 1
            status_char = "✓"
        else:
            results["struct_fail"] += 1
            status_char = "✗"
        
        # Progress output
        print(f"  {status_char} {token} ({agg_key[:8]}...): struct_ok={audit_record['struct_ok']}, recommendation={audit_record['recommendation']}")
    
    # Summary
    print()
    print("=" * 60)
    print("Phase 7C Audit Summary")
    print("=" * 60)
    print(f"Total audited:    {results['audited']}")
    print(f"Structure OK:     {results['struct_ok']}")
    print(f"Structure FAIL:   {results['struct_fail']}")
    print(f"Audit log:        {audit_log_file}")
    print()
    print("Phase 7C guarantees:")
    print("  - NO winner determined")
    print("  - NO state modified")
    print("  - Audit log is append-only")
    print("  - Same input produces same output")
    
    return results


# =============================================================================
# Phase 7C': Triple Audit with Drift Detection (LLM-based)
# =============================================================================

# QwQ32B LLM Configuration
LLM_HOST = "http://100.107.6.119:8001/v1"
LLM_MODEL = "qwq32b_tp2_long32k_existing"
LLM_TIMEOUT = 180  # seconds (QwQ needs time for reasoning)

# System prompts for each profile
SYSTEM_PROMPT_BASE = """あなたは「監査器」であり「決定器ではない」。

絶対ルール：
- winner を決めてはいけない
- 「正解」「最も可能性が高い」「おそらく」等の断定表現禁止
- rationale は必ず evidence / signals / volatility を参照すること
- スキーマ外のキーを追加禁止

出力形式（厳守）:
思考が終わったら、必ず「FINAL:」という行を書き、その直後にJSONのみを出力すること。
FINAL: の前に思考過程を書いてもよいが、FINAL: の後はJSONのみ。

出力スキーマ:
{
  "status_suggestion": "candidate" または "defer" または "quarantine",
  "risk_flags": ["source_conflict", "entity_ambiguous", "typo_plausible", "title_like", "template_risk"] から該当するもの,
  "route_emphasis": ["A", "B", "C", "D"] から注目すべきもの,
  "volatility_estimate": 0.0〜1.0 の数値,
  "rationale": "200字以内。signals/evidenceを参照した説明",
  "needs_human_review": true または false,
  "next_actions": ["observe_more", "online_search", "do_not_patch", "human_review", "quarantine_review"] から該当するもの
}

例:
（思考過程...）
FINAL:
{"status_suggestion": "defer", "risk_flags": ["source_conflict"], "route_emphasis": ["A", "B"], "volatility_estimate": 0.45, "rationale": "A/Bが競合状態で揺れ度0.45", "needs_human_review": true, "next_actions": ["observe_more"]}
"""

SYSTEM_PROMPT_CONSERVATIVE = SYSTEM_PROMPT_BASE + """
あなたのプロファイル: AUDIT_CONSERVATIVE（保守的監査器）

特性：
- 断定を極力避ける
- quarantine/observe 側に寄せてよい
- 新規解釈を出さない
- 迷ったら defer または quarantine を提案
- リスクフラグは積極的に追加
"""

SYSTEM_PROMPT_BALANCED = SYSTEM_PROMPT_BASE + """
あなたのプロファイル: AUDIT_BALANCED（標準監査器）

特性：
- 標準的な評価
- 7C の narrative を忠実に踏襲
- バイアスなく中立的に判断
- 既存の status を尊重しつつ評価
"""

SYSTEM_PROMPT_EXPLORATORY = SYSTEM_PROMPT_BASE + """
あなたのプロファイル: AUDIT_EXPLORATORY（探索的監査器）

特性：
- 仮説の幅を広げる
- ただし winner=null を絶対維持
- "あり得る競合" を積極的に指摘
- 閾値以下でも潜在的な仮説を注視
- ただし断定は禁止
"""


def build_audit_prompt(aggregate: Dict[str, Any], audit_7c: Dict[str, Any]) -> str:
    """Build user prompt for LLM audit."""
    token = aggregate.get("token_norm", "unknown")
    status = aggregate.get("status", "unknown")
    volatility_avg = aggregate.get("volatility_avg", 0)
    volatility_max = aggregate.get("volatility_max", 0)
    competing_routes = aggregate.get("competing_routes", [])
    competing_count = aggregate.get("competing_count", 0)
    
    # Build hypothesis summary
    hypotheses = aggregate.get("hypotheses", {})
    hyp_lines = []
    for route in ["A", "B", "C", "D"]:
        if route in hypotheses:
            h = hypotheses[route]
            score_avg = h.get("score_avg", 0)
            score_max = h.get("score_max", 0)
            reasons = h.get("reasons", [])
            signals = h.get("signals", [])
            hyp_lines.append(f"  {route}: score_avg={score_avg}, score_max={score_max}, reasons={reasons}, signals={signals}")
    
    hyp_summary = "\n".join(hyp_lines) if hyp_lines else "  (なし)"
    
    # Build 7C narrative if available
    narrative = ""
    if audit_7c:
        narrative = audit_7c.get("narrative", "")
        sayable = audit_7c.get("sayable", [])
        if sayable:
            narrative += "\n  sayable: " + "; ".join(sayable)
    
    prompt = f"""以下の aggregate record を監査し、JSON形式で出力してください。

トークン: {token}
現在のステータス: {status}
揺れ度 (avg): {volatility_avg}
揺れ度 (max): {volatility_max}
競合ルート数: {competing_count}
競合ルート: {competing_routes}

仮説スコア:
{hyp_summary}

7C監査結果:
  {narrative if narrative else "(なし)"}

この情報に基づき、指定されたJSONスキーマで出力してください。
"""
    return prompt


def call_llm(system_prompt: str, user_prompt: str) -> Optional[str]:
    """
    Call QwQ32B LLM via OpenAI-compatible API.
    
    QwQ32B outputs reasoning process followed by final answer.
    The final answer is marked with "FINAL:" prefix.
    
    Returns raw response text or None on error.
    """
    import urllib.request
    import urllib.error
    
    url = f"{LLM_HOST}/chat/completions"
    
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 4096  # QwQ needs more tokens for reasoning + final answer
    }
    
    data = json.dumps(payload).encode('utf-8')
    
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json"
        }
    )
    
    try:
        with urllib.request.urlopen(req, timeout=LLM_TIMEOUT) as response:
            result = json.loads(response.read().decode('utf-8'))
            choices = result.get("choices", [])
            if choices:
                message = choices[0].get("message", {})
                content = message.get("content", "")
                
                # QwQ32B may also have reasoning_content field
                reasoning = message.get("reasoning_content", "")
                
                if content:
                    return content
                elif reasoning:
                    return reasoning
                    
    except urllib.error.URLError as e:
        print(f"  [LLM] Connection error: {e}")
    except Exception as e:
        print(f"  [LLM] Error: {e}")
    
    return None


def extract_final_answer(response: str) -> Optional[str]:
    """
    Extract final answer from QwQ32B response.
    
    QwQ32B outputs reasoning process followed by "FINAL:" marker.
    This function extracts the content after FINAL: marker.
    
    If no FINAL: marker, tries to find JSON in the response.
    """
    if not response:
        return None
    
    # Try to find FINAL: marker (case-insensitive)
    final_markers = ["FINAL:", "Final:", "final:", "FINAL：", "Final："]
    
    for marker in final_markers:
        if marker in response:
            parts = response.split(marker, 1)
            if len(parts) > 1:
                return parts[1].strip()
    
    # No FINAL marker - try to find JSON directly
    # Look for JSON object pattern
    import re
    
    # Find all potential JSON objects
    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.findall(json_pattern, response, re.DOTALL)
    
    # Return the last JSON-like match (likely the final answer)
    for match in reversed(matches):
        try:
            json.loads(match)
            return match
        except:
            continue
    
    # Return full response as fallback
    return response


def parse_llm_response(response: str, aggregate_key: str, profile_id: str) -> Optional[Dict[str, Any]]:
    """
    Parse LLM response into vote structure.
    
    Handles JSON extraction from potentially wrapped response.
    """
    if not response:
        return None
    
    # Try to extract JSON from response
    text = response.strip()
    
    # Remove markdown code blocks if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Find start and end of JSON block
        start = 0
        end = len(lines)
        for i, line in enumerate(lines):
            if line.startswith("```") and i > 0:
                end = i
                break
            if line.startswith("```"):
                start = i + 1
        text = "\n".join(lines[start:end])
    
    # Try to parse JSON
    try:
        vote_data = json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON object in text
        import re
        match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if match:
            try:
                vote_data = json.loads(match.group())
            except:
                return None
        else:
            return None
    
    # Validate required fields
    required_fields = ["status_suggestion", "risk_flags", "route_emphasis", 
                       "volatility_estimate", "rationale", "needs_human_review", "next_actions"]
    
    for field in required_fields:
        if field not in vote_data:
            # Add default for missing field
            if field == "status_suggestion":
                vote_data[field] = "defer"
            elif field in ["risk_flags", "route_emphasis", "next_actions"]:
                vote_data[field] = []
            elif field == "volatility_estimate":
                vote_data[field] = 0.5
            elif field == "rationale":
                vote_data[field] = "(LLM出力不完全)"
            elif field == "needs_human_review":
                vote_data[field] = True
    
    # Validate status_suggestion
    if vote_data["status_suggestion"] not in ["candidate", "defer", "quarantine"]:
        vote_data["status_suggestion"] = "defer"
    
    # Truncate rationale to 200 chars
    vote_data["rationale"] = vote_data["rationale"][:200]
    
    return {
        "aggregate_key": aggregate_key,
        "profile_id": profile_id,
        "vote": vote_data
    }


def generate_audit_vote_llm(aggregate: Dict[str, Any],
                             audit_7c: Dict[str, Any],
                             profile_id: str,
                             system_prompt: str) -> Optional[Dict[str, Any]]:
    """
    Generate audit vote using LLM.
    """
    agg_key = aggregate.get("aggregate_key", "unknown")
    
    # Build prompt
    user_prompt = build_audit_prompt(aggregate, audit_7c)
    
    # Call LLM
    response = call_llm(system_prompt, user_prompt)
    
    if response is None:
        return None
    
    # Extract final answer from QwQ reasoning output
    final_answer = extract_final_answer(response)
    
    if final_answer is None:
        return None
    
    # Parse final answer (not full response with reasoning)
    return parse_llm_response(final_answer, agg_key, profile_id)


def generate_audit_vote_fallback(aggregate: Dict[str, Any],
                                  audit_7c: Dict[str, Any],
                                  profile_id: str) -> Dict[str, Any]:
    """
    Fallback vote generation when LLM is unavailable.
    Uses simple rule-based logic.
    """
    agg_key = aggregate.get("aggregate_key", "unknown")
    status = aggregate.get("status", "defer")
    volatility_avg = aggregate.get("volatility_avg", 0)
    volatility_max = aggregate.get("volatility_max", 0)
    competing_routes = aggregate.get("competing_routes", [])
    competing_count = aggregate.get("competing_count", 0)
    
    risk_flags = []
    if volatility_max >= 0.5:
        risk_flags.append("source_conflict")
    if competing_count >= 2:
        risk_flags.append("entity_ambiguous")
    
    # Profile-specific adjustments
    if profile_id == PROFILE_CONSERVATIVE:
        status_suggestion = "quarantine" if status == "defer" else "defer" if status == "candidate" else "quarantine"
        volatility_estimate = min(1.0, volatility_avg + 0.15)
        next_actions = ["observe_more", "do_not_patch"]
        rationale = f"(フォールバック)保守的評価。競合{competing_count}件、揺れ{volatility_avg:.2f}"
    elif profile_id == PROFILE_EXPLORATORY:
        status_suggestion = status
        volatility_estimate = volatility_avg
        next_actions = ["observe_more", "online_search"]
        rationale = f"(フォールバック)探索的評価。潜在的競合を注視。揺れ{volatility_avg:.2f}"
    else:  # BALANCED
        status_suggestion = status
        volatility_estimate = volatility_avg
        next_actions = ["human_review"] if status == "candidate" else ["observe_more"]
        rationale = f"(フォールバック)標準評価。現状維持。揺れ{volatility_avg:.2f}"
    
    return {
        "aggregate_key": agg_key,
        "profile_id": profile_id,
        "vote": {
            "status_suggestion": status_suggestion,
            "risk_flags": risk_flags,
            "route_emphasis": competing_routes[:3],
            "volatility_estimate": round(volatility_estimate, 3),
            "rationale": rationale[:200],
            "needs_human_review": len(risk_flags) > 0 or profile_id == PROFILE_CONSERVATIVE,
            "next_actions": next_actions
        }
    }


def compute_drift(votes: List[Dict[str, Any]], success_votes: int) -> Dict[str, Any]:
    """
    Compute drift from audit votes.
    
    Handles degraded/unavailable states when LLM votes are missing.
    
    Returns transparent drift data including:
    - stability (stable/medium/volatile/degraded/unavailable)
    - status_votes (vote distribution)
    - risk_flag_union (all flags across votes)
    - risk_flag_conflicts (flags that differ between votes)
    - axis_drift (analysis axes)
    """
    # Handle unavailable case (0 successful votes)
    if success_votes == 0:
        return {
            "stability": "unavailable",
            "degraded": False,
            "unavailable": True,
            "success_votes": 0,
            "status_votes": {},
            "risk_flag_union": [],
            "risk_flag_conflicts": [],
            "axis_drift": {}  # No axis computation when unavailable
        }
    
    # Handle degraded case (1-2 successful votes)
    degraded = success_votes < 3
    
    if len(votes) == 0:
        return {
            "stability": "unavailable",
            "degraded": False,
            "unavailable": True,
            "success_votes": success_votes,
            "status_votes": {},
            "risk_flag_union": [],
            "risk_flag_conflicts": [],
            "axis_drift": {}
        }
    
    # Extract vote details
    status_suggestions = [v["vote"]["status_suggestion"] for v in votes]
    volatility_estimates = [v["vote"]["volatility_estimate"] for v in votes]
    all_risk_flags = [set(v["vote"]["risk_flags"]) for v in votes]
    human_review_votes = [v["vote"]["needs_human_review"] for v in votes]
    
    # Compute status_votes distribution
    status_votes = {}
    for s in status_suggestions:
        status_votes[s] = status_votes.get(s, 0) + 1
    
    # Compute risk_flag_union and conflicts
    risk_flag_union = set()
    for flags in all_risk_flags:
        risk_flag_union.update(flags)
    
    # Conflicts: flags that appear in some but not all votes
    risk_flag_intersection = all_risk_flags[0].copy() if all_risk_flags else set()
    for flags in all_risk_flags[1:]:
        risk_flag_intersection &= flags
    risk_flag_conflicts = list(risk_flag_union - risk_flag_intersection)
    
    # Compute stability
    unique_statuses = set(status_suggestions)
    
    # Check for major axis conflicts (typo_plausible vs entity_ambiguous)
    major_conflict_pairs = [
        ("typo_plausible", "entity_ambiguous"),
        ("source_conflict", "template_risk")
    ]
    has_major_conflict = False
    for a, b in major_conflict_pairs:
        if a in risk_flag_conflicts and b in risk_flag_conflicts:
            has_major_conflict = True
            break
    
    if len(unique_statuses) == 1 and not has_major_conflict:
        stability = "stable"
    elif len(unique_statuses) == 2:
        stability = "medium"
    elif len(unique_statuses) >= 3 or has_major_conflict:
        stability = "volatile"
    else:
        stability = "medium"
    
    # If degraded (1-2 votes), mark it but keep computed stability
    if degraded:
        stability = "degraded"
    
    # Compute axis_drift
    axis_drift = {}
    
    # focus_axis
    if len(risk_flag_union) > 0 and len(risk_flag_conflicts) > 0:
        if "source_conflict" in risk_flag_union and "entity_ambiguous" in risk_flag_union:
            axis_drift["focus_axis"] = "dispersion_vs_sense_conflict"
        elif "typo_plausible" in risk_flag_conflicts:
            axis_drift["focus_axis"] = "short_vs_tier1"
        else:
            axis_drift["focus_axis"] = "general_divergence"
    else:
        axis_drift["focus_axis"] = "aligned"
    
    # abstention_axis
    if all(human_review_votes):
        axis_drift["abstention_axis"] = "strict"
    elif not any(human_review_votes):
        axis_drift["abstention_axis"] = "loose"
    else:
        axis_drift["abstention_axis"] = "mixed"
    
    # granularity_axis
    rationale_lens = [len(v["vote"]["rationale"]) for v in votes]
    len_variance = max(rationale_lens) - min(rationale_lens) if rationale_lens else 0
    if len_variance > 50:
        axis_drift["granularity_axis"] = "detailed"
    else:
        axis_drift["granularity_axis"] = "brief"
    
    # template_fidelity_axis
    template_flags = sum(1 for flags in all_risk_flags if "template_risk" in flags)
    if template_flags == 0:
        axis_drift["template_fidelity_axis"] = "pass"
    elif template_flags <= 1:
        axis_drift["template_fidelity_axis"] = "warn"
    else:
        axis_drift["template_fidelity_axis"] = "fail"
    
    # action_conservatism_axis
    observe_count = sum(1 for v in votes if "observe_more" in v["vote"]["next_actions"])
    review_count = sum(1 for v in votes if "human_review" in v["vote"]["next_actions"])
    if observe_count > review_count:
        axis_drift["action_conservatism_axis"] = "observe_bias"
    else:
        axis_drift["action_conservatism_axis"] = "review_bias"
    
    return {
        "stability": stability,
        "degraded": degraded,
        "unavailable": False,
        "success_votes": success_votes,
        "status_votes": status_votes,
        "risk_flag_union": list(risk_flag_union),
        "risk_flag_conflicts": risk_flag_conflicts,
        "axis_drift": axis_drift
    }


# High-risk flags that trigger human review (title_like removed per spec)
HIGH_RISK_FLAGS = {"source_conflict", "entity_ambiguous", "template_risk"}

# Threshold for source_conflict detection
SOURCE_CONFLICT_THRESHOLD = 0.3


def normalize_risk_flags(raw_flags: Set[str], 
                         aggregate: Dict[str, Any]) -> Tuple[Set[str], Dict[str, str]]:
    """
    Normalize risk flags based on hard evidence rules.
    
    LLM can output any flags, but we only keep high_risk flags
    that have machine-verifiable evidence in the aggregate.
    
    Rules:
    - title_like: Not a high_risk flag anymore (always drop from high_risk)
    - source_conflict: Only if conflict_components.source_conflict >= threshold
    - entity_ambiguous: Only if hypotheses.B has multi_entity signals
    - template_risk: Only if explicit template match signal exists
    
    Returns:
        (normalized_flags, dropped_flags_with_reasons)
    """
    normalized = set()
    dropped = {}
    
    for flag in raw_flags:
        # title_like is no longer high_risk
        if flag == "title_like":
            dropped[flag] = "not_high_risk_flag"
            # Still add to normalized as it's a valid flag, just not high_risk
            normalized.add(flag)
            continue
        
        # source_conflict: requires conflict_components evidence
        if flag == "source_conflict":
            conflict_components = aggregate.get("conflict_components", {})
            source_conflict_val = conflict_components.get("source_conflict", 0)
            
            # Also check volatility as proxy for source conflict
            volatility_max = aggregate.get("volatility_max", 0)
            
            if source_conflict_val >= SOURCE_CONFLICT_THRESHOLD:
                normalized.add(flag)
            elif volatility_max >= 0.5:
                # High volatility can indicate source conflict
                normalized.add(flag)
            else:
                dropped[flag] = f"no_evidence:conflict_components.source_conflict={source_conflict_val},volatility_max={volatility_max}"
            continue
        
        # entity_ambiguous: requires multi-entity evidence
        if flag == "entity_ambiguous":
            hypotheses = aggregate.get("hypotheses", {})
            h_b = hypotheses.get("B", {})
            
            # Check for explicit multi-entity signals
            notes = h_b.get("notes", [])
            signals = h_b.get("signals", [])
            reasons = h_b.get("reasons", [])
            
            has_multi_entity_signal = False
            
            # Check notes for multi_entity indicators
            multi_entity_keywords = ["multi_entity", "dual_entity", "multiple_entity", "entity_types"]
            for note in (notes if isinstance(notes, list) else [notes]):
                if isinstance(note, str):
                    if any(kw in note.lower() for kw in multi_entity_keywords):
                        has_multi_entity_signal = True
                        break
            
            # Check signals
            for signal in (signals if isinstance(signals, list) else [signals]):
                if isinstance(signal, str):
                    if any(kw in signal.lower() for kw in multi_entity_keywords):
                        has_multi_entity_signal = True
                        break
            
            # Check if competing_routes includes both B and other routes (ambiguity indicator)
            competing_routes = aggregate.get("competing_routes", [])
            competing_count = aggregate.get("competing_count", 0)
            
            if "B" in competing_routes and competing_count >= 2:
                has_multi_entity_signal = True
            
            if has_multi_entity_signal:
                normalized.add(flag)
            else:
                dropped[flag] = "no_multi_entity_evidence_in_hypotheses_B"
            continue
        
        # template_risk: requires explicit template match signal
        if flag == "template_risk":
            hypotheses = aggregate.get("hypotheses", {})
            h_c = hypotheses.get("C", {})  # Novel/template hypothesis
            
            # Check for explicit template signals
            signals = h_c.get("signals", [])
            reasons = h_c.get("reasons", [])
            score_max = h_c.get("score_max", 0)
            
            has_template_signal = False
            
            template_keywords = ["template", "pattern", "formulaic", "boilerplate"]
            for signal in (signals if isinstance(signals, list) else [signals]):
                if isinstance(signal, str):
                    if any(kw in signal.lower() for kw in template_keywords):
                        has_template_signal = True
                        break
            
            # Also check if C hypothesis has significant score
            if score_max >= 0.3:
                has_template_signal = True
            
            if has_template_signal:
                normalized.add(flag)
            else:
                dropped[flag] = "no_template_signal_in_hypotheses_C"
            continue
        
        # typo_plausible: not a high_risk flag, but keep it
        if flag == "typo_plausible":
            normalized.add(flag)
            continue
        
        # Any other flag: keep as-is
        normalized.add(flag)
    
    return normalized, dropped


def should_send_to_human_review(drift: Dict[str, Any], 
                                 votes: List[Dict[str, Any]],
                                 aggregate: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Determine if aggregate should be sent to human review queue.
    
    Decision logic (from spec):
    - volatile → true
    - unavailable → true (with unavailable reason)
    - degraded → true (with degraded reason)
    - medium + any high_risk_flags → true
    - medium + no high_risk_flags → false
    - stable + 2+ high_risk_flags → true
    - stable + 0-1 high_risk_flags → false
    """
    reasons = []
    stability = drift.get("stability", "unknown")
    
    # Unavailable case
    if drift.get("unavailable", False):
        reasons.append("llm_unavailable")
        return True, reasons
    
    # Degraded case
    if drift.get("degraded", False):
        reasons.append(f"degraded:success_votes={drift.get('success_votes', 0)}")
        return True, reasons
    
    # Volatile case
    if stability == "volatile":
        reasons.append("stability_volatile")
        return True, reasons
    
    # Get high-risk flags from union
    risk_flag_union = set(drift.get("risk_flag_union", []))
    high_risk_found = risk_flag_union & HIGH_RISK_FLAGS
    high_risk_count = len(high_risk_found)
    
    # Medium case
    if stability == "medium":
        if high_risk_count >= 1:
            reasons.append(f"medium_with_high_risk:{','.join(high_risk_found)}")
            return True, reasons
        else:
            return False, []
    
    # Stable case
    if stability == "stable":
        if high_risk_count >= 2:
            reasons.append(f"stable_with_multiple_high_risk:{','.join(high_risk_found)}")
            return True, reasons
        else:
            return False, []
    
    # Unknown stability - be conservative
    reasons.append(f"unknown_stability:{stability}")
    return True, reasons


def truncate_vote(vote: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """
    Truncate vote fields to spec limits.
    
    Limits:
    - rationale: 200 chars
    - risk_flags: 5 items
    - next_actions: 3 items
    - route_emphasis: 3 items
    
    Returns: (truncated_vote, list of truncated field names)
    """
    truncated_fields = []
    vote_data = vote["vote"]
    
    # Truncate rationale
    if len(vote_data.get("rationale", "")) > 200:
        vote_data["rationale"] = vote_data["rationale"][:200]
        truncated_fields.append("rationale")
    
    # Truncate risk_flags
    if len(vote_data.get("risk_flags", [])) > 5:
        vote_data["risk_flags"] = vote_data["risk_flags"][:5]
        truncated_fields.append("risk_flags")
    
    # Truncate next_actions
    if len(vote_data.get("next_actions", [])) > 3:
        vote_data["next_actions"] = vote_data["next_actions"][:3]
        truncated_fields.append("next_actions")
    
    # Truncate route_emphasis
    if len(vote_data.get("route_emphasis", [])) > 3:
        vote_data["route_emphasis"] = vote_data["route_emphasis"][:3]
        truncated_fields.append("route_emphasis")
    
    return vote, truncated_fields


def run_7cprime_audit(aggregate: Dict[str, Any],
                       audit_7c: Dict[str, Any],
                       state_entry: Dict[str, Any],
                       use_llm: bool = True) -> Dict[str, Any]:
    """
    Run Phase 7C' triple audit on a single aggregate.
    
    Uses LLM for vote generation if available, falls back to rules if not.
    Tracks success_votes for degraded/unavailable detection.
    """
    now = datetime.now(timezone.utc)
    agg_key = aggregate.get("aggregate_key", "unknown")
    
    votes = []
    llm_success_count = 0
    llm_errors = []
    all_truncated_fields = []
    
    profiles = [
        (PROFILE_CONSERVATIVE, SYSTEM_PROMPT_CONSERVATIVE),
        (PROFILE_BALANCED, SYSTEM_PROMPT_BALANCED),
        (PROFILE_EXPLORATORY, SYSTEM_PROMPT_EXPLORATORY)
    ]
    
    for profile_id, system_prompt in profiles:
        vote = None
        
        if use_llm:
            vote = generate_audit_vote_llm(aggregate, audit_7c, profile_id, system_prompt)
            if vote:
                llm_success_count += 1
                # Apply truncation
                vote, truncated = truncate_vote(vote)
                if truncated:
                    all_truncated_fields.extend(truncated)
            else:
                llm_errors.append(f"{profile_id}:failed")
        
        if vote is None:
            vote = generate_audit_vote_fallback(aggregate, audit_7c, profile_id)
            # Fallback votes also need truncation
            vote, truncated = truncate_vote(vote)
            if truncated:
                all_truncated_fields.extend(truncated)
        
        votes.append(vote)
    
    # Compute drift with success_votes for degraded/unavailable detection
    drift = compute_drift(votes, llm_success_count)
    
    # Normalize risk flags based on aggregate evidence
    raw_flags = set(drift.get("risk_flag_union", []))
    normalized_flags, dropped_flags = normalize_risk_flags(raw_flags, aggregate)
    
    # Update drift with raw/normalized/dropped info
    drift["risk_flag_union_raw"] = list(raw_flags)
    drift["risk_flag_union"] = list(normalized_flags & HIGH_RISK_FLAGS)  # Only high_risk for review
    drift["risk_flag_all"] = list(normalized_flags)  # All normalized flags
    drift["dropped_flags"] = dropped_flags
    
    # Add LLM error info to drift
    drift["llm_errors"] = llm_errors
    if all_truncated_fields:
        drift["truncated_fields"] = list(set(all_truncated_fields))
    
    # Determine human review requirement (uses normalized high_risk flags)
    should_review, review_reasons = should_send_to_human_review(drift, votes, aggregate)
    
    return {
        "aggregate_key": agg_key,
        "token_norm": aggregate.get("token_norm"),
        "spec_version": SPEC_VERSION_7CPRIME,
        "engine_version": VERSION,
        "votes": votes,
        "drift": drift,
        "human_review": {
            "required": should_review,
            "reasons": review_reasons
        },
        "llm_success_count": llm_success_count,
        "timestamp": now.isoformat()
    }


def load_audit_7c_records(file_path: str) -> Dict[str, Dict[str, Any]]:
    """Load 7C audit records indexed by aggregate_key."""
    records = {}
    
    if not Path(file_path).exists():
        return records
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    agg_key = record.get("aggregate_key")
                    if agg_key:
                        records[agg_key] = record
    except Exception as e:
        print(f"[7C'] Error loading 7C audit log: {e}")
    
    return records


def append_jsonl(record: Dict[str, Any], file_path: str) -> None:
    """Append record to JSONL file (append-only)."""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')


def save_json(data: Dict[str, Any], file_path: str) -> None:
    """Save JSON file."""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def check_llm_connection() -> bool:
    """Check if LLM is reachable."""
    import urllib.request
    import urllib.error
    
    try:
        url = f"{LLM_HOST}/models"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=5) as response:
            return response.status == 200
    except:
        return False


def run_7cprime_mode(state_file: str,
                      aggregated_file: str,
                      audit_7c_file: str,
                      votes_file: str,
                      drift_file: str,
                      human_review_file: str,
                      qc_metrics_file: str) -> Dict[str, Any]:
    """
    Run Phase 7C' triple audit mode.
    
    Uses QwQ32B LLM for generating 3 different audit perspectives.
    Falls back to rule-based logic if LLM unavailable.
    """
    print("=" * 60)
    print(f"ESDE Engine v{VERSION} - Phase 7C' Triple Audit Mode")
    print("=" * 60)
    print()
    print("Phase 7C': Triple audit with drift detection")
    print("Profiles: CONSERVATIVE / BALANCED / EXPLORATORY")
    print("NO winner determination. NO state modification.")
    print()
    
    # Check LLM connection
    print(f"[7C'] Checking LLM connection: {LLM_HOST}")
    use_llm = check_llm_connection()
    if use_llm:
        print(f"[7C'] ✓ LLM available: {LLM_MODEL}")
    else:
        print(f"[7C'] ✗ LLM unavailable - using fallback rules")
    print()
    
    # Load state (read-only)
    agg_state = AggregateStateManager(state_file)
    agg_state.load()
    print(f"[7C'] Loaded aggregate state: {state_file}")
    
    # Load aggregated queue (7B+ output)
    aggregates = load_aggregated_queue(aggregated_file)
    print(f"[7C'] Loaded {len(aggregates)} aggregates from: {aggregated_file}")
    
    # Load 7C audit records
    audit_7c_records = load_audit_7c_records(audit_7c_file)
    print(f"[7C'] Loaded {len(audit_7c_records)} 7C audit records from: {audit_7c_file}")
    
    if not aggregates:
        print("[7C'] No aggregates to audit. Exiting.")
        return {"audited": 0}
    
    # Results tracking
    results = {
        "audited": 0,
        "stable": 0,
        "medium": 0,
        "volatile": 0,
        "human_review_required": 0,
        "llm_calls_success": 0,
        "llm_calls_total": 0
    }
    
    print(f"\n[7C'] Running triple audit on {len(aggregates)} aggregates...")
    print()
    
    for aggregate in aggregates:
        agg_key = aggregate.get("aggregate_key", "unknown")
        token = aggregate.get("token_norm", "unknown")
        
        # Get related records (read-only)
        state_entry = agg_state.get_entry(agg_key) or {}
        audit_7c = audit_7c_records.get(agg_key, {})
        
        # Run 7C' audit
        audit_result = run_7cprime_audit(aggregate, audit_7c, state_entry, use_llm=use_llm)
        
        # Track LLM usage
        results["llm_calls_total"] += 3
        results["llm_calls_success"] += audit_result.get("llm_success_count", 0)
        
        # Write votes (3 votes)
        for vote in audit_result["votes"]:
            append_jsonl(vote, votes_file)
        
        # Write drift with all required fields per spec
        drift_data = audit_result["drift"]
        drift_record = {
            "aggregate_key": agg_key,
            "token_norm": token,
            # Core stability info
            "stability": drift_data.get("stability"),
            "degraded": drift_data.get("degraded", False),
            "unavailable": drift_data.get("unavailable", False),
            "success_votes": drift_data.get("success_votes", 0),
            # Transparency fields
            "status_votes": drift_data.get("status_votes", {}),
            # Risk flags (raw/normalized/dropped)
            "risk_flag_union_raw": drift_data.get("risk_flag_union_raw", []),
            "risk_flag_union": drift_data.get("risk_flag_union", []),  # Normalized high_risk only
            "risk_flag_all": drift_data.get("risk_flag_all", []),  # All normalized flags
            "dropped_flags": drift_data.get("dropped_flags", {}),
            "risk_flag_conflicts": drift_data.get("risk_flag_conflicts", []),
            # Axis drift
            "axis_drift": drift_data.get("axis_drift", {}),
            # Human review info
            "human_review_required": audit_result["human_review"]["required"],
            "human_review_reason_codes": audit_result["human_review"]["reasons"],
            # LLM errors if any
            "llm_errors": drift_data.get("llm_errors", []),
            # Truncation info
            "truncated_fields": drift_data.get("truncated_fields", []),
            # Timestamp
            "timestamp": audit_result["timestamp"]
        }
        append_jsonl(drift_record, drift_file)
        
        # Write to human review queue if required
        if audit_result["human_review"]["required"]:
            review_record = {
                "aggregate_key": agg_key,
                "token_norm": token,
                "reasons": audit_result["human_review"]["reasons"],
                "stability": audit_result["drift"]["stability"],
                "success_votes": audit_result["drift"].get("success_votes", 0),
                "timestamp": audit_result["timestamp"]
            }
            append_jsonl(review_record, human_review_file)
            results["human_review_required"] += 1
        
        # Track results
        results["audited"] += 1
        stability = audit_result["drift"]["stability"]
        if stability == "stable":
            results["stable"] += 1
        elif stability == "medium":
            results["medium"] += 1
        elif stability == "volatile":
            results["volatile"] += 1
        elif stability == "degraded":
            results["degraded"] = results.get("degraded", 0) + 1
        elif stability == "unavailable":
            results["unavailable"] = results.get("unavailable", 0) + 1
        
        # Progress output
        stability_char = {
            "stable": "●", 
            "medium": "◐", 
            "volatile": "○",
            "degraded": "⚡",
            "unavailable": "✗"
        }.get(stability, "?")
        hr_mark = "⚠" if audit_result["human_review"]["required"] else "✓"
        llm_mark = f"LLM:{audit_result.get('llm_success_count', 0)}/3" if use_llm else "RULE"
        print(f"  {stability_char}{hr_mark} {token} ({agg_key[:8]}...): stability={stability} [{llm_mark}]")
    
    # Save QC metrics
    qc_metrics = {
        "spec_version": SPEC_VERSION_7CPRIME,
        "engine_version": VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_audited": results["audited"],
        "stability_distribution": {
            "stable": results["stable"],
            "medium": results["medium"],
            "volatile": results["volatile"],
            "degraded": results.get("degraded", 0),
            "unavailable": results.get("unavailable", 0)
        },
        "human_review_required": results["human_review_required"],
        "human_review_not_required": results["audited"] - results["human_review_required"],
        "human_review_rate": round(results["human_review_required"] / max(1, results["audited"]), 3),
        "llm_usage": {
            "enabled": use_llm,
            "model": LLM_MODEL if use_llm else None,
            "calls_total": results["llm_calls_total"],
            "calls_success": results["llm_calls_success"],
            "success_rate": round(results["llm_calls_success"] / max(1, results["llm_calls_total"]), 3)
        }
    }
    save_json(qc_metrics, qc_metrics_file)
    
    # Summary
    print()
    print("=" * 60)
    print("Phase 7C' Triple Audit Summary")
    print("=" * 60)
    print(f"Total audited:        {results['audited']}")
    print(f"Stability distribution:")
    print(f"  ● Stable:           {results['stable']}")
    print(f"  ◐ Medium:           {results['medium']}")
    print(f"  ○ Volatile:         {results['volatile']}")
    if results.get("degraded", 0) > 0:
        print(f"  ⚡ Degraded:         {results['degraded']}")
    if results.get("unavailable", 0) > 0:
        print(f"  ✗ Unavailable:      {results['unavailable']}")
    
    not_required = results["audited"] - results["human_review_required"]
    print(f"\nHuman review:")
    print(f"  Required:           {results['human_review_required']} ({qc_metrics['human_review_rate']*100:.1f}%)")
    print(f"  Not required:       {not_required} ({(1-qc_metrics['human_review_rate'])*100:.1f}%)")
    print()
    if use_llm:
        print(f"LLM Usage:")
        print(f"  Model:              {LLM_MODEL}")
        print(f"  Calls:              {results['llm_calls_success']}/{results['llm_calls_total']} success")
        print(f"  Success rate:       {qc_metrics['llm_usage']['success_rate']*100:.1f}%")
    else:
        print(f"LLM Usage:            Disabled (using fallback rules)")
    print()
    print(f"Output files:")
    print(f"  Votes:        {votes_file}")
    print(f"  Drift:        {drift_file}")
    print(f"  Human review: {human_review_file}")
    print(f"  QC metrics:   {qc_metrics_file}")
    print()
    print("Phase 7C' guarantees:")
    print("  - winner=null ALWAYS maintained")
    print("  - NO state modified")
    print("  - NO patch applied")
    print("  - All outputs are append-only")
    print("  - Same input produces same output")
    
    return results


# =============================================================================
# Statistics
# =============================================================================

def show_stats(agg_state: AggregateStateManager, cache: SearchCache) -> None:
    """Show 7B+ specific statistics."""
    print("\n" + "=" * 60)
    print("Aggregate State Statistics (v5.3.4)")
    print("=" * 60)
    
    stats = agg_state.get_stats()
    print(f"Total aggregates: {stats['total']}")
    print(f"Seen (evaluated): {stats['seen_count']}")
    print(f"Finalized (human-reviewed): {stats['finalized_count']}")
    print(f"Needs observation: {stats['needs_observation_count']}")
    print(f"Avg observe count: {stats['avg_observe_count']}")
    
    print("\nBy status:")
    for status, count in stats["by_status"].items():
        print(f"  {status}: {count}")
    
    print("\n" + "=" * 60)
    print("Cache Statistics")
    print("=" * 60)
    cache_stats = cache.get_stats()
    print(f"Memory entries: {cache_stats['memory_entries']}")
    print(f"File entries: {cache_stats['file_entries']}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="ESDE Phase 7B+/7C/7C': Unknown Queue Resolver and Auditor"
    )
    parser.add_argument("--mode", choices=["7bplus", "7c", "7cprime"], default="7bplus",
                        help="Execution mode: 7bplus (default), 7c (audit), or 7cprime (triple audit)")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--reprocess", action="store_true",
                        help="Include finalized entries in processing")
    parser.add_argument("--min-sources", type=int, default=3)
    parser.add_argument("--max-sources", type=int, default=8)
    parser.add_argument("--cache-dir", default=CACHE_DIR)
    parser.add_argument("--stats", action="store_true")
    parser.add_argument("--queue-file", default=QUEUE_FILE)
    parser.add_argument("--state-file", default=AGG_STATE_FILE)
    parser.add_argument("--run-id", default=None)
    # Phase 7C specific options
    parser.add_argument("--aggregated-file", default=AGGREGATED_QUEUE_FILE,
                        help="7B+ aggregated output file (for 7C/7C' mode)")
    parser.add_argument("--audit-log-file", default=AUDIT_LOG_FILE_7C,
                        help="7C audit log file (input for 7C', output for 7C)")
    # Phase 7C' specific options
    parser.add_argument("--votes-file", default=AUDIT_VOTES_FILE_7CPRIME,
                        help="7C' audit votes output file")
    parser.add_argument("--drift-file", default=AUDIT_DRIFT_FILE_7CPRIME,
                        help="7C' drift output file")
    parser.add_argument("--human-review-file", default=HUMAN_REVIEW_QUEUE_7CPRIME,
                        help="7C' human review queue output file")
    parser.add_argument("--qc-metrics-file", default=QC_METRICS_FILE_7CPRIME,
                        help="7C' QC metrics output file")
    
    args = parser.parse_args()
    
    # ==========================================================================
    # Phase 7C' Mode: Triple Audit
    # ==========================================================================
    if args.mode == "7cprime":
        run_7cprime_mode(
            state_file=args.state_file,
            aggregated_file=args.aggregated_file,
            audit_7c_file=args.audit_log_file,
            votes_file=args.votes_file,
            drift_file=args.drift_file,
            human_review_file=args.human_review_file,
            qc_metrics_file=args.qc_metrics_file
        )
        return
    
    # ==========================================================================
    # Phase 7C Mode: Audit only
    # ==========================================================================
    if args.mode == "7c":
        run_7c_mode(
            state_file=args.state_file,
            aggregated_file=args.aggregated_file,
            audit_log_file=args.audit_log_file
        )
        return
    
    # ==========================================================================
    # Phase 7B+ Mode: Normal processing
    # ==========================================================================
    print("=" * 60)
    print(f"ESDE Engine v{VERSION} - Phase 7B+ Resolver (v5.3.4 Spec)")
    print("=" * 60)
    print()
    print("Key: ⊘=quarantine  ◐=defer  ○=candidate")
    print("v5.3.4: Two-stage processing (seen/finalized)")
    print("v5.3.4: Observation model in state")
    print()
    
    # Initialize
    agg_state = AggregateStateManager(args.state_file)
    legacy_state = QueueStateManager(LEGACY_STATE_FILE)
    cache = SearchCache(args.cache_dir)
    search_provider = SearXNGProvider()  #仕様変更 online_v4.py　2026/01/10
    audit_ledger = AuditLedger(LEDGER_FILE)

    print("[7B+] Using MultiSourceProvider (online_v4)") #online_v4.py 仕様変更に伴う修正　2026/01/10

    agg_state.load()
    legacy_state.load()
    
    if args.stats:
        show_stats(agg_state, cache)
        return
    
    # Load queue
    print(f"\n[7B+] Loading queue: {args.queue_file}")
    records = load_queue_records(args.queue_file)
    print(f"[7B+] Loaded {len(records)} records")
    
    if not records:
        print("[7B+] No records to process. Exiting.")
        return
    
    # Process
    results = process_batch(
        records=records,
        agg_state=agg_state,
        legacy_state=legacy_state,
        audit_ledger=audit_ledger,
        search_provider=search_provider,
        cache=cache,
        limit=args.limit,
        min_sources=args.min_sources,
        max_sources=args.max_sources,
        dry_run=args.dry_run,
        reprocess=args.reprocess,
        run_id=args.run_id
    )
    
    # Summary
    print("\n" + "=" * 60)
    print("7B+ Resolution Summary (v5.3.4)")
    print("=" * 60)
    
    skip = results.get("skip_info", {})
    print(f"\n--- Pipeline Accountability ---")
    print(f"Candidates total:    {skip.get('candidates_total', 0)}")
    print(f"Skipped (by route):  {skip.get('skipped_by_route', 0)}")
    print(f"Skipped (finalized): {skip.get('skipped_finalized', 0)}")
    print(f"Skipped (cooldown):  {skip.get('skipped_cooldown', 0)}")
    print(f"Skipped (observed):  {skip.get('skipped_observed', 0)}")
    print(f"Skipped (by limit):  {results.get('skipped_by_filter', 0)}")
    print(f"Skipped (by valid):  {results.get('skipped_by_validation', 0)}")
    print(f"Eligible pending:    {skip.get('eligible_pending', 0)}")
    
    # Accounting check
    print(f"\n--- Accounting Integrity Check ---")
    total = skip.get('candidates_total', 0)
    by_route = skip.get('skipped_by_route', 0)
    finalized = skip.get('skipped_finalized', 0)
    cooldown = skip.get('skipped_cooldown', 0)
    observed = skip.get('skipped_observed', 0)
    eligible = skip.get('eligible_pending', 0)
    
    expected1 = by_route + finalized + cooldown + observed + eligible
    if total == expected1:
        print(f"✓ Total breakdown OK: {total} = {by_route}+{finalized}+{cooldown}+{observed}+{eligible}")
    else:
        print(f"✗ ACCOUNTING ERROR: {total} != {expected1}")
    
    processed = results['processed']
    by_filter = results.get('skipped_by_filter', 0)
    by_valid = results.get('skipped_by_validation', 0)
    expected2 = processed + by_filter + by_valid
    
    if eligible == expected2:
        print(f"✓ Eligible breakdown OK: {eligible} = {processed}+{by_filter}+{by_valid}")
    else:
        print(f"✗ ACCOUNTING ERROR: {eligible} != {expected2}")
    
    print(f"\n--- Processing Results ---")
    print(f"Records processed: {results['processed']}")
    print(f"Aggregated to:     {results['aggregated']} unique keys")
    
    print(f"\n--- Aggregate Status ---")
    print(f"Quarantined:   {results['quarantined']}")
    print(f"Deferred:      {results['deferred']}")
    print(f"Candidate:     {results['candidate']}")
    
    print(f"\n--- Volatility Analysis ---")
    print(f"Average volatility: {results['avg_volatility']:.3f}")
    print(f"With competing hypotheses: {results['competing_hypotheses']}")
    print(f"COMPETE_TH = {COMPETE_TH}")
    
    # Save state
    if not args.dry_run:
        print("\n[7B+] Saving aggregate state...")
        agg_state.save()
    else:
        print("\n[7B+] Dry run - no state saved")
    
    print("\n[7B+] Done.")
    print()
    print("v5.3.4 Guarantees:")
    print("  - winner=null ALWAYS maintained")
    print("  - State tracks seen/finalized separately")
    print("  - Ledger is append-only audit trail")
    print("  - Re-evaluation controlled by observation model")


if __name__ == "__main__":
    main()
