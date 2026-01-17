#!/usr/bin/env python3
"""
ESDE Engine - Phase 7D: Meta-Auditor (Rule Calibration by Log-Only)

"ESDEはなぜ今この世界を、こう見ているのか？"を問い直すフェーズ。
対象はデータではなく「法則（ルール）」であり、意味ではなく「意味が決まらない理由の扱い方」。

7D I/O Contract:
    Input:  7B+/7C/7C' の出力ログのみ
    Output: ルール変更提案（候補値＋根拠＋影響範囲＋リスク）
    No-touch: 意味決定しない / winner作らない / config書き換えない

Philosophy (Arism/ESDE v3.3):
    - 対等性: 単一の勝者で世界を閉じない（候補レンジで提示）
    - ε保持: ズレは誤差ではなく構造が生きている証拠
    - 第三項: 7Dは7B+/7C/7C'の"観測者"として機能

Usage:
    python esde_meta_auditor.py [options]
    python esde_meta_auditor.py --data-dir ./data --output-dir ./data/7d_output
"""

import json
import hashlib
import argparse
import statistics
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Set, Tuple
from collections import defaultdict
from dataclasses import dataclass, field, asdict


# =============================================================================
# Version & Constants
# =============================================================================

VERSION = "7D-1.0"
SPEC_VERSION = "7D-1.0"

# Input file definitions
REQUIRED_INPUTS = [
    "unknown_queue_7bplus.jsonl"
]

OPTIONAL_INPUTS = [
    "audit_log_7c.jsonl",
    "audit_drift_7cprime.jsonl",
    "audit_votes_7cprime.jsonl",
    "human_review_queue_7cprime.jsonl",
    "qc_metrics_7cprime.json"
]

# Field aliases for schema flexibility
FIELD_ALIASES = {
    "volatility_avg": ["volatility_avg", "vol_avg", "volatility.mean", "avg_volatility", "global_volatility"],
    "volatility_max": ["volatility_max", "vol_max", "volatility.max", "max_volatility"],
    "competing_count": ["competing_count", "compete_count", "num_competing"],
    "competing_routes": ["competing_routes", "competing", "routes_competing"],
    "status": ["status", "final_status", "recommendation"],
    "aggregate_key": ["aggregate_key", "agg_key", "key"],
    "token_norm": ["token_norm", "token", "normalized_token"],
    "stability": ["stability", "drift_stability"],
    "risk_flag_union": ["risk_flag_union", "risk_flags", "flags"],
    "struct_ok": ["struct_ok", "structure_ok", "valid_structure"],
    "human_review_required": ["human_review_required", "needs_human_review", "human_review"],
}

# Proxy hierarchy for "missed danger" detection
PROXY_HIERARCHY = [
    ("stability_critical", lambda r: r.get("stability") in ("volatile", "unavailable")),
    ("stability_degraded", lambda r: r.get("stability") == "degraded"),
    ("struct_failed", lambda r: r.get("struct_ok") is False),
    ("risk_flagged", lambda r: len(r.get("risk_flag_union", [])) > 0),
    ("axis_drifted", lambda r: bool(r.get("axis_drift"))),
]

# Percentile candidates for threshold proposals
PERCENTILE_CANDIDATES = [70, 80, 90]


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class InputStats:
    """Statistics about loaded input files."""
    file_name: str
    exists: bool
    record_count: int = 0
    load_error: Optional[str] = None
    schema_warnings: List[str] = field(default_factory=list)


@dataclass
class VolatilityDistribution:
    """Volatility distribution statistics."""
    count: int
    valid_count: int  # Records with valid volatility
    missing_count: int  # Records with missing volatility
    mean: float
    std: float
    min_val: float
    max_val: float
    percentiles: Dict[int, float]  # {70: 0.45, 80: 0.52, ...}
    vol_high_rate: float  # Rate >= VOL_HIGH_TH
    vol_low_rate: float   # Rate < VOL_LOW_TH
    sample_size_adequate: bool  # True if n >= 10 for reliable percentiles


@dataclass
class CompeteDistribution:
    """Competing hypothesis distribution."""
    count: int
    mean: float
    by_count: Dict[int, int]  # {0: 5, 1: 10, 2: 15, ...}
    multi_compete_rate: float  # Rate with competing_count >= 2


@dataclass
class HumanReviewStats:
    """Human review statistics."""
    total_reviewed: int
    total_not_reviewed: int
    review_rate: float
    by_stability: Dict[str, int]
    missed_danger_count: int  # proxy立っているのにreviewされていない
    missed_danger_details: List[Dict[str, Any]]


@dataclass
class Proposal:
    """A single rule adjustment proposal."""
    id: str
    touch_surface: str  # config.py | hypothesis.py | 7cprime
    parameter: str
    current_value: Any
    candidates: List[Any]
    rationale: Dict[str, Any]
    impact_scope: List[str]
    risk: Dict[str, Any]
    confidence: str = "medium"  # high | medium | low


def unique_candidates(candidates: List[float], current: float, delta: float = 0.05) -> Tuple[List[float], bool]:
    """
    Ensure candidates are unique and have at least 2 options.
    
    Returns:
        (unique_candidates, was_補完ed)
    """
    # Filter None and make unique
    unique = list(dict.fromkeys([c for c in candidates if c is not None]))
    
    # Sort
    unique.sort()
    
    was_補完 = False
    
    # If less than 2 candidates, supplement with ±delta
    if len(unique) < 2:
        was_補完 = True
        base = unique[0] if unique else current
        
        # Add candidates at ±delta intervals
        for d in [delta, -delta, delta * 2, -delta * 2]:
            new_val = round_float(base + d)
            # Clip to [0, 1]
            new_val = max(0.0, min(1.0, new_val))
            if new_val not in unique and new_val != current:
                unique.append(new_val)
            if len(unique) >= 2:
                break
        
        unique.sort()
    
    return unique, was_補完


@dataclass
class AxisCandidate:
    """An axis candidate (structure only, no meaning)."""
    axis_id: str
    members: List[str]  # token_norms or aggregate_keys
    cohesion: float
    stability: float
    volatility_association: float
    operator_signature: Dict[str, float]  # ¬, ×, ⊕ rates (if available)
    note: str  # Structure memo only


# =============================================================================
# Utility Functions
# =============================================================================

def compute_file_hash(file_path: str) -> str:
    """Compute SHA256 hash of file for fingerprinting."""
    if not Path(file_path).exists():
        return "file_not_found"
    
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def compute_record_id(record: Dict[str, Any]) -> str:
    """Compute deterministic ID for a record."""
    # Try aggregate_key first
    agg_key = get_field(record, "aggregate_key")
    if agg_key:
        return agg_key
    
    # Fallback to hash of record content
    content = json.dumps(record, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def get_field(record: Dict[str, Any], field_name: str) -> Any:
    """Get field value with alias fallback."""
    aliases = FIELD_ALIASES.get(field_name, [field_name])
    
    for alias in aliases:
        # Direct access
        if alias in record:
            return record[alias]
        
        # Nested access (e.g., "volatility.mean")
        if "." in alias:
            parts = alias.split(".")
            val = record
            try:
                for part in parts:
                    val = val[part]
                return val
            except (KeyError, TypeError):
                continue
    
    return None


def round_float(value: float, precision: int = 6) -> float:
    """Round float with fixed precision for determinism."""
    if value is None:
        return None
    return round(value, precision)


def percentile(values: List[float], p: int) -> float:
    """Compute percentile (0-100)."""
    if not values:
        return 0.0
    
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * p / 100
    f = int(k)
    c = f + 1 if f + 1 < len(sorted_vals) else f
    
    if f == c:
        return sorted_vals[f]
    
    return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)


# =============================================================================
# Input Loading
# =============================================================================

def load_jsonl(file_path: str) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """Load JSONL file with error handling."""
    records = []
    
    if not Path(file_path).exists():
        return records, "file_not_found"
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        return records, f"json_error_line_{line_num}: {str(e)}"
    except Exception as e:
        return records, f"read_error: {str(e)}"
    
    return records, None


def load_json(file_path: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Load JSON file with error handling."""
    if not Path(file_path).exists():
        return None, "file_not_found"
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f), None
    except Exception as e:
        return None, f"read_error: {str(e)}"


def load_all_inputs(data_dir: str) -> Tuple[Dict[str, Any], List[InputStats]]:
    """
    Load all input files with existence check and schema validation.
    
    Returns:
        data: Dict with loaded records by file type
        stats: List of InputStats for each file
    """
    data_path = Path(data_dir)
    data = {}
    stats = []
    
    all_files = REQUIRED_INPUTS + OPTIONAL_INPUTS
    
    for file_name in all_files:
        file_path = data_path / file_name
        is_required = file_name in REQUIRED_INPUTS
        
        stat = InputStats(
            file_name=file_name,
            exists=file_path.exists()
        )
        
        if not stat.exists:
            if is_required:
                stat.load_error = "required_file_missing"
            stats.append(stat)
            continue
        
        # Load based on extension
        if file_name.endswith('.jsonl'):
            records, error = load_jsonl(str(file_path))
            if error:
                stat.load_error = error
            else:
                stat.record_count = len(records)
                # Sort by record_id for determinism
                records.sort(key=lambda r: compute_record_id(r))
                data[file_name] = records
                
                # Check schema (sample first 5 records)
                stat.schema_warnings = check_schema(records[:5], file_name)
        
        elif file_name.endswith('.json'):
            content, error = load_json(str(file_path))
            if error:
                stat.load_error = error
            else:
                stat.record_count = 1
                data[file_name] = content
        
        stats.append(stat)
    
    return data, stats


def check_schema(records: List[Dict[str, Any]], file_name: str) -> List[str]:
    """Check schema and return warnings for missing expected fields."""
    warnings = []
    
    if not records:
        return warnings
    
    # Expected fields by file type
    expected_fields = {
        "unknown_queue_7bplus.jsonl": ["aggregate_key", "volatility_avg", "volatility_max", 
                                        "competing_count", "status"],
        "audit_log_7c.jsonl": ["aggregate_key", "struct_ok"],
        "audit_drift_7cprime.jsonl": ["aggregate_key", "stability", "risk_flag_union"],
        "human_review_queue_7cprime.jsonl": ["aggregate_key", "reasons"],
    }
    
    expected = expected_fields.get(file_name, [])
    
    for field_name in expected:
        found = False
        for record in records:
            if get_field(record, field_name) is not None:
                found = True
                break
        
        if not found:
            warnings.append(f"field_not_found:{field_name}")
    
    return warnings


# =============================================================================
# Analysis Functions
# =============================================================================

def analyze_volatility_distribution(records: List[Dict[str, Any]], 
                                     vol_high_th: float = 0.50,
                                     vol_low_th: float = 0.25) -> VolatilityDistribution:
    """Analyze volatility distribution from 7B+ records."""
    vol_values = []
    missing_count = 0
    
    for record in records:
        vol = get_field(record, "volatility_avg")
        if vol is not None:
            vol_values.append(float(vol))
        else:
            missing_count += 1
    
    total_count = len(records)
    valid_count = len(vol_values)
    sample_adequate = valid_count >= 10  # Minimum for reliable percentiles
    
    if not vol_values:
        return VolatilityDistribution(
            count=total_count, valid_count=0, missing_count=missing_count,
            mean=0.0, std=0.0, min_val=0.0, max_val=0.0,
            percentiles={}, vol_high_rate=0.0, vol_low_rate=0.0,
            sample_size_adequate=False
        )
    
    # Compute percentiles
    percentiles = {p: round_float(percentile(vol_values, p)) for p in PERCENTILE_CANDIDATES}
    
    # Compute rates
    vol_high_count = sum(1 for v in vol_values if v >= vol_high_th)
    vol_low_count = sum(1 for v in vol_values if v < vol_low_th)
    
    return VolatilityDistribution(
        count=total_count,
        valid_count=valid_count,
        missing_count=missing_count,
        mean=round_float(statistics.mean(vol_values)),
        std=round_float(statistics.stdev(vol_values) if len(vol_values) > 1 else 0.0),
        min_val=round_float(min(vol_values)),
        max_val=round_float(max(vol_values)),
        percentiles=percentiles,
        vol_high_rate=round_float(vol_high_count / len(vol_values)),
        vol_low_rate=round_float(vol_low_count / len(vol_values)),
        sample_size_adequate=sample_adequate
    )


def analyze_compete_distribution(records: List[Dict[str, Any]],
                                  compete_th: float = 0.15) -> CompeteDistribution:
    """Analyze competing hypothesis distribution."""
    compete_counts = []
    by_count = defaultdict(int)
    
    for record in records:
        count = get_field(record, "competing_count")
        if count is not None:
            compete_counts.append(int(count))
            by_count[int(count)] += 1
    
    if not compete_counts:
        return CompeteDistribution(
            count=0, mean=0.0, by_count={}, multi_compete_rate=0.0
        )
    
    multi_count = sum(1 for c in compete_counts if c >= 2)
    
    return CompeteDistribution(
        count=len(compete_counts),
        mean=round_float(statistics.mean(compete_counts)),
        by_count=dict(by_count),
        multi_compete_rate=round_float(multi_count / len(compete_counts))
    )


def analyze_human_review(data: Dict[str, Any]) -> HumanReviewStats:
    """Analyze human review patterns and missed dangers."""
    # Get records from different sources
    aggregates = data.get("unknown_queue_7bplus.jsonl", [])
    drift_records = data.get("audit_drift_7cprime.jsonl", [])
    hr_queue = data.get("human_review_queue_7cprime.jsonl", [])
    
    # Build set of reviewed aggregate_keys
    reviewed_keys = set()
    for record in hr_queue:
        key = get_field(record, "aggregate_key")
        if key:
            reviewed_keys.add(key)
    
    # Build drift index
    drift_by_key = {}
    for record in drift_records:
        key = get_field(record, "aggregate_key")
        if key:
            drift_by_key[key] = record
    
    # Analyze each aggregate
    by_stability = defaultdict(int)
    missed_dangers = []
    total_not_reviewed = 0
    
    for agg in aggregates:
        key = get_field(agg, "aggregate_key")
        if not key:
            continue
        
        is_reviewed = key in reviewed_keys
        drift = drift_by_key.get(key, {})
        stability = get_field(drift, "stability") or "unknown"
        
        if is_reviewed:
            by_stability[stability] += 1
        else:
            total_not_reviewed += 1
            
            # Check proxy conditions
            triggered_proxies = []
            for proxy_name, check_fn in PROXY_HIERARCHY:
                if check_fn(drift) or check_fn(agg):
                    triggered_proxies.append(proxy_name)
            
            if triggered_proxies:
                missed_dangers.append({
                    "aggregate_key": key,
                    "token_norm": get_field(agg, "token_norm"),
                    "triggered_proxies": triggered_proxies,
                    "stability": stability,
                    "volatility_avg": get_field(agg, "volatility_avg")
                })
    
    total_reviewed = len(reviewed_keys)
    total = total_reviewed + total_not_reviewed
    
    return HumanReviewStats(
        total_reviewed=total_reviewed,
        total_not_reviewed=total_not_reviewed,
        review_rate=round_float(total_reviewed / max(1, total)),
        by_stability=dict(by_stability),
        missed_danger_count=len(missed_dangers),
        missed_danger_details=missed_dangers[:20]  # Limit for output size
    )


def analyze_status_distribution(records: List[Dict[str, Any]]) -> Dict[str, int]:
    """Analyze status distribution."""
    distribution = defaultdict(int)
    
    for record in records:
        status = get_field(record, "status")
        if status:
            distribution[status] += 1
    
    return dict(distribution)


def analyze_stability_distribution(drift_records: List[Dict[str, Any]]) -> Dict[str, int]:
    """Analyze stability distribution from drift records."""
    distribution = defaultdict(int)
    
    for record in drift_records:
        stability = get_field(record, "stability")
        if stability:
            distribution[stability] += 1
    
    return dict(distribution)


def analyze_risk_flags(drift_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze risk flag patterns."""
    flag_counts = defaultdict(int)
    flag_cooccurrence = defaultdict(lambda: defaultdict(int))
    total_with_flags = 0
    
    for record in drift_records:
        flags = get_field(record, "risk_flag_union") or []
        if isinstance(flags, list) and len(flags) > 0:
            total_with_flags += 1
            for flag in flags:
                flag_counts[flag] += 1
                for other_flag in flags:
                    if flag != other_flag:
                        flag_cooccurrence[flag][other_flag] += 1
    
    # Convert cooccurrence to list format
    cooccurrence_list = []
    seen_pairs = set()
    for f1, others in flag_cooccurrence.items():
        for f2, count in others.items():
            pair = tuple(sorted([f1, f2]))
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                cooccurrence_list.append({
                    "flags": list(pair),
                    "count": count
                })
    
    cooccurrence_list.sort(key=lambda x: x["count"], reverse=True)
    
    return {
        "total_with_flags": total_with_flags,
        "total_records": len(drift_records),
        "flag_rate": round_float(total_with_flags / max(1, len(drift_records))),
        "flag_counts": dict(flag_counts),
        "top_cooccurrences": cooccurrence_list[:10]
    }


# =============================================================================
# Proposal Generation
# =============================================================================

def generate_vol_high_proposal(vol_dist: VolatilityDistribution,
                                current_th: float = 0.50,
                                risk_flag_rate: float = 0.0) -> List[Proposal]:
    """
    Generate VOL_HIGH_TH adjustment proposal(s).
    
    Returns list of proposals (may include hypothesis.py proposal if risk_flag_rate is high).
    """
    proposals = []
    
    if vol_dist.valid_count == 0:
        return proposals
    
    # Determine confidence based on sample size
    if not vol_dist.sample_size_adequate:
        base_confidence = "low"
        sample_note = f"n={vol_dist.valid_count} too small for reliable percentiles"
    else:
        base_confidence = "medium"
        sample_note = None
    
    # Check if distribution is skewed (too many HIGH)
    if vol_dist.vol_high_rate > 0.60:
        # Too many quarantined - threshold may be too low
        raw_candidates = [
            vol_dist.percentiles.get(70),
            vol_dist.percentiles.get(80),
            vol_dist.percentiles.get(90)
        ]
        raw_candidates = [c for c in raw_candidates if c and c > current_th]
        
        candidates, was_補完 = unique_candidates(raw_candidates, current_th, delta=0.05)
        
        if not candidates:
            return proposals
        
        why_now = f"VOL_HIGHが過密({vol_dist.vol_high_rate*100:.1f}%)で弁別力低下"
        if sample_note:
            why_now += f"; {sample_note}"
        if was_補完:
            why_now += "; candidates supplemented due to percentile collapse"
        
        proposals.append(Proposal(
            id="P-VOL_HIGH-001",
            touch_surface="config.py",
            parameter="VOL_HIGH_TH",
            current_value=current_th,
            candidates=candidates,
            rationale={
                "evidence": [
                    {"metric": "vol_high_rate", "value": vol_dist.vol_high_rate},
                    {"metric": "vol_p80", "value": vol_dist.percentiles.get(80)},
                    {"metric": "vol_mean", "value": vol_dist.mean},
                    {"metric": "vol_valid", "value": vol_dist.valid_count},
                    {"metric": "vol_missing", "value": vol_dist.missing_count}
                ],
                "why_now": why_now
            },
            impact_scope=["vol_labeling", "quarantine_gate", "human_review_rate"],
            risk={
                "type": "false_negative_increase",
                "level": "medium",
                "mitigation": "閾値上昇でquarantine減→drift_gateを同時強化しない"
            },
            confidence=base_confidence
        ))
    
    elif vol_dist.vol_high_rate < 0.10:
        # Too few quarantined - threshold may be too high
        raw_candidates = [
            vol_dist.percentiles.get(90),
            vol_dist.percentiles.get(80)
        ]
        raw_candidates = [c for c in raw_candidates if c and c < current_th]
        
        candidates, was_補完 = unique_candidates(raw_candidates, current_th, delta=-0.05)
        
        if not candidates:
            # Still generate if we can create candidates via補完
            candidates, was_補完 = unique_candidates([], current_th, delta=-0.05)
        
        why_now = f"VOL_HIGHが希薄({vol_dist.vol_high_rate*100:.1f}%)で検出力不足の可能性"
        if sample_note:
            why_now += f"; {sample_note}"
        if was_補完:
            why_now += "; candidates supplemented due to percentile collapse"
        
        # Check if risk_flag_rate is high - suggests volatility formula may not capture proxy danger
        if risk_flag_rate > 0.50:
            why_now += f"; risk_flag_rate={risk_flag_rate*100:.1f}%高→volatility式がproxy危険を捉えていない可能性"
            
            # Add hypothesis.py proposal for volatility formula review
            proposals.append(Proposal(
                id="P-VOL_FORMULA-001",
                touch_surface="hypothesis.py",
                parameter="volatility_weights_or_formula",
                current_value={
                    "sense_conflict": 0.5,
                    "source_conflict": 0.3,
                    "evidence_dispersion": 0.2
                },
                candidates=[
                    "increase_source_conflict_weight",
                    "add_risk_flag_correlation_term",
                    "normalize_by_evidence_count",
                    "log_component_correlation_for_analysis"
                ],
                rationale={
                    "evidence": [
                        {"metric": "vol_high_rate", "value": vol_dist.vol_high_rate},
                        {"metric": "risk_flag_rate", "value": risk_flag_rate},
                        {"metric": "discrepancy", "value": round_float(risk_flag_rate - vol_dist.vol_high_rate)}
                    ],
                    "why_now": f"risk_flag_rate({risk_flag_rate*100:.1f}%)とvol_high_rate({vol_dist.vol_high_rate*100:.1f}%)の乖離が大きい→volatility式がproxy危険を捉えていない疑い"
                },
                impact_scope=["volatility_calc", "sense_conflict", "source_conflict", "evidence_dispersion"],
                risk={
                    "type": "formula_change_cascade",
                    "level": "high",
                    "mitigation": "まず相関ログを出して分析してから重み調整"
                },
                confidence="low"
            ))
        
        proposals.append(Proposal(
            id="P-VOL_HIGH-002",
            touch_surface="config.py",
            parameter="VOL_HIGH_TH",
            current_value=current_th,
            candidates=candidates,
            rationale={
                "evidence": [
                    {"metric": "vol_high_rate", "value": vol_dist.vol_high_rate},
                    {"metric": "vol_p90", "value": vol_dist.percentiles.get(90)},
                    {"metric": "vol_valid", "value": vol_dist.valid_count},
                    {"metric": "vol_missing", "value": vol_dist.missing_count}
                ],
                "why_now": why_now
            },
            impact_scope=["vol_labeling", "quarantine_gate"],
            risk={
                "type": "false_positive_increase",
                "level": "low",
                "mitigation": "過剰quarantineはhuman_review負荷増"
            },
            confidence=base_confidence
        ))
    
    return proposals


def generate_vol_low_proposal(vol_dist: VolatilityDistribution,
                               current_th: float = 0.25) -> Optional[Proposal]:
    """Generate VOL_LOW_TH adjustment proposal."""
    if vol_dist.valid_count == 0:
        return None
    
    # Determine confidence based on sample size
    if not vol_dist.sample_size_adequate:
        base_confidence = "low"
        sample_note = f"n={vol_dist.valid_count} too small"
    else:
        base_confidence = "medium"
        sample_note = None
    
    # Check distribution balance
    if vol_dist.vol_low_rate > 0.70:
        # Too many candidates - threshold may be too high
        raw_candidates = [
            round_float(current_th * 0.8),
            round_float(current_th * 0.6)
        ]
        candidates, was_補完 = unique_candidates(raw_candidates, current_th, delta=-0.05)
        
        why_now = f"candidate過多({vol_dist.vol_low_rate*100:.1f}%)でdefer帯が圧縮"
        if sample_note:
            why_now += f"; {sample_note}"
        if was_補完:
            why_now += "; candidates supplemented"
        
        return Proposal(
            id="P-VOL_LOW-001",
            touch_surface="config.py",
            parameter="VOL_LOW_TH",
            current_value=current_th,
            candidates=candidates,
            rationale={
                "evidence": [
                    {"metric": "vol_low_rate", "value": vol_dist.vol_low_rate},
                    {"metric": "vol_mean", "value": vol_dist.mean},
                    {"metric": "vol_valid", "value": vol_dist.valid_count},
                    {"metric": "vol_missing", "value": vol_dist.missing_count}
                ],
                "why_now": why_now
            },
            impact_scope=["candidate_gate", "defer_distribution"],
            risk={
                "type": "defer_increase",
                "level": "low",
                "mitigation": "candidateからdeferへの移行は安全側"
            },
            confidence=base_confidence
        )
    
    return None


def generate_compete_th_proposal(compete_dist: CompeteDistribution,
                                  hr_stats: HumanReviewStats,
                                  current_th: float = 0.15) -> Optional[Proposal]:
    """Generate COMPETE_TH adjustment proposal."""
    if compete_dist.count == 0:
        return None
    
    # Determine confidence based on sample size
    sample_adequate = compete_dist.count >= 10
    base_confidence = "medium" if sample_adequate else "low"
    sample_note = None if sample_adequate else f"n={compete_dist.count} too small"
    
    # Check if too many are competing
    if compete_dist.multi_compete_rate > 0.50:
        # Many records have competing hypotheses - threshold may be too low
        raw_candidates = [
            round_float(current_th + 0.05),
            round_float(current_th + 0.10),
            round_float(current_th + 0.15)
        ]
        candidates, was_補完 = unique_candidates(raw_candidates, current_th, delta=0.05)
        
        why_now = f"競合過多({compete_dist.multi_compete_rate*100:.1f}%)でsense_conflict膨張"
        if sample_note:
            why_now += f"; {sample_note}"
        if was_補完:
            why_now += "; candidates supplemented"
        
        return Proposal(
            id="P-COMPETE_TH-001",
            touch_surface="config.py",
            parameter="COMPETE_TH",
            current_value=current_th,
            candidates=candidates,
            rationale={
                "evidence": [
                    {"metric": "multi_compete_rate", "value": compete_dist.multi_compete_rate},
                    {"metric": "compete_mean", "value": compete_dist.mean},
                    {"metric": "missed_danger_count", "value": hr_stats.missed_danger_count},
                    {"metric": "sample_count", "value": compete_dist.count}
                ],
                "why_now": why_now
            },
            impact_scope=["competing_count", "sense_conflict", "volatility_calc"],
            risk={
                "type": "under_detection",
                "level": "medium",
                "mitigation": "閾値上昇で見逃し増リスク→proxy監視強化"
            },
            confidence=base_confidence
        )
    
    elif compete_dist.multi_compete_rate < 0.10 and hr_stats.missed_danger_count > 0:
        # Few competing but missed dangers exist - threshold may be too high
        raw_candidates = [
            round_float(current_th - 0.03),
            round_float(current_th - 0.05)
        ]
        candidates, was_補完 = unique_candidates(raw_candidates, current_th, delta=-0.03)
        
        why_now = f"競合検出希薄({compete_dist.multi_compete_rate*100:.1f}%)×見逃し{hr_stats.missed_danger_count}件"
        if sample_note:
            why_now += f"; {sample_note}"
        if was_補完:
            why_now += "; candidates supplemented"
        
        return Proposal(
            id="P-COMPETE_TH-002",
            touch_surface="config.py",
            parameter="COMPETE_TH",
            current_value=current_th,
            candidates=candidates,
            rationale={
                "evidence": [
                    {"metric": "multi_compete_rate", "value": compete_dist.multi_compete_rate},
                    {"metric": "missed_danger_count", "value": hr_stats.missed_danger_count},
                    {"metric": "sample_count", "value": compete_dist.count}
                ],
                "why_now": why_now
            },
            impact_scope=["competing_count", "sense_conflict"],
            risk={
                "type": "over_detection",
                "level": "low",
                "mitigation": "競合増はvolatility上昇→安全側"
            },
            confidence=base_confidence
        )
    
    return None


def generate_human_review_proposal(hr_stats: HumanReviewStats,
                                    stability_dist: Dict[str, int]) -> Optional[Proposal]:
    """Generate human review gate adjustment proposal."""
    total = hr_stats.total_reviewed + hr_stats.total_not_reviewed
    if total == 0:
        return None
    
    # Determine confidence based on sample size
    sample_adequate = total >= 10
    base_confidence = "medium" if sample_adequate else "low"
    sample_note = None if sample_adequate else f"n={total} too small"
    
    # Check for excessive escalation
    if hr_stats.review_rate > 0.50:
        why_now = f"過剰エスカレーション({hr_stats.review_rate*100:.1f}%)で人間負荷過大"
        if sample_note:
            why_now += f"; {sample_note}"
        
        return Proposal(
            id="P-HR_GATE-001",
            touch_surface="7cprime",
            parameter="human_review_conditions",
            current_value="current_rules",
            candidates=[
                "require_stability_volatile_only",
                "require_multi_risk_flags",
                "require_vol_high_and_risk_flag"
            ],
            rationale={
                "evidence": [
                    {"metric": "review_rate", "value": hr_stats.review_rate},
                    {"metric": "stability_distribution", "value": stability_dist},
                    {"metric": "sample_count", "value": total}
                ],
                "why_now": why_now
            },
            impact_scope=["human_review_queue", "7cprime_gate"],
            risk={
                "type": "missed_danger",
                "level": "medium",
                "mitigation": "条件絞り込みは見逃しリスク→proxy監視必須"
            },
            confidence=base_confidence
        )
    
    # Check for missed dangers
    if hr_stats.missed_danger_count > 5:
        why_now = f"潜在見逃し{hr_stats.missed_danger_count}件検出"
        if sample_note:
            why_now += f"; {sample_note}"
        
        return Proposal(
            id="P-HR_GATE-002",
            touch_surface="7cprime",
            parameter="human_review_conditions",
            current_value="current_rules",
            candidates=[
                "add_proxy_trigger",
                "lower_stability_threshold",
                "include_risk_flag_any"
            ],
            rationale={
                "evidence": [
                    {"metric": "missed_danger_count", "value": hr_stats.missed_danger_count},
                    {"metric": "review_rate", "value": hr_stats.review_rate},
                    {"metric": "sample_missed", "value": hr_stats.missed_danger_details[:3]},
                    {"metric": "sample_count", "value": total}
                ],
                "why_now": why_now
            },
            impact_scope=["human_review_queue", "7cprime_gate"],
            risk={
                "type": "escalation_increase",
                "level": "low",
                "mitigation": "見逃し減少が優先"
            },
            confidence=base_confidence
        )
    
    return None


def generate_all_proposals(data: Dict[str, Any],
                           current_config: Dict[str, float]) -> List[Proposal]:
    """Generate all rule adjustment proposals."""
    proposals = []
    
    # Get records
    aggregates = data.get("unknown_queue_7bplus.jsonl", [])
    drift_records = data.get("audit_drift_7cprime.jsonl", [])
    
    # Analyze distributions
    vol_dist = analyze_volatility_distribution(
        aggregates, 
        current_config.get("VOL_HIGH_TH", 0.50),
        current_config.get("VOL_LOW_TH", 0.25)
    )
    
    compete_dist = analyze_compete_distribution(
        aggregates,
        current_config.get("COMPETE_TH", 0.15)
    )
    
    hr_stats = analyze_human_review(data)
    status_dist = analyze_status_distribution(aggregates)
    stability_dist = analyze_stability_distribution(drift_records)
    risk_analysis = analyze_risk_flags(drift_records)
    
    # Get risk_flag_rate for vol_high proposal
    risk_flag_rate = risk_analysis.get("flag_rate", 0.0)
    
    # Generate proposals - vol_high now returns list
    vol_high_props = generate_vol_high_proposal(
        vol_dist, 
        current_config.get("VOL_HIGH_TH", 0.50),
        risk_flag_rate
    )
    proposals.extend(vol_high_props)
    
    vol_low_prop = generate_vol_low_proposal(vol_dist, current_config.get("VOL_LOW_TH", 0.25))
    if vol_low_prop:
        proposals.append(vol_low_prop)
    
    compete_prop = generate_compete_th_proposal(
        compete_dist, hr_stats, current_config.get("COMPETE_TH", 0.15)
    )
    if compete_prop:
        proposals.append(compete_prop)
    
    hr_prop = generate_human_review_proposal(hr_stats, stability_dist)
    if hr_prop:
        proposals.append(hr_prop)
    
    return proposals


# =============================================================================
# Axis Candidate Extraction
# =============================================================================

def extract_axis_candidates(data: Dict[str, Any]) -> List[AxisCandidate]:
    """
    Extract axis candidates from log patterns.
    
    NOTE: This extracts STRUCTURE only, not meaning.
    Axes are identified by co-occurrence and stability patterns.
    
    IMPORTANT: members must never contain null. Missing keys are replaced
    with "UNKNOWN:<hash>" or excluded.
    """
    aggregates = data.get("unknown_queue_7bplus.jsonl", [])
    drift_records = data.get("audit_drift_7cprime.jsonl", [])
    
    if not aggregates:
        return []
    
    # Build drift index
    drift_by_key = {}
    for record in drift_records:
        key = get_field(record, "aggregate_key")
        if key:
            drift_by_key[key] = record
    
    # Group by status + volatility band
    axis_groups = defaultdict(list)
    
    for agg in aggregates:
        key = get_field(agg, "aggregate_key")
        
        # Skip records with null/missing aggregate_key
        if not key:
            # Generate fallback key from record hash
            record_hash = hashlib.sha256(
                json.dumps(agg, sort_keys=True, ensure_ascii=False).encode()
            ).hexdigest()[:12]
            key = f"UNKNOWN:{record_hash}"
        
        status = get_field(agg, "status") or "unknown"
        vol = get_field(agg, "volatility_avg")
        
        # Skip if volatility is missing (can't determine band)
        if vol is None:
            continue
        
        vol = float(vol)
        
        # Volatility band
        if vol < 0.25:
            vol_band = "low"
        elif vol < 0.50:
            vol_band = "medium"
        else:
            vol_band = "high"
        
        # Competing pattern
        competing = get_field(agg, "competing_routes") or []
        compete_sig = "_".join(sorted(competing)) if competing else "none"
        
        # Group key
        group_key = f"{status}|{vol_band}|{compete_sig}"
        axis_groups[group_key].append({
            "aggregate_key": key,  # Guaranteed non-null now
            "token_norm": get_field(agg, "token_norm") or f"token_{key[:8]}",
            "volatility": vol,
            "stability": get_field(drift_by_key.get(key, {}), "stability")
        })
    
    # Convert to axis candidates
    candidates = []
    
    for group_key, members in axis_groups.items():
        if len(members) < 2:
            continue  # Need at least 2 members for an axis
        
        parts = group_key.split("|")
        status, vol_band, compete_sig = parts[0], parts[1], parts[2]
        
        # Compute cohesion (how similar are volatilities)
        vols = [m["volatility"] for m in members if m["volatility"] is not None]
        if vols and len(vols) > 1:
            vol_std = statistics.stdev(vols)
            cohesion = max(0.0, 1.0 - vol_std)
        else:
            cohesion = 1.0
        
        # Compute stability score
        stabilities = [m["stability"] for m in members if m["stability"]]
        if stabilities:
            stable_count = sum(1 for s in stabilities if s == "stable")
            stability_score = stable_count / len(stabilities)
        else:
            stability_score = 0.5
        
        # Volatility association
        vol_assoc = statistics.mean(vols) if vols else 0.0
        
        # Create axis ID
        axis_id = hashlib.sha256(group_key.encode()).hexdigest()[:12]
        
        # Extract member keys - guaranteed non-null strings
        member_keys = [m["aggregate_key"] for m in members[:10]]
        # Final safety check - should never trigger but defensive
        member_keys = [k for k in member_keys if k is not None and isinstance(k, str)]
        
        if len(member_keys) < 2:
            continue  # Skip if not enough valid members after filtering
        
        candidates.append(AxisCandidate(
            axis_id=axis_id,
            members=member_keys,
            cohesion=round_float(cohesion),
            stability=round_float(stability_score),
            volatility_association=round_float(vol_assoc),
            operator_signature={},  # Not available in current logs
            note=f"status={status}|vol_band={vol_band}|compete={compete_sig}|n={len(members)}"
        ))
    
    # Sort by cohesion descending
    candidates.sort(key=lambda x: x.cohesion, reverse=True)
    
    return candidates[:20]  # Limit output


# =============================================================================
# Output Generation
# =============================================================================

def generate_output(data: Dict[str, Any],
                    input_stats: List[InputStats],
                    data_dir: str,
                    current_config: Dict[str, float]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Generate 7D output files.
    
    Returns:
        audit_rules_review: Main proposal output
        axis_candidates: Axis candidate output
    """
    now = datetime.now(timezone.utc)
    
    # Compute input fingerprint
    fingerprints = {}
    for stat in input_stats:
        if stat.exists:
            file_path = Path(data_dir) / stat.file_name
            fingerprints[stat.file_name] = compute_file_hash(str(file_path))
    
    # Missing inputs
    missing_inputs = [s.file_name for s in input_stats if not s.exists]
    
    # Schema warnings - start with input-level warnings
    all_warnings = []
    for stat in input_stats:
        for warn in stat.schema_warnings:
            all_warnings.append(f"{stat.file_name}:{warn}")
    
    # Get records
    aggregates = data.get("unknown_queue_7bplus.jsonl", [])
    drift_records = data.get("audit_drift_7cprime.jsonl", [])
    
    # Analyze
    vol_dist = analyze_volatility_distribution(aggregates)
    compete_dist = analyze_compete_distribution(aggregates)
    hr_stats = analyze_human_review(data)
    status_dist = analyze_status_distribution(aggregates)
    stability_dist = analyze_stability_distribution(drift_records)
    risk_analysis = analyze_risk_flags(drift_records)
    
    # Add volatility field coverage warnings
    if vol_dist.count > 0:
        if vol_dist.missing_count > 0:
            coverage_pct = round_float(vol_dist.valid_count / vol_dist.count * 100)
            all_warnings.append(
                f"volatility_coverage:vol_valid={vol_dist.valid_count},"
                f"vol_missing={vol_dist.missing_count},"
                f"coverage={coverage_pct}%"
            )
        if not vol_dist.sample_size_adequate:
            all_warnings.append(
                f"volatility_sample_size:n={vol_dist.valid_count}<10,percentiles_unreliable"
            )
    
    # Generate proposals
    proposals = generate_all_proposals(data, current_config)
    
    # Build audit_rules_review
    audit_rules_review = {
        "meta": {
            "phase": "7D",
            "spec_version": SPEC_VERSION,
            "engine_version": VERSION,
            "generated_at": now.isoformat(),
            "determinism": {
                "seed": 0,
                "sort_keys": True,
                "float_precision": 6
            },
            "input_fingerprints": fingerprints,
            "missing_inputs": missing_inputs,
            "schema_warnings": all_warnings
        },
        "current_config": current_config,
        "summary": {
            "signals": {
                "vol_distribution_health": "degraded" if vol_dist.vol_high_rate > 0.60 else "ok",
                "human_review_rate": hr_stats.review_rate,
                "vol_high_rate": vol_dist.vol_high_rate,
                "vol_low_rate": vol_dist.vol_low_rate,
                "multi_compete_rate": compete_dist.multi_compete_rate,
                "missed_danger_count": hr_stats.missed_danger_count,
                "notable_operator_shift": False,  # Not available in current logs
                "risk_flag_rate": risk_analysis.get("flag_rate", 0.0)
            },
            "distributions": {
                "volatility": {
                    "count": vol_dist.count,
                    "valid_count": vol_dist.valid_count,
                    "missing_count": vol_dist.missing_count,
                    "sample_size_adequate": vol_dist.sample_size_adequate,
                    "mean": vol_dist.mean,
                    "std": vol_dist.std,
                    "min": vol_dist.min_val,
                    "max": vol_dist.max_val,
                    "percentiles": vol_dist.percentiles
                },
                "competing": {
                    "count": compete_dist.count,
                    "mean": compete_dist.mean,
                    "by_count": compete_dist.by_count
                },
                "status": status_dist,
                "stability": stability_dist
            },
            "human_review": {
                "total_reviewed": hr_stats.total_reviewed,
                "total_not_reviewed": hr_stats.total_not_reviewed,
                "review_rate": hr_stats.review_rate,
                "by_stability": hr_stats.by_stability,
                "missed_danger_count": hr_stats.missed_danger_count
            },
            "risk_flags": risk_analysis
        },
        "proposals": [asdict(p) for p in proposals],
        "missed_danger_samples": hr_stats.missed_danger_details
    }
    
    # Generate axis candidates
    axis_candidates_list = extract_axis_candidates(data)
    
    axis_candidates = {
        "meta": {
            "phase": "7D",
            "spec_version": SPEC_VERSION,
            "generated_at": now.isoformat(),
            "note": "Structure only. No meaning interpretation."
        },
        "candidates": [asdict(a) for a in axis_candidates_list]
    }
    
    return audit_rules_review, axis_candidates


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="ESDE Phase 7D: Meta-Auditor (Rule Calibration by Log-Only)"
    )
    parser.add_argument("--data-dir", default="./data",
                        help="Directory containing 7B+/7C/7C' output logs")
    parser.add_argument("--output-dir", default="./data",
                        help="Directory for 7D output files")
    parser.add_argument("--vol-high-th", type=float, default=0.50,
                        help="Current VOL_HIGH_TH value")
    parser.add_argument("--vol-low-th", type=float, default=0.25,
                        help="Current VOL_LOW_TH value")
    parser.add_argument("--compete-th", type=float, default=0.15,
                        help="Current COMPETE_TH value")
    parser.add_argument("--dry-run", action="store_true",
                        help="Analyze without writing output files")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed analysis")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print(f"ESDE Engine - Phase 7D Meta-Auditor v{VERSION}")
    print("=" * 60)
    print()
    print("Philosophy: Rule Calibration by Log-Only")
    print("  - NO meaning determination")
    print("  - NO winner creation")
    print("  - NO config auto-modification")
    print("  - Proposals only (human review required)")
    print()
    
    # Current config
    current_config = {
        "VOL_HIGH_TH": args.vol_high_th,
        "VOL_LOW_TH": args.vol_low_th,
        "COMPETE_TH": args.compete_th
    }
    
    print(f"Current config: {current_config}")
    print()
    
    # Load inputs
    print(f"[7D] Loading inputs from: {args.data_dir}")
    data, input_stats = load_all_inputs(args.data_dir)
    
    print(f"\n[7D] Input file status:")
    required_ok = True
    for stat in input_stats:
        status = "✓" if stat.exists else "✗"
        req = "[REQ]" if stat.file_name in REQUIRED_INPUTS else "[OPT]"
        
        if stat.exists:
            print(f"  {status} {req} {stat.file_name}: {stat.record_count} records")
            if stat.schema_warnings:
                for warn in stat.schema_warnings:
                    print(f"      ⚠ {warn}")
        else:
            print(f"  {status} {req} {stat.file_name}: not found")
            if stat.file_name in REQUIRED_INPUTS:
                required_ok = False
    
    if not required_ok:
        print("\n[7D] ERROR: Required input file missing. Exiting.")
        return 1
    
    # Analyze
    print(f"\n[7D] Running analysis...")
    
    aggregates = data.get("unknown_queue_7bplus.jsonl", [])
    drift_records = data.get("audit_drift_7cprime.jsonl", [])
    
    vol_dist = analyze_volatility_distribution(aggregates)
    compete_dist = analyze_compete_distribution(aggregates)
    hr_stats = analyze_human_review(data)
    
    if args.verbose:
        print(f"\n--- Volatility Distribution ---")
        print(f"  Total records: {vol_dist.count}")
        print(f"  Valid (with vol): {vol_dist.valid_count}")
        print(f"  Missing (no vol): {vol_dist.missing_count}")
        print(f"  Sample adequate: {vol_dist.sample_size_adequate}")
        print(f"  Mean: {vol_dist.mean}")
        print(f"  Std: {vol_dist.std}")
        print(f"  Percentiles: {vol_dist.percentiles}")
        print(f"  VOL_HIGH rate: {vol_dist.vol_high_rate*100:.1f}%")
        print(f"  VOL_LOW rate: {vol_dist.vol_low_rate*100:.1f}%")
        
        print(f"\n--- Competing Distribution ---")
        print(f"  Count: {compete_dist.count}")
        print(f"  Mean: {compete_dist.mean}")
        print(f"  Multi-compete rate: {compete_dist.multi_compete_rate*100:.1f}%")
        
        print(f"\n--- Human Review ---")
        print(f"  Review rate: {hr_stats.review_rate*100:.1f}%")
        print(f"  Missed dangers: {hr_stats.missed_danger_count}")
    
    # Generate output
    print(f"\n[7D] Generating proposals...")
    audit_rules_review, axis_candidates = generate_output(
        data, input_stats, args.data_dir, current_config
    )
    
    proposals = audit_rules_review.get("proposals", [])
    print(f"  Generated {len(proposals)} proposal(s)")
    
    for p in proposals:
        print(f"    - {p['id']}: {p['parameter']} ({p['touch_surface']})")
    
    axis_count = len(axis_candidates.get("candidates", []))
    print(f"  Extracted {axis_count} axis candidate(s)")
    
    # Write output
    if not args.dry_run:
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        audit_file = output_path / "audit_rules_review_7d.json"
        axis_file = output_path / "axis_candidates.json"
        
        with open(audit_file, 'w', encoding='utf-8') as f:
            json.dump(audit_rules_review, f, ensure_ascii=False, indent=2, sort_keys=True)
        
        with open(axis_file, 'w', encoding='utf-8') as f:
            json.dump(axis_candidates, f, ensure_ascii=False, indent=2, sort_keys=True)
        
        print(f"\n[7D] Output files written:")
        print(f"  - {audit_file}")
        print(f"  - {axis_file}")
    else:
        print(f"\n[7D] Dry run - no files written")
    
    # Summary
    print()
    print("=" * 60)
    print("Phase 7D Meta-Audit Summary")
    print("=" * 60)
    
    signals = audit_rules_review["summary"]["signals"]
    print(f"\nHealth Signals:")
    print(f"  Vol distribution: {signals['vol_distribution_health']}")
    print(f"  VOL_HIGH rate: {signals['vol_high_rate']*100:.1f}%")
    print(f"  Multi-compete rate: {signals['multi_compete_rate']*100:.1f}%")
    print(f"  Human review rate: {signals['human_review_rate']*100:.1f}%")
    print(f"  Missed dangers: {signals['missed_danger_count']}")
    
    print(f"\nProposals Generated: {len(proposals)}")
    for p in proposals:
        print(f"  [{p['id']}] {p['parameter']}")
        print(f"      Current: {p['current_value']}")
        print(f"      Candidates: {p['candidates']}")
        print(f"      Risk: {p['risk']['level']} ({p['risk']['type']})")
    
    print()
    print("7D Guarantees:")
    print("  - winner=null ALWAYS maintained")
    print("  - NO meaning determined")
    print("  - NO config modified (proposals only)")
    print("  - Deterministic output (same input → same output)")
    print("  - Missing inputs recorded, not fatal")
    
    return 0


if __name__ == "__main__":
    exit(main())
