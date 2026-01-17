"""
ESDE Engine v5.3.3 - Phase 7B+: Multi-Hypothesis Evaluation

Evaluates A/B/C/D hypotheses WITHOUT determining a winner.
Volatility is the primary output, not confidence.

Key Principles:
- Multiple hypotheses can be simultaneously valid
- "No winner" is a normal, expected outcome
- Volatility (uncertainty) is first-class output
- Decision is NOT to decide when hypotheses compete
"""
import math
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict, field

from .online import EvidenceCollection, EvidenceItem

# Import centralized constants from config (SINGLE SOURCE OF TRUTH)
from ..config import (
    COMPETE_TH,
    ROUTE_A_MIN_SCORE,
    VOL_LOW_TH,
    VOL_HIGH_TH,
    DEFAULT_LOW_VOLATILITY_THRESHOLD,
    DEFAULT_HIGH_VOLATILITY_THRESHOLD,
    DEFAULT_CONFIDENCE_FLOOR
)


# =============================================================================
# Configuration (Weights only - thresholds come from config)
# =============================================================================

# Weights for global volatility calculation
VOLATILITY_WEIGHTS = {
    "sense_conflict": 0.5,       # Inter-hypothesis score proximity
    "source_conflict": 0.3,      # Tier conflict
    "evidence_dispersion": 0.2   # Claim type scatter
}

# Legacy exports for backward compatibility
DEFAULT_CONFIDENCE_THRESHOLD = 0.70
DEFAULT_VOLATILITY_THRESHOLD = 0.35


# =============================================================================
# Data Structures (7B+ style)
# =============================================================================

@dataclass
class HypothesisResult:
    """
    Result for a single hypothesis (A/B/C/D).
    
    NOT a "score to compete" but an independent evaluation.
    Both confidence and volatility are valid outputs.
    
    IMPORTANT: reason is MANDATORY even if score = 0.
    Human reviewers must be able to trace "why this hypothesis was scored this way".
    """
    hypothesis: str              # A, B, C, D
    score: float                 # 0-1 continuous (renamed from confidence for clarity)
    reason: str                  # MANDATORY: Why this score? (e.g., "EDIT_DIST_MATCH", "NO_CANDIDATES")
    evidence_count: int          # Number of evidence items supporting this
    volatility: float            # Internal uncertainty (0-1)
    signals: List[str]           # Evidence signals used
    notes: str = ""
    
    @property
    def confidence(self) -> float:
        """Alias for backward compatibility."""
        return self.score
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "hypothesis": self.hypothesis,
            "score": self.score,
            "reason": self.reason,
            "evidence_count": self.evidence_count,
            "volatility": self.volatility,
            "signals": self.signals,
            "notes": self.notes
        }


@dataclass
class EvaluationReport:
    """
    Full evaluation report for a token.
    
    DOES NOT have a "winner" field - winner is ALWAYS null.
    All hypotheses are presented equally.
    
    Key outputs:
    - hypotheses: All 4 hypotheses with scores and reasons
    - global_volatility: Primary uncertainty measure
    - competing_count: Number of hypotheses with score >= COMPETE_TH
    - competing_routes: List of route names that are competing
    - status: defer | quarantine | candidate (NOT "resolved")
    """
    token: str
    hypotheses: Dict[str, HypothesisResult]  # {A: result, B: result, ...}
    global_volatility: float                  # Primary output
    competing_count: int                      # Number of hypotheses >= COMPETE_TH
    competing_routes: List[str]               # Which routes are competing
    sense_conflict: float                     # How much hypotheses disagree
    source_conflict: float                    # Tier conflict
    evidence_dispersion: float                # Scatter of evidence
    status: str                               # defer | quarantine | candidate
    notes: str = ""
    consistency_warnings: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate consistency invariants."""
        self._check_consistency()
    
    def _check_consistency(self) -> None:
        """
        Check consistency between conflict and competing_count.
        
        Invariants:
        - If sense_conflict > 0, competing_count >= 2
        - If competing_count == 0, sense_conflict == 0
        """
        warnings = []
        
        # Check: conflict > 0 requires competing_count >= 2
        if self.sense_conflict > 0.1 and self.competing_count < 2:
            warnings.append(
                f"INCONSISTENCY: sense_conflict={self.sense_conflict:.2f} > 0 "
                f"but competing_count={self.competing_count} < 2"
            )
        
        # Check: competing_count == 0 requires conflict near 0
        if self.competing_count == 0 and self.sense_conflict > 0.15:
            warnings.append(
                f"INCONSISTENCY: competing_count=0 but sense_conflict={self.sense_conflict:.2f}"
            )
        
        # Check: competing_routes matches competing_count
        if len(self.competing_routes) != self.competing_count:
            warnings.append(
                f"INCONSISTENCY: competing_routes={self.competing_routes} "
                f"but competing_count={self.competing_count}"
            )
        
        self.consistency_warnings = warnings
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dict for JSON serialization.
        
        NOTE: Explicitly includes winner=null to enforce 7B+ philosophy.
        """
        return {
            "token": self.token,
            # Scores breakdown
            "scores": {k: v.score for k, v in self.hypotheses.items()},
            # Full hypothesis details
            "hypotheses": {k: v.to_dict() for k, v in self.hypotheses.items()},
            # Competing analysis
            "competing": self.competing_routes,
            "competing_count": self.competing_count,
            "compete_th": COMPETE_TH,
            # EXPLICIT: No winner in 7B+
            "winner": None,
            # Volatility breakdown
            "global_volatility": self.global_volatility,
            "conflict_components": {
                "sense_conflict": self.sense_conflict,
                "source_conflict": self.source_conflict,
                "evidence_dispersion": self.evidence_dispersion
            },
            # Status
            "status": self.status,
            "notes": self.notes,
            # Audit
            "consistency_warnings": self.consistency_warnings
        }
    
    def format_scores(self) -> str:
        """Format scores as [A:x B:y C:z D:w] for logging."""
        parts = []
        for route in ["A", "B", "C", "D"]:
            if route in self.hypotheses:
                score = self.hypotheses[route].score
                parts.append(f"{route}:{score:.2f}")
        return "[" + " ".join(parts) + "]"
    
    def format_log_line(self) -> str:
        """
        Format a complete log line with all required audit fields.
        
        Format: scores=[A:x B:y C:z D:w] competing=[A,B] compete_th=0.15 winner=null vol=X.XX
        """
        scores_str = self.format_scores()
        competing_str = ",".join(self.competing_routes) if self.competing_routes else "none"
        return (
            f"scores={scores_str} "
            f"competing=[{competing_str}] "
            f"compete_th={COMPETE_TH} "
            f"winner=null "
            f"vol={self.global_volatility:.2f}"
        )
    
    def get_strongest_hypothesis(self) -> Optional[str]:
        """
        Returns the hypothesis with highest score, IF it's clearly dominant.
        Returns None if no clear winner (which is expected and normal).
        """
        if not self.hypotheses:
            return None
        
        sorted_hyps = sorted(
            self.hypotheses.items(),
            key=lambda x: x[1].score,
            reverse=True
        )
        
        if len(sorted_hyps) < 2:
            return sorted_hyps[0][0] if sorted_hyps else None
        
        top = sorted_hyps[0]
        second = sorted_hyps[1]
        
        # Only return if margin is significant AND global volatility is low
        margin = top[1].score - second[1].score
        if margin > 0.3 and self.global_volatility < DEFAULT_LOW_VOLATILITY_THRESHOLD:
            return top[0]
        
        return None  # No clear winner - this is normal


# Legacy dataclass for backward compatibility
@dataclass
class ScoringDecision:
    """Legacy format for backward compatibility."""
    token: str
    scores: Dict[str, float]
    winner: str
    confidence: float
    volatility: float
    signals: Dict[str, Any]
    reasoning: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# Hypothesis Evaluation (7B+ style - independent evaluation, not scoring)
# =============================================================================

def evaluate_hypothesis_a(token: str,
                          evidence: EvidenceCollection,
                          typo_candidates: List[Dict],
                          queue_metrics: Dict[str, float]) -> HypothesisResult:
    """
    Evaluate Route A (Typo/Alias) hypothesis.
    
    IMPORTANT RULES:
    1. If edit_dist <= 2 and dictionary candidate exists, score >= ROUTE_A_MIN_SCORE
    2. Even WITHOUT explicit candidates, if token is short and plausible typo,
       give minimum score (A should not die just because we lack data)
    3. A is NOT zero just because title/entity signals exist - they can coexist
    
    reason is MANDATORY.
    """
    signals = []
    score = 0.0
    volatility = 0.0
    notes_parts = []
    reason = "NO_CANDIDATES"  # Default reason
    
    # Check if token could plausibly be a typo (even without explicit candidates)
    # Short words (2-5 chars) that aren't all caps are plausible typos
    is_plausible_typo = (
        2 <= len(token) <= 5 and 
        not token.isupper() and 
        not token.isdigit()
    )
    
    if not typo_candidates:
        # No explicit candidates, but check if plausibly a typo
        if is_plausible_typo:
            # Give minimum score - A should not die
            score = ROUTE_A_MIN_SCORE
            reason = f"PLAUSIBLE_TYPO:len={len(token)}"
            signals.append(f"plausible_typo_by_length:{len(token)}")
            volatility = 0.3  # Higher volatility since we're guessing
            notes_parts.append("no_explicit_candidates_but_plausible")
            
            return HypothesisResult(
                hypothesis="A",
                score=round(score, 3),
                reason=reason,
                evidence_count=0,
                volatility=round(volatility, 3),
                signals=signals,
                notes="; ".join(notes_parts)
            )
        else:
            # Long word or special format - truly no typo hypothesis
            return HypothesisResult(
                hypothesis="A",
                score=0.0,
                reason=f"NO_CANDIDATES:len={len(token)}",
                evidence_count=0,
                volatility=0.0,
                signals=[],
                notes="No typo candidates and not a plausible typo"
            )
    
    # Typo candidate exists
    best_typo = typo_candidates[0]
    edit_dist = best_typo.get("dist", best_typo.get("distance", 99))
    base_conf = best_typo.get("confidence", 0.0)
    candidate = best_typo.get("candidate", "")
    
    # CRITICAL: If edit_dist <= 2 and candidate exists, A >= ROUTE_A_MIN_SCORE
    if edit_dist <= 2 and candidate:
        score = max(ROUTE_A_MIN_SCORE, base_conf * 0.5)
        reason = f"EDIT_DIST_MATCH:dist={edit_dist}:candidate={candidate}"
        signals.append(f"typo_candidate:{candidate}:{base_conf:.2f}")
    else:
        score = max(ROUTE_A_MIN_SCORE * 0.5, base_conf * 0.3)  # Still some minimum
        reason = f"WEAK_MATCH:dist={edit_dist}"
        signals.append(f"weak_typo:{candidate}:{base_conf:.2f}")
    
    # Evidence support
    typo_mentions = 0
    conflicting_signals = 0
    
    for item in evidence.items:
        if item.signals.get("mentions_typo"):
            typo_mentions += 1
            tier_boost = {1: 0.15, 2: 0.08, 3: 0.03}.get(item.source_tier, 0.03)
            score += tier_boost
            signals.append(f"typo_evidence_tier{item.source_tier}")
        
        # Conflicting signals (title, entity) - increase volatility, NOT reduce score
        if item.signals.get("is_work_title") or item.signals.get("is_person"):
            conflicting_signals += 1
    
    # Volatility: increased if conflicting signals exist
    # NOTE: Conflicting signals mean A coexists with B, not that A is wrong
    if conflicting_signals > 0:
        volatility = min(0.8, conflicting_signals * 0.25)
        notes_parts.append(f"coexists_with_B:{conflicting_signals}")
    
    # Multiple typo candidates with similar confidence = volatility
    if len(typo_candidates) > 1:
        conf_spread = max(c.get("confidence", 0) for c in typo_candidates) - \
                      min(c.get("confidence", 0) for c in typo_candidates)
        if conf_spread < 0.15:  # Similar confidences
            volatility = max(volatility, 0.3)
            notes_parts.append("multiple_similar_candidates")
    
    # Clamp and round
    score = min(1.0, max(0.0, score))
    volatility = min(1.0, max(0.0, volatility))
    
    return HypothesisResult(
        hypothesis="A",
        score=round(score, 3),
        reason=reason,
        evidence_count=typo_mentions,
        volatility=round(volatility, 3),
        signals=signals,
        notes="; ".join(notes_parts) if notes_parts else ""
    )


def evaluate_hypothesis_b(token: str,
                          evidence: EvidenceCollection,
                          queue_metrics: Dict[str, float]) -> HypothesisResult:
    """
    Evaluate Route B (Proper Noun / Title / Entity) hypothesis.
    
    reason is MANDATORY.
    """
    signals = []
    score = 0.0
    volatility = 0.0
    notes_parts = []
    reason = "NO_ENTITY_SIGNALS"  # Default
    
    # Count entity signals
    entity_signals = {
        "person": 0,
        "character": 0,
        "place": 0,
        "work_title": 0
    }
    
    tier1_entities = []
    tier2_entities = []
    
    for item in evidence.items:
        if item.signals.get("is_person"):
            entity_signals["person"] += 1
            signals.append(f"person_tier{item.source_tier}")
        if item.signals.get("is_character"):
            entity_signals["character"] += 1
            signals.append(f"character_tier{item.source_tier}")
        if item.signals.get("is_place"):
            entity_signals["place"] += 1
            signals.append(f"place_tier{item.source_tier}")
        if item.signals.get("is_work_title"):
            entity_signals["work_title"] += 1
            signals.append(f"title_tier{item.source_tier}")
        
        # Track by tier
        for claim in item.claims:
            if claim.get("type") == "entity_type":
                if item.source_tier == 1:
                    tier1_entities.append(claim.get("text", ""))
                else:
                    tier2_entities.append(claim.get("text", ""))
    
    total_signals = sum(entity_signals.values())
    
    if total_signals == 0:
        return HypothesisResult(
            hypothesis="B",
            score=0.0,
            reason="NO_ENTITY_SIGNALS",
            evidence_count=0,
            volatility=0.2,  # Some volatility for unknown
            signals=[],
            notes="No entity signals found"
        )
    
    # Determine dominant entity type for reason
    dominant_type = max(entity_signals.keys(), key=lambda k: entity_signals[k])
    reason = f"ENTITY_DETECTED:{dominant_type}:{entity_signals[dominant_type]}"
    
    # Score from signal count
    score = min(0.8, total_signals * 0.15)
    
    # Tier-1 boost
    tier1_count = evidence.tier1_count
    if tier1_count >= 2:
        score += 0.15
        signals.append(f"tier1_count:{tier1_count}")
    
    # Volatility: multiple entity types = high volatility
    active_types = sum(1 for v in entity_signals.values() if v > 0)
    if active_types >= 3:
        volatility = 0.7
        notes_parts.append(f"multi_entity_types:{active_types}")
    elif active_types == 2:
        volatility = 0.4
        notes_parts.append(f"dual_entity_types:{active_types}")
    
    # Tier conflict
    if tier1_entities and tier2_entities:
        tier1_set = set(tier1_entities)
        tier2_set = set(tier2_entities)
        if tier1_set != tier2_set:
            volatility = max(volatility, 0.5)
            notes_parts.append("tier_conflict")
    
    score = min(1.0, max(0.0, score))
    volatility = min(1.0, max(0.0, volatility))
    
    return HypothesisResult(
        hypothesis="B",
        score=round(score, 3),
        reason=reason,
        evidence_count=total_signals,
        volatility=round(volatility, 3),
        signals=signals,
        notes="; ".join(notes_parts) if notes_parts else ""
    )


def evaluate_hypothesis_c(token: str,
                          evidence: EvidenceCollection,
                          queue_metrics: Dict[str, float]) -> HypothesisResult:
    """
    Evaluate Route C (Molecule / Novel Concept) hypothesis.
    
    reason is MANDATORY.
    """
    signals = []
    score = 0.0
    volatility = 0.0
    notes_parts = []
    reason_parts = []
    
    tier1_count = evidence.tier1_count
    tier2_count = evidence.tier2_count
    total = len(evidence.items)
    
    # Weak external presence suggests novel/internal term
    if tier1_count == 0:
        score += 0.4
        signals.append("no_tier1")
        reason_parts.append("NO_TIER1")
    elif tier1_count == 1 and total >= 3:
        score += 0.2
        signals.append("weak_tier1")
        reason_parts.append("WEAK_TIER1")
    
    # Slang/informal mentions
    slang_count = sum(1 for item in evidence.items if item.signals.get("mentions_slang"))
    if slang_count > 0:
        score += 0.15
        volatility += 0.3  # Slang = volatile
        signals.append(f"slang:{slang_count}")
        notes_parts.append("slang_detected")
        reason_parts.append(f"SLANG:{slang_count}")
    
    # Acronym
    acronym_count = sum(1 for item in evidence.items if item.signals.get("mentions_acronym"))
    if acronym_count > 0:
        score += 0.1
        signals.append(f"acronym:{acronym_count}")
        reason_parts.append(f"ACRONYM:{acronym_count}")
    
    # No clear definition = more likely novel
    definition_count = sum(1 for item in evidence.items if item.signals.get("is_definition"))
    if definition_count == 0 and total > 0:
        score += 0.1
        signals.append("no_definition")
        reason_parts.append("NO_DEFINITION")
    elif definition_count > 0:
        # Has definition but weak tier-1 = uncertain
        volatility += 0.2
    
    # Inherent volatility for C (novel concepts are uncertain by nature)
    volatility = max(volatility, 0.35)
    
    score = min(1.0, max(0.0, score))
    volatility = min(1.0, max(0.0, volatility))
    
    # Build reason
    if not reason_parts:
        reason = "UNKNOWN_NOVEL"
    else:
        reason = "+".join(reason_parts)
    
    return HypothesisResult(
        hypothesis="C",
        score=round(score, 3),
        reason=reason,
        evidence_count=slang_count + acronym_count,
        volatility=round(volatility, 3),
        signals=signals,
        notes="; ".join(notes_parts) if notes_parts else "novel_concept_candidate"
    )


def evaluate_hypothesis_d(token: str,
                          evidence: EvidenceCollection,
                          queue_metrics: Dict[str, float]) -> HypothesisResult:
    """
    Evaluate Route D (Noise) hypothesis.
    
    reason is MANDATORY.
    """
    signals = []
    score = 0.0
    volatility = 0.0
    reason_parts = []
    
    # Short tokens
    if len(token) <= 2:
        score += 0.5
        signals.append(f"very_short:{len(token)}")
        reason_parts.append(f"VERY_SHORT:{len(token)}")
    elif len(token) == 3:
        score += 0.2
        signals.append("short:3")
        reason_parts.append("SHORT:3")
    
    # Numeric
    if token.isdigit():
        score += 0.6
        signals.append("numeric")
        reason_parts.append("NUMERIC")
    
    # No evidence at all
    if len(evidence.items) == 0:
        score += 0.3
        signals.append("no_evidence")
        reason_parts.append("NO_EVIDENCE")
    
    # Only Tier-3
    if evidence.tier1_count == 0 and evidence.tier2_count == 0 and evidence.tier3_count > 0:
        score += 0.15
        signals.append("only_tier3")
        reason_parts.append("ONLY_TIER3")
    
    # D has low volatility (noise is noise)
    volatility = 0.1
    
    score = min(1.0, max(0.0, score))
    
    # Build reason
    if not reason_parts:
        reason = "NO_NOISE_SIGNALS"
    else:
        reason = "+".join(reason_parts)
    
    return HypothesisResult(
        hypothesis="D",
        score=round(score, 3),
        reason=reason,
        evidence_count=0,
        volatility=round(volatility, 3),
        signals=signals,
        notes=""
    )


# =============================================================================
# Global Volatility Calculation (7B+ Spec)
# =============================================================================

def compute_competing_info(hypotheses: Dict[str, HypothesisResult]) -> Tuple[int, List[str]]:
    """
    Get competing hypotheses info.
    
    Returns:
        (count, routes) - count of competing hypotheses and their route names
    
    A hypothesis is "competing" if score >= COMPETE_TH.
    competing_count >= 2 means there's conflict between hypotheses.
    """
    competing_routes = [
        route for route, h in hypotheses.items() 
        if h.score >= COMPETE_TH
    ]
    return len(competing_routes), sorted(competing_routes)


def compute_competing_count(hypotheses: Dict[str, HypothesisResult]) -> int:
    """Legacy wrapper - returns just the count."""
    count, _ = compute_competing_info(hypotheses)
    return count


def compute_sense_conflict(hypotheses: Dict[str, HypothesisResult]) -> float:
    """
    Compute sense conflict (仮説間スコア近接).
    
    High conflict = multiple hypotheses have similar scores above COMPETE_TH.
    
    Formula spec:
    - weight = 0.5 in global volatility
    """
    scores = [h.score for h in hypotheses.values()]
    
    if len(scores) < 2:
        return 0.0
    
    # Sort descending
    sorted_scores = sorted(scores, reverse=True)
    
    # Count competing hypotheses
    competing = sum(1 for s in sorted_scores if s >= COMPETE_TH)
    
    if competing < 2:
        return 0.1  # No significant conflict
    
    # Top two competing scores
    top1, top2 = sorted_scores[0], sorted_scores[1]
    
    if top2 < COMPETE_TH:
        return 0.1  # Only one hypothesis is competing
    
    margin = top1 - top2
    
    # Conflict based on margin
    if margin < 0.10:
        conflict = 0.9  # Very high conflict: almost tied
    elif margin < 0.15:
        conflict = 0.7  # High conflict
    elif margin < 0.25:
        conflict = 0.5  # Medium conflict
    elif margin < 0.35:
        conflict = 0.3  # Low conflict
    else:
        conflict = 0.1  # Clear leader
    
    # Bonus for 3+ competing hypotheses
    if competing >= 3:
        conflict = min(1.0, conflict + 0.2)
    
    return conflict


def compute_source_conflict(evidence: EvidenceCollection) -> float:
    """
    Compute source conflict (情報源Tier間矛盾).
    
    High conflict = Tier-1 and Tier-2 sources disagree.
    
    Formula spec:
    - weight = 0.3 in global volatility
    """
    tier1_signals = set()
    tier2_signals = set()
    
    for item in evidence.items:
        active_signals = {k for k, v in item.signals.items() if v}
        if item.source_tier == 1:
            tier1_signals.update(active_signals)
        elif item.source_tier == 2:
            tier2_signals.update(active_signals)
    
    if not tier1_signals or not tier2_signals:
        return 0.1  # Can't compare
    
    # Check for contradictory signals
    tier1_only = tier1_signals - tier2_signals
    tier2_only = tier2_signals - tier1_signals
    
    # Key contradictions
    contradiction_pairs = [
        ("is_definition", "mentions_slang"),
        ("is_person", "is_character"),
        ("is_work_title", "is_definition"),
    ]
    
    conflict = 0.0
    
    for sig1, sig2 in contradiction_pairs:
        if (sig1 in tier1_signals and sig2 in tier2_signals) or \
           (sig2 in tier1_signals and sig1 in tier2_signals):
            conflict += 0.3
    
    # General disagreement
    disagreement_ratio = len(tier1_only | tier2_only) / max(1, len(tier1_signals | tier2_signals))
    conflict += disagreement_ratio * 0.4
    
    return min(1.0, conflict)


def compute_evidence_dispersion(evidence: EvidenceCollection) -> float:
    """
    Compute evidence dispersion (主張タイプの散り).
    
    High dispersion = claims are scattered across many types.
    
    Formula spec:
    - weight = 0.2 in global volatility
    """
    if not evidence.items:
        return 0.5  # Unknown = some dispersion
    
    # Count different signal types
    signal_types = set()
    claim_types = set()
    
    for item in evidence.items:
        for signal, value in item.signals.items():
            if value:
                signal_types.add(signal)
        for claim in item.claims:
            claim_types.add(claim.get("type", "unknown"))
    
    total_types = len(signal_types) + len(claim_types)
    
    # More types = more dispersion
    if total_types >= 6:
        return 0.7
    elif total_types >= 4:
        return 0.5
    elif total_types >= 2:
        return 0.3
    
    return 0.1


def compute_global_volatility(hypotheses: Dict[str, HypothesisResult],
                               evidence: EvidenceCollection) -> Tuple[float, Dict[str, float]]:
    """
    Compute global volatility from all components.
    
    Formula (from spec):
        volatility = 0.5 * sense_conflict
                   + 0.3 * source_conflict
                   + 0.2 * dispersion
    
    Returns: (global_volatility, component_breakdown)
    """
    sense = compute_sense_conflict(hypotheses)
    source = compute_source_conflict(evidence)
    dispersion = compute_evidence_dispersion(evidence)
    
    global_vol = (
        VOLATILITY_WEIGHTS["sense_conflict"] * sense +
        VOLATILITY_WEIGHTS["source_conflict"] * source +
        VOLATILITY_WEIGHTS["evidence_dispersion"] * dispersion
    )
    
    return (
        round(min(1.0, max(0.0, global_vol)), 3),
        {
            "sense_conflict": round(sense, 3),
            "source_conflict": round(source, 3),
            "evidence_dispersion": round(dispersion, 3)
        }
    )


# =============================================================================
# Status Determination (NOT winner selection)
# =============================================================================

def determine_status(global_volatility: float,
                     competing_count: int) -> str:
    """
    Determine status based on volatility and competing count.
    
    This is NOT about finding a winner.
    It's about categorizing the uncertainty level.
    
    Rules (from spec):
        - volatility >= 0.50 → quarantine
        - 0.25 <= volatility < 0.50 → defer
        - volatility < 0.25 → candidate
    
    Additional:
        - competing_count >= 2 with low volatility → defer (conflict exists)
    
    Returns:
        - "candidate": Low volatility, MIGHT be resolvable (but not auto-resolved)
        - "defer": Medium volatility, needs more observation
        - "quarantine": High volatility, competing hypotheses, DO NOT RESOLVE
    """
    # High volatility = quarantine
    if global_volatility >= DEFAULT_HIGH_VOLATILITY_THRESHOLD:
        return "quarantine"
    
    # Medium volatility = defer
    if global_volatility >= DEFAULT_LOW_VOLATILITY_THRESHOLD:
        return "defer"
    
    # Low volatility but competing hypotheses = defer
    if competing_count >= 2:
        return "defer"
    
    # Low volatility and single hypothesis = candidate
    return "candidate"


# =============================================================================
# Main Evaluation Function (7B+ style)
# =============================================================================

def evaluate_all_hypotheses(token: str,
                            evidence: EvidenceCollection,
                            typo_candidates: List[Dict] = None,
                            queue_metrics: Dict[str, float] = None) -> EvaluationReport:
    """
    Evaluate all hypotheses for a token.
    
    DOES NOT determine a winner.
    Returns independent evaluations for all hypotheses.
    
    Args:
        token: The token being evaluated
        evidence: Evidence collection from online search
        typo_candidates: List of typo candidates (optional)
        queue_metrics: Metrics from queue record (optional)
    
    Returns:
        EvaluationReport with all hypotheses and global volatility
    """
    typo_candidates = typo_candidates or []
    queue_metrics = queue_metrics or {}
    
    # Evaluate each hypothesis independently
    hyp_a = evaluate_hypothesis_a(token, evidence, typo_candidates, queue_metrics)
    hyp_b = evaluate_hypothesis_b(token, evidence, queue_metrics)
    hyp_c = evaluate_hypothesis_c(token, evidence, queue_metrics)
    hyp_d = evaluate_hypothesis_d(token, evidence, queue_metrics)
    
    hypotheses = {
        "A": hyp_a,
        "B": hyp_b,
        "C": hyp_c,
        "D": hyp_d
    }
    
    # Compute competing count and identify competing routes
    competing_count = compute_competing_count(hypotheses)
    competing_routes = [r for r, h in hypotheses.items() if h.score >= COMPETE_TH]
    
    # Compute global volatility
    global_vol, components = compute_global_volatility(hypotheses, evidence)
    sense_conflict = components["sense_conflict"]
    
    # ==========================================================================
    # INTEGRITY ASSERTIONS (v5.3.3)
    # ==========================================================================
    warnings = []
    
    # Rule 1: If sense_conflict > 0.1, competing_count must be >= 2
    if sense_conflict > 0.1 and competing_count < 2:
        warnings.append(f"INTEGRITY_WARN:sense_conflict={sense_conflict:.2f}_but_competing={competing_count}")
        # Force sense_conflict to 0.1 if competing < 2 (correct the inconsistency)
        components["sense_conflict"] = 0.1
        sense_conflict = 0.1
        # Recalculate global volatility
        global_vol = (
            VOLATILITY_WEIGHTS["sense_conflict"] * sense_conflict +
            VOLATILITY_WEIGHTS["source_conflict"] * components["source_conflict"] +
            VOLATILITY_WEIGHTS["evidence_dispersion"] * components["evidence_dispersion"]
        )
        global_vol = round(min(1.0, max(0.0, global_vol)), 3)
    
    # Rule 2: If competing_count == 0, sense_conflict should be minimal
    if competing_count == 0 and sense_conflict > 0.1:
        warnings.append(f"INTEGRITY_WARN:competing=0_but_sense_conflict={sense_conflict:.2f}")
    
    # Determine status (NOT winner)
    status = determine_status(global_vol, competing_count)
    
    # Build notes with full audit trail
    scores_str = " ".join([f"{r}:{h.score:.2f}" for r, h in hypotheses.items()])
    notes_parts = [
        f"scores=[{scores_str}]",
        f"competing={competing_count}",
        f"competing_routes=[{','.join(competing_routes)}]" if competing_routes else "competing_routes=[]",
        f"compete_th={COMPETE_TH}",
        "winner=null"  # Explicit: no winner
    ]
    
    # Add status explanation
    if status == "quarantine":
        notes_parts.append("high_volatility:competing_hypotheses")
    elif status == "defer":
        notes_parts.append("medium_volatility:needs_observation")
    else:
        notes_parts.append("low_volatility:candidate_for_review")
    
    # Add warnings if any
    if warnings:
        notes_parts.extend(warnings)
    
    return EvaluationReport(
        token=token,
        hypotheses=hypotheses,
        global_volatility=global_vol,
        competing_count=competing_count,
        competing_routes=competing_routes,
        sense_conflict=sense_conflict,
        source_conflict=components["source_conflict"],
        evidence_dispersion=components["evidence_dispersion"],
        status=status,
        notes="; ".join(notes_parts)
    )


# =============================================================================
# Legacy Functions (for backward compatibility with 7B-online)
# =============================================================================

def score_all_hypotheses(token: str,
                         evidence: EvidenceCollection,
                         typo_candidates: List[Dict] = None,
                         queue_metrics: Dict[str, float] = None) -> ScoringDecision:
    """
    Legacy function for backward compatibility.
    
    Internally uses evaluate_all_hypotheses but converts to old format.
    """
    report = evaluate_all_hypotheses(token, evidence, typo_candidates, queue_metrics)
    
    # Convert to legacy format (using .score which aliases .confidence)
    scores = {k: v.score for k, v in report.hypotheses.items()}
    
    # Determine legacy "winner" (highest score, even if not meaningful)
    winner = max(scores.keys(), key=lambda k: scores[k])
    
    return ScoringDecision(
        token=token,
        scores=scores,
        winner=winner,
        confidence=scores[winner],
        volatility=report.global_volatility,
        signals={
            "sense_conflict": report.sense_conflict,
            "source_conflict": report.source_conflict,
            "evidence_dispersion": report.evidence_dispersion,
            "competing_count": report.competing_count,
            "status": report.status
        },
        reasoning=report.notes
    )


def determine_action(decision: ScoringDecision,
                     conf_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
                     vol_threshold: float = DEFAULT_VOLATILITY_THRESHOLD,
                     quarantine_on_vol: bool = True) -> str:
    """
    Legacy function for backward compatibility.
    
    Maps status to action.
    """
    status = decision.signals.get("status", "defer")
    
    if status == "quarantine":
        return "quarantine"
    elif status == "candidate":
        # Even candidates need review in 7B+
        return "defer"  # Changed from "patch" to "defer"
    else:
        return "defer"


# =============================================================================
# Exports for __init__.py
# =============================================================================

# Legacy aliases for backward compatibility
HypothesisScore = HypothesisResult
compute_inter_hypothesis_conflict = compute_sense_conflict
compute_intra_hypothesis_volatility = lambda h: 0.0  # Deprecated
