"""
ESDE Engine v5.3.2 - Unknown Token Router (Multi-Hypothesis)
"""
from typing import Dict, List, Any, Optional

from nltk.corpus import wordnet as wn

from .config import (
    UNKNOWN_MARGIN_TH, UNKNOWN_ENTROPY_TH,
    CONTEXT_TITLE_LIKE_BOOST, CONTEXT_CAPITALIZED_BOOST,
    CONTEXT_QUOTE_BOOST, CONTEXT_TYPO_PENALTY_TITLE,
    TITLE_LIKE_PHRASES, STOPWORDS, MIN_TOKEN_LENGTH
)
from .utils import (
    find_all_typo_candidates, is_proper_noun_candidate, compute_entropy
)


class UnknownTokenRouter:
    """
    Multi-hypothesis routing with variance gate.
    
    Evaluates ALL routes (A/B/C/D) in parallel and uses variance detection
    to decide whether to apply a route or abstain (queue for later).
    
    Key principle: "Not deciding" (abstain) is the preferred outcome when
    hypotheses compete. This prevents typo over-correction and preserves
    potential titles/quotes/proper nouns for human review.
    """
    
    def __init__(self,
                 margin_threshold: float = UNKNOWN_MARGIN_TH,
                 entropy_threshold: float = UNKNOWN_ENTROPY_TH):
        self.margin_threshold = margin_threshold
        self.entropy_threshold = entropy_threshold
    
    def _detect_context_features(self,
                                  token: str,
                                  original_text: str,
                                  tokens: List[str],
                                  token_index: int) -> Dict[str, Any]:
        """Detect context features that might indicate title/quote/proper noun."""
        features = {
            "capitalized": False,
            "has_quotes": False,
            "title_like_ngram": False,
            "title_phrase_matched": None
        }
        
        # Check if original text has capitalization pattern (Title Case)
        words = original_text.split()
        if words:
            cap_count = sum(1 for w in words if w and w[0].isupper())
            features["capitalized"] = (cap_count / len(words)) > 0.5
        
        # Check for quotes
        features["has_quotes"] = ('"' in original_text or "'" in original_text or
                                   '「' in original_text or '」' in original_text)
        
        # Check for title-like n-grams (window around token)
        start = max(0, token_index - 2)
        end = min(len(tokens), token_index + 3)
        window = tokens[start:end]
        
        for n in range(2, min(6, len(window) + 1)):
            for i in range(len(window) - n + 1):
                ngram = ' '.join(window[i:i+n])
                if ngram in TITLE_LIKE_PHRASES:
                    features["title_like_ngram"] = True
                    features["title_phrase_matched"] = ngram
                    break
            if features["title_like_ngram"]:
                break
        
        return features
    
    def _score_route_a(self,
                        token: str,
                        typo_candidates: List[Dict],
                        context: Dict[str, Any]) -> Dict[str, Any]:
        """Score Route A (typo correction)."""
        if not typo_candidates:
            return {
                "route": "A",
                "label": "typo",
                "candidate": None,
                "score": 0.0,
                "evidence": {"no_candidates": True}
            }
        
        best = typo_candidates[0]
        base_score = best.get("confidence", 0.0)
        
        # Apply penalty if title-like context
        if context.get("title_like_ngram"):
            base_score -= CONTEXT_TYPO_PENALTY_TITLE
        
        # Clamp to [0, 1]
        score = max(0.0, min(1.0, base_score))
        
        return {
            "route": "A",
            "label": "typo",
            "candidate": best.get("candidate"),
            "score": round(score, 3),
            "evidence": {
                "edit_dist": best.get("dist"),
                "original_confidence": best.get("confidence"),
                "wn_exists": True,
                "title_penalty_applied": context.get("title_like_ngram", False)
            }
        }
    
    def _score_route_b(self,
                        token: str,
                        synset_ids: List[str],
                        has_edges: bool,
                        context: Dict[str, Any]) -> Dict[str, Any]:
        """Score Route B (proper noun / title / external knowledge needed)."""
        base_score = 0.3  # Base probability
        evidence = {
            "shape": "short_token" if len(token) <= 5 else "normal_token",
            "synset_count": len(synset_ids),
            "has_edges": has_edges
        }
        
        # Check for proper noun pattern in synsets
        if synset_ids and is_proper_noun_candidate(synset_ids):
            base_score += 0.25
            evidence["proper_noun_pattern"] = True
        
        # Boost for title-like context
        if context.get("title_like_ngram"):
            base_score += CONTEXT_TITLE_LIKE_BOOST
            evidence["title_like_ngram"] = True
            evidence["title_phrase"] = context.get("title_phrase_matched")
        
        # Boost for capitalization
        if context.get("capitalized"):
            base_score += CONTEXT_CAPITALIZED_BOOST
            evidence["capitalized"] = True
        
        # Boost for quotes
        if context.get("has_quotes"):
            base_score += CONTEXT_QUOTE_BOOST
            evidence["has_quotes"] = True
        
        # Clamp to [0, 1]
        score = max(0.0, min(1.0, base_score))
        
        return {
            "route": "B",
            "label": "proper_noun_or_title",
            "candidate": None,
            "score": round(score, 3),
            "evidence": evidence
        }
    
    def _score_route_c(self,
                        token: str,
                        synset_ids: List[str],
                        has_edges: bool) -> Dict[str, Any]:
        """Score Route C (molecule candidate / new word)."""
        # Higher score if no synsets at all
        if not synset_ids:
            score = 0.6
            evidence = {"no_synsets": True, "novelty": 0.7}
        elif not has_edges:
            score = 0.35
            evidence = {"synsets_but_no_edges": True, "novelty": 0.4}
        else:
            score = 0.15
            evidence = {"has_edges": True, "novelty": 0.2}
        
        return {
            "route": "C",
            "label": "molecule_candidate",
            "candidate": None,
            "score": round(score, 3),
            "evidence": evidence
        }
    
    def _score_route_d(self, token: str) -> Dict[str, Any]:
        """Score Route D (noise / ignore)."""
        is_stop = token.lower() in STOPWORDS
        too_short = len(token) < MIN_TOKEN_LENGTH
        
        if is_stop or too_short:
            score = 0.9
        else:
            score = 0.05
        
        return {
            "route": "D",
            "label": "noise",
            "candidate": None,
            "score": round(score, 3),
            "evidence": {
                "stopword": is_stop,
                "too_short": too_short
            }
        }
    
    def _compute_decision(self, hypotheses: List[Dict]) -> Dict[str, Any]:
        """Apply variance gate to decide: apply winner or abstain."""
        scores = [h["score"] for h in hypotheses]
        
        # Normalize to probability distribution
        total = sum(scores)
        if total <= 0:
            return {
                "action": "abstain",
                "reason": "zero_scores",
                "winner": None,
                "margin": 0.0,
                "entropy": 0.0
            }
        
        probs = [s / total for s in scores]
        
        # Compute margin and entropy
        sorted_probs = sorted(probs, reverse=True)
        margin = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else sorted_probs[0]
        entropy = compute_entropy(probs)
        
        # Variance gate: abstain if uncertain
        if margin < self.margin_threshold or entropy > self.entropy_threshold:
            return {
                "action": "abstain",
                "reason": "variance_high",
                "winner": None,
                "margin": round(margin, 3),
                "entropy": round(entropy, 3)
            }
        
        # Find winner
        winner_idx = scores.index(max(scores))
        winner = hypotheses[winner_idx]
        
        action_map = {
            "A": "apply_A",
            "B": "apply_B",
            "C": "apply_C",
            "D": "ignore"
        }
        
        return {
            "action": action_map.get(winner["route"], "abstain"),
            "reason": "winner_clear",
            "winner": winner["route"],
            "margin": round(margin, 3),
            "entropy": round(entropy, 3)
        }
    
    def evaluate(self,
                  token: str,
                  original_text: str,
                  tokens: List[str],
                  token_index: int,
                  synset_ids: List[str],
                  has_direct_edges: bool,
                  has_proxy_edges: bool,
                  typo_candidates: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Evaluate all routes and return RoutingReport.
        
        Returns a structured report with:
        - All hypotheses with scores and evidence
        - Decision (action, reason, winner, margin, entropy)
        - Context features used
        """
        # Get typo candidates if not provided
        if typo_candidates is None:
            typo_candidates = find_all_typo_candidates(token)
        
        # Detect context features
        context = self._detect_context_features(token, original_text, tokens, token_index)
        
        # Has any edges?
        has_edges = has_direct_edges or has_proxy_edges
        
        # Score all routes
        hypotheses = [
            self._score_route_a(token, typo_candidates, context),
            self._score_route_b(token, synset_ids, has_edges, context),
            self._score_route_c(token, synset_ids, has_edges),
            self._score_route_d(token)
        ]
        
        # Compute decision
        decision = self._compute_decision(hypotheses)
        
        return {
            "token": token,
            "pos": "u",  # Will be filled by caller if known
            "context": {
                "text": original_text,
                "tokens": tokens,
                "window": tokens[max(0, token_index-2):min(len(tokens), token_index+3)],
                "capitalized": context.get("capitalized", False),
                "has_quotes": context.get("has_quotes", False),
                "title_like_ngram": context.get("title_like_ngram", False),
                "title_phrase_matched": context.get("title_phrase_matched")
            },
            "hypotheses": hypotheses,
            "decision": decision
        }
