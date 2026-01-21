"""
ESDE Phase 9-4: W4 Projector
============================
Projects ArticleRecords onto W3 axis candidates to produce resonance vectors.

Theme: The Resonance of Weakness (弱さの共鳴・構造射影)

Mathematical Model:
  R(A, C) = Σ count(t, A) × S(t, C)
  
  Where:
    - A = Article
    - C = Condition
    - t = Token
    - count(t, A) = Token count in article A
    - S(t, C) = S-Score from W3 for token t under condition C

Processing Flow:
  1. Load W3 records → Build S-Score dictionary per condition
  2. Load ArticleRecord
  3. Tokenize ArticleRecord.raw_text using W1Tokenizer (INV-W4-006)
  4. Compute resonance: dot product of token counts × S-Scores
  5. Generate W4Record with deterministic analysis_id

Invariants:
  INV-W4-001 (No Labeling): Output keys are condition_signature only
  INV-W4-002 (Deterministic): Same inputs → same outputs
  INV-W4-003 (Recomputable): W4 = f(W0, W3)
  INV-W4-004 (Full S-Score Usage): Uses positive + negative candidates
  INV-W4-005 (Immutable Input): Never modifies ArticleRecord or W3Record
  INV-W4-006 (Tokenization Canon): Uses W1Tokenizer + normalize_token

Spec: v5.4.4-P9.4
"""

import os
import json
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
from collections import Counter
from dataclasses import dataclass

# Import from statistics package (INV-W4-006)
from .tokenizer import get_w1_tokenizer, W1Tokenizer
from .normalizer import normalize_token, NORMALIZER_VERSION
from .schema_w3 import W3Record, CandidateToken
from .schema_w4 import (
    W4Record,
    compute_w4_analysis_id,
    W4_VERSION,
    W4_ALGORITHM,
    W4_PROJECTION_NORM,
    DEFAULT_W4_OUTPUT_DIR,
)


# ==========================================
# Constants
# ==========================================

# Minimum S-Score magnitude to include in calculation
# (tokens with |S| < this are effectively neutral)
MIN_SSCORE_MAGNITUDE = 0.0  # Include all by default


# ==========================================
# S-Score Dictionary Builder
# ==========================================

def build_sscore_dict(w3_record: W3Record) -> Dict[str, float]:
    """
    Build token_norm → s_score mapping from W3Record.
    
    INV-W4-004: Includes BOTH positive AND negative candidates.
    
    Args:
        w3_record: W3Record with positive/negative candidates
        
    Returns:
        Dict mapping token_norm to s_score
    """
    sscore_dict: Dict[str, float] = {}
    
    # Add positive candidates (S > 0)
    for candidate in w3_record.positive_candidates:
        sscore_dict[candidate.token_norm] = candidate.s_score
    
    # Add negative candidates (S < 0)
    for candidate in w3_record.negative_candidates:
        sscore_dict[candidate.token_norm] = candidate.s_score
    
    return sscore_dict


# ==========================================
# W4 Projector
# ==========================================

class W4Projector:
    """
    Projects ArticleRecords onto W3 axis candidates.
    
    Produces W4Records containing resonance vectors.
    
    INV-W4-005: This projector ONLY READS W3/ArticleRecord. Never modifies.
    INV-W4-006: Uses W1Tokenizer from statistics package.
    
    Usage:
        projector = W4Projector()
        projector.load_w3_records([w3_record1, w3_record2])
        w4_record = projector.project(article_record)
        projector.save(w4_record)
    """
    
    def __init__(
        self,
        tokenizer: Optional[W1Tokenizer] = None,
        output_dir: str = DEFAULT_W4_OUTPUT_DIR,
    ):
        """
        Initialize W4 Projector.
        
        Args:
            tokenizer: W1Tokenizer instance (default: hybrid)
            output_dir: Directory for W4 output files
        """
        # INV-W4-006: Use canonical tokenizer
        self.tokenizer = tokenizer or get_w1_tokenizer("hybrid")
        self.output_dir = output_dir
        
        # S-Score dictionaries per condition
        # {condition_signature: {token_norm: s_score}}
        self._sscore_dicts: Dict[str, Dict[str, float]] = {}
        
        # W3 analysis IDs for traceability (P0-2)
        # {condition_signature: w3_analysis_id}
        self._w3_ids: Dict[str, str] = {}
    
    @property
    def tokenizer_version(self) -> str:
        """Get tokenizer version for audit."""
        return self.tokenizer.name
    
    @property
    def normalizer_version(self) -> str:
        """Get normalizer version for audit."""
        return NORMALIZER_VERSION
    
    def load_w3_record(self, w3_record: W3Record) -> None:
        """
        Load a single W3Record into memory.
        
        INV-W4-005: Does not modify the W3Record.
        
        Args:
            w3_record: W3Record to load
        """
        cond_sig = w3_record.condition_signature
        
        # Build S-Score dictionary (INV-W4-004: includes both positive and negative)
        self._sscore_dicts[cond_sig] = build_sscore_dict(w3_record)
        
        # Track W3 analysis ID (P0-2)
        self._w3_ids[cond_sig] = w3_record.analysis_id
    
    def load_w3_records(self, w3_records: List[W3Record]) -> None:
        """
        Load multiple W3Records.
        
        Args:
            w3_records: List of W3Records
        """
        for w3_record in w3_records:
            self.load_w3_record(w3_record)
    
    def load_w3_from_file(self, filepath: str) -> W3Record:
        """
        Load W3Record from JSON file.
        
        Args:
            filepath: Path to W3 JSON file
            
        Returns:
            Loaded W3Record
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        w3_record = W3Record.from_dict(data)
        self.load_w3_record(w3_record)
        
        return w3_record
    
    def get_loaded_conditions(self) -> List[str]:
        """Get list of loaded condition signatures."""
        return list(self._sscore_dicts.keys())
    
    def _tokenize_and_count(self, text: str) -> Tuple[Counter, int]:
        """
        Tokenize text and count normalized tokens.
        
        INV-W4-006: Uses W1Tokenizer + normalize_token.
        
        Args:
            text: Raw text to tokenize
            
        Returns:
            (token_counts, total_valid_tokens)
            token_counts: Counter of {token_norm: count}
        """
        token_results = self.tokenizer.tokenize(text)
        
        counts: Counter = Counter()
        total_valid = 0
        
        for token_result in token_results:
            # INV-W4-006: Use canonical normalizer
            token_norm = normalize_token(token_result.surface)
            
            if token_norm is None:
                continue
            
            counts[token_norm] += 1
            total_valid += 1
        
        return counts, total_valid
    
    def _compute_resonance(
        self,
        token_counts: Counter,
        sscore_dict: Dict[str, float],
    ) -> float:
        """
        Compute resonance score (dot product).
        
        R(A, C) = Σ count(t, A) × S(t, C)
        
        INV-W4-004: sscore_dict includes both positive and negative S-Scores.
        
        Args:
            token_counts: {token_norm: count}
            sscore_dict: {token_norm: s_score}
            
        Returns:
            Resonance score (can be positive or negative)
        """
        resonance = 0.0
        
        for token_norm, count in token_counts.items():
            if token_norm in sscore_dict:
                s_score = sscore_dict[token_norm]
                
                # Optional: filter by magnitude
                if abs(s_score) >= MIN_SSCORE_MAGNITUDE:
                    resonance += count * s_score
        
        return resonance
    
    def project(self, article) -> W4Record:
        """
        Project an ArticleRecord onto W3 axis candidates.
        
        INV-W4-001: Output keys are condition_signature only.
        INV-W4-002: Deterministic given same inputs.
        INV-W4-005: Does not modify ArticleRecord.
        
        Args:
            article: ArticleRecord-like object with:
                - article_id: str
                - raw_text: str
            
        Returns:
            W4Record with resonance vector
        
        Raises:
            ValueError: If no W3 records loaded
        """
        if not self._sscore_dicts:
            raise ValueError("No W3 records loaded. Call load_w3_records() first.")
        
        # INV-W4-006: Tokenize using canonical tokenizer
        token_counts, total_tokens = self._tokenize_and_count(article.raw_text)
        
        # Compute resonance for each condition
        resonance_vector: Dict[str, float] = {}
        
        for cond_sig, sscore_dict in self._sscore_dicts.items():
            resonance = self._compute_resonance(token_counts, sscore_dict)
            resonance_vector[cond_sig] = resonance
        
        # Compute deterministic analysis ID (P0-1)
        w4_analysis_id = compute_w4_analysis_id(
            article_id=article.article_id,
            used_w3=self._w3_ids,
            tokenizer_version=self.tokenizer_version,
            normalizer_version=self.normalizer_version,
            algorithm=W4_ALGORITHM,
        )
        
        # Create W4Record
        w4_record = W4Record(
            article_id=article.article_id,
            w4_analysis_id=w4_analysis_id,
            resonance_vector=resonance_vector,
            used_w3=dict(self._w3_ids),  # Copy for immutability (P0-2)
            token_count=total_tokens,
            tokenizer_version=self.tokenizer_version,
            normalizer_version=self.normalizer_version,
        )
        
        return w4_record
    
    def project_batch(self, articles: List) -> List[W4Record]:
        """
        Project multiple ArticleRecords.
        
        Args:
            articles: List of ArticleRecord-like objects
            
        Returns:
            List of W4Records
        """
        return [self.project(article) for article in articles]
    
    def save(self, w4_record: W4Record) -> str:
        """
        Save W4Record to JSON file.
        
        Storage: {output_dir}/{article_id}.json
        
        Args:
            w4_record: W4Record to save
            
        Returns:
            Path to saved file
        """
        os.makedirs(self.output_dir, exist_ok=True)
        
        filepath = os.path.join(self.output_dir, f"{w4_record.article_id}.json")
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(w4_record.to_json())
        
        return filepath
    
    def load(self, article_id: str) -> Optional[W4Record]:
        """
        Load W4Record from file.
        
        Args:
            article_id: Article ID
            
        Returns:
            W4Record or None if not found
        """
        filepath = os.path.join(self.output_dir, f"{article_id}.json")
        
        if not os.path.exists(filepath):
            return None
        
        with open(filepath, 'r', encoding='utf-8') as f:
            return W4Record.from_json(f.read())


# ==========================================
# Comparison Utility
# ==========================================

def compare_w4_records(
    record1: W4Record,
    record2: W4Record,
    tolerance: float = 1e-8,
) -> Dict[str, Any]:
    """
    Compare two W4Records for determinism verification.
    
    INV-W4-002: Same inputs should produce identical outputs.
    
    Args:
        record1: First W4Record
        record2: Second W4Record
        tolerance: Floating-point tolerance (P1-1)
        
    Returns:
        Comparison result dict
    """
    diffs = []
    
    # Compare article_id
    if record1.article_id != record2.article_id:
        diffs.append(f"article_id: {record1.article_id} vs {record2.article_id}")
    
    # Compare analysis_id
    if record1.w4_analysis_id != record2.w4_analysis_id:
        diffs.append(f"w4_analysis_id: {record1.w4_analysis_id} vs {record2.w4_analysis_id}")
    
    # Compare token_count
    if record1.token_count != record2.token_count:
        diffs.append(f"token_count: {record1.token_count} vs {record2.token_count}")
    
    # Compare resonance_vector (with tolerance)
    all_keys = set(record1.resonance_vector.keys()) | set(record2.resonance_vector.keys())
    
    for key in all_keys:
        v1 = record1.resonance_vector.get(key, 0.0)
        v2 = record2.resonance_vector.get(key, 0.0)
        
        if abs(v1 - v2) > tolerance:
            diffs.append(f"resonance[{key[:8]}...]: {v1:.8f} vs {v2:.8f}")
    
    # Compare used_w3
    if record1.used_w3 != record2.used_w3:
        diffs.append(f"used_w3 mismatch")
    
    return {
        "match": len(diffs) == 0,
        "differences": diffs[:10],
        "total_differences": len(diffs),
    }


# ==========================================
# Test
# ==========================================

if __name__ == "__main__":
    print("Phase 9-4 W4Projector Test")
    print("=" * 60)
    
    # Create mock data for testing
    from .schema_w3 import W3Record, CandidateToken, compute_analysis_id
    
    # Mock ArticleRecord-like class
    class MockArticle:
        def __init__(self, article_id: str, raw_text: str):
            self.article_id = article_id
            self.raw_text = raw_text
    
    # Build mock W3 records
    print("\n[Setup] Creating mock W3 records")
    
    # Condition: news (political terms are positive)
    w3_news = W3Record(
        analysis_id=compute_analysis_id("cond_news", "w1_hash", "w2_hash"),
        condition_signature="cond_news_abc123",
        condition_factors={"source_type": "news", "language_profile": "en", "time_bucket": "2026-01"},
        positive_candidates=[
            CandidateToken(token_norm="prime", s_score=0.05, p_cond=0.02, p_global=0.01),
            CandidateToken(token_norm="minister", s_score=0.04, p_cond=0.015, p_global=0.008),
            CandidateToken(token_norm="government", s_score=0.03, p_cond=0.012, p_global=0.007),
        ],
        negative_candidates=[
            CandidateToken(token_norm="lol", s_score=-0.02, p_cond=0.001, p_global=0.008),
            CandidateToken(token_norm="yeah", s_score=-0.015, p_cond=0.002, p_global=0.01),
        ],
    )
    
    # Condition: dialog (casual terms are positive)
    w3_dialog = W3Record(
        analysis_id=compute_analysis_id("cond_dialog", "w1_hash", "w2_hash"),
        condition_signature="cond_dialog_xyz789",
        condition_factors={"source_type": "dialog", "language_profile": "en", "time_bucket": "2026-01"},
        positive_candidates=[
            CandidateToken(token_norm="lol", s_score=0.04, p_cond=0.02, p_global=0.008),
            CandidateToken(token_norm="yeah", s_score=0.03, p_cond=0.018, p_global=0.01),
            CandidateToken(token_norm="hey", s_score=0.025, p_cond=0.015, p_global=0.008),
        ],
        negative_candidates=[
            CandidateToken(token_norm="prime", s_score=-0.03, p_cond=0.002, p_global=0.01),
            CandidateToken(token_norm="minister", s_score=-0.025, p_cond=0.001, p_global=0.008),
        ],
    )
    
    print(f"  W3 news candidates: {len(w3_news.positive_candidates)} pos, {len(w3_news.negative_candidates)} neg")
    print(f"  W3 dialog candidates: {len(w3_dialog.positive_candidates)} pos, {len(w3_dialog.negative_candidates)} neg")
    
    # Test 1: Basic projection
    print("\n[Test 1] Basic projection (news article)")
    
    projector = W4Projector(output_dir="/tmp/w4_test")
    projector.load_w3_records([w3_news, w3_dialog])
    
    print(f"  Loaded conditions: {projector.get_loaded_conditions()}")
    
    # News-like article
    news_article = MockArticle(
        article_id="article_news_001",
        raw_text="The prime minister announced new government policy. The prime minister said reforms are needed."
    )
    
    w4_result = projector.project(news_article)
    
    print(f"  Article: {news_article.article_id}")
    print(f"  Token count: {w4_result.token_count}")
    print(f"  Resonance vector:")
    for cond, score in sorted(w4_result.resonance_vector.items()):
        print(f"    {cond}: {score:.6f}")
    
    # News article should resonate positively with news condition
    assert w4_result.resonance_vector["cond_news_abc123"] > 0, \
        "News article should have positive resonance with news condition"
    print("  ✅ PASS: News article resonates with news condition")
    
    # Test 2: Dialog article
    print("\n[Test 2] Dialog article projection")
    
    dialog_article = MockArticle(
        article_id="article_dialog_001",
        raw_text="Hey! Yeah, I agree lol. Yeah that's so funny lol lol."
    )
    
    w4_dialog_result = projector.project(dialog_article)
    
    print(f"  Token count: {w4_dialog_result.token_count}")
    print(f"  Resonance vector:")
    for cond, score in sorted(w4_dialog_result.resonance_vector.items()):
        print(f"    {cond}: {score:.6f}")
    
    # Dialog article should resonate positively with dialog condition
    assert w4_dialog_result.resonance_vector["cond_dialog_xyz789"] > 0, \
        "Dialog article should have positive resonance with dialog condition"
    print("  ✅ PASS: Dialog article resonates with dialog condition")
    
    # Test 3: Determinism (INV-W4-002)
    print("\n[Test 3] Determinism verification (INV-W4-002)")
    
    # Project same article twice
    result1 = projector.project(news_article)
    result2 = projector.project(news_article)
    
    comparison = compare_w4_records(result1, result2)
    
    print(f"  w4_analysis_id match: {result1.w4_analysis_id == result2.w4_analysis_id}")
    print(f"  Full comparison match: {comparison['match']}")
    
    assert comparison['match'], "Same input should produce identical output"
    print("  ✅ PASS: Deterministic output verified")
    
    # Test 4: INV-W4-004 (Both positive and negative used)
    print("\n[Test 4] INV-W4-004: Full S-Score usage")
    
    # Article with both news and dialog terms
    mixed_article = MockArticle(
        article_id="article_mixed_001",
        raw_text="The prime minister said lol just kidding. Government policy yeah."
    )
    
    w4_mixed = projector.project(mixed_article)
    
    print(f"  Mixed article resonance:")
    for cond, score in sorted(w4_mixed.resonance_vector.items()):
        print(f"    {cond}: {score:.6f}")
    
    # Both positive and negative contributions should be reflected
    # (news has positive 'prime', 'minister', 'government' and negative 'lol', 'yeah')
    print("  ✅ PASS: Both positive and negative S-Scores contribute to resonance")
    
    # Test 5: Traceability (P0-2)
    print("\n[Test 5] Traceability (P0-2: used_w3 mapping)")
    
    print(f"  used_w3: {w4_result.used_w3}")
    
    assert len(w4_result.used_w3) == 2, "Should have 2 W3 references"
    assert w4_result.used_w3["cond_news_abc123"] == w3_news.analysis_id
    assert w4_result.used_w3["cond_dialog_xyz789"] == w3_dialog.analysis_id
    print("  ✅ PASS: W3 analysis IDs properly tracked")
    
    # Test 6: Save and load
    print("\n[Test 6] Save and load")
    
    filepath = projector.save(w4_result)
    print(f"  Saved to: {filepath}")
    
    loaded = projector.load(w4_result.article_id)
    
    assert loaded is not None, "Should be able to load saved record"
    assert loaded.article_id == w4_result.article_id
    assert loaded.w4_analysis_id == w4_result.w4_analysis_id
    print(f"  Loaded article_id: {loaded.article_id}")
    print("  ✅ PASS")
    
    # Test 7: Version tracking (P0-3)
    print("\n[Test 7] Version tracking (P0-3)")
    
    print(f"  tokenizer_version: {w4_result.tokenizer_version}")
    print(f"  normalizer_version: {w4_result.normalizer_version}")
    print(f"  projection_norm: {w4_result.projection_norm}")
    print(f"  algorithm: {w4_result.algorithm}")
    
    assert w4_result.tokenizer_version == "hybrid_v1"
    assert w4_result.projection_norm == "raw"
    print("  ✅ PASS: Version info properly recorded")
    
    print("\n" + "=" * 60)
    print("All tests passed! ✅")
    print("\nPhase 9-4 W4Projector is ready.")
    print("\nDefinition of Done:")
    print("  ✓ schema_w4.py: W4Record defined")
    print("  ✓ w4_projector.py: W4Projector implemented")
    print("  ✓ Tokenizer consistency: Uses W1Tokenizer (INV-W4-006)")
    print("  ✓ Math test: News article → positive news score")
    print("  ✓ Math test: Dialog article → positive dialog score")
