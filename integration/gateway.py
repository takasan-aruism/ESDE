"""
ESDE Phase 9-0: Content Gateway
===============================
The entry point for observation. Routes content to appropriate pipelines.

Design Principles:
  - Gateway is a "switchboard", not an interpreter
  - Emits W0 (ObservationEvent) for every segment
  - No meaning, no judgment, no aggregation

Spec: v5.4.0-P9.0

GATEWAY MUST NOT:
  - Interpret meaning
  - Decide importance
  - Aggregate statistics (beyond per-segment counts)
  - Create or modify Axis
  - Make routing decisions based on content semantics
"""

import hashlib
import uuid
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from .schema import (
    ArticleRecord,
    ObservationEvent,
    EnvironmentRecord,
    CanonicalMolecule,
    UnknownToken,
    DiagnosticType,
    create_observation_event,
    ENGINE_VERSION,
)
from .segmenter import Segmenter, get_segmenter


# ==========================================
# Gateway Configuration
# ==========================================

@dataclass
class GatewayConfig:
    """Configuration for ContentGateway."""
    segmenter_type: str = "sentence"
    synapse_version: str = "v3.0"
    glossary_version: str = "v5.3"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "segmenter_type": self.segmenter_type,
            "synapse_version": self.synapse_version,
            "glossary_version": self.glossary_version,
        }


# ==========================================
# Content Gateway
# ==========================================

class ContentGateway:
    """
    Entry point for content observation.
    
    Responsibilities:
        1. Ingest: Receive text, create ArticleRecord
        2. Segment: Split text into observation units
        3. Emit W0: Create ObservationEvent for each segment
        4. Return: ArticleRecord with W0 observations (no molecules yet)
    
    GATEWAY MUST NOT:
        - Interpret meaning
        - Decide importance
        - Aggregate statistics (beyond per-segment counts)
        - Create or modify Axis
        - Make routing decisions based on content semantics
    
    GATEWAY DOES NOT (Phase 9-0):
        - Call MoleculeGenerator (that's pipeline's job)
        - Resolve unknown tokens (that's Phase 7's job)
        - Calculate W1/W2/Axis (that's Phase 9-1+'s job)
    """
    
    def __init__(
        self,
        config: Optional[GatewayConfig] = None,
        synapse_hash: Optional[str] = None,
    ):
        """
        Initialize ContentGateway.
        
        Args:
            config: Gateway configuration
            synapse_hash: Pre-computed synapse file hash (for reproducibility)
        """
        self.config = config or GatewayConfig()
        self._synapse_hash = synapse_hash or "not_provided"
        self._segmenter: Optional[Segmenter] = None
        
    @property
    def segmenter(self) -> Segmenter:
        """Lazy-load segmenter."""
        if self._segmenter is None:
            self._segmenter = get_segmenter(self.config.segmenter_type)
        return self._segmenter
    
    def _create_environment(self) -> EnvironmentRecord:
        """Create environment record for this ingestion."""
        # Compute config hash for reproducibility
        config_str = f"{self.config.segmenter_type}:{self.config.synapse_version}:{self.config.glossary_version}"
        env_id = f"env_{hashlib.sha256(config_str.encode()).hexdigest()[:12]}"
        
        return EnvironmentRecord(
            env_id=env_id,
            synapse_version=self.config.synapse_version,
            synapse_hash=self._synapse_hash,
            glossary_version=self.config.glossary_version,
            parameters={
                "segmenter": self.segmenter.name,
                "engine_version": ENGINE_VERSION,
            },
        )
    
    def _compute_segment_stats(self, text: str) -> Dict[str, Any]:
        """
        Compute factual statistics for a segment.
        
        ALLOWED: Pure counts (no interpretation)
        FORBIDDEN: Importance scores, meaning labels, routing hints
        """
        return {
            "char_count": len(text),
            "token_count": len(text.split()),  # Simple whitespace tokenization
            "has_punctuation": any(c in text for c in '.!?'),
        }
    
    def ingest(
        self,
        text: str,
        source_url: Optional[str] = None,
        source_id: Optional[str] = None,
        source_meta: Optional[Dict[str, Any]] = None,
    ) -> ArticleRecord:
        """
        Ingest text and create ArticleRecord with W0 observations.
        
        This is the main entry point for content processing.
        
        Args:
            text: Raw input text
            source_url: Optional source URL
            source_id: Optional pre-assigned article ID
            source_meta: Optional metadata for W2 condition factors
                Expected keys (v1):
                - source_type: "news" | "dialog" | "paper" | "social" | "unknown"
                - language_profile: "en" | "ja" | "mixed" | "unknown"
        
        Returns:
            ArticleRecord with:
                - raw_text: Original text
                - segments: List of (start, end) positions
                - observations: W0 events for each segment
                - molecules: Empty (filled by pipeline)
                - unknowns: Empty (filled by Phase 7)
        
        Example:
            gateway = ContentGateway()
            article = gateway.ingest(
                "I love you.",
                source_meta={"source_type": "dialog", "language_profile": "en"}
            )
        """
        # Generate IDs
        article_id = source_id or str(uuid.uuid4())
        ingestion_time = datetime.now(timezone.utc).isoformat()
        
        # Create environment record
        environment = self._create_environment()
        
        # Segment text
        segments = self.segmenter.segment(text)
        
        # Create W0 observations for each segment
        observations: List[ObservationEvent] = []
        
        for index, (start, end) in enumerate(segments):
            segment_text = text[start:end]
            
            # Check for empty segment (diagnostic)
            if not segment_text.strip():
                obs = create_observation_event(
                    source_id=article_id,
                    segment_index=index,
                    segment_span=(start, end),
                    segment_text=segment_text,
                    context_meta={},
                )
                obs.diagnostic_type = DiagnosticType.EMPTY_SEGMENT
                observations.append(obs)
                continue
            
            # Compute factual stats
            stats = self._compute_segment_stats(segment_text)
            
            # Create W0 observation
            obs = create_observation_event(
                source_id=article_id,
                segment_index=index,
                segment_span=(start, end),
                segment_text=segment_text,
                context_meta=stats,
            )
            observations.append(obs)
        
        # Assemble ArticleRecord
        article = ArticleRecord(
            article_id=article_id,
            source_url=source_url,
            ingestion_time=ingestion_time,
            environment=environment,
            source_meta=source_meta or {},
            raw_text=text,
            segments=segments,
            observations=observations,
            molecules=[],    # Empty - filled by pipeline
            unknowns=[],     # Empty - filled by Phase 7
        )
        
        return article
    
    def ingest_batch(
        self,
        texts: List[str],
        source_urls: Optional[List[str]] = None,
    ) -> List[ArticleRecord]:
        """
        Batch ingestion for multiple texts.
        
        Args:
            texts: List of input texts
            source_urls: Optional list of source URLs (same length as texts)
        
        Returns:
            List of ArticleRecords
        """
        if source_urls and len(source_urls) != len(texts):
            raise ValueError("source_urls must have same length as texts")
        
        results = []
        for i, text in enumerate(texts):
            url = source_urls[i] if source_urls else None
            results.append(self.ingest(text, source_url=url))
        
        return results


# ==========================================
# Test
# ==========================================

if __name__ == "__main__":
    print("Phase 9-0 Gateway Test")
    print("=" * 60)
    
    # Test 1: Basic ingestion
    print("\n[Test 1] Basic ingestion")
    gateway = ContentGateway()
    article = gateway.ingest("I love you. You love me.")
    
    print(f"  article_id: {article.article_id}")
    print(f"  raw_text: '{article.raw_text}'")
    print(f"  segments: {article.segments}")
    print(f"  observations count: {len(article.observations)}")
    
    assert len(article.observations) == 2, "Should have 2 observations"
    assert article.observations[0].observer == "Gateway"
    assert article.observations[0].segment_span == (0, 11)
    print("  ✅ PASS")
    
    # Test 2: W0 structure
    print("\n[Test 2] W0 observation structure")
    obs = article.observations[0]
    print(f"  observation_id: {obs.observation_id}")
    print(f"  timestamp: {obs.timestamp}")
    print(f"  segment_index: {obs.segment_index}")
    print(f"  segment_span: {obs.segment_span}")
    print(f"  segment_text: '{obs.segment_text}'")
    print(f"  observer: {obs.observer}")
    print(f"  context_meta: {obs.context_meta}")
    
    assert "char_count" in obs.context_meta
    assert "token_count" in obs.context_meta
    assert obs.diagnostic_type == DiagnosticType.NONE
    print("  ✅ PASS")
    
    # Test 3: Environment record
    print("\n[Test 3] Environment record")
    env = article.environment
    print(f"  env_id: {env.env_id}")
    print(f"  synapse_version: {env.synapse_version}")
    print(f"  parameters: {env.parameters}")
    
    assert env.synapse_version == "v3.0"
    assert "segmenter" in env.parameters
    print("  ✅ PASS")
    
    # Test 4: Segment text extraction
    print("\n[Test 4] Segment text extraction")
    for i, obs in enumerate(article.observations):
        extracted = article.get_segment_text(i)
        print(f"  [{i}] segment_text: '{obs.segment_text}'")
        print(f"  [{i}] extracted:    '{extracted}'")
        assert obs.segment_text == extracted, "segment_text should match extraction"
    print("  ✅ PASS")
    
    # Test 5: Long text with multiple sentences
    print("\n[Test 5] Long text processing")
    long_text = """
    The quick brown fox jumps over the lazy dog. This is a famous pangram.
    Dr. Smith went to the store. He bought some apples!
    What time is it? It's 3 p.m. in New York.
    """
    article = gateway.ingest(long_text.strip())
    print(f"  Input length: {len(long_text)} chars")
    print(f"  Segments found: {len(article.segments)}")
    print(f"  Observations: {len(article.observations)}")
    
    for i, obs in enumerate(article.observations):
        print(f"    [{i}] '{obs.segment_text[:40]}...' " if len(obs.segment_text) > 40 else f"    [{i}] '{obs.segment_text}'")
    
    assert len(article.observations) >= 4, "Should have at least 4 observations"
    print("  ✅ PASS")
    
    # Test 6: Serialization
    print("\n[Test 6] Serialization")
    import json
    article_dict = article.to_dict()
    json_str = json.dumps(article_dict, indent=2, ensure_ascii=False)
    print(f"  JSON length: {len(json_str)} chars")
    
    # Verify structure
    parsed = json.loads(json_str)
    assert "observations" in parsed
    assert len(parsed["observations"]) == len(article.observations)
    assert parsed["observations"][0]["observer"] == "Gateway"
    print("  ✅ PASS")
    
    # Test 7: Empty molecules and unknowns
    print("\n[Test 7] Empty molecules/unknowns (as expected)")
    assert article.molecules == [], "molecules should be empty"
    assert article.unknowns == [], "unknowns should be empty"
    assert article.w1_matrix is None, "w1_matrix should be None"
    assert article.w2_matrix is None, "w2_matrix should be None"
    assert article.aruism_axis is None, "aruism_axis should be None"
    print("  ✅ PASS")
    
    # Test 8: Batch ingestion
    print("\n[Test 8] Batch ingestion")
    texts = [
        "Hello world.",
        "Goodbye world!",
        "What is this?",
    ]
    articles = gateway.ingest_batch(texts)
    print(f"  Input count: {len(texts)}")
    print(f"  Output count: {len(articles)}")
    
    assert len(articles) == 3
    for i, article in enumerate(articles):
        assert len(article.observations) >= 1
        print(f"    [{i}] observations: {len(article.observations)}")
    print("  ✅ PASS")
    
    print("\n" + "=" * 60)
    print("All tests passed! ✅")
    print()
    print("Phase 9-0 Gateway is ready for integration.")
    print("Next steps:")
    print("  1. Connect to SynapseLoader for routing (Phase 9-1)")
    print("  2. Connect to MoleculeGenerator for processing (Phase 9-1)")
    print("  3. Implement W1/W2 aggregation (Phase 9-2+)")