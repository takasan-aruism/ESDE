"""
ESDE Phase 9-0: Content Gateway (Substrate Layer Extended)
==========================================================
The entry point for observation. Routes content to appropriate pipelines.

Changes in v5.4.7-SUB.1:
  - Added Substrate Layer integration
  - source_meta → legacy:* traces → SubstrateRegistry
  - ArticleRecord now includes substrate_ref

Design Principles:
  - Gateway is a "switchboard", not an interpreter
  - Emits W0 (ObservationEvent) for every segment
  - No meaning, no judgment, no aggregation
  - Substrate registration is optional (backward compatible)

Spec: v5.4.7-SUB.1
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

# Substrate Layer (optional import for backward compatibility)
try:
    from substrate import (
        SubstrateRegistry,
        create_context_record,
        convert_source_meta_to_traces,
    )
    SUBSTRATE_AVAILABLE = True
except ImportError:
    SUBSTRATE_AVAILABLE = False
    SubstrateRegistry = None  # Type hint placeholder


# ==========================================
# Gateway Configuration
# ==========================================

@dataclass
class GatewayConfig:
    """Configuration for ContentGateway."""
    segmenter_type: str = "sentence"
    synapse_version: str = "v3.0"
    glossary_version: str = "v5.3"
    
    # Substrate Layer integration (v0.1.0+)
    enable_substrate: bool = True  # Enable Substrate registration
    substrate_registry_path: str = "data/substrate/context_registry.jsonl"
    substrate_capture_version: str = "v1.0"  # P0-W0-001: Fixed version
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "segmenter_type": self.segmenter_type,
            "synapse_version": self.synapse_version,
            "glossary_version": self.glossary_version,
            "enable_substrate": self.enable_substrate,
            "substrate_registry_path": self.substrate_registry_path,
            "substrate_capture_version": self.substrate_capture_version,
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
        4. Register Substrate: Convert source_meta to traces (v0.1.0+)
        5. Return: ArticleRecord with W0 observations and substrate_ref
    
    GATEWAY MUST NOT:
        - Interpret meaning
        - Decide importance
        - Aggregate statistics (beyond per-segment counts)
        - Create or modify Axis
        - Make routing decisions based on content semantics
    """
    
    def __init__(
        self,
        config: Optional[GatewayConfig] = None,
        synapse_hash: Optional[str] = None,
        substrate_registry: Optional['SubstrateRegistry'] = None,
    ):
        """
        Initialize ContentGateway.
        
        Args:
            config: Gateway configuration
            synapse_hash: Pre-computed synapse file hash (for reproducibility)
            substrate_registry: Optional pre-initialized SubstrateRegistry
        """
        self.config = config or GatewayConfig()
        self._synapse_hash = synapse_hash or "not_provided"
        self._segmenter: Optional[Segmenter] = None
        
        # Substrate Layer (lazy initialization)
        self._substrate_registry = substrate_registry
        self._substrate_initialized = False
    
    @property
    def segmenter(self) -> Segmenter:
        """Lazy-load segmenter."""
        if self._segmenter is None:
            self._segmenter = get_segmenter(self.config.segmenter_type)
        return self._segmenter
    
    def _ensure_substrate(self) -> Optional['SubstrateRegistry']:
        """
        Lazy-initialize SubstrateRegistry.
        
        Returns:
            SubstrateRegistry instance or None if disabled/unavailable
        """
        if not self.config.enable_substrate:
            return None
        
        if not SUBSTRATE_AVAILABLE:
            return None
        
        if not self._substrate_initialized:
            if self._substrate_registry is None:
                self._substrate_registry = SubstrateRegistry(
                    storage_path=self.config.substrate_registry_path
                )
            self._substrate_initialized = True
        
        return self._substrate_registry
    
    def _create_environment(self) -> EnvironmentRecord:
        """Create environment record for this ingestion."""
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
                "substrate_enabled": self.config.enable_substrate,
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
            "token_count": len(text.split()),
            "has_punctuation": any(c in text for c in '.!?'),
        }
    
    def _register_to_substrate(
        self,
        source_meta: Dict[str, Any],
        source_url: Optional[str] = None,
    ) -> Optional[str]:
        """
        Register source_meta as traces in Substrate Layer.
        
        P0-AR-001: Lossless conversion (source_meta → legacy:* traces)
        P0-W0-001: Deterministic (same source_meta → same context_id)
        P0-W0-002: Deduplication handled by SubstrateRegistry
        
        Args:
            source_meta: Original source metadata
            source_url: Optional retrieval path
            
        Returns:
            context_id if registered, None if Substrate disabled/unavailable
        """
        registry = self._ensure_substrate()
        if registry is None:
            return None
        
        # P0-AR-001: Lossless migration (source_meta → legacy:* traces)
        traces = convert_source_meta_to_traces(source_meta)
        
        # Add retrieval info if available
        if source_url:
            traces["meta:retrieval_url"] = source_url
        
        # Register to Substrate (P0-W0-002: deduplication handled internally)
        context_id = registry.register_traces(
            traces=traces,
            retrieval_path=source_url,
            capture_version=self.config.substrate_capture_version,
        )
        
        return context_id
    
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
                - source_meta: Original metadata (preserved)
                - substrate_ref: context_id from Substrate (if enabled)
                - molecules: Empty (filled by pipeline)
                - unknowns: Empty (filled by Phase 7)
        
        Example:
            gateway = ContentGateway()
            article = gateway.ingest(
                "I love you. You love me.",
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
        
        # Register to Substrate Layer (P0-AR-001, P0-W0-001)
        substrate_ref = self._register_to_substrate(
            source_meta=source_meta or {},
            source_url=source_url,
        )
        
        # Assemble ArticleRecord
        article = ArticleRecord(
            article_id=article_id,
            source_url=source_url,
            ingestion_time=ingestion_time,
            environment=environment,
            source_meta=source_meta or {},
            substrate_ref=substrate_ref,  # NEW: Substrate Layer reference
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
        source_metas: Optional[List[Dict[str, Any]]] = None,
    ) -> List[ArticleRecord]:
        """
        Batch ingestion for multiple texts.
        
        Args:
            texts: List of raw texts
            source_urls: Optional list of URLs (same length as texts)
            source_metas: Optional list of metadata dicts (same length as texts)
        
        Returns:
            List of ArticleRecords
        """
        results = []
        
        for i, text in enumerate(texts):
            url = source_urls[i] if source_urls and i < len(source_urls) else None
            meta = source_metas[i] if source_metas and i < len(source_metas) else None
            
            article = self.ingest(
                text=text,
                source_url=url,
                source_meta=meta,
            )
            results.append(article)
        
        return results


# ==========================================
# Test
# ==========================================

if __name__ == "__main__":
    print("Phase 9-0 Gateway Test (Substrate Extended)")
    print("=" * 60)
    
    # Test 1: Basic ingestion with Substrate disabled
    print("\n[Test 1] Basic ingestion (Substrate disabled)")
    
    config = GatewayConfig(enable_substrate=False)
    gateway = ContentGateway(config=config)
    
    article = gateway.ingest(
        text="I love you. You love me.",
        source_meta={"source_type": "dialog", "language_profile": "en"},
    )
    
    print(f"  article_id: {article.article_id[:16]}...")
    print(f"  segments: {article.segments}")
    print(f"  observations: {len(article.observations)}")
    print(f"  substrate_ref: {article.substrate_ref}")
    print(f"  source_meta: {article.source_meta}")
    
    assert article.substrate_ref is None, "Should be None when disabled"
    assert article.source_meta == {"source_type": "dialog", "language_profile": "en"}
    print("  ✅ PASS")
    
    # Test 2: Ingestion with Substrate enabled (if available)
    print("\n[Test 2] Ingestion with Substrate enabled")
    
    if SUBSTRATE_AVAILABLE:
        import tempfile
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            temp_path = f.name
        
        try:
            config2 = GatewayConfig(
                enable_substrate=True,
                substrate_registry_path=temp_path,
            )
            gateway2 = ContentGateway(config=config2)
            
            article2 = gateway2.ingest(
                text="Apple releases new iPhone.",
                source_url="https://example.com/news",
                source_meta={"source_type": "news", "language_profile": "en"},
            )
            
            print(f"  article_id: {article2.article_id[:16]}...")
            print(f"  substrate_ref: {article2.substrate_ref}")
            
            assert article2.substrate_ref is not None, "Should have substrate_ref"
            assert len(article2.substrate_ref) == 32, "Should be 32 hex chars"
            print("  ✅ PASS")
        finally:
            import os
            if os.path.exists(temp_path):
                os.remove(temp_path)
    else:
        print("  ⚠️ SKIPPED: Substrate not available")
    
    print("\n" + "=" * 60)
    print("All tests passed! ✅")
