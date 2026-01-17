"""
ESDE Sensor V2 (Facade)
Phase 8: Synapse-Integrated Concept Extraction

This is the thin orchestration layer that calls modular components in sensor/.

Usage:
    from esde_sensor_v2 import ESDESensorV2
    sensor = ESDESensorV2()
    result = sensor.analyze("I love you")

GPT Audit Compliance:
  - Config values injected explicitly
  - ALLOWED_POS includes 's'
  - determinism_hash includes config_snapshot
"""
import json
import re
from typing import Dict, List, Any, Optional

# Import modular components
from sensor import (
    SynapseLoader,
    SynsetExtractor,
    CandidateRanker,
    LegacyTriggerMatcher,
    AuditTracer,
)


# ==========================================
# Version
# ==========================================
VERSION = "2.0.0"


# ==========================================
# Config Import
# ==========================================
try:
    import config as cfg
    CONFIG_IMPORTED = True
except ImportError:
    cfg = None
    CONFIG_IMPORTED = False


def get_config_value(key: str, default: Any) -> Any:
    """Get config value with priority: config.py > default."""
    if CONFIG_IMPORTED and cfg and hasattr(cfg, key):
        return getattr(cfg, key)
    return default


# ==========================================
# Default Configuration
# ==========================================
DEFAULT_CONFIG = {
    "SENSOR_TOP_K": get_config_value("SENSOR_TOP_K", 5),
    "SENSOR_MAX_SYNSETS_PER_TOKEN": get_config_value("SENSOR_MAX_SYNSETS_PER_TOKEN",
                                                      get_config_value("MAX_SYNSETS_PER_TOKEN", 3)),
    "STRICT_SYNAPSE_ONLY": get_config_value("STRICT_SYNAPSE_ONLY", False),
    "ALLOWED_POS": get_config_value("SENSOR_ALLOWED_POS", {'n', 'v', 'a', 'r', 's'}),
    "MIN_SCORE_THRESHOLD": get_config_value("MIN_SCORE_THRESHOLD", 0.3),
    "SYNAPSE_FILE": get_config_value("SYNAPSE_FILE", "esde_synapses_v2_1.json"),
    "GLOSSARY_FILE": get_config_value("GLOSSARY_FILE", "glossary_results.json"),
}

# GPT Audit: Ensure 's' is always in ALLOWED_POS
if 's' not in DEFAULT_CONFIG["ALLOWED_POS"]:
    DEFAULT_CONFIG["ALLOWED_POS"] = set(DEFAULT_CONFIG["ALLOWED_POS"]) | {'s'}


# ==========================================
# Glossary Loader (simple)
# ==========================================
def load_glossary(filepath: str) -> Dict[str, Any]:
    """Load glossary from file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, dict):
            if "glossary" in data:
                glossary = data["glossary"]
                if isinstance(glossary, list):
                    return {item["concept_id"]: item for item in glossary 
                           if isinstance(item, dict) and "concept_id" in item}
                return glossary
            return data
        return {}
    except Exception as e:
        print(f"[Glossary] Warning: {e}")
        return {}


# ==========================================
# ESDESensorV2 (Facade)
# ==========================================
class ESDESensorV2:
    """
    ESDE Sensor V2 - Synapse-Integrated Concept Extraction
    
    This is a thin Facade that orchestrates modular components.
    """
    
    def __init__(self,
                 synapse_file: str = None,
                 glossary_file: str = None,
                 config: Dict[str, Any] = None):
        """
        Initialize Sensor V2.
        
        Args:
            synapse_file: Path to synapse JSON
            glossary_file: Path to glossary JSON (for fallback)
            config: Configuration overrides
        """
        # Merge config with defaults
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        
        # Initialize components
        self.synapse_loader = SynapseLoader(
            filepath=synapse_file or self.config["SYNAPSE_FILE"]
        )
        self.synapse_loader.load()
        
        self.synset_extractor = SynsetExtractor(
            max_synsets_per_token=self.config["SENSOR_MAX_SYNSETS_PER_TOKEN"],
            allowed_pos=self.config["ALLOWED_POS"]
        )
        
        # Top-K priority: config > meta > default
        top_k = self.config.get("SENSOR_TOP_K")
        if top_k is None:
            top_k = self.synapse_loader.get_meta_top_k() or 5
        
        self.ranker = CandidateRanker(top_k=top_k)
        self.audit = AuditTracer()
        
        # Legacy fallback (Hybrid mode)
        self.legacy_matcher = None
        glossary_path = glossary_file or self.config.get("GLOSSARY_FILE")
        if not self.config["STRICT_SYNAPSE_ONLY"] and glossary_path:
            glossary = load_glossary(glossary_path)
            if glossary:
                self.legacy_matcher = LegacyTriggerMatcher(glossary)
        
        print(f"[SensorV2] Initialized (ALLOWED_POS={self.config['ALLOWED_POS']}, "
              f"TOP_K={top_k}, STRICT={self.config['STRICT_SYNAPSE_ONLY']})")
    
    def tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        text = text.lower()
        text = re.sub(r"[^a-z0-9'\s]", " ", text)
        tokens = text.split()
        return [t.strip("'") for t in tokens if t.strip("'") and len(t.strip("'")) >= 2]
    
    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Analyze text and extract concept candidates.
        
        Returns:
            {
                "candidates": [...],
                "meta": {...}
            }
        """
        timestamp = self.audit.create_timestamp()
        
        # Step 1: Tokenize
        tokens = self.tokenize(text)
        tokens_with_index = list(enumerate(tokens))
        
        if not tokens:
            return self._empty_result(timestamp, "no_tokens")
        
        # Step 2: Extract synsets
        token_synsets = self.synset_extractor.extract_all(tokens)
        
        # Step 3: Synapse lookup & aggregation
        concept_scores, synapse_counters = self.ranker.aggregate_scores(
            token_synsets=token_synsets,
            synapse_loader=self.synapse_loader,
            tokens_with_index=tokens_with_index
        )
        
        # Step 4: Rank
        candidates = self.ranker.rank(concept_scores)
        
        # Step 5: Fallback
        engine_type = "v2_synapse"
        fallback_candidates_count = 0
        
        if not candidates and not self.config["STRICT_SYNAPSE_ONLY"]:
            if self.legacy_matcher:
                fallback_result = self.legacy_matcher.match(text)
                fallback_candidates_count = len(fallback_result)
                candidates = fallback_result[:self.ranker.top_k]
                engine_type = "v2_synapse+v1_fallback"
        
        # Step 6: Add rank and top_evidence
        for i, cand in enumerate(candidates):
            cand["rank"] = i + 1
        candidates = self.audit.add_top_evidence(candidates)
        
        # Step 7: Config snapshot
        config_snapshot = self.audit.build_config_snapshot(
            top_k=self.ranker.top_k,
            max_synsets=self.config["SENSOR_MAX_SYNSETS_PER_TOKEN"],
            strict_mode=self.config["STRICT_SYNAPSE_ONLY"],
            allowed_pos=self.config["ALLOWED_POS"]
        )
        
        # Step 8: Determinism hash
        hash_full, hash_short = self.ranker.compute_determinism_hash(
            candidates=candidates,
            config_snapshot=config_snapshot,
            synapse_hash=self.synapse_loader.get_file_hash()
        )
        
        # Step 9: Counters
        counters = self.audit.build_counters(
            input_tokens=len(tokens),
            tokens_with_synsets=len(token_synsets),
            synapse_counters=synapse_counters,
            concepts_aggregated=len(concept_scores),
            candidates_returned=len(candidates),
            fallback_candidates=fallback_candidates_count
        )
        
        # Step 10: Build meta
        meta = self.audit.build_meta(
            engine_type=engine_type,
            version=VERSION,
            timestamp=timestamp,
            hash_short=hash_short,
            hash_full=hash_full,
            config_snapshot=config_snapshot,
            counters=counters
        )
        
        return {
            "candidates": candidates,
            "meta": meta
        }
    
    def _empty_result(self, timestamp: str, reason: str) -> Dict[str, Any]:
        """Return empty result structure."""
        config_snapshot = self.audit.build_config_snapshot(
            top_k=self.ranker.top_k,
            max_synsets=self.config["SENSOR_MAX_SYNSETS_PER_TOKEN"],
            strict_mode=self.config["STRICT_SYNAPSE_ONLY"],
            allowed_pos=self.config["ALLOWED_POS"]
        )
        return {
            "candidates": [],
            "meta": self.audit.build_empty_meta(
                version=VERSION,
                timestamp=timestamp,
                config_snapshot=config_snapshot,
                reason=reason
            )
        }
    
    def get_allowed_atoms(self) -> set:
        """Get all concept IDs available in synapses (for validation)."""
        return self.synapse_loader.get_all_concept_ids()


# ==========================================
# Test
# ==========================================
if __name__ == "__main__":
    print("=" * 60)
    print(f"ESDE Sensor V2 Test (Version {VERSION})")
    print("Modular Architecture")
    print("=" * 60)
    
    sensor = ESDESensorV2(
        synapse_file="esde_synapses_v2_1.json",
        glossary_file="glossary_results.json"
    )
    
    test_texts = [
        "I love you",
        "The law requires obedience",
        "I cannot forgive you",
        "apprenticed to a master",
    ]
    
    for text in test_texts:
        print(f"\n{'='*60}")
        print(f"[Input] {text}")
        result = sensor.analyze(text)
        
        meta = result['meta']
        print(f"Engine: {meta['engine']}")
        print(f"Candidates: {len(result['candidates'])}")
        
        counters = meta.get('counters', {})
        print(f"Counters: synsets_checked={counters.get('synsets_checked')}, "
              f"with_edges={counters.get('synsets_with_edges')}")
        
        for cand in result['candidates'][:2]:
            print(f"  #{cand['rank']}: {cand['concept_id']} (score={cand['score']:.4f})")
            if cand.get('top_evidence'):
                print(f"       via {cand['top_evidence']['synset']}")
    
    print("\n" + "=" * 60)
    print("Test Complete")
    print("=" * 60)
