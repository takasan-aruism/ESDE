"""
ESDE Sensor - Synapse Loader
Loads and manages synapse data from JSON file.

Design:
  - Singleton pattern for memory efficiency
  - File hash for audit trail
  - Meta config access for Top-K defaults
"""
import json
import hashlib
import os
from typing import Dict, List, Any, Optional


def compute_file_hash(filepath: str) -> str:
    """Compute SHA256 hash of a file."""
    if not os.path.exists(filepath):
        return "file_not_found"
    try:
        with open(filepath, 'rb') as f:
            return f"sha256:{hashlib.sha256(f.read()).hexdigest()[:16]}"
    except Exception:
        return "hash_error"


class SynapseLoader:
    """
    Loads synapse data from JSON file.
    Singleton-like usage recommended for memory efficiency.
    """
    
    _instance: Optional['SynapseLoader'] = None
    
    def __init__(self, filepath: str = "esde_synapses_v2_1.json"):
        self.filepath = filepath
        self.synapses: Dict[str, List[Dict]] = {}
        self.meta: Dict[str, Any] = {}
        self._loaded = False
        self._file_hash: Optional[str] = None
    
    @classmethod
    def get_instance(cls, filepath: str = None) -> 'SynapseLoader':
        """Get singleton instance."""
        if cls._instance is None or (filepath and cls._instance.filepath != filepath):
            cls._instance = cls(filepath or "esde_synapses_v2_1.json")
        return cls._instance
    
    @classmethod
    def reset_instance(cls):
        """Reset singleton (for testing)."""
        cls._instance = None
    
    def load(self) -> bool:
        """Load synapse data from file."""
        if self._loaded:
            return True
        
        if not os.path.exists(self.filepath):
            print(f"[SynapseLoader] File not found: {self.filepath}")
            return False
        
        try:
            self._file_hash = compute_file_hash(self.filepath)
            with open(self.filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.synapses = data.get("synapses", {})
            self.meta = data.get("_meta", {})
            self._loaded = True
            print(f"[SynapseLoader] Loaded {len(self.synapses)} synsets")
            return True
        except Exception as e:
            print(f"[SynapseLoader] Error: {e}")
            return False
    
    def get_edges(self, synset_id: str) -> List[Dict]:
        """Get edges for a synset ID."""
        return self.synapses.get(synset_id, [])
    
    def has_synset(self, synset_id: str) -> bool:
        """Check if synset exists in synapses."""
        return synset_id in self.synapses
    
    def get_file_hash(self) -> str:
        """Get file hash for audit."""
        return self._file_hash or "not_loaded"
    
    def get_meta_top_k(self) -> Optional[int]:
        """Get global_top_k from meta if available."""
        config = self.meta.get("config", {})
        return config.get("global_top_k")
    
    def get_all_concept_ids(self) -> set:
        """Get all unique concept IDs in synapses (for validation)."""
        concept_ids = set()
        for edges in self.synapses.values():
            for edge in edges:
                cid = edge.get("concept_id")
                if cid:
                    concept_ids.add(cid)
        return concept_ids
