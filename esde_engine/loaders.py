"""
ESDE Engine v5.3.2 - Data Loaders
"""
import os
import json
from typing import Dict, List, Any, Optional

from .config import SYNAPSE_FILE, GLOSSARY_FILE
from .utils import compute_file_hash


class SynapseLoader:
    """Load and manage synapse map data."""
    
    def __init__(self, filepath: str = SYNAPSE_FILE):
        self.filepath = filepath
        self.synapses: Dict[str, List[Dict]] = {}
        self.meta: Dict[str, Any] = {}
        self._loaded = False
        self._file_hash: Optional[str] = None
    
    def load(self) -> bool:
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
        return self.synapses.get(synset_id, [])
    
    def has_synset(self, synset_id: str) -> bool:
        return synset_id in self.synapses
    
    def get_file_hash(self) -> str:
        return self._file_hash or "not_loaded"


class GlossaryLoader:
    """Load and manage glossary data."""
    
    def __init__(self, filepath: str = GLOSSARY_FILE):
        self.filepath = filepath
        self.glossary: Dict[str, Any] = {}
        self._loaded = False
        self._file_hash: Optional[str] = None
    
    def load(self) -> bool:
        if not os.path.exists(self.filepath):
            print(f"[GlossaryLoader] File not found: {self.filepath}")
            return False
        try:
            self._file_hash = compute_file_hash(self.filepath)
            with open(self.filepath, 'r', encoding='utf-8') as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                if "glossary" in raw:
                    glossary = raw["glossary"]
                    if isinstance(glossary, dict):
                        self.glossary = glossary
                    elif isinstance(glossary, list):
                        self.glossary = {
                            item["concept_id"]: item for item in glossary
                            if isinstance(item, dict) and "concept_id" in item
                        }
                else:
                    self.glossary = raw
            elif isinstance(raw, list):
                self.glossary = {
                    item["concept_id"]: item for item in raw
                    if isinstance(item, dict) and "concept_id" in item
                }
            self._loaded = True
            print(f"[GlossaryLoader] Loaded {len(self.glossary)} concepts")
            return True
        except Exception as e:
            print(f"[GlossaryLoader] Error: {e}")
            return False
    
    def get_concept(self, concept_id: str) -> Optional[Dict]:
        return self.glossary.get(concept_id)
    
    def get_file_hash(self) -> str:
        return self._file_hash or "not_loaded"
