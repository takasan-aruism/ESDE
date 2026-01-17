"""
ESDE Sensor - Synset Extractor
Extracts WordNet synsets from tokens.

GPT Audit: ALLOWED_POS must include 's' for Satellite Adjective coverage.
"""
from typing import Dict, List, Set, Optional

import nltk
from nltk.corpus import wordnet as wn


def ensure_nltk_data():
    """Download WordNet if not present."""
    try:
        wn.synsets('test')
    except LookupError:
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)


# Default ALLOWED_POS includes 's' (Satellite Adjective)
DEFAULT_ALLOWED_POS = {'n', 'v', 'a', 'r', 's'}
DEFAULT_MAX_SYNSETS = 3


class SynsetExtractor:
    """
    Extract WordNet synsets from tokens.
    GPT Audit: ALLOWED_POS must include 's' for Satellite Adjective coverage.
    """
    
    def __init__(self, 
                 max_synsets_per_token: int = None,
                 allowed_pos: Set[str] = None):
        self.max_synsets = max_synsets_per_token or DEFAULT_MAX_SYNSETS
        self.allowed_pos = allowed_pos or DEFAULT_ALLOWED_POS
        
        # Ensure 's' is always included
        if 's' not in self.allowed_pos:
            self.allowed_pos = set(self.allowed_pos) | {'s'}
            print(f"[SynsetExtractor] WARNING: Added 's' to ALLOWED_POS")
        
        # Ensure NLTK data
        ensure_nltk_data()
    
    def extract_synsets(self, token: str) -> List[str]:
        """
        Extract synset IDs for a single token.
        
        Returns:
            List of synset IDs (e.g., ['apprenticed.s.01', 'bound.a.01'])
        """
        synsets = wn.synsets(token)
        
        # Filter by POS
        synsets = [s for s in synsets if s.pos() in self.allowed_pos]
        
        # Limit count
        synsets = synsets[:self.max_synsets]
        
        return [s.name() for s in synsets]
    
    def extract_all(self, tokens: List[str]) -> Dict[str, List[str]]:
        """
        Extract synsets for all tokens.
        
        Returns:
            {token: [synset_ids]}
        """
        result = {}
        for token in tokens:
            synset_ids = self.extract_synsets(token)
            if synset_ids:
                result[token] = synset_ids
        return result
    
    def get_synset_definition(self, synset_id: str) -> Optional[str]:
        """Get definition for a synset (for validation/debugging)."""
        try:
            synset = wn.synset(synset_id)
            return synset.definition()
        except Exception:
            return None
