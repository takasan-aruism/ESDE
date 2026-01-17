"""
ESDE Engine v5.3.2 - Synset Extraction
"""
from typing import Dict, List, Tuple, Set
from nltk.corpus import wordnet as wn

from .config import MAX_SYNSETS_PER_TOKEN, ALLOWED_POS
from .utils import is_stopword_or_noise, find_typo_candidate


class SynsetExtractor:
    """Token routing with A/B/C/D classification."""
    
    def __init__(self, max_synsets_per_token: int = MAX_SYNSETS_PER_TOKEN,
                 allowed_pos: Set[str] = None):
        self.max_synsets = max_synsets_per_token
        self.allowed_pos = allowed_pos or ALLOWED_POS
    
    def extract(self, tokens: List[str]) -> Tuple[Dict[str, List[str]], Dict[str, List[Dict]]]:
        """
        Extract synsets from tokens with routing classification.
        
        Returns:
            token_synsets: {token: [synset_ids]} for tokens with synsets
            routing_queue: {route: [records]} for A/B/C classification
        """
        result = {}
        routing_queue = {
            "typo_candidates": [],
            "proper_noun_candidates": [],
            "molecule_candidates": [],
            "skipped": []
        }
        
        for idx, token in enumerate(tokens):
            # Route D: Skip entirely
            if is_stopword_or_noise(token):
                routing_queue["skipped"].append({
                    "token": token,
                    "token_index": idx,
                    "reason": "stopword_or_noise"
                })
                continue
            
            # Route A: Check for typo
            typo = find_typo_candidate(token)
            if typo:
                corrected = typo["suggestion"]
                synsets = wn.synsets(corrected)
                synsets = [s for s in synsets if s.pos() in self.allowed_pos]
                synsets = synsets[:self.max_synsets]
                if synsets:
                    result[token] = [s.name() for s in synsets]
                    routing_queue["typo_candidates"].append({
                        "token": token,
                        "token_index": idx,
                        "typo_info": typo,
                        "used_synsets": [s.name() for s in synsets]
                    })
                    continue
            
            # Normal WordNet lookup
            synsets = wn.synsets(token)
            synsets = [s for s in synsets if s.pos() in self.allowed_pos]
            synsets = synsets[:self.max_synsets]
            synset_ids = [s.name() for s in synsets]
            
            if synset_ids:
                result[token] = synset_ids
            else:
                # Route C: No synsets found
                routing_queue["molecule_candidates"].append({
                    "token": token,
                    "token_index": idx,
                    "reason": "no_synsets_in_wordnet"
                })
        
        return result, routing_queue
