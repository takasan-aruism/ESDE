"""
ESDE Phase 9: N-gram Collector
===============================
Collects bigram/trigram statistics as a separate data stream.

Philosophy: "Describe, but do not decide."

This module is SEPARATE from the 20-dim feature vector.
N-grams capture "structural patterns" (syntactic templates)
rather than "token properties".

Spec: Phase 9 W1 Feature Extraction v1.0
"""

import json
from collections import Counter
from dataclasses import dataclass
from typing import List, Dict, Optional, Any, Iterator, Tuple, TYPE_CHECKING
from pathlib import Path

if TYPE_CHECKING:
    from .feature_extractor import TokenFeature


# ==========================================
# Constants
# ==========================================

NGRAM_COLLECTOR_VERSION = "v1.0.0"

NGRAM_SEP = "\t"
BOS = "<BOS>"
EOS = "<EOS>"


# ==========================================
# N-gram Record
# ==========================================

@dataclass
class NgramRecord:
    """Statistics for a single n-gram pattern."""
    ngram_id: str
    ngram_type: str
    tokens: Tuple[str, ...]
    count: int = 0
    pos_mean: float = 0.0
    pos_std: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "ngram_id": self.ngram_id,
            "type": self.ngram_type,
            "tokens": list(self.tokens),
            "count": self.count,
            "pos_mean": round(self.pos_mean, 4),
            "pos_std": round(self.pos_std, 4),
        }
    
    def to_jsonl(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, separators=(',', ':'))
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NgramRecord":
        return cls(
            ngram_id=data["ngram_id"],
            ngram_type=data["type"],
            tokens=tuple(data["tokens"]),
            count=data.get("count", 0),
            pos_mean=data.get("pos_mean", 0.0),
            pos_std=data.get("pos_std", 0.0),
        )


# ==========================================
# N-gram Collector
# ==========================================

class NgramCollector:
    """
    Collects n-gram statistics from tokenized text.
    
    Usage:
        collector = NgramCollector()
        for sent_tokens in sentences:
            collector.process_sentence(sent_tokens)
        
        bigrams = collector.get_bigram_stats()
    """
    
    def __init__(
        self,
        lowercase: bool = True,
        include_boundaries: bool = True,
    ):
        self.lowercase = lowercase
        self.include_boundaries = include_boundaries
        
        self._bigram_counts: Counter = Counter()
        self._trigram_counts: Counter = Counter()
        self._bigram_positions: Dict[str, List[float]] = {}
        self._trigram_positions: Dict[str, List[float]] = {}
        
        self._stats = {
            "sentences_processed": 0,
            "tokens_processed": 0,
        }
    
    def _normalize(self, token: str) -> str:
        if self.lowercase:
            return token.lower()
        return token
    
    def _make_ngram_id(self, tokens: Tuple[str, ...]) -> str:
        return NGRAM_SEP.join(tokens)
    
    def process_sentence(
        self,
        tokens: List[str],
        positions: Optional[List[float]] = None,
    ) -> None:
        if not tokens:
            return
        
        self._stats["sentences_processed"] += 1
        self._stats["tokens_processed"] += len(tokens)
        
        normalized = [self._normalize(t) for t in tokens]
        
        if self.include_boundaries:
            normalized = [BOS] + normalized + [EOS]
            if positions:
                positions = [0.0] + positions + [1.0]
        
        # Bigrams
        for i in range(len(normalized) - 1):
            bigram = (normalized[i], normalized[i + 1])
            ngram_id = self._make_ngram_id(bigram)
            self._bigram_counts[ngram_id] += 1
            
            if positions:
                pos = positions[i] if i < len(positions) else 0.5
                if ngram_id not in self._bigram_positions:
                    self._bigram_positions[ngram_id] = []
                self._bigram_positions[ngram_id].append(pos)
        
        # Trigrams
        for i in range(len(normalized) - 2):
            trigram = (normalized[i], normalized[i + 1], normalized[i + 2])
            ngram_id = self._make_ngram_id(trigram)
            self._trigram_counts[ngram_id] += 1
            
            if positions:
                pos = positions[i + 1] if i + 1 < len(positions) else 0.5
                if ngram_id not in self._trigram_positions:
                    self._trigram_positions[ngram_id] = []
                self._trigram_positions[ngram_id].append(pos)
    
    def process_tokens_with_features(
        self,
        token_features: List["TokenFeature"],
    ) -> None:
        """Process from TokenFeature list (from feature_extractor)."""
        sentences: Dict[int, List[Tuple[str, float]]] = {}
        
        for feat in token_features:
            sent_idx = feat.sentence_idx
            if sent_idx not in sentences:
                sentences[sent_idx] = []
            pos = feat.vector[0] if feat.vector else 0.5
            sentences[sent_idx].append((feat.token, pos))
        
        for sent_idx in sorted(sentences.keys()):
            items = sentences[sent_idx]
            tokens = [t for t, p in items]
            positions = [p for t, p in items]
            self.process_sentence(tokens, positions)
    
    def _compute_stats(self, positions: List[float]) -> Tuple[float, float]:
        if not positions:
            return 0.0, 0.0
        n = len(positions)
        mean = sum(positions) / n
        if n < 2:
            return mean, 0.0
        variance = sum((p - mean) ** 2 for p in positions) / (n - 1)
        std = variance ** 0.5
        return mean, std
    
    def get_bigram_stats(
        self,
        min_count: int = 1,
        top_k: Optional[int] = None,
    ) -> List[NgramRecord]:
        records = []
        for ngram_id, count in self._bigram_counts.most_common():
            if count < min_count:
                continue
            tokens = tuple(ngram_id.split(NGRAM_SEP))
            positions = self._bigram_positions.get(ngram_id, [])
            mean, std = self._compute_stats(positions)
            record = NgramRecord(
                ngram_id=ngram_id,
                ngram_type="bigram",
                tokens=tokens,
                count=count,
                pos_mean=mean,
                pos_std=std,
            )
            records.append(record)
            if top_k and len(records) >= top_k:
                break
        return records
    
    def get_trigram_stats(
        self,
        min_count: int = 1,
        top_k: Optional[int] = None,
    ) -> List[NgramRecord]:
        records = []
        for ngram_id, count in self._trigram_counts.most_common():
            if count < min_count:
                continue
            tokens = tuple(ngram_id.split(NGRAM_SEP))
            positions = self._trigram_positions.get(ngram_id, [])
            mean, std = self._compute_stats(positions)
            record = NgramRecord(
                ngram_id=ngram_id,
                ngram_type="trigram",
                tokens=tokens,
                count=count,
                pos_mean=mean,
                pos_std=std,
            )
            records.append(record)
            if top_k and len(records) >= top_k:
                break
        return records
    
    def get_all_stats(self, min_count: int = 1) -> Iterator[NgramRecord]:
        yield from self.get_bigram_stats(min_count)
        yield from self.get_trigram_stats(min_count)
    
    def save(self, path: str, min_count: int = 1) -> int:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with open(output, 'w', encoding='utf-8') as f:
            for record in self.get_all_stats(min_count):
                f.write(record.to_jsonl() + "\n")
                count += 1
        return count
    
    @classmethod
    def load(cls, path: str) -> "NgramCollector":
        collector = cls()
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                record = NgramRecord.from_dict(data)
                if record.ngram_type == "bigram":
                    collector._bigram_counts[record.ngram_id] = record.count
                elif record.ngram_type == "trigram":
                    collector._trigram_counts[record.ngram_id] = record.count
        return collector
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "version": NGRAM_COLLECTOR_VERSION,
            "sentences_processed": self._stats["sentences_processed"],
            "tokens_processed": self._stats["tokens_processed"],
            "unique_bigrams": len(self._bigram_counts),
            "unique_trigrams": len(self._trigram_counts),
            "total_bigram_occurrences": sum(self._bigram_counts.values()),
            "total_trigram_occurrences": sum(self._trigram_counts.values()),
        }
    
    def merge(self, other: "NgramCollector") -> None:
        self._bigram_counts.update(other._bigram_counts)
        self._trigram_counts.update(other._trigram_counts)
        
        for ngram_id, positions in other._bigram_positions.items():
            if ngram_id not in self._bigram_positions:
                self._bigram_positions[ngram_id] = []
            self._bigram_positions[ngram_id].extend(positions)
        
        for ngram_id, positions in other._trigram_positions.items():
            if ngram_id not in self._trigram_positions:
                self._trigram_positions[ngram_id] = []
            self._trigram_positions[ngram_id].extend(positions)


# ==========================================
# Convenience Function
# ==========================================

def collect_ngrams_from_text(text: str, lowercase: bool = True) -> NgramCollector:
    """Quick n-gram collection from raw text."""
    import re
    collector = NgramCollector(lowercase=lowercase)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    for sent in sentences:
        tokens = re.findall(r'\S+', sent)
        if tokens:
            collector.process_sentence(tokens)
    return collector
