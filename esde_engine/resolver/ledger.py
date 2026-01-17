"""
ESDE Engine v5.3.2 - Phase 7B: Evidence Ledger Management

Records all resolution decisions with full audit trail.
"""
import json
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
from pathlib import Path


class EvidenceLedger:
    """
    Manages evidence_ledger.jsonl for audit trail.
    
    Each entry records:
    - What was resolved (token, context)
    - What sources were consulted (local/web/llm)
    - What decision was made (route, action)
    - Confidence metrics (reliability, volatility)
    - Why (evidence, reasoning)
    """
    
    LEDGER_FILE = "./data/evidence_ledger.jsonl"
    
    # Source types
    SOURCE_LOCAL = "local"       # Local dictionary/WordNet
    SOURCE_WEB = "web"           # Web search
    SOURCE_LLM = "llm"           # LLM inference
    SOURCE_HUMAN = "human"       # Human annotation
    SOURCE_FREQUENCY = "frequency"  # Frequency-based
    
    # Action types
    ACTION_ALIAS_ADD = "alias_add"
    ACTION_SYNAPSE_ADD = "synapse_add"
    ACTION_STOPWORD_ADD = "stopword_add"
    ACTION_MOLECULE_DRAFT = "molecule_draft"
    ACTION_QUARANTINE = "quarantine"
    ACTION_IGNORE = "ignore"
    ACTION_DEFER = "defer"
    
    def __init__(self, ledger_file: str = None):
        self.ledger_file = ledger_file or self.LEDGER_FILE
        self._buffer: List[Dict[str, Any]] = []
        self._ensure_directory()
    
    def _ensure_directory(self):
        """Create ledger directory if needed."""
        Path(self.ledger_file).parent.mkdir(parents=True, exist_ok=True)
    
    def create_entry(self,
                     # Identity
                     token_norm: str,
                     token_raw: str,
                     hash_key: str,
                     
                     # Context
                     input_text: str,
                     context_window: List[str],
                     
                     # Resolution
                     route_winner: str,
                     action: str,
                     
                     # Scores
                     hypotheses: Dict[str, float],
                     margin: float,
                     entropy: float,
                     
                     # Confidence
                     reliability: float,
                     volatility: float,
                     
                     # Sources
                     sources: List[Dict[str, Any]],
                     
                     # Output
                     patch_type: Optional[str] = None,
                     patch_data: Optional[Dict[str, Any]] = None,
                     
                     # Notes
                     reasoning: str = "",
                     notes: str = "") -> Dict[str, Any]:
        """
        Create a ledger entry for a resolution decision.
        """
        now = datetime.now(timezone.utc)
        
        entry = {
            # Metadata
            "timestamp": now.isoformat(),
            "ledger_version": "1.0",
            
            # Identity
            "token_norm": token_norm,
            "token_raw": token_raw,
            "hash_key": hash_key,
            
            # Context
            "input_text": input_text,
            "context_window": context_window,
            
            # Resolution decision
            "route_winner": route_winner,
            "action": action,
            
            # Hypothesis scores (from 7A+)
            "hypotheses": hypotheses,
            "margin": round(margin, 4),
            "entropy": round(entropy, 4),
            
            # Confidence metrics (separate!)
            "reliability": round(reliability, 4),  # Evidence quality
            "volatility": round(volatility, 4),    # Concept stability
            
            # Sources consulted
            "sources": sources,
            
            # Patch output (if any)
            "patch_type": patch_type,
            "patch_data": patch_data,
            
            # Human-readable reasoning
            "reasoning": reasoning,
            "notes": notes,
            
            # Flags
            "is_auto_approved": reliability >= 0.75 and volatility < 0.6,
            "needs_review": reliability < 0.75 or volatility >= 0.6
        }
        
        return entry
    
    def add(self, entry: Dict[str, Any]) -> None:
        """Add entry to buffer."""
        self._buffer.append(entry)
    
    def flush(self) -> int:
        """Write buffered entries to file. Returns count written."""
        if not self._buffer:
            return 0
        
        try:
            with open(self.ledger_file, 'a', encoding='utf-8') as f:
                for entry in self._buffer:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
            count = len(self._buffer)
            self._buffer.clear()
            return count
        except Exception as e:
            print(f"[EvidenceLedger] Flush error: {e}")
            return 0
    
    def read_all(self) -> List[Dict[str, Any]]:
        """Read all entries from ledger file."""
        entries = []
        
        if not Path(self.ledger_file).exists():
            return entries
        
        try:
            with open(self.ledger_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entries.append(json.loads(line))
        except Exception as e:
            print(f"[EvidenceLedger] Read error: {e}")
        
        return entries
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics from ledger."""
        entries = self.read_all()
        
        stats = {
            "total_entries": len(entries),
            "by_action": {},
            "by_route": {},
            "auto_approved": 0,
            "needs_review": 0,
            "avg_reliability": 0.0,
            "avg_volatility": 0.0
        }
        
        if not entries:
            return stats
        
        total_reliability = 0.0
        total_volatility = 0.0
        
        for entry in entries:
            action = entry.get("action", "unknown")
            route = entry.get("route_winner", "unknown")
            
            stats["by_action"][action] = stats["by_action"].get(action, 0) + 1
            stats["by_route"][route] = stats["by_route"].get(route, 0) + 1
            
            if entry.get("is_auto_approved"):
                stats["auto_approved"] += 1
            if entry.get("needs_review"):
                stats["needs_review"] += 1
            
            total_reliability += entry.get("reliability", 0)
            total_volatility += entry.get("volatility", 0)
        
        stats["avg_reliability"] = round(total_reliability / len(entries), 4)
        stats["avg_volatility"] = round(total_volatility / len(entries), 4)
        
        return stats


def compute_reliability(sources: List[Dict[str, Any]], 
                        consistency: float = 1.0) -> float:
    """
    Compute reliability score from sources.
    
    Factors:
    - Source quality (local > web > llm)
    - Number of sources agreeing
    - Consistency of information
    """
    if not sources:
        return 0.0
    
    # Source quality weights
    quality_weights = {
        EvidenceLedger.SOURCE_LOCAL: 0.9,
        EvidenceLedger.SOURCE_FREQUENCY: 0.7,
        EvidenceLedger.SOURCE_WEB: 0.6,
        EvidenceLedger.SOURCE_LLM: 0.5,
        EvidenceLedger.SOURCE_HUMAN: 1.0
    }
    
    # Average source quality
    total_quality = 0.0
    for src in sources:
        src_type = src.get("type", "unknown")
        total_quality += quality_weights.get(src_type, 0.3)
    
    avg_quality = total_quality / len(sources)
    
    # Multi-source bonus (more sources = more reliable)
    source_bonus = min(1.0, 0.5 + (len(sources) * 0.1))
    
    # Combine with consistency
    reliability = avg_quality * source_bonus * consistency
    
    return min(1.0, reliability)


def compute_volatility(token: str, 
                       sources: List[Dict[str, Any]],
                       is_proper_noun: bool = False,
                       is_slang: bool = False,
                       is_trending: bool = False,
                       source_disagreement: float = 0.0) -> float:
    """
    Compute volatility score (how unstable the concept is).
    
    High volatility = concept may change meaning over time.
    """
    volatility = 0.0
    
    # Proper nouns are somewhat volatile (people change roles)
    if is_proper_noun:
        volatility += 0.3
    
    # Slang is highly volatile
    if is_slang:
        volatility += 0.5
    
    # Trending topics are volatile
    if is_trending:
        volatility += 0.4
    
    # Source disagreement indicates instability
    volatility += source_disagreement * 0.5
    
    # Short tokens tend to be more ambiguous
    if len(token) <= 3:
        volatility += 0.2
    
    # LLM-only sources are less stable
    if sources and all(s.get("type") == EvidenceLedger.SOURCE_LLM for s in sources):
        volatility += 0.3
    
    return min(1.0, volatility)
