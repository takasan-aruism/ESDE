"""
ESDE Sensor - Legacy Trigger Matcher
Fallback to v1 trigger-based matching when synapse lookup returns no results.
"""
import re
from typing import Dict, List, Any
from collections import defaultdict


class LegacyTriggerMatcher:
    """
    Fallback to legacy trigger-based matching.
    Used in Hybrid mode when synapse lookup returns no results.
    """
    
    def __init__(self, glossary: Dict[str, Any]):
        self.glossary = glossary
        self.trigger_index = self._build_trigger_index()
    
    def _build_trigger_index(self) -> Dict[str, List[str]]:
        """Build index from triggers to concept IDs."""
        index = {}
        
        for concept_id, content in self.glossary.items():
            # Handle nested structure
            final = content.get("final_json", content)
            
            if not isinstance(final, dict):
                continue
            
            # Get triggers from axes
            axes = final.get("axes", {})
            if not isinstance(axes, dict):
                continue
            
            for axis_name, levels in axes.items():
                if not isinstance(levels, dict):
                    continue
                
                for level_name, details in levels.items():
                    if not isinstance(details, dict):
                        continue
                    
                    triggers = details.get("triggers_en", [])
                    for trigger in triggers:
                        key = trigger.lower().strip()
                        if key not in index:
                            index[key] = []
                        if concept_id not in index[key]:
                            index[key].append(concept_id)
        
        return index
    
    def match(self, text: str) -> List[Dict]:
        """
        Match text against triggers.
        
        Returns:
            List of matched concepts with evidence
        """
        text_lower = text.lower()
        words = re.findall(r'[a-z]+', text_lower)
        
        matches = defaultdict(lambda: {"score": 0.0, "evidence": []})
        
        for word in words:
            if word in self.trigger_index:
                for concept_id in self.trigger_index[word]:
                    matches[concept_id]["score"] += 0.5  # Base score for trigger match
                    matches[concept_id]["evidence"].append({
                        "trigger": word,
                        "type": "legacy_trigger"
                    })
        
        result = []
        for concept_id, data in matches.items():
            result.append({
                "concept_id": concept_id,
                "axis": None,  # Unknown from trigger
                "level": None,
                "score": round(data["score"], 8),
                "evidence": data["evidence"]
            })
        
        # Sort by score DESC
        result.sort(key=lambda x: -x["score"])
        
        return result
    
    def get_trigger_count(self) -> int:
        """Get total trigger count (for stats)."""
        return len(self.trigger_index)
