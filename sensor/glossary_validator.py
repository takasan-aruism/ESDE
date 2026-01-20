"""
ESDE Sensor - Glossary Validator
Validates atoms and coordinates against Glossary definitions.

This is a neutral module used by both canonical (v8.3) and legacy validators.
No dependency on legacy modules.
"""
from typing import Dict, List, Any, Set


class GlossaryValidator:
    """
    Validates atoms and coordinates against Glossary definitions.
    
    Responsibilities:
      - Check if atom exists in glossary
      - Check if axis is valid for a given atom
      - Check if level is valid for a given atom+axis
      - Extract glossary subset for LLM context
    """
    
    def __init__(self, glossary: Dict[str, Any]):
        """
        Initialize with glossary data.
        
        Args:
            glossary: Glossary dict. Supports both flat and nested formats:
                      - Flat: {concept_id: {final_json: {...}}}
                      - Nested: {glossary: {concept_id: {...}}}
        """
        # Handle nested glossary structure (glossary_results.json format)
        if "glossary" in glossary and isinstance(glossary["glossary"], dict):
            self.glossary = glossary["glossary"]
        else:
            self.glossary = glossary
        self._build_indexes()
    
    def _build_indexes(self):
        """Build lookup indexes for fast validation."""
        self.valid_atoms: Set[str] = set()
        self.atom_axes: Dict[str, Set[str]] = {}  # atom -> set of valid axes
        self.axis_levels: Dict[str, Dict[str, Set[str]]] = {}  # atom -> axis -> set of valid levels
        
        for concept_id, content in self.glossary.items():
            # Skip non-dict entries
            if not isinstance(content, dict):
                continue
            
            self.valid_atoms.add(concept_id)
            
            # Handle both direct and final_json wrapped formats
            final = content.get("final_json", content)
            if not isinstance(final, dict):
                continue
            
            axes = final.get("axes", {})
            if not isinstance(axes, dict):
                continue
            
            self.atom_axes[concept_id] = set()
            self.axis_levels[concept_id] = {}
            
            for axis_name, levels in axes.items():
                self.atom_axes[concept_id].add(axis_name)
                
                if isinstance(levels, dict):
                    self.axis_levels[concept_id][axis_name] = set(levels.keys())
    
    def is_valid_atom(self, atom: str) -> bool:
        """Check if atom exists in glossary."""
        return atom in self.valid_atoms
    
    def is_valid_axis(self, atom: str, axis: str) -> bool:
        """Check if axis exists for this atom."""
        if atom not in self.atom_axes:
            return False
        return axis in self.atom_axes[atom]
    
    def is_valid_level(self, atom: str, axis: str, level: str) -> bool:
        """Check if level exists for this atom+axis."""
        if atom not in self.axis_levels:
            return False
        if axis not in self.axis_levels[atom]:
            return False
        return level in self.axis_levels[atom][axis]
    
    def get_valid_axes(self, atom: str) -> Set[str]:
        """Get valid axes for an atom."""
        return self.atom_axes.get(atom, set())
    
    def get_valid_levels(self, atom: str, axis: str) -> Set[str]:
        """Get valid levels for an atom+axis."""
        return self.axis_levels.get(atom, {}).get(axis, set())
    
    def get_glossary_subset(self, atoms: List[str]) -> Dict[str, Any]:
        """
        Extract glossary subset for given atoms.
        Used to create lightweight context for LLM.
        
        Args:
            atoms: List of concept_ids to extract
            
        Returns:
            Dict with atom definitions (axes only, for brevity)
        """
        subset = {}
        for atom in atoms:
            if atom in self.glossary:
                content = self.glossary[atom]
                final = content.get("final_json", content)
                if isinstance(final, dict) and "axes" in final:
                    subset[atom] = {"axes": final["axes"]}
        return subset
    
    def get_atom_count(self) -> int:
        """Get total number of valid atoms."""
        return len(self.valid_atoms)
