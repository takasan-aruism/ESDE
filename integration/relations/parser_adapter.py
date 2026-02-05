"""
ESDE Integration - Parser Adapter
==================================
Extracts SVO (Subject-Verb-Object) triples from text using spaCy
dependency parsing. Deterministic. No LLM.

Design Principle: "Describe, but do not decide."
  - Extracts structural facts only (who did what to whom)
  - Does not interpret meaning
  - Handles passive voice, negation, conjunctions
  - Returns raw triples for downstream grounding

Spec: Phase 8 Integration Design Brief, Step 1
3AI Agreement: Operator ▷ (ACT) only for initial prototype.

Usage:
    adapter = ParserAdapter()
    triples = adapter.extract_svo("Nobunaga attacked the Azai clan.")
    # [SVOTriple(subject="Nobunaga", verb="attacked", object="Azai clan",
    #            verb_lemma="attack", negated=False, passive=False)]
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple

try:
    import spacy
    from spacy.tokens import Doc, Span, Token
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False


# ==========================================
# Data Structures
# ==========================================

@dataclass
class SVOTriple:
    """
    A single Subject-Verb-Object relation extracted from text.
    
    All fields are raw text. Semantic grounding (Atom mapping)
    happens downstream in relation_logger.py.
    """
    subject: str               # Raw subject text (e.g., "Nobunaga")
    verb: str                  # Raw verb text (e.g., "attacked")
    object: str                # Raw object text (e.g., "the Azai clan")
    verb_lemma: str            # Lemmatized verb (e.g., "attack")
    negated: bool = False      # Whether the verb is negated
    passive: bool = False      # Whether the construction is passive
    
    # Source tracking
    sentence_text: str = ""    # Full sentence for audit
    sentence_idx: int = 0      # Sentence index within section
    subject_char_span: Tuple[int, int] = (0, 0)  # Character offsets in sentence
    object_char_span: Tuple[int, int] = (0, 0)
    verb_char_span: Tuple[int, int] = (0, 0)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "subject": self.subject,
            "verb": self.verb,
            "object": self.object,
            "verb_lemma": self.verb_lemma,
            "negated": self.negated,
            "passive": self.passive,
            "sentence_text": self.sentence_text,
            "sentence_idx": self.sentence_idx,
            "subject_char_span": list(self.subject_char_span),
            "object_char_span": list(self.object_char_span),
            "verb_char_span": list(self.verb_char_span),
        }


@dataclass
class ExtractionResult:
    """Result of SVO extraction for a section of text."""
    triples: List[SVOTriple]
    sentences_processed: int = 0
    sentences_with_triples: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "triples": [t.to_dict() for t in self.triples],
            "sentences_processed": self.sentences_processed,
            "sentences_with_triples": self.sentences_with_triples,
        }


# ==========================================
# Negation Detection
# ==========================================

# spaCy dependency labels that indicate negation
NEGATION_DEPS = {"neg"}

# Tokens that indicate negation even without dep label
NEGATION_TOKENS = {"not", "never", "no", "neither", "nor", "n't", "cannot"}


def _is_verb_negated(verb_token: "Token") -> bool:
    """
    Check if a verb token is negated.
    
    Checks:
      1. Direct negation child (dep=neg)
      2. Negation tokens in children
    """
    for child in verb_token.children:
        if child.dep_ in NEGATION_DEPS:
            return True
        if child.text.lower() in NEGATION_TOKENS:
            return True
    return False


# ==========================================
# Subtree Text Extraction
# ==========================================

def _get_span_text(token: "Token", doc: "Doc") -> str:
    """
    Get the full text of a token's subtree, preserving word order.
    
    For compound subjects/objects like "the Azai clan" or 
    "Oda Nobunaga", this captures the full noun phrase.
    """
    subtree = sorted(token.subtree, key=lambda t: t.i)
    
    # Filter out tokens that belong to relative clauses or other verb-attached subtrees
    # Keep only the nominal phrase, not deeply nested clauses
    filtered = []
    for t in subtree:
        # Stop at clause boundaries (verbs that aren't part of the noun phrase)
        if t.pos_ == "VERB" and t != token and t.dep_ not in ("amod", "compound"):
            continue
        # Skip subordinate conjunctions and relative pronouns that start clauses
        if t.dep_ in ("mark", "relcl", "advcl", "ccomp", "xcomp") and t != token:
            continue
        filtered.append(t)
    
    if not filtered:
        return token.text
    
    return " ".join(t.text for t in filtered)


def _get_entity_text(token: "Token", doc: "Doc") -> str:
    """
    Get the best text representation for a subject/object token.
    
    Priority:
      1. Named entity span (if token is part of one)
      2. Noun chunk (if token is root of one)
      3. Compound + token (for compound nouns)
      4. Token text alone
    """
    # Check if part of a named entity
    if token.ent_type_:
        # Find the full entity span
        for ent in doc.ents:
            if token.i >= ent.start and token.i < ent.end:
                return ent.text
    
    # Check noun chunks
    for chunk in doc.noun_chunks:
        if token.i >= chunk.start and token.i < chunk.end:
            # Return chunk but strip leading determiners for cleaner entities
            text = chunk.text
            # Optionally strip leading "the", "a", "an" for entity names
            # But keep them for non-entity phrases
            if token.ent_type_:
                for prefix in ("the ", "a ", "an "):
                    if text.lower().startswith(prefix):
                        text = text[len(prefix):]
                        break
            return text
    
    # Fallback: compound nouns
    compounds = [child for child in token.children if child.dep_ == "compound"]
    if compounds:
        parts = sorted(compounds + [token], key=lambda t: t.i)
        return " ".join(t.text for t in parts)
    
    return token.text


def _get_char_span(token: "Token", doc: "Doc", text: str) -> Tuple[int, int]:
    """Get character span for entity text within sentence."""
    entity_text = text
    sent_start = token.sent.start_char
    
    # Try to find the text in the sentence
    sent_text = token.sent.text
    idx = sent_text.find(entity_text)
    if idx >= 0:
        return (sent_start + idx, sent_start + idx + len(entity_text))
    
    # Fallback to token's own span
    return (token.idx, token.idx + len(token.text))


# ==========================================
# SVO Extraction Core
# ==========================================

def _extract_svo_from_sentence(sent: "Span", doc: "Doc") -> List[SVOTriple]:
    """
    Extract SVO triples from a single spaCy Sentence span.
    
    Handles:
      - Active voice: nsubj → VERB → dobj
      - Passive voice: nsubjpass → VERB (by → agent) → pobj
      - Conjunction subjects/objects: conj dependencies
      - Negation: neg dependency or negation tokens
    
    Does NOT handle (deferred to priority 2):
      - Relative clauses (relcl)
      - Clausal complements (ccomp, xcomp)
      - Prepositional objects beyond "by" agent in passive
    """
    triples = []
    
    for token in sent:
        # Only process main verbs (not auxiliaries or particles)
        if token.pos_ not in ("VERB",):
            continue
        if token.dep_ in ("aux", "auxpass"):
            continue
        
        # Detect negation
        negated = _is_verb_negated(token)
        
        # Collect subjects and objects
        subjects = []
        objects = []
        passive = False
        
        for child in token.children:
            dep = child.dep_
            
            # Active subject
            if dep == "nsubj":
                subjects.append(child)
                # Also check for conjunct subjects ("A and B attacked C")
                for conj in child.children:
                    if conj.dep_ == "conj":
                        subjects.append(conj)
            
            # Passive subject (the entity being acted upon)
            elif dep == "nsubjpass":
                # In passive, the grammatical subject is actually the patient
                objects.append(child)
                passive = True
                for conj in child.children:
                    if conj.dep_ == "conj":
                        objects.append(conj)
            
            # Direct object
            elif dep == "dobj":
                objects.append(child)
                for conj in child.children:
                    if conj.dep_ == "conj":
                        objects.append(conj)
            
            # "by" agent in passive construction
            elif dep == "agent":
                # "defeated by Nobunaga" → agent = Nobunaga
                for pobj in child.children:
                    if pobj.dep_ == "pobj":
                        subjects.append(pobj)
                        for conj in pobj.children:
                            if conj.dep_ == "conj":
                                subjects.append(conj)
        
        # Generate triples for all subject × object combinations
        if subjects and objects:
            for subj in subjects:
                subj_text = _get_entity_text(subj, doc)
                subj_span = _get_char_span(subj, doc, subj_text)
                
                for obj in objects:
                    obj_text = _get_entity_text(obj, doc)
                    obj_span = _get_char_span(obj, doc, obj_text)
                    
                    triple = SVOTriple(
                        subject=subj_text,
                        verb=token.text,
                        object=obj_text,
                        verb_lemma=token.lemma_,
                        negated=negated,
                        passive=passive,
                        sentence_text=sent.text,
                        sentence_idx=0,  # Set by caller
                        subject_char_span=subj_span,
                        object_char_span=obj_span,
                        verb_char_span=(token.idx, token.idx + len(token.text)),
                    )
                    triples.append(triple)
        
        # Intransitive with subject only (no object) → skip for now
        # Design decision: ▷ only, need both S and O
    
    return triples


# ==========================================
# Parser Adapter (Public API)
# ==========================================

class ParserAdapter:
    """
    Extracts SVO triples from text using spaCy dependency parsing.
    
    Deterministic. No LLM. Parser-exchangeable layer.
    
    Usage:
        adapter = ParserAdapter()
        result = adapter.extract("Nobunaga attacked the Azai.", section_name="battle")
    """
    
    def __init__(self, spacy_model: str = "en_core_web_sm"):
        """
        Initialize with spaCy model.
        
        Args:
            spacy_model: Model name. Must support dependency parsing.
        """
        if not SPACY_AVAILABLE:
            raise RuntimeError(
                "spaCy is required for ParserAdapter. "
                "Install: pip install spacy && python -m spacy download en_core_web_sm"
            )
        
        self.spacy_model_name = spacy_model
        self._nlp = None
        self._stats = {
            "texts_processed": 0,
            "sentences_processed": 0,
            "triples_extracted": 0,
        }
    
    @property
    def nlp(self):
        """Lazy load spaCy model."""
        if self._nlp is None:
            try:
                self._nlp = spacy.load(self.spacy_model_name)
            except OSError:
                raise RuntimeError(
                    f"spaCy model '{self.spacy_model_name}' not found. "
                    f"Run: python -m spacy download {self.spacy_model_name}"
                )
        return self._nlp
    
    def extract(self, text: str, section_name: str = "") -> ExtractionResult:
        """
        Extract SVO triples from text.
        
        Args:
            text: Input text (can be multi-sentence).
            section_name: Optional section identifier for logging.
        
        Returns:
            ExtractionResult with list of SVOTriple.
        """
        if not text or not text.strip():
            return ExtractionResult(triples=[], sentences_processed=0)
        
        self._stats["texts_processed"] += 1
        
        doc = self.nlp(text)
        all_triples = []
        sentences_with_triples = 0
        
        for sent_idx, sent in enumerate(doc.sents):
            self._stats["sentences_processed"] += 1
            
            triples = _extract_svo_from_sentence(sent, doc)
            
            # Set sentence index on all triples
            for t in triples:
                t.sentence_idx = sent_idx
            
            if triples:
                sentences_with_triples += 1
                all_triples.extend(triples)
        
        self._stats["triples_extracted"] += len(all_triples)
        
        return ExtractionResult(
            triples=all_triples,
            sentences_processed=len(list(doc.sents)),
            sentences_with_triples=sentences_with_triples,
        )
    
    def extract_batch(
        self,
        sections: List[Dict[str, str]],
    ) -> Dict[str, ExtractionResult]:
        """
        Extract SVO triples from multiple sections.
        
        Args:
            sections: List of {"title": "...", "content": "..."} dicts.
        
        Returns:
            {section_title: ExtractionResult}
        """
        results = {}
        for sec in sections:
            title = sec.get("title", "untitled")
            content = sec.get("content", "")
            results[title] = self.extract(content, section_name=title)
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Return extraction statistics."""
        return {
            "spacy_model": self.spacy_model_name,
            **self._stats,
        }


# ==========================================
# Quick Test
# ==========================================

if __name__ == "__main__":
    print("=" * 60)
    print("ESDE Parser Adapter - SVO Extraction Test")
    print("=" * 60)
    
    adapter = ParserAdapter()
    
    test_sentences = [
        # Basic SVO
        "Nobunaga attacked the Azai clan.",
        # Passive voice
        "The Azai were defeated by Nobunaga.",
        # Negation
        "Tokugawa did not betray the alliance.",
        # Multiple objects
        "Nobunaga conquered Mino and Omi provinces.",
        # Conjunction subjects
        "Nobunaga and Tokugawa formed an alliance.",
        # Complex
        "After the battle, Hideyoshi seized control of the government.",
        # Intransitive (should produce no triple)
        "The empire collapsed.",
        # Passive without agent
        "The castle was destroyed.",
    ]
    
    for text in test_sentences:
        print(f"\n  Input: \"{text}\"")
        result = adapter.extract(text)
        
        if result.triples:
            for t in result.triples:
                neg = " [NEG]" if t.negated else ""
                pas = " [PASSIVE]" if t.passive else ""
                print(f"    → {t.subject} ▷ {t.verb_lemma}{neg}{pas} ▷ {t.object}")
        else:
            print(f"    → (no SVO triple extracted)")
    
    print(f"\n  Stats: {adapter.get_stats()}")
