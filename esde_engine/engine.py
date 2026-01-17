"""
ESDE Engine v5.3.2 - Main Engine
"""
from datetime import datetime, timezone
from typing import Dict, List, Any

from nltk.corpus import wordnet as wn

from .config import (
    VERSION, SYNAPSE_FILE, GLOSSARY_FILE, MAX_SYNSETS_PER_TOKEN,
    TOP_CONCEPTS, AXIS_TOP_LEVELS, DEDUP_PER_TOKEN_CONCEPT,
    ENABLE_PROXIMITY_EXPANSION, MAX_PROXY_DEPTH, DEBUG_PROXIMITY,
    QUEUE_FILE_PATH, QUEUE_INCLUDE_NOISE, UNKNOWN_MARGIN_TH, UNKNOWN_ENTROPY_TH,
    STOPWORDS, TYPO_MAX_EDIT_DISTANCE, TYPO_MIN_CONFIDENCE, TITLE_LIKE_PHRASES
)
from .utils import ensure_nltk_data, tokenize, find_all_typo_candidates, is_proper_noun_candidate
from .loaders import SynapseLoader, GlossaryLoader
from .extractors import SynsetExtractor
from .collectors import ActivationCollector, OutputGate
from .routing import UnknownTokenRouter
from .queue import UnknownQueueWriter


class ESDEEngine:
    """ESDE Engine with Phase 7A+ Multi-Hypothesis Routing."""
    
    def __init__(self, synapse_file: str = SYNAPSE_FILE,
                 glossary_file: str = GLOSSARY_FILE,
                 max_synsets_per_token: int = MAX_SYNSETS_PER_TOKEN,
                 top_concepts: int = TOP_CONCEPTS,
                 axis_top_levels: int = AXIS_TOP_LEVELS,
                 dedup_per_token: bool = DEDUP_PER_TOKEN_CONCEPT,
                 enable_proximity: bool = ENABLE_PROXIMITY_EXPANSION,
                 max_proxy_depth: int = MAX_PROXY_DEPTH,
                 debug_proximity: bool = DEBUG_PROXIMITY,
                 queue_path: str = QUEUE_FILE_PATH,
                 queue_include_noise: bool = QUEUE_INCLUDE_NOISE):
        self.synapse_file = synapse_file
        self.glossary_file = glossary_file
        self.synapse_loader = SynapseLoader(synapse_file)
        self.glossary_loader = GlossaryLoader(glossary_file)
        self.synset_extractor = SynsetExtractor(max_synsets_per_token)
        self.activation_collector = None
        self.output_gate = OutputGate(top_concepts, axis_top_levels)
        self.dedup = dedup_per_token
        self.enable_proximity = enable_proximity
        self.max_proxy_depth = max_proxy_depth
        self.debug_proximity = debug_proximity
        
        # Phase 7A: Unknown Queue
        self.queue_writer = UnknownQueueWriter(
            queue_path=queue_path,
            include_noise=queue_include_noise
        )
        
        # Phase 7A+: Unknown Token Router
        self.unknown_router = UnknownTokenRouter(
            margin_threshold=UNKNOWN_MARGIN_TH,
            entropy_threshold=UNKNOWN_ENTROPY_TH
        )
        
        self._initialized = False
    
    def initialize(self) -> bool:
        """Initialize the engine."""
        print(f"\n[ESDEEngine v{VERSION}] Initializing...")
        print(f"  Phase 6A: {'ENABLED' if self.enable_proximity else 'DISABLED'}")
        print(f"  Phase 7A+: Multi-Hypothesis Routing ENABLED")
        print(f"  Variance Gate: margin<{UNKNOWN_MARGIN_TH}, entropy>{UNKNOWN_ENTROPY_TH}")
        print(f"  Title-Like Phrases: {len(TITLE_LIKE_PHRASES)} patterns")
        print(f"  Stopword Filter: {len(STOPWORDS)} words (SKIP mode)")
        print(f"  Typo Detection: edit_dist<={TYPO_MAX_EDIT_DISTANCE}, confidence>={TYPO_MIN_CONFIDENCE}")
        print(f"  Queue Path: {self.queue_writer.queue_path}")
        
        ensure_nltk_data()
        
        if not self.synapse_loader.load():
            print("[ESDEEngine] FATAL: Cannot load synapse map")
            return False
        self.glossary_loader.load()
        
        self.activation_collector = ActivationCollector(
            self.synapse_loader, self.dedup, self.enable_proximity,
            self.max_proxy_depth, self.debug_proximity
        )
        
        self._initialized = True
        print("[ESDEEngine] Initialized successfully")
        return True
    
    def _get_config_meta(self) -> Dict[str, Any]:
        """Get configuration metadata for queue records."""
        return {
            "artifacts": {
                "synapse_file": self.synapse_file,
                "glossary_file": self.glossary_file,
                "synapse_hash": self.synapse_loader.get_file_hash(),
                "glossary_hash": self.glossary_loader.get_file_hash()
            },
            "settings": {
                "proximity_enabled": self.enable_proximity,
                "max_proxy_depth": self.max_proxy_depth,
                "unknown_margin_th": UNKNOWN_MARGIN_TH,
                "unknown_entropy_th": UNKNOWN_ENTROPY_TH
            }
        }
    
    def process(self, text: str, include_evidence: bool = True) -> Dict[str, Any]:
        """Process input text and return semantic analysis results."""
        if not self._initialized:
            return {"error": "Engine not initialized"}
        
        start_time = datetime.now(timezone.utc)
        tokens = tokenize(text)
        
        if not tokens:
            return {
                "input": text, "tokens": [], "top_concepts": [], "top_levels": [],
                "routing_queue": {
                    "skipped": [], "typo_candidates": [],
                    "proper_noun_candidates": [], "molecule_candidates": []
                },
                "_meta": {"status": "empty_input", "version": VERSION}
            }
        
        # Start a run for this input
        run_id = self.queue_writer.start_run(self._get_config_meta())
        
        token_synsets, routing_queue = self.synset_extractor.extract(tokens)
        activations = self.activation_collector.collect(token_synsets, routing_queue, tokens)
        gated = self.output_gate.apply(activations)
        
        evidence = activations.get("evidence", [])
        direct_count = sum(1 for e in evidence if e.get("mode") == "direct" and e.get("selected"))
        proxy_count = sum(1 for e in evidence if e.get("mode") == "proxy" and e.get("selected"))
        
        final_routing_queue = activations.get("routing_queue", routing_queue)
        active_token_set = set(token_synsets.keys())
        
        # Collect routing decisions for display
        routing_decisions = []
        
        # Queue unknown records with multi-hypothesis routing
        unknown_records = activations.get("unknown_records", [])
        for ur in unknown_records:
            # Build wn_synsets in proper format
            wn_synsets = []
            synset_ids = ur.get("wordnet_synsets", [])
            for sid in synset_ids:
                try:
                    pos = wn.synset(sid).pos() if sid else "u"
                except:
                    pos = "u"
                wn_synsets.append({"id": sid, "pos": pos})
            
            # Get typo candidates for routing
            typo_candidates = find_all_typo_candidates(ur["token"])
            
            # Evaluate with multi-hypothesis router
            routing_report = self.unknown_router.evaluate(
                token=ur["token"],
                original_text=text,
                tokens=tokens,
                token_index=ur["token_index"],
                synset_ids=synset_ids,
                has_direct_edges=ur.get("direct_edges_found", 0) > 0,
                has_proxy_edges=ur.get("proxy_edges_found", 0) > 0,
                typo_candidates=typo_candidates
            )
            
            # Extract decision
            decision = routing_report.get("decision", {})
            route = decision.get("winner") or "C"
            
            routing_decisions.append({
                "token": ur["token"],
                "action": decision.get("action"),
                "winner": decision.get("winner"),
                "margin": decision.get("margin"),
                "entropy": decision.get("entropy"),
                "reason": decision.get("reason"),
                "hypotheses": {h["route"]: h["score"] for h in routing_report.get("hypotheses", [])}
            })
            
            record = self.queue_writer.create_record(
                input_text=text,
                token=ur["token"],
                token_raw=ur["token"],
                token_index=ur["token_index"],
                reason=ur["reason"],
                route=route,
                context_window=tokens,
                pos_guess=ur.get("pos", "u"),
                is_active=ur["token"] in active_token_set,
                wn_synsets=wn_synsets,
                wn_selected=None,
                has_direct_edges=ur.get("direct_edges_found", 0) > 0,
                has_proxy_edges=ur.get("proxy_edges_found", 0) > 0,
                direct_edge_count=ur.get("direct_edges_found", 0),
                proxy_edge_count=ur.get("proxy_edges_found", 0),
                proxy_evidence=ur.get("proxy_trace", []),
                abstain_reasons=[ur["reason"]],
                typo_candidates=typo_candidates,
                routing_report=routing_report
            )
            self.queue_writer.enqueue(record)
        
        # Queue molecule candidates from extraction phase with routing
        for mc in final_routing_queue.get("molecule_candidates", []):
            if mc.get("reason") == "no_synsets_in_wordnet":
                typo_candidates = find_all_typo_candidates(mc["token"])
                routing_report = self.unknown_router.evaluate(
                    token=mc["token"],
                    original_text=text,
                    tokens=tokens,
                    token_index=mc.get("token_index", -1),
                    synset_ids=[],
                    has_direct_edges=False,
                    has_proxy_edges=False,
                    typo_candidates=typo_candidates
                )
                
                decision = routing_report.get("decision", {})
                routing_decisions.append({
                    "token": mc["token"],
                    "action": decision.get("action"),
                    "winner": decision.get("winner"),
                    "margin": decision.get("margin"),
                    "entropy": decision.get("entropy"),
                    "reason": decision.get("reason"),
                    "hypotheses": {h["route"]: h["score"] for h in routing_report.get("hypotheses", [])}
                })
                
                record = self.queue_writer.create_record(
                    input_text=text,
                    token=mc["token"],
                    token_raw=mc["token"],
                    token_index=mc.get("token_index", -1),
                    reason="no_synsets_in_wordnet",
                    route=decision.get("winner") or "C",
                    context_window=tokens,
                    pos_guess="u",
                    is_active=mc["token"] in active_token_set,
                    wn_synsets=[],
                    abstain_reasons=["no_synsets_in_wordnet"],
                    typo_candidates=typo_candidates,
                    routing_report=routing_report,
                    notes="No WordNet synsets found"
                )
                self.queue_writer.enqueue(record)
        
        # Optionally queue stopwords if configured
        if self.queue_writer.include_noise:
            for sk in final_routing_queue.get("skipped", []):
                record = self.queue_writer.create_record(
                    input_text=text,
                    token=sk["token"],
                    token_raw=sk["token"],
                    token_index=sk.get("token_index", -1),
                    reason="stopword_or_noise",
                    route="D",
                    context_window=tokens,
                    pos_guess="u",
                    is_active=False,
                    abstain_reasons=["stopword_or_noise"]
                )
                self.queue_writer.enqueue(record)
        
        # End run and get summary
        queue_summary = self.queue_writer.end_run()
        
        result = {
            "input": text,
            "tokens": tokens,
            "active_tokens": list(token_synsets.keys()),
            "synsets_found": sum(len(v) for v in token_synsets.values()),
            "top_concepts": gated["top_concepts"],
            "top_levels": gated["top_levels"],
            "routing_queue": final_routing_queue,
            "routing_decisions": routing_decisions,
            "abstain_summary": activations.get("abstain_summary", {}),
            "queue_summary": queue_summary,
            "_meta": {
                "version": VERSION,
                "timestamp": start_time.isoformat(),
                "status": "success",
                "run_id": run_id,
                "direct_hits": direct_count,
                "proxy_hits": proxy_count,
                "skipped_count": len(final_routing_queue.get("skipped", [])),
                "proximity_enabled": self.enable_proximity
            }
        }
        if include_evidence:
            result["evidence"] = evidence
        return result
    
    def process_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Process multiple texts."""
        return [self.process(text) for text in texts]
