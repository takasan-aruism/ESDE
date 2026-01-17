"""
ESDE Phase 8-8: Core Pipeline
=============================

End-to-End統合パイプライン。

Sensor → Index(Read) → Modulator → Generator → Ledger(Write) → Index(Update)

Spec v8.8:
- 既存モジュールを接続してループを閉じる
- Ledgerスキーマは変更しない
- 既存テストとの互換性を維持
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime, timezone

from feedback import (
    Modulator,
    GenerationStrategy,
    StrategyMode,
    get_target_atom_from_candidates,
    STRATEGY_NEUTRAL,
)


# ==========================================
# Logger
# ==========================================
logger = logging.getLogger("esde.pipeline")


# ==========================================
# Pipeline Result
# ==========================================
@dataclass
class PipelineResult:
    """パイプライン実行結果"""
    success: bool
    input_text: str
    
    # Sensor
    candidates: List[Dict] = field(default_factory=list)
    target_atom: Optional[str] = None
    
    # Modulator
    strategy: Optional[GenerationStrategy] = None
    rigidity: Optional[float] = None
    
    # Generator
    molecule: Optional[Dict] = None
    generation_error: Optional[str] = None
    abstained: bool = False
    
    # Ledger
    ledger_entry: Optional[Dict] = None
    
    # Index
    index_updated: bool = False
    
    # Alert
    alert: Optional[Dict] = None
    
    # Meta
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    def to_dict(self) -> Dict:
        return {
            "success": self.success,
            "input_text": self.input_text,
            "candidates_count": len(self.candidates),
            "target_atom": self.target_atom,
            "strategy": self.strategy.to_dict() if self.strategy else None,
            "rigidity": self.rigidity,
            "molecule": self.molecule,
            "generation_error": self.generation_error,
            "abstained": self.abstained,
            "ledger_entry_seq": self.ledger_entry.get("seq") if self.ledger_entry else None,
            "index_updated": self.index_updated,
            "alert": self.alert,
            "timestamp": self.timestamp
        }


# ==========================================
# Modulated Generator Wrapper
# ==========================================
class ModulatedGenerator:
    """
    MoleculeGeneratorLiveのラッパー。
    
    Strategyに基づいてtemperatureとprompt_suffixを調整する。
    既存のMoleculeGeneratorLiveを変更せずに拡張。
    """
    
    def __init__(self, base_generator: Any):
        """
        Args:
            base_generator: MoleculeGeneratorLive instance
        """
        self.base = base_generator
        self._original_call_llm = base_generator._call_llm
    
    def generate_with_strategy(
        self,
        original_text: str,
        candidates: List[Dict],
        strategy: GenerationStrategy
    ) -> Any:
        """
        Strategyを適用して生成する。
        
        Args:
            original_text: 入力テキスト
            candidates: 候補リスト
            strategy: 適用する戦略
            
        Returns:
            LiveGenerationResult
        """
        # _call_llmを一時的にオーバーライド
        def modulated_call_llm(system_prompt: str, user_prompt: str) -> str:
            # System promptにsuffixを追加
            if strategy.prompt_suffix:
                system_prompt = system_prompt + f"\n\nAdditional Instruction: {strategy.prompt_suffix}"
            
            # 元のメソッドを呼び出し（temperatureを調整）
            return self._call_with_temperature(
                system_prompt,
                user_prompt,
                strategy.temperature
            )
        
        # 一時的に置き換え
        self.base._call_llm = modulated_call_llm
        
        try:
            result = self.base.generate(original_text, candidates)
        finally:
            # 元に戻す
            self.base._call_llm = self._original_call_llm
        
        return result
    
    def _call_with_temperature(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float
    ) -> str:
        """
        指定されたtemperatureでLLMを呼び出す。
        """
        import requests
        
        url = f"{self.base.llm_host}/chat/completions"
        
        payload = {
            "model": self.base.llm_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": temperature,  # Strategyから指定
            "max_tokens": 16000
        }
        
        response = requests.post(
            url,
            json=payload,
            timeout=self.base.llm_timeout,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        
        data = response.json()
        return data["choices"][0]["message"]["content"]


# ==========================================
# ESDE Pipeline
# ==========================================
class ESDEPipeline:
    """
    ESDE End-to-End Pipeline
    
    観測 → 記録 → 内省 → 戦略変更 → 次の観測
    
    Usage:
        pipeline = ESDEPipeline(
            sensor=sensor,
            generator=generator,
            ledger=ledger,
            index=index
        )
        result = pipeline.run("I love you")
    """
    
    def __init__(
        self,
        sensor: Any,
        generator: Any,
        ledger: Any,
        index: Any,
        modulator: Modulator = None,
        projector: Any = None,
        enable_alerts: bool = True
    ):
        """
        Args:
            sensor: ESDESensorV2 instance
            generator: MoleculeGeneratorLive instance
            ledger: PersistentLedger instance
            index: SemanticIndex or QueryAPI instance
            modulator: Modulator instance (default: new instance)
            projector: Projector instance (for index update)
            enable_alerts: アラート出力を有効にするか
        """
        self.sensor = sensor
        self.generator = generator
        self.ledger = ledger
        self.index = index
        self.modulator = modulator or Modulator()
        self.projector = projector
        self.enable_alerts = enable_alerts
        
        # GeneratorをWrap
        self.modulated_generator = ModulatedGenerator(generator)
    
    def run(self, text: str, direction: str = "=>+") -> PipelineResult:
        """
        パイプラインを実行する。
        
        Args:
            text: 入力テキスト
            direction: 創発方向性
            
        Returns:
            PipelineResult
        """
        result = PipelineResult(success=False, input_text=text)
        
        try:
            # ===========================================
            # Step 1: Scan (Sensor V2)
            # ===========================================
            sensor_result = self.sensor.analyze(text)
            candidates = sensor_result.get("candidates", [])
            result.candidates = candidates
            
            if not candidates:
                result.abstained = True
                result.generation_error = "No candidates from Sensor"
                logger.info(f"[Pipeline] No candidates for: {text[:50]}...")
                return result
            
            # ===========================================
            # Step 2: Get Target Atom (top-1)
            # ===========================================
            target_atom = get_target_atom_from_candidates(candidates)
            result.target_atom = target_atom
            
            if not target_atom:
                result.abstained = True
                result.generation_error = "No target atom"
                return result
            
            # ===========================================
            # Step 3: Introspect (Index Read)
            # ===========================================
            rigidity = self._get_rigidity(target_atom)
            result.rigidity = rigidity
            
            # ===========================================
            # Step 4: Modulate (Strategy Decision)
            # ===========================================
            strategy = self.modulator.decide_strategy(target_atom, self.index)
            result.strategy = strategy
            
            logger.info(
                f"[Pipeline] {target_atom}: R={rigidity}, "
                f"mode={strategy.mode.value}, temp={strategy.temperature}"
            )
            
            # ===========================================
            # Step 5: Generate (LLM with Strategy)
            # ===========================================
            gen_result = self.modulated_generator.generate_with_strategy(
                original_text=text,
                candidates=candidates,
                strategy=strategy
            )
            
            if not gen_result.success:
                result.abstained = gen_result.abstained
                result.generation_error = gen_result.error
                logger.warning(f"[Pipeline] Generation failed: {gen_result.error}")
                return result
            
            result.molecule = gen_result.molecule
            
            # ===========================================
            # Step 6: Record (Ledger Write)
            # ===========================================
            # Direction調整（Disruptiveモードなら-|>を検討）
            actual_direction = direction
            if strategy.mode == StrategyMode.DISRUPTIVE:
                actual_direction = "-|>"  # 破壊的創発
            
            ledger_entry = self.ledger.observe_molecule(
                source_text=text,
                molecule=gen_result.molecule,
                direction=actual_direction,
                actor="Pipeline"
            )
            result.ledger_entry = ledger_entry
            
            # ===========================================
            # Step 7: Update (Index Update)
            # ===========================================
            if self.projector:
                self.projector.on_event(ledger_entry)
                result.index_updated = True
            
            # ===========================================
            # Step 8: Alert Check
            # ===========================================
            if self.enable_alerts:
                alert = self.modulator.check_alert(target_atom, self.index)
                if alert:
                    result.alert = alert
            
            result.success = True
            logger.info(f"[Pipeline] Success: {target_atom} → seq={ledger_entry['seq']}")
            
        except Exception as e:
            result.generation_error = f"Pipeline error: {str(e)}"
            logger.error(f"[Pipeline] Error: {e}", exc_info=True)
        
        return result
    
    def _get_rigidity(self, atom_id: str) -> Optional[float]:
        """IndexからRigidityを取得"""
        if hasattr(self.index, 'get_rigidity'):
            return self.index.get_rigidity(atom_id)
        
        if hasattr(self.index, 'get_atom_stats'):
            from index.rigidity import calculate_rigidity
            stats = self.index.get_atom_stats(atom_id)
            if stats:
                return calculate_rigidity(stats)
        
        return None
    
    def status(self) -> Dict[str, Any]:
        """パイプライン状態を取得"""
        return {
            "sensor": type(self.sensor).__name__,
            "generator": type(self.generator).__name__,
            "ledger_seq": self.ledger.seq if hasattr(self.ledger, 'seq') else None,
            "index_events": self.index.total_events if hasattr(self.index, 'total_events') else None,
            "modulator_thresholds": {
                "rigidity_high": self.modulator.rigidity_high,
                "rigidity_low": self.modulator.rigidity_low
            },
            "alerts_enabled": self.enable_alerts
        }
