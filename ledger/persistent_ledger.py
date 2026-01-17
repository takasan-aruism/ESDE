"""
ESDE Phase 8-6: Persistent Semantic Ledger
==========================================

意味時間の結晶化（Semantic Time Crystallization）

Hash Chain技術を用いて、ESDEの意味生成履歴を改竄検知可能な
形で永続化し、後から"なぜそうなったか"を検証できる状態にする。

Spec v8.6.1 準拠:
- Atomic Append: flush + fsync
- Canonical Serialization: バイト一致保証
- 5種の整合性検証（T861〜T865）
- Rehydration: Ledger Replay原則

GPT監査修正（4点）:
1. validate()は"行文字列"をcanonicalとして扱い、JSON再dumpでハッシュ入力を作らない
2. 最終行破損はsalvageなしで停止
3. event_idからmetaを除外（意味同一性に寄せる）
4. rehydrationはledger replayを原則とする
"""

import os
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Iterator, Optional
from dataclasses import dataclass

from .ephemeral_ledger import EphemeralLedger
from .chain_crypto import (
    compute_hashes, compute_event_id, compute_self_hash, GENESIS_PREV
)
from .canonical import canonicalize, parse_canonical_line
from .memory_math import generate_fingerprint, get_tau_from_molecule


class IntegrityError(Exception):
    """改竄・破損検知エラー"""
    pass


@dataclass
class ValidationReport:
    """検証レポート"""
    valid: bool
    total_entries: int
    errors: list
    
    def __str__(self) -> str:
        status = "PASS" if self.valid else "FAIL"
        error_summary = f", errors: {self.errors}" if self.errors else ""
        return f"ValidationReport({status}, entries={self.total_entries}{error_summary})"


class PersistentLedger:
    """
    永続化意味Ledger。
    
    Hash Chainを用いた改竄検知可能な履歴管理。
    起動時に整合性検証を行い、破損があれば起動を停止する。
    """
    
    VERSION = 1
    LEDGER_ID = "esde-semantic-ledger"
    HASH_ALGO = "sha256"
    
    def __init__(
        self,
        path: str,
        engine_version: str = "5.3.6-P8.6",
        auto_initialize: bool = True
    ):
        """
        Args:
            path: Ledgerファイルのパス
            engine_version: エンジンバージョン
            auto_initialize: 自動初期化（Genesis/Rehydration）
        """
        self.path = Path(path)
        self.engine_version = engine_version
        self.ephemeral = EphemeralLedger()
        
        self._seq = -1
        self._prev_hash = GENESIS_PREV
        self._initialized = False
        
        if auto_initialize:
            self._initialize()
    
    def _initialize(self) -> None:
        """起動時初期化: Genesis作成 or Validation + Rehydration"""
        if not self.path.exists():
            self._write_genesis()
        else:
            report = self.validate()
            if not report.valid:
                raise IntegrityError(
                    f"Ledger integrity check failed: {report.errors}"
                )
            self._rehydrate()
        
        self._initialized = True
    
    def _write_genesis(self) -> None:
        """Genesis Block（始原ブロック）を生成する。"""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        
        genesis = self._create_entry(
            event_type="genesis",
            direction="=>",
            data={"message": "Aru - There is"},
            actor="System"
        )
        
        self._append_entry(genesis)
        
        self._seq = genesis["seq"]
        self._prev_hash = genesis["hash"]["self"]
    
    def _create_entry(
        self,
        event_type: str,
        direction: str,
        data: dict,
        actor: str = "Engine"
    ) -> dict:
        """エントリを作成する（まだ書き込まない）。"""
        self._seq += 1
        
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        
        entry_without_hash = {
            "v": self.VERSION,
            "ledger_id": self.LEDGER_ID,
            "seq": self._seq,
            "ts": ts,
            "event_type": event_type,
            "direction": direction,
            "data": data,
            "meta": {
                "engine_version": self.engine_version,
                "actor": actor
            }
        }
        
        hash_block = compute_hashes(entry_without_hash, self._prev_hash)
        
        entry = entry_without_hash.copy()
        entry["hash"] = hash_block
        
        return entry
    
    def _append_entry(self, entry: dict) -> None:
        """
        エントリを安全に追記する。
        
        Atomic Append: flush + fsync
        """
        line = canonicalize(entry) + "\n"
        
        with open(self.path, 'a', encoding='utf-8') as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
        
        self._prev_hash = entry["hash"]["self"]
    
    def append(
        self,
        event_type: str,
        direction: str,
        data: dict,
        actor: str = "Engine"
    ) -> dict:
        """新しいイベントを追記する。"""
        if not self._initialized:
            raise RuntimeError("Ledger not initialized")
        
        entry = self._create_entry(event_type, direction, data, actor)
        self._append_entry(entry)
        
        # メモリ（Ephemeral）も更新
        if "molecule" in data:
            source_text = data.get("source_text", data.get("source_hash", "unknown"))
            self.ephemeral.upsert(
                molecule=data["molecule"],
                source_text=source_text
            )
        
        return entry
    
    def observe_molecule(
        self,
        source_text: str,
        molecule: dict,
        direction: str = "=>+",
        audit: Optional[dict] = None,
        actor: str = "SensorV2"
    ) -> dict:
        """意味分子の観測を記録する。"""
        # フィンガープリント計算（既存関数を使用）
        fingerprint = generate_fingerprint(molecule)
        
        # 現在の重みを取得（メモリから）
        existing = self.ephemeral.get_entry(fingerprint)
        current_weight = existing.weight if existing else 1.0
        
        data = {
            "source_text": source_text,
            "source_hash": source_text[:16] if len(source_text) > 16 else source_text,
            "molecule": molecule,
            "weight": current_weight,
            "audit": audit or {"validator_pass": True, "coerced": False}
        }
        
        return self.append(
            event_type="molecule.observe",
            direction=direction,
            data=data,
            actor=actor
        )
    
    def validate(self) -> ValidationReport:
        """
        Ledgerの整合性を検証する。
        
        T861: chain_validates
        T862: prev_linkage
        T863: truncation
        T864: monotonic_seq
        T865: header_match
        """
        errors = []
        prev_hash = GENESIS_PREV
        prev_seq = -1
        total = 0
        
        if not self.path.exists():
            return ValidationReport(valid=True, total_entries=0, errors=[])
        
        try:
            with open(self.path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception as e:
            errors.append(f"T863: File read error: {e}")
            return ValidationReport(valid=False, total_entries=0, errors=errors)
        
        for i, line in enumerate(lines):
            line_num = i + 1
            
            # T863: Truncation
            try:
                entry = parse_canonical_line(line)
            except json.JSONDecodeError as e:
                errors.append(f"T863: Line {line_num} parse error: {e}")
                continue
            
            total += 1
            
            # T865: Header Match
            if entry.get("v") != self.VERSION:
                errors.append(f"T865: Line {line_num} version mismatch")
            if entry.get("ledger_id") != self.LEDGER_ID:
                errors.append(f"T865: Line {line_num} ledger_id mismatch")
            if entry.get("hash", {}).get("algo") != self.HASH_ALGO:
                errors.append(f"T865: Line {line_num} hash algo mismatch")
            
            # T864: Monotonic Seq
            seq = entry.get("seq", -1)
            expected_seq = prev_seq + 1
            if seq != expected_seq:
                errors.append(f"T864: Line {line_num} seq={seq}, expected={expected_seq}")
            
            # T862: Prev Linkage
            entry_prev = entry.get("hash", {}).get("prev", "")
            if entry_prev != prev_hash:
                errors.append(f"T862: Line {line_num} prev mismatch")
            
            # T861: Chain Validates
            entry_for_hash = {
                "v": entry["v"],
                "ledger_id": entry["ledger_id"],
                "seq": entry["seq"],
                "ts": entry["ts"],
                "event_type": entry["event_type"],
                "direction": entry["direction"],
                "data": entry["data"],
                "meta": entry["meta"],
                "hash": {
                    "algo": entry["hash"]["algo"],
                    "prev": entry["hash"]["prev"],
                    "event_id": entry["hash"]["event_id"]
                }
            }
            
            expected_self = compute_self_hash(entry_for_hash, prev_hash)
            actual_self = entry.get("hash", {}).get("self", "")
            
            if actual_self != expected_self:
                errors.append(f"T861: Line {line_num} self hash mismatch")
            
            prev_hash = actual_self
            prev_seq = seq
        
        return ValidationReport(
            valid=len(errors) == 0,
            total_entries=total,
            errors=errors
        )
    
    def _rehydrate(self) -> None:
        """起動時復元: Ledgerをリプレイしてメモリを再構築する。"""
        for entry in self.iter_entries():
            self._seq = entry["seq"]
            self._prev_hash = entry["hash"]["self"]
            
            if entry["event_type"] == "molecule.observe":
                data = entry.get("data", {})
                molecule = data.get("molecule")
                
                if molecule:
                    source_text = data.get("source_text", data.get("source_hash", "unknown"))
                    # タイムスタンプを渡してupsert
                    ts_str = entry["ts"]
                    try:
                        ts = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
                    except:
                        ts = None
                    
                    self.ephemeral.upsert(
                        molecule=molecule,
                        source_text=source_text,
                        timestamp=ts
                    )
    
    def iter_entries(self) -> Iterator[dict]:
        """全エントリをイテレートする。"""
        if not self.path.exists():
            return
        
        with open(self.path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    yield parse_canonical_line(line)
    
    @property
    def seq(self) -> int:
        return self._seq
    
    @property
    def entry_count(self) -> int:
        return self._seq + 1 if self._seq >= 0 else 0
    
    def status(self) -> dict:
        return {
            "path": str(self.path),
            "initialized": self._initialized,
            "seq": self._seq,
            "prev_hash": self._prev_hash[:16] + "..." if self._prev_hash else None,
            "memory_entries": len(self.ephemeral),
            "memory_stats": self.ephemeral.get_stats().to_dict()
        }
