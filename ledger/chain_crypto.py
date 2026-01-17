"""
ESDE Phase 8-6: Hash Chain Cryptography
=======================================

ハッシュチェーンの計算ロジック。

Spec v8.6.1 準拠:
- Event ID: 意味同一性のハッシュ（metaを除外）
- Self Hash: 歴史的連続性のハッシュ（prevとの連鎖）

GPT監査修正:
- event_idからmetaを除外（actor/engine_versionは同一性に不要）
"""

from .canonical import canonicalize, sha256_hex


# Genesis Block の prev_hash（64文字のゼロ）
GENESIS_PREV = "0" * 64


def compute_event_id(entry: dict) -> str:
    """
    Event ID（内容同一性ハッシュ）を計算する。
    
    将来的なMerge/De-duplication用。同じ意味イベントは
    タイムスタンプやメタ情報が異なっても同じevent_idを持つ。
    
    Spec v8.6.1 (GPT監査修正):
    - Target: {ledger_id, event_type, direction, data} + v
    - 除外: ts, seq, hash, meta（意味同一性に不要）
    
    Args:
        entry: エントリdict（hashフィールドなし or あり両対応）
        
    Returns:
        64文字の16進数ハッシュ文字列
    """
    target = {
        "v": entry.get("v"),
        "ledger_id": entry.get("ledger_id"),
        "event_type": entry.get("event_type"),
        "direction": entry.get("direction"),
        "data": entry.get("data")
    }
    return sha256_hex(canonicalize(target))


def compute_self_hash(entry: dict, prev_hash: str) -> str:
    """
    Self Hash（チェーンハッシュ）を計算する。
    
    歴史的連続性を保証。過去のいかなるデータを1ビットでも
    書き換えると、それ以降すべてのハッシュが整合しなくなる。
    
    Spec v8.6.1:
    - Target: {v, ledger_id, seq, ts, event_type, direction, data, meta, 
               hash: {algo, prev, event_id}} (selfを除く)
    - self_hash = SHA256(prev_hash + "\\n" + Canonical(Target))
    
    Args:
        entry: hashフィールドにalgo/prev/event_idが含まれたdict
        prev_hash: 直前のエントリのself hash（Genesisなら GENESIS_PREV）
        
    Returns:
        64文字の16進数ハッシュ文字列
    """
    target = {
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
    
    canonical_body = canonicalize(target)
    return sha256_hex(prev_hash + "\n" + canonical_body)


def compute_hashes(entry_without_hash: dict, prev_hash: str) -> dict:
    """
    エントリのハッシュフィールド全体を計算する。
    
    Args:
        entry_without_hash: hashフィールドを含まないエントリdict
        prev_hash: 直前のエントリのself hash
        
    Returns:
        {algo, prev, event_id, self} を含むhash dict
    """
    # Step 1: event_id を計算
    event_id = compute_event_id(entry_without_hash)
    
    # Step 2: hash フィールドを仮構築（selfなし）
    entry_with_partial_hash = entry_without_hash.copy()
    entry_with_partial_hash["hash"] = {
        "algo": "sha256",
        "prev": prev_hash,
        "event_id": event_id
    }
    
    # Step 3: self hash を計算
    self_hash = compute_self_hash(entry_with_partial_hash, prev_hash)
    
    return {
        "algo": "sha256",
        "prev": prev_hash,
        "event_id": event_id,
        "self": self_hash
    }
