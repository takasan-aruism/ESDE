"""
ESDE Phase 8-6: Canonical JSON Serialization
=============================================

JSON正規化ロジック。ハッシュ計算とファイル保存の整合性を保証する。

Spec v8.6.1 準拠:
- Keys: ソート済み (sort_keys=True)
- Encoding: UTF-8 (ensure_ascii=False)
- Separators: 空白なし (separators=(',', ':'))

Critical: validate()は"行文字列"をcanonicalとして扱い、
         JSON再dumpでハッシュ入力を作らないこと。
"""

import json
import hashlib
from typing import Any


def canonicalize(obj: Any) -> str:
    """
    オブジェクトをCanonical JSON文字列に変換する。
    
    この関数で生成された文字列は:
    1. ハッシュ計算の入力として使用される
    2. ファイルに保存される（改行なし）
    
    Args:
        obj: シリアライズ対象のPythonオブジェクト
        
    Returns:
        正規化されたJSON文字列（末尾改行なし）
    """
    return json.dumps(
        obj,
        sort_keys=True,
        ensure_ascii=False,
        separators=(',', ':'),
        default=str
    )


def sha256_hex(data: str) -> str:
    """
    文字列のSHA256ハッシュを16進数文字列で返す。
    
    Args:
        data: ハッシュ対象の文字列（UTF-8エンコード）
        
    Returns:
        64文字の16進数ハッシュ文字列
    """
    return hashlib.sha256(data.encode('utf-8')).hexdigest()


def parse_canonical_line(line: str) -> dict:
    """
    ファイルから読み込んだ行をパースする。
    
    Args:
        line: ファイルから読み込んだ1行（改行含む可能性あり）
        
    Returns:
        パースされたdict
        
    Raises:
        json.JSONDecodeError: パース失敗時
    """
    return json.loads(line.strip())
