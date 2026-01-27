# ESDE Cell Architecture Proposal

**Version:** Draft 0.1  
**Date:** 2026-01-27  
**Authors:** Taka (Human) + Claude (AI)  
**Status:** RFC (Request for Comments from Gemini/GPT)

---

## 1. Executive Summary

本提案は、ESDE Phase 8（強い意味系）とPhase 9（弱い意味系）を統合し、「細胞（Cell）」を形成するアーキテクチャを定義する。

**核心的洞察:**
- Phase 8とPhase 9は**別々の系**であり、混ぜてはならない
- **条件因子（Condition Factor）**が「引力」として機能し、両者を結合する
- LLMによるセグメント境界は**動的条件因子**として定義可能

---

## 2. 物理学的アナロジー

### 2.1 原子構造との対応

```
物理学:
  原子核（陽子・中性子）  ←  強い力で結合
  電子                    ←  別の存在、別の法則
  
  これらは別々だが、電磁力で引き合って「原子」を形成


ESDE:
  Molecule（強い意味）    ←  Phase 8（326 Atoms基準）
  Island（弱い意味）      ←  Phase 9（統計的パターン）
  
  これらは別々だが、条件因子で引き合って「Cell」を形成
```

### 2.2 設計原則

| 原則 | 説明 |
|------|------|
| **非混合** | Phase 8とPhase 9は互いに侵食しない |
| **自然な結合** | 無理に粒度を合わせず、結合可能なものが自然に結合 |
| **引力としての条件因子** | 条件因子がなければ、ただの2つの独立した観測結果 |

---

## 3. 階層構造

### 3.1 完全な階層定義

```
Atom（326個）
    ↓ Phase 8: Synapse + LLM
Molecule（セグメント単位の意味構造）
    ↓
    │
    │   ←──── 条件因子（引力）────→   Island（統計的クラスタ）
    │                                      ↑
    │                               Phase 9: W4-W5-W6
    ↓
Cell（条件因子で結合されたMolecule + Island）
    ↓ 条件因子の階層でグルーピング
Organ（同一上位条件因子のCell群）
    ↓ LLM統合
Ecosystem（全体の意味構造 + 言語化レポート）
```

### 3.2 各層の定義

| 層 | 定義 | 生成元 |
|----|------|--------|
| **Atom** | 326個の最小意味単位（163対称ペア） | Foundation Layer |
| **Molecule** | セグメント単位の意味構造（Atom + Formula） | Phase 8 |
| **Island** | 共鳴ベクトルが類似した記事群のクラスタ | Phase 9 (W5) |
| **Cell** | 条件因子で結合されたMolecule + Island | 統合層 |
| **Organ** | 上位条件因子でグループ化されたCell群 | 統合層 |
| **Ecosystem** | 全体の意味構造 + LLMによる言語化 | 出力層 |

---

## 4. 条件因子の再定義

### 4.1 現状の条件因子（静的・環境的）

```python
# schema_w2.py より
SOURCE_TYPES = {"news", "dialog", "paper", "social", "unknown"}
LANGUAGE_PROFILES = {"en", "ja", "mixed", "unknown"}

# 抽出される因子
{
    "source_type": "news",
    "language_profile": "en",
    "time_bucket": "2026-01",
}
```

**特徴:** データ取得時に決定される環境属性

### 4.2 動的条件因子（提案）

**Phase 8のLLMセグメント境界を条件因子として定義:**

```python
{
    # 静的（環境）
    "source_type": "news",
    "language_profile": "en",
    "time_bucket": "2026-01",
    
    # 動的（構造）- Phase 8が生成
    "segment_id": "seg_0042",
    "segment_boundary_type": "llm_semantic",  # LLMによる意味境界
}
```

### 4.3 条件因子の役割

```
条件因子なし:
  Phase 8 → Molecule群（バラバラ）
  Phase 9 → Island群（バラバラ）
  = ただの2つの独立した観測結果

条件因子あり:
  条件因子「segment_id」で引く
    → そのセグメントのMolecule + そのセグメントの共鳴パターン
    → Cell形成

  条件因子「章」で引く
    → その章に属するCell群
    → Organ形成
```

---

## 5. 処理フロー

### 5.1 並列処理モデル

```
┌─────────────────────────────────────────────────────────────────────┐
│                        入力テキスト                                  │
└─────────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ↓                               ↓
┌─────────────────────────┐     ┌─────────────────────────┐
│  Phase 8（強い意味）     │     │  Phase 9（弱い意味）     │
│                         │     │                         │
│  1. セグメント境界検出    │     │  W0: データ正規化        │
│     （LLM）              │     │  W1: 全体統計            │
│  2. 原子化（WordNet）    │     │  W2: 条件別統計          │
│  3. 分子化（LLM）        │     │  W3: S-Score軸候補       │
│                         │     │  W4: 共鳴ベクトル         │
│  出力: Molecule群        │     │  W5: 島形成              │
│       + segment_id      │     │  W6: エビデンス抽出       │
│       （動的条件因子）    │     │                         │
│                         │     │  出力: Island群           │
│                         │     │        + Evidence        │
└─────────────────────────┘     └─────────────────────────┘
              │                               │
              │      互いに侵食しない          │
              │      別々の観測結果            │
              └───────────────┬───────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    統合層（条件因子による結合）                        │
│                                                                     │
│  条件因子「segment_id」で引く:                                       │
│    Molecule(seg_0042) + Island(seg_0042に共鳴) → Cell               │
│                                                                     │
│  条件因子「chapter」で引く:                                          │
│    Cell群(chapter_3に属する) → Organ                                │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    出力層（LLM言語化）                                │
│                                                                     │
│  ESDEの観測結果を自然言語レポートに変換                               │
│  ※ 自己流の解釈を加えない（材料外の推測をしない）                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.2 重要な制約

| 制約 | 説明 |
|------|------|
| **Phase 8 → Phase 9 への流入禁止** | Moleculeの情報がW層に影響しない |
| **Phase 9 → Phase 8 への流入禁止** | Islandの情報がMolecule生成に影響しない |
| **統合は条件因子のみで行う** | 結合ロジックに意味解釈を含めない |

---

## 6. Cellの構造定義

### 6.1 Cell スキーマ（案）

```python
@dataclass
class Cell:
    """
    条件因子で結合されたMolecule + Island。
    
    Phase 8とPhase 9の観測結果を並列で保持。
    両者は混ぜない。
    """
    
    # Identity
    cell_id: str
    
    # 結合に使用した条件因子
    binding_factor: Dict[str, Any]  # e.g., {"segment_id": "seg_0042"}
    
    # Phase 8 からの観測（強い意味）
    molecules: List[Molecule]  # この条件因子に属するMolecule群
    
    # Phase 9 からの観測（弱い意味）
    resonance_pattern: Optional[Dict[str, float]]  # この条件因子での共鳴
    related_islands: List[str]  # 関連するisland_id群
    
    # メタデータ
    source_segment: Optional[str]  # 元のテキストセグメント
    created_at: str
```

### 6.2 Cell形成ロジック

```python
def form_cell(
    segment_id: str,
    molecules: List[Molecule],
    w4_records: List[W4Record],
    w5_structure: W5Structure,
) -> Cell:
    """
    条件因子（segment_id）でMoleculeとIslandを結合。
    
    Phase 8とPhase 9のデータを混ぜない。
    並列で保持するだけ。
    """
    
    # Phase 8: このsegmentのMolecule
    segment_molecules = [m for m in molecules if m.segment_id == segment_id]
    
    # Phase 9: このsegmentに関連するIsland
    related_islands = find_related_islands(segment_id, w4_records, w5_structure)
    
    # 結合（混ぜない、並列保持）
    return Cell(
        cell_id=compute_cell_id(segment_id, segment_molecules, related_islands),
        binding_factor={"segment_id": segment_id},
        molecules=segment_molecules,
        related_islands=related_islands,
        ...
    )
```

---

## 7. ESDEとLLMの分業

### 7.1 役割分担

```
┌─────────────────────────────────────────────────────────────────────┐
│  ESDE（観測層）                                                      │
│                                                                     │
│  責務:                                                              │
│    ・Phase 8: Molecule生成（強い意味の構造化）                        │
│    ・Phase 9: Island/Evidence抽出（弱い意味のパターン）               │
│    ・条件因子による結合（Cell/Organ形成）                             │
│                                                                     │
│  出力: 構造化されたデータ（JSON, スキーマ準拠, 検証可能）              │
│                                                                     │
│  哲学: "記述せよ、しかし決定するな"                                   │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
                    解釈の材料を徹底的に提供
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│  LLM（言語化層）                                                     │
│                                                                     │
│  責務:                                                              │
│    ・ESDEの観測結果に基づいてレポート生成                             │
│    ・人間に分かりやすい自然言語で出力                                 │
│                                                                     │
│  制約:                                                              │
│    ・自己流の解釈を加えない                                          │
│    ・材料外の推測をしない                                            │
│    ・ESDEが提供した情報の範囲内で言語化                               │
│                                                                     │
│  出力: 自然言語レポート（柔軟、読みやすさ重視）                        │
└─────────────────────────────────────────────────────────────────────┘
```

### 7.2 厳密さのグラデーション

| 層 | 厳密さ | 理由 |
|---|---|---|
| Atom/Molecule/Cell | **厳密** | 機械的に検証可能、再現性必須 |
| Island/Evidence | **厳密** | 統計的根拠、トレーサビリティ |
| 最終レポート | **柔軟** | 人間が読むもの、自然言語の強みを活かす |

### 7.3 なぜこの分業が重要か

```
問題: LLMは自己流の解釈を加えがち

解決: ESDEが「ありとあらゆるデータの解釈」を先に与える
      → LLMは与えられた材料の範囲内で言語化するだけ
      → 材料にないことは推測しない
      → 推測の余地を最小化
```

---

## 8. 最小Cell単位の考察

### 8.1 センテンスと原子の非対称性

```
物理学:
  分子になり得ない原子がある
  細胞になり得ないタンパク質がある

ESDE:
  Cellになり得ないMoleculeがある可能性
  Cellになり得ないIslandがある可能性
```

**設計方針:** Phase 8とPhase 9を「意図的に一致させる」のは最適解ではない。自然と統合が可能なものを統合できるようにこちらが合わせる。

### 8.2 動的な最小単位

```
現状:
  入力単位 = ArticleRecord（記事全体）
  → 「データ取得行為」が一つの単位として条件因子に記録される

提案:
  Phase 8のLLMセグメント境界 = 動的に決定される最小単位
  → 「意味の区切り」が条件因子として機能
  → センテンスより大きくても小さくてもよい
```

---

## 9. Substrateとの関係

### 9.1 現状の条件因子（NAMESPACES.md）

```python
# 環境的（静的）
"legacy:source_type": "news"
"legacy:language_profile": "en"

# 構造的（現状はboolのみ）
"html:has_h1": True
"html:has_h2": True
"struct:paragraph_count": 5
```

**意図的な制限:** h1やタイトルの「値」は保持しない
- PDF/Docsでは定義しづらい
- HTMLのタイトルが意味上のタイトルとして正しいかは厳密には不明
- 最小構成で拡張性を残す

### 9.2 拡張の方向性

```python
# 静的条件因子（Substrate）
{
    "legacy:source_type": "legal",
    "legacy:language_profile": "ja",
    "html:has_h1": True,
}

# 動的条件因子（Phase 8が生成）
{
    "esde:segment_id": "seg_0042",
    "esde:segment_boundary": "llm_semantic_v1",
}
```

---

## 10. 未解決の課題

### 10.1 Phase 9の入力単位

```
現状: ArticleRecord（記事単体）
疑問: セグメント単位に変更すべきか？

考察:
  - Phase 9は「記事間」のクラスタリングを行う（W5）
  - セグメント単位にすると、セグメント間のクラスタリングになる
  - どちらが適切かは用途による
```

### 10.2 Island と Segment の対応

```
一つの記事から複数のIslandが出る可能性がある
一つのSegmentが複数のIslandに関連する可能性がある

→ 多対多の関係をどう扱うか？
```

### 10.3 条件因子の階層

```
segment_id < paragraph_id < section_id < chapter_id < document_id

どの階層で結合するかで、異なるCell/Organが形成される
→ ユースケースに応じた選択が必要
```

---

## 11. レビュー依頼事項

Gemini/GPTへの質問:

1. **Phase 8/9の非混合原則**について、物理学的アナロジーは適切か？他に良いモデルがあるか？

2. **条件因子を「引力」として定義する**アプローチは妥当か？潜在的な問題点は？

3. **LLMセグメント境界を動的条件因子として扱う**提案について、実装上の懸念は？

4. **ESDEとLLMの分業**（観測 vs 言語化）の境界設定は適切か？

5. **最小Cell単位**を固定しない（Phase 8/9を意図的に一致させない）アプローチのリスクは？

6. その他、アーキテクチャ上の矛盾や改善提案があれば。

---

## 12. 次のステップ（案）

1. Gemini/GPTからのフィードバック収集
2. フィードバックに基づく設計修正
3. Cell/Organ スキーマの正式定義
4. 統合層の実装設計
5. プロトタイプ実装とテスト

---

*Document generated from discussion between Taka (Human) and Claude (AI)*
*Philosophy: Aruism - "Describe, but do not decide"*
