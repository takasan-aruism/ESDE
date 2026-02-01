# ESDE Cell Architecture

**Version:** 2.0  
**Date:** 2026-02-02  
**Authors:** Taka (Human) + Claude (AI)  
**Status:** Phase 9 完了時点の設計記録  
**Previous:** Draft 0.1 (2026-01-27) — 全面改訂

---

## 変更履歴

| 版 | 日付 | 概要 |
|----|------|------|
| 0.1 | 2026-01-27 | 初版（RFC、概念設計のみ） |
| 2.0 | 2026-02-02 | Phase 9 実装完了に基づく全面改訂。W層再定義、Lens導入、Threshold 3層化、Mutual-kNN + k-sweep 追加。旧0.1の未解決課題の大半が解決済み。 |

---

## 1. Executive Summary

本文書は、ESDE Phase 8（強い意味系）と Phase 9（弱い意味系）の統合アーキテクチャ「Cell」を定義する。

**核心的洞察（v0.1から不変）:**
- Phase 8 と Phase 9 は**別々の系**であり、混ぜてはならない
- **条件因子（Condition Factor）**が「引力」として機能し、両者を結合する

**v2.0 での主要変化:**
- 条件因子は外部メタデータ（source_type等）ではなく、**テキスト内部構造**（セクション名・ドキュメント名）から抽出される
- Phase 9 の分析単位は「記事」ではなく**「セクション」**
- **Lens（レンズ）**が導入され、同じデータを異なる観点（構造/意味/混合）で観測可能に
- **Island** は「書き方が統計的に類似したセクション群のクラスタ」
- 閾値・エッジ選択・クラスタリングの全工程が**動的・トレーサブル**

---

## 2. 物理学的アナロジー

### 2.1 原子構造との対応

```
物理学:
  原子核（陽子・中性子）  ←  強い力で結合
  電子                    ←  別の存在、別の法則
  
  これらは別々だが、電磁力で引き合って「原子」を形成


ESDE:
  Molecule（強い意味）    ←  Phase 8（326 Atoms + Synapse v3.0）
  Island（弱い意味）      ←  Phase 9（統計的パターン → セクション群のクラスタ）
  
  これらは別々だが、条件因子で引き合って「Cell」を形成
```

### 2.2 光学的アナロジー（v2.0 追加）

Phase 9 に**レンズ**の概念が導入された。同じデータを異なる「焦点」で観測する。

```
顕微鏡:
  対物レンズを変えると、同じ標本から異なる構造が見える
  4x → 組織全体の構造
  40x → 個々の細胞
  100x → 細胞内小器官

ESDE Phase 9:
  Lens を変えると、同じテキストから異なるパターンが見える
  Structure Lens → Wikipedia のテンプレート構造（Hub/Narrative/Institutional）
  Semantic Lens  → 主題の意味的類似性（戦争/哲学/都市）
  Hybrid Lens    → セクション内の意味的偏り（書き方の個性）
```

さらに Mutual-kNN の k パラメータは**レンズの焦点距離**として機能する:

```
k 大（広角） → 大域的テンプレート構造が見える（島が少ない、巨大成分が支配的）
k 小（望遠） → 局所的な主題クラスタが見える（島が多い、高結束の小集団）

k=2 : 望遠 → 108 islands（微細構造、noise 30%）
k=3 : 標準 → 61 islands（中粒度、noise 16%）
k≥4 : 臨界点超過 → 連鎖（gcr > 0.65、巨大成分が全体を飲み込む）
```

### 2.3 設計原則

| 原則 | 説明 |
|------|------|
| **非混合** | Phase 8 と Phase 9 は互いに侵食しない |
| **自然な結合** | 無理に粒度を合わせず、結合可能なものが自然に結合 |
| **引力としての条件因子** | 条件因子がなければ、ただの2つの独立した観測結果 |
| **記述せよ、決定するな** | 全ての判断は trace として記録され、後から検証可能 |
| **不確実性は結果** | 分類できない（noise）は正当な観測結果であり、排除しない |

---

## 3. 階層構造

### 3.1 完全な階層定義

```
Atom（326個）
    ↓ Phase 8: Synapse v3.0 + LLM
Molecule（セグメント単位の意味構造）
    ↓
    │
    │   ←─── 条件因子（引力）───→   Island（統計的クラスタ）
    │                                     ↑
    │                              Phase 9: W1→W2→W3→W4→W5→W6
    │                              （Lens × k-sweep × Threshold）
    ↓
Cell（条件因子で結合された Molecule + Island）
    ↓ 条件因子の階層でグルーピング
Organ（同一上位条件因子の Cell 群）
    ↓ LLM 統合
Ecosystem（全体の意味構造 + 言語化レポート）
```

### 3.2 各層の定義

| 層 | 定義 | 生成元 | 粒度 |
|----|------|--------|------|
| **Atom** | 326個の最小意味単位（163対称ペア） | Foundation Layer | 固定 |
| **Molecule** | セグメント単位の意味構造（Atom + Formula） | Phase 8 | セグメント |
| **Island** | 書き方が統計的に類似した**セクション群**のクラスタ | Phase 9 (W5) | セクション |
| **Cell** | 条件因子で結合された Molecule + Island | 統合層 | 可変 |
| **Organ** | 上位条件因子でグループ化された Cell 群 | 統合層 | 可変 |
| **Ecosystem** | 全体の意味構造 + LLM による言語化 | 出力層 | 全体 |

**v0.1 → v2.0 の変化:** Island の定義が「共鳴ベクトルが類似した**記事群**」から「書き方が統計的に類似した**セクション群**」に変わった。Phase 9 の分析単位がセクション単位に確定したことによる。

---

## 4. Phase 9 パイプライン（実装済み）

### 4.1 W 層の定義（v2.0 確定版）

v0.1 では W0〜W6 が概念的に定義されていたが、実装を通じて以下に確定した:

| W層 | 名称 | 入力 | 出力 | 説明 |
|-----|------|------|------|------|
| **W1** | Feature Extraction | 生テキスト | 20次元トークン特徴 | spaCy による品詞/形態素解析。各トークンに20次元ベクトルを付与 |
| **W2** | Conditional Aggregation | W1 特徴 + 条件因子 | 条件別統計 | ConditionProvider が抽出した条件（セクション名等）ごとに特徴を集約 |
| **W3** | Profile Computation | W2 統計 | z-score プロファイル | 条件間の偏差を z-score で正規化。セクションの「個性」を数値化 |
| **W4** | Similarity Computation | W3 プロファイル | 全ペア類似度行列 | コサイン類似度（z-score ベース）。N 条件から N(N-1)/2 ペアを計算 |
| **W5** | Island Formation | W4 類似度 + Threshold + EdgeFilter | Island 構造 | 閾値フィルタ → エッジ選択 → 連結成分 → Island |
| **W6** | Export | W5 構造 | JSON/MD/CSV | 構造化データ + 人間可読レポート + k-sweep テーブル |

**W0（旧定義: データ正規化）は削除。** Harvester モジュールが担当する前処理であり、W 層の責務ではない。

### 4.2 条件因子（v2.0: 内部構造ベース）

v0.1 では条件因子を `source_type` / `language_profile` / `time_bucket` といった**外部メタデータ**で定義していた。v2.0 では**テキストの内部構造**から動的に抽出される:

```python
# v0.1（旧: 外部メタデータ）
{
    "source_type": "news",        # データ取得時に決定
    "language_profile": "en",     # 環境属性
    "time_bucket": "2026-01",     # 時間属性
}

# v2.0（現: 内部構造）
# SectionConditionProvider → セクション名が条件
"cao_cao__early_life"         # article_id__section_name
"san_francisco__demographics"

# DocumentConditionProvider → ドキュメント名が条件
"cao_cao"                     # article_id
"san_francisco"
```

**利用可能な ConditionProvider:**

| Provider | 条件軸 | 用途 |
|----------|--------|------|
| `SectionConditionProvider` | セクション名 | セクション横断のパターン発見 |
| `DocumentConditionProvider` | ドキュメント名 | 記事レベルの意味的類似性 |
| `PassiveConditionProvider` | 受動態(0/1) | 文体分析 |
| `ParenthesesConditionProvider` | 括弧内(0/1) | 注釈パターン |
| `QuoteConditionProvider` | 引用文内(0/1) | 引用パターン |
| `ProperNounConditionProvider` | 固有名詞有無(0/1) | 人名・地名の影響 |

### 4.3 Lens（レンズ）

Lens = (ConditionProvider, FeatureMode) のペア。同じデータを異なる角度から観測する。

| Lens | Condition | Feature Mode | 何が見えるか |
|------|-----------|-------------|-------------|
| **Structure** | Section | Token (S-Score) | Wikipedia のテンプレート構造 |
| **Semantic** | Document | Vector (20-dim) | 記事間の主題的類似性 |
| **Hybrid** | Section | Vector (20-dim) | セクション内の意味的偏り |

**Feature Mode の違い:**
- **Token mode**: トークン頻度 → S-Score → 共鳴ベクトル（次元数 = Atom 数）
- **Vector mode**: 20次元特徴ベクトル平均 → z-score プロファイル → コサイン類似度

### 4.4 Threshold（閾値: 動的3層構造）

v0.1 にはなかった設計。固定閾値（0.9 等）が Lens ごとに機能しない問題を解決:

```
t_abs = Q_global(q)     # 全履歴の分位（蓄積型）
t_rel = Q_run(q)        # 今回実行の分位（データ適応型）
t_resolved = max(t_abs, t_rel, floor)  # 合成（安全優先）

初回実行: t_abs = fallback（データなし）
2回目〜: t_abs が蓄積データから算出される → 経験的な普遍閾値
```

全ての決定は trace として記録:
- mode (quantile/fixed)
- t_abs, t_rel, t_resolved
- 分布統計 (min/mean/max/std)
- サンプル数 (n_pairs)
- abs_source (global/fallback, n_global)

### 4.5 Edge Selection（Mutual-kNN + k-sweep）

v0.1 にはなかった設計。単連結（single-linkage）の連鎖問題を解決:

**問題:** 閾値だけでは「A-B-C-D-...-Z」と一方向の弱い類似で全ノードが一つの巨大成分に合流する（Hybrid lens で 475/492 が 1 island に）。

**解決: Mutual-kNN**
- エッジ (i,j) を保持する条件: j が i の top-k 近傍 **かつ** i が j の top-k 近傍 **かつ** sim(i,j) ≥ threshold
- 一方向の弱い親和性チェーンが切断される

**k-sweep（EdgePolicyResolver）:**
- 候補 k = [2, 3, 4, 5, 7, 9, 12, 15]
- 各 k で W5 を実行し、指標を観測（islands, gcr, mean_intra_sim 等）
- **最小の k で制約を満たすものを選択**（連鎖回避 + 最大分解能）
- 制約: max_giant_ratio ≤ 0.20, min_mean_intra ≥ 0.25

**実験結果（混合データセット 15記事 492セクション, Hybrid lens）:**

```
  k   edges  islands  noise  largest     gcr  mean_intra   ok
  2     242      108    147       10  0.0203      0.5884    ✓ ←
  3     387       61     80       62  0.1260      0.3639    ✓
  4     540       22     49      323  0.6565      0.0110     
  5     670       10     32      440  0.8943      0.0040     
```

**相転移点の発見:** k=3 → k=4 で largest island が 62 → 323 に急増（gcr: 0.13 → 0.66）。これはパラメータチューニングではなく、ネットワークの**固有のパーコレーション閾値**。

---

## 5. 処理フロー

### 5.1 並列処理モデル（v2.0 更新）

```
┌──────────────────────────────────────────────────────────────┐
│                      入力テキスト群                             │
│         （Wikipedia 記事 × N、各記事にセクション群）              │
└──────────────────────────────────────────────────────────────┘
                             │
             ┌───────────────┴───────────────┐
             ↓                               ↓
┌──────────────────────────┐   ┌──────────────────────────────┐
│  Phase 8（強い意味）       │   │  Phase 9（弱い意味）           │
│                          │   │                              │
│  1. セグメント境界検出      │   │  Lens 選択                    │
│     （LLM）               │   │    ↓                         │
│  2. 原子化（WordNet/326） │   │  W1: 20次元トークン特徴抽出     │
│  3. 分子化（LLM + Synapse）│   │    ↓                         │
│                          │   │  W2: ConditionProvider で集約   │
│  出力: Molecule 群         │   │    ↓                         │
│       + segment_id       │   │  W3: z-score プロファイル       │
│       （動的条件因子）      │   │    ↓                         │
│                          │   │  W4: コサイン類似度行列         │
│                          │   │    ↓                         │
│                          │   │  W5: Threshold + Mutual-kNN    │
│                          │   │      + k-sweep → Island 形成   │
│                          │   │    ↓                         │
│                          │   │  W6: Export (JSON/MD/CSV)      │
│                          │   │                              │
│                          │   │  出力: Island 群               │
│                          │   │       + Chaining Metrics      │
│                          │   │       + Threshold Trace       │
│                          │   │       + k-sweep Table         │
└──────────────────────────┘   └──────────────────────────────┘
             │                               │
             │      互いに侵食しない            │
             │      別々の観測結果              │
             └───────────────┬───────────────┘
                             ↓
┌──────────────────────────────────────────────────────────────┐
│                 統合層（条件因子による結合）                       │
│                                                              │
│  条件因子「section_name」で引く:                                │
│    Molecule(early_life) + Island(early_lifeが属す島)            │
│    → Cell 形成                                                │
│                                                              │
│  条件因子「article_id」で引く:                                  │
│    Cell群(cao_caoに属する) → Organ                             │
└──────────────────────────────────────────────────────────────┘
                             ↓
┌──────────────────────────────────────────────────────────────┐
│                 出力層（LLM 言語化）                             │
│                                                              │
│  ESDE の観測結果を自然言語レポートに変換                          │
│  ※ 自己流の解釈を加えない（材料外の推測をしない）                  │
└──────────────────────────────────────────────────────────────┘
```

### 5.2 重要な制約

| 制約 | 説明 |
|------|------|
| **Phase 8 → Phase 9 への流入禁止** | Molecule の情報が W 層に影響しない |
| **Phase 9 → Phase 8 への流入禁止** | Island の情報が Molecule 生成に影響しない |
| **統合は条件因子のみで行う** | 結合ロジックに意味解釈を含めない |
| **全判断は trace として記録** | Threshold 決定、k 選択、エッジフィルタの全てが再現・検証可能 |

---

## 6. Cell の構造定義

### 6.1 Cell スキーマ（v2.0 更新案）

```python
@dataclass
class Cell:
    """
    条件因子で結合された Molecule + Island。
    
    Phase 8 と Phase 9 の観測結果を並列で保持。
    両者は混ぜない。
    """
    
    # Identity
    cell_id: str
    
    # 結合に使用した条件因子
    binding_factor: Dict[str, Any]
    # e.g., {"section_name": "early_life", "article_id": "cao_cao"}
    
    # Phase 8 からの観測（強い意味）
    molecules: List[Molecule]
    
    # Phase 9 からの観測（弱い意味）
    z_score_profile: Optional[Dict[str, float]]  # 20次元 z-score
    island_membership: Optional[str]  # 所属 island_id（noise なら None）
    cohesion_score: Optional[float]   # 島内結束度
    
    # Lens 情報（どのレンズで観測したか）
    lens_name: str   # "structure" / "semantic" / "hybrid"
    k_used: int      # Mutual-kNN の k（焦点距離）
    
    # メタデータ
    source_segment: Optional[str]
    created_at: str
```

### 6.2 v0.1 → v2.0 の変化

| 項目 | v0.1 | v2.0 |
|------|------|------|
| binding_factor | `{"segment_id": "seg_0042"}` | `{"section_name": "early_life", "article_id": "cao_cao"}` |
| Phase 9 側 | `resonance_pattern` (dict) + `related_islands` (list) | `z_score_profile` (20-dim) + `island_membership` (single) + `cohesion_score` |
| Lens 情報 | なし | `lens_name` + `k_used` |

**重要な変化:** 1 セクション = 1 Island に帰属（多対多ではなく多対一）。セクションは高々1つの Island に属するか、noise（どこにも属さない）。これにより v0.1 で未解決だった「多対多の関係をどう扱うか」が解消された。

---

## 7. ESDE と LLM の分業

### 7.1 役割分担（v0.1 から不変）

```
┌──────────────────────────────────────────────────────────────┐
│  ESDE（観測層）                                                │
│                                                              │
│  責務:                                                        │
│    ・Phase 8: Molecule 生成（強い意味の構造化）                   │
│    ・Phase 9: Island/Evidence 抽出（弱い意味のパターン）           │
│    ・条件因子による結合（Cell/Organ 形成）                        │
│                                                              │
│  出力: 構造化されたデータ（JSON, スキーマ準拠, 検証可能）           │
│                                                              │
│  哲学: "記述せよ、しかし決定するな"                               │
└──────────────────────────────────────────────────────────────┘
                             ↓
                   解釈の材料を徹底的に提供
                             ↓
┌──────────────────────────────────────────────────────────────┐
│  LLM（言語化層）                                               │
│                                                              │
│  責務:                                                        │
│    ・ESDE の観測結果に基づいてレポート生成                        │
│    ・人間に分かりやすい自然言語で出力                              │
│                                                              │
│  制約:                                                        │
│    ・自己流の解釈を加えない                                      │
│    ・材料外の推測をしない                                        │
│    ・ESDE が提供した情報の範囲内で言語化                          │
│                                                              │
│  出力: 自然言語レポート（柔軟、読みやすさ重視）                    │
└──────────────────────────────────────────────────────────────┘
```

### 7.2 厳密さのグラデーション

| 層 | 厳密さ | 理由 |
|----|--------|------|
| Atom / Molecule | **厳密** | 機械的に検証可能、再現性必須 |
| Island / Threshold / Edge Policy | **厳密** | 統計的根拠、全工程が trace 記録 |
| Cell / Organ | **厳密** | 条件因子による機械的結合、混ぜない |
| 最終レポート | **柔軟** | 人間が読むもの、自然言語の強みを活かす |

---

## 8. Substrate との関係

### 8.1 Substrate の役割

Substrate Layer は Phase 9 パイプラインの**横断的基盤**として機能する:

- **Context Fabric**: 全 trace の append-only 格納
- **決定論的 context_id 生成**: 入力データから一意に ID を計算
- **条件因子の管理**: ConditionProvider が生成した条件の記録

### 8.2 trace として記録されるもの

| trace 種別 | 内容 | 記録タイミング |
|-----------|------|-------------|
| **threshold_trace** | t_abs / t_rel / t_resolved / 分布統計 / abs_source | W5 実行時 |
| **edge_filter_trace** | selector 名 / k 値 / edges_before / edges_after | W5 エッジフィルタ時 |
| **edge_policy_trace** | k_chosen / sweep_summary / selection_reason / policy 制約 | k-sweep 実行時 |
| **chaining_metrics** | gcr / mean_intra / edge_sparsity / chaining_detected | W5 完了後 |
| **global_model** | lens/feature_mode 別の累積類似度分布 | 実行ごとに追記 |

---

## 9. 実験的発見

Phase 9 実装を通じて得られた知見（設計判断の根拠となるもの）:

### 9.1 Wikipedia のテンプレートトポロジー

Structure lens で発見された3層構造:
- **Hub 層**: 同心円的セクション配置（都市記事: demographics, economy, transport...）
- **Narrative 層**: 時系列セクション配置（戦国武将: early life, campaign, legacy...）
- **Institutional 層**: 制度的セクション配置（組織・法律系）

**含意:** Phase 9 が検出しているのは「ジャンル分類」ではなく「編集パターンの構造的類型」。

### 9.2 相転移点（k=3 → k=4）

Mutual-kNN の k-sweep で発見:
- k ≤ 3: 分解状態（意味のある島が形成される）
- k ≥ 4: 連鎖状態（巨大成分が全体を支配する）
- **臨界点は k=4**（gcr が 0.13 → 0.66 に急増）

**含意:** この臨界点はデータ固有の性質であり、パラメータチューニングとは無関係。「レンズの焦点距離」としての k の解釈を裏付ける。

### 9.3 z-score ベースの類似度

W4 でコサイン類似度の入力を `mean_vector`（生平均）から `z_score_vector`（標準化偏差）に変更:
- **生平均の問題:** 大数の法則により全条件の平均ベクトルが global baseline に収束 → 全ペアの類似度が 1.0 に
- **z-score の効果:** baseline からの**偏差**を比較 → 意味のある分布（-0.82〜0.77）が出現

**含意:** 「何を知っているか」ではなく「何が偏っているか」を比較するのが正しい。

---

## 10. 未解決の課題

### 10.1 解決済み（v0.1 からの移行）

| v0.1 課題 | 解決策 |
|-----------|--------|
| Phase 9 の入力単位（記事 vs セグメント） | **セクション単位に確定**。Lens の ConditionProvider が粒度を決定 |
| Island と Segment の多対多関係 | **多対一に確定**。各セクションは高々1つの Island に帰属、または noise |
| 条件因子の階層（どの階層で結合するか） | **Lens が決定**。Structure/Hybrid = section、Semantic = document |

### 10.2 現存する課題

**Cell 形成の実装:**
- Phase 8 (Molecule) と Phase 9 (Island) を条件因子で結合するコードは未実装
- Cell スキーマは設計段階。Phase 10 以降の課題

**Edge Policy プリセット（profile）:**
- purity / balanced / overview 等の目的別プリセットは未実装
- 複数データセットの蓄積が前提。現状は k-sweep テーブルを人間が見て判断
- Phase 9 の範囲外（使いやすさの改善であり、設計思想の完成には不要）

**Lens 間の統合:**
- 3つの Lens が独立に Island を生成する。Lens 間の関係は未定義
- 同じセクションが Structure lens では island A、Hybrid lens では island B に属する場合の扱い

**global_model の成熟:**
- t_abs（絶対閾値）は累積データに基づく。現状は mixed dataset 1つ分のみ
- 十分なデータが蓄積されるまで fallback に頼る場面が多い

---

## 11. モジュール構成（実装リファレンス）

```
statistics/pipeline/
├── run_full_pipeline.py    # メインパイプライン（CLI）
├── lens.py                 # Lens 定義（Structure/Semantic/Hybrid）
├── condition_provider.py   # ConditionProvider 群（Section/Document/Passive/...）
├── w2_aggregator.py        # W2: 条件別集約
├── w3_calculator.py        # W3: S-Score（token mode）
├── w3_vector.py            # W3: z-score プロファイル（vector mode）
├── w4_projector.py         # W4: 共鳴ベクトル投影（token mode）
├── w4_vector.py            # W4: コサイン類似度（vector mode）
├── w5_w6_adapter.py        # W5: SimpleCondensator（島形成）+ W6 export
├── threshold.py            # ThresholdResolver（t_abs/t_rel/t_resolved）
├── global_model.py         # GlobalThresholdModel（累積分布管理）
├── edge_selector.py        # MutualKNNSelector / NoOpSelector
├── edge_policy.py          # EdgePolicyResolver（k-sweep + 自動選定）
└── chaining_metrics.py     # 連鎖診断指標（gcr, mean_intra, etc.）
```

---

## 12. 用語集（本文書で使用する主要概念）

| 用語 | 定義 |
|------|------|
| **Atom** | 326個の最小意味単位。163対称ペアで構成 |
| **Molecule** | Phase 8 が生成するセグメント単位の意味構造 |
| **Island** | Phase 9 (W5) が生成する、書き方が類似したセクション群のクラスタ |
| **Cell** | 条件因子で Molecule と Island を結合した統合単位 |
| **Lens** | (ConditionProvider, FeatureMode) のペア。観測の視点 |
| **Condition Factor** | テキスト内部構造から抽出される分類軸（セクション名等） |
| **z-score Profile** | global baseline からの偏差を標準化した20次元ベクトル |
| **Threshold (3層)** | t_abs（全履歴）+ t_rel（今回実行）→ t_resolved（合成） |
| **Mutual-kNN** | 双方向 k 近傍フィルタ。一方向の連鎖を防止 |
| **k-sweep** | 複数の k で W5 を試行し、最小適格 k を選定 |
| **Percolation Threshold** | k-sweep で観測される相転移点（島 → 巨大成分の臨界） |
| **Giant Component Ratio (gcr)** | 最大島サイズ / 総ノード数。連鎖の程度を示す |
| **Noise** | どの Island にも属さないセクション。正当な観測結果 |

---

*Document generated from implementation record of ESDE Phase 9 (v1.0→v1.9)*  
*Philosophy: Aruism — "Describe, but do not decide"*
