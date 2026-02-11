# Project Lexicon: 統一実装仕様書

**Version**: 1.1.1  
**Date**: 2026-02-09  
**Status**: Taka 承認済み → 実装開始  
**Sources**: Gemini 設計判断 + GPT 監査統合案 + Claude 実装知見  
**v1.1 Changes**: GPT 監査 7 点パッチ適用（words 型統一、lemma::pos キー、汎用 Grounder、synonym status 管理）  
**v1.1.1 Changes**: Final Patch A(POS正規形), B(Index=core only), C(正規化2段階防御), D(Evidence/wordsキー整合), E(Masterメタ)

---

## 0. 憲法（この 1 文が全ての判断基準）

> **Atom は座標（位置）であり意味ではない。Lexicon は自然言語を座標へ接続する入口。**
> **WordNet/コーパスは入口の「検証・補強・拡張」に限定する。**
> **Grounder は Lexicon Core のみを参照し、Evidence は監査と改善のためだけに使う。**

---

## 1. データ資産の分離

### 1.1 三資産構成

```
(A) Definition Pack（不変資産）
    ├── categories_v1.json      ← Layer 1: 24 カテゴリ定義
    ├── axes_levels_v1.json     ← Layer 3-4: 10 軸 × 48 レベル定義
    └── atoms_v1.json           ← Layer 2: 326 Atom（漢字意味場 + 対称ペア）

(B) Lexicon Master（運用資産）
    ├── lexicon_master.json     ← 全 Atom × 48 slot + evidence（重くてよい）
    └── lexicon_conflicts.json  ← 対称ペア衝突・多義語衝突の記録

(C) Runtime Index（生成物）
    └── lexicon_index.json      ← lemma → (atom, axis, level) の逆引き（軽量）
```

**Master と Index を最初から分ける**（GPT 案）。
Grounder は (C) だけ読む。(B) は監査・拡張時のみ参照。

### 1.2 正規化ルールとキー体系（GPT Patch #1,#2,#4 + Final Patch A,C）

全データ資産を通じて以下の正規化ルールを適用する:

```
Lemma 正規化:
  - lowercase
  - 空白正規化（連続スペース → 単一スペース、trim）
  - WordNet 由来の underscore → space（"ice_cream" → "ice cream"）
  - spaCy lemma を正とする

POS 正規形（Final Patch A）:
  - 内部 POS は n/v/adj/adv の 4 種のみ
  - WordNet a/s → adj, r → adv に強制マップ
  - spaCy PROPN → n 扱い（固有名詞は名詞として正規化）
  - 変換テーブル: {"n":"n", "v":"v", "a":"adj", "s":"adj", "r":"adv"}

正規化の責務（Final Patch C）:
  - 正規化は「候補取り込み時」と「Index 生成時」の 2 段階で必ず実施（二重防御）
  - Master 内の words[].w も正規化済みを原則とする（入力時に正規化）
  - 同一 slot 内で正規化後に重複する語は 1 つに統合

Key 体系:
  - Master の words: dict 形式 {w, pos, reason, status} に統一
  - Evidence のキーと words の (w, pos) は必ず同一形式を使用（Final Patch D）
    例: words に {"w":"infatuation","pos":"n"} → evidence キーは "infatuation::n"
  - Runtime Index のキー: "{lemma}::{pos}"（pos は正規形 n/v/adj/adv）
  - 対称衝突チェック: lemma::pos 単位で判定
  - Synonym expansion: status:"proposed" で登録。Taka 承認後に "core" 化

Status 値:
  - "core": 人間が承認済み。Grounder が参照する
  - "proposed": LLM/WordNet が提案。レビュー待ち
  - "rejected": レビューで却下。記録として保持
```

### 1.3 ディレクトリ構成

```
integration/lexicon/
├── definitions/
│   ├── categories_v1.json
│   ├── axes_levels_v1.json
│   └── atoms_v1.json
├── master/
│   └── lexicon_master.json
├── runtime/
│   └── lexicon_index.json
├── scripts/
│   ├── generate_candidates.py      ← QwQ 候補生成
│   ├── wordnet_evidence_builder.py ← WordNet 検証 + 同義語拡張
│   ├── conflict_checker.py         ← 対称ペア衝突 + カテゴリ不整合検出
│   ├── build_runtime_index.py      ← Master → Index 変換
│   └── pilot_report.py             ← レビュー用レポート生成
├── grounder.py                     ← LexiconGrounder（SynapseGrounder 置換）
└── README.md
```

---

## 2. Definition Pack スキーマ

### 2.1 categories_v1.json

```json
{
  "version": "1.0",
  "categories": {
    "EMO": {
      "name_en": "Emotion",
      "name_ja": "感情",
      "description_en": "Emotional states and feelings",
      "description_ja": "感情の状態と情動"
    },
    "ACT": {
      "name_en": "Action",
      "name_ja": "行為",
      "description_en": "Actions and movements",
      "description_ja": "行為と運動"
    }
  }
}
```

### 2.2 axes_levels_v1.json

```json
{
  "version": "1.0",
  "axes": {
    "temporal": {
      "name_ja": "時間的条件",
      "definition_ja": "物事が時間の経過と共にどのように存在し、変化し、あるいはその状態を維持するかのパターンを問う視点。",
      "definition_en": "The pattern of how things exist, change, or maintain their state over time.",
      "levels": {
        "emergence": {
          "name_ja": "発生",
          "definition_ja": "物事が初めて現れる、生じる瞬間",
          "definition_en": "The moment when something first appears or comes into being"
        },
        "indication": {
          "name_ja": "兆候",
          "definition_ja": "まだ顕在化していないが、変化の予兆が現れている段階",
          "definition_en": "The stage where signs of change appear but have not yet fully manifested"
        },
        "influence": {
          "name_ja": "影響",
          "definition_ja": "現れた物事が周囲に作用を及ぼし始める段階",
          "definition_en": "The stage where the emerged thing begins to exert effects on its surroundings"
        },
        "transformation": {
          "name_ja": "変容",
          "definition_ja": "影響を受けた物事が質的に変化する段階",
          "definition_en": "The stage where things undergo qualitative change due to influence"
        },
        "establishment": {
          "name_ja": "定着",
          "definition_ja": "変容した状態が安定し、定まる段階",
          "definition_en": "The stage where the transformed state stabilizes and settles"
        },
        "continuation": {
          "name_ja": "継続",
          "definition_ja": "定着した状態が持続する段階",
          "definition_en": "The stage where the established state persists"
        },
        "permanence": {
          "name_ja": "永続",
          "definition_ja": "時間を超えて存在し続ける、恒久的な段階",
          "definition_en": "The stage of enduring existence beyond time, permanence"
        }
      }
    },
    "scale": {
      "name_ja": "空間的・スケール的条件",
      "definition_ja": "ある事象が、どの範囲・規模において観測され、影響を及ぼすかを問う視点。",
      "definition_en": "The scope and scale at which a phenomenon is observed and exerts influence.",
      "levels": {
        "individual": {
          "name_ja": "個人",
          "definition_ja": "一個人の内面や行動の範囲",
          "definition_en": "The scope of a single individual's inner world and actions"
        },
        "community": {
          "name_ja": "共同体",
          "definition_ja": "家族・近隣・集団など直接的な関係の範囲",
          "definition_en": "The scope of direct relationships: family, neighborhood, groups"
        },
        "society": {
          "name_ja": "社会",
          "definition_ja": "社会制度・国家・文化圏の範囲",
          "definition_en": "The scope of social institutions, nations, cultural spheres"
        },
        "ecosystem": {
          "name_ja": "生態系",
          "definition_ja": "生態系・地球環境の範囲",
          "definition_en": "The scope of ecosystems and global environment"
        },
        "stellar": {
          "name_ja": "星系",
          "definition_ja": "惑星系・恒星系の範囲",
          "definition_en": "The scope of planetary and stellar systems"
        },
        "cosmic": {
          "name_ja": "宇宙",
          "definition_ja": "宇宙全体・存在の根源的スケール",
          "definition_en": "The scope of the entire cosmos, the fundamental scale of existence"
        }
      }
    },
    "epistemological": {
      "name_ja": "認識論的条件",
      "definition_ja": "ある存在が、意識によってどのように捉えられ、意味を与えられ、新たな認識へと至るかのプロセスを問う視点。",
      "definition_en": "The process by which consciousness apprehends, assigns meaning, and arrives at new understanding.",
      "levels": {
        "perception": {
          "name_ja": "知覚",
          "definition_ja": "感覚を通じて存在を捉える段階",
          "definition_en": "The stage of apprehending existence through the senses"
        },
        "identification": {
          "name_ja": "識別",
          "definition_ja": "知覚したものを区別し、名付ける段階",
          "definition_en": "The stage of distinguishing and naming what has been perceived"
        },
        "understanding": {
          "name_ja": "理解",
          "definition_ja": "識別したものの意味や構造を把握する段階",
          "definition_en": "The stage of grasping the meaning and structure of what has been identified"
        },
        "experience": {
          "name_ja": "体験",
          "definition_ja": "理解を超えて自己の中に統合する段階",
          "definition_en": "The stage of integrating understanding into oneself beyond mere comprehension"
        },
        "creation": {
          "name_ja": "創造",
          "definition_ja": "体験から新たな認識・概念を生み出す段階",
          "definition_en": "The stage of generating new cognition and concepts from experience"
        }
      }
    },
    "ontological": {
      "name_ja": "存在論的条件",
      "definition_ja": "ある存在が、どのようなレイヤーで構成されているのか、その成り立ちそのものを問う視点。",
      "definition_en": "The layers of which an entity is composed — its very constitution.",
      "levels": {
        "material": {
          "name_ja": "物質",
          "definition_ja": "物理的・物質的なレイヤー",
          "definition_en": "The physical, material layer"
        },
        "informational": {
          "name_ja": "情報",
          "definition_ja": "情報・データとしてのレイヤー",
          "definition_en": "The layer of information and data"
        },
        "relational": {
          "name_ja": "関係",
          "definition_ja": "他の存在との関係性のレイヤー",
          "definition_en": "The layer of relationships with other entities"
        },
        "structural": {
          "name_ja": "構造",
          "definition_ja": "パターン・構造としてのレイヤー",
          "definition_en": "The layer of patterns and structures"
        },
        "semantic": {
          "name_ja": "意味",
          "definition_ja": "意味・価値としてのレイヤー",
          "definition_en": "The layer of meaning and value"
        }
      }
    },
    "interconnection": {
      "name_ja": "連動性の条件",
      "definition_ja": "複数の存在が、互いにどのように影響を与え合い、関係性が深化・発展していくかの動的なプロセスを問う視点。",
      "definition_en": "The dynamic process by which multiple entities influence each other and deepen their relationships.",
      "levels": {
        "independent": {
          "name_ja": "独立的",
          "definition_ja": "他の存在から独立して存在する段階",
          "definition_en": "The stage of existing independently from other entities"
        },
        "catalytic": {
          "name_ja": "触発的",
          "definition_ja": "他の存在に触発され、変化の契機が生まれる段階",
          "definition_en": "The stage where contact with another entity triggers the opportunity for change"
        },
        "chained": {
          "name_ja": "連鎖的",
          "definition_ja": "因果の連鎖として影響が波及する段階",
          "definition_en": "The stage where influence propagates as a chain of cause and effect"
        },
        "synchronous": {
          "name_ja": "同期的",
          "definition_ja": "複数の存在が同期的に変化する段階",
          "definition_en": "The stage where multiple entities change synchronously"
        },
        "resonant": {
          "name_ja": "共振的",
          "definition_ja": "存在同士が深く共鳴し合う段階",
          "definition_en": "The stage where entities deeply resonate with each other"
        }
      }
    },
    "resonance": {
      "name_ja": "共鳴度の条件",
      "definition_ja": "ある繋がりや関係性が、どの程度の深さで本質に触れているのか、その質的な度合いを問う視点。",
      "definition_en": "The qualitative depth to which a connection or relationship touches the essence.",
      "levels": {
        "superficial": {
          "name_ja": "表層的",
          "definition_ja": "表面的・一時的な繋がり",
          "definition_en": "A surface-level, temporary connection"
        },
        "structural": {
          "name_ja": "構造的",
          "definition_ja": "構造レベルでの対応・類似",
          "definition_en": "Correspondence or similarity at the structural level"
        },
        "essential": {
          "name_ja": "本質的",
          "definition_ja": "本質に触れる深い繋がり",
          "definition_en": "A deep connection that touches the essence"
        },
        "existential": {
          "name_ja": "存在的",
          "definition_ja": "存在そのものに関わる根源的な繋がり",
          "definition_en": "A fundamental connection involving existence itself"
        }
      }
    },
    "symmetry": {
      "name_ja": "対称性との関係条件",
      "definition_ja": "対立または補完しあう対称的な力が、どのような相互作用を生み出すかを問う視点。",
      "definition_en": "The interactions produced by opposing or complementary symmetric forces.",
      "levels": {
        "destructive": {
          "name_ja": "破壊的",
          "definition_ja": "対称的な力が互いを否定・破壊する関係",
          "definition_en": "A relationship where symmetric forces negate or destroy each other"
        },
        "inclusive": {
          "name_ja": "包含的",
          "definition_ja": "一方が他方を包含・吸収する関係",
          "definition_en": "A relationship where one subsumes or absorbs the other"
        },
        "transformative": {
          "name_ja": "変容的",
          "definition_ja": "両者の相互作用により質的変容が生じる関係",
          "definition_en": "A relationship where interaction between both produces qualitative transformation"
        },
        "generative": {
          "name_ja": "生成的",
          "definition_ja": "対称的な力から新たなものが生成される関係",
          "definition_en": "A relationship where something new is generated from symmetric forces"
        },
        "cyclical": {
          "name_ja": "循環的",
          "definition_ja": "対称的な力が循環的に交替する関係",
          "definition_en": "A relationship where symmetric forces alternate cyclically"
        }
      }
    },
    "lawfulness": {
      "name_ja": "法則性の条件",
      "definition_ja": "ある現象を支配するルールの性質が、単純な因果律か、複雑系か、あるいは根源的な必然性かを問う視点。",
      "definition_en": "The nature of the rules governing a phenomenon: simple causality, complex emergence, or fundamental necessity.",
      "levels": {
        "predictable": {
          "name_ja": "予測可能",
          "definition_ja": "単純な因果律で予測可能な法則",
          "definition_en": "Laws predictable through simple causality"
        },
        "emergent": {
          "name_ja": "創発的",
          "definition_ja": "複雑系から創発する予測困難な法則",
          "definition_en": "Laws that emerge from complex systems and are difficult to predict"
        },
        "contingent": {
          "name_ja": "偶発的",
          "definition_ja": "偶然や条件依存で生じる法則",
          "definition_en": "Laws that arise from chance or conditional dependencies"
        },
        "necessary": {
          "name_ja": "必然的",
          "definition_ja": "存在の根源に由来する必然的法則",
          "definition_en": "Laws arising necessarily from the very foundation of existence"
        }
      }
    },
    "experience": {
      "name_ja": "体験の質的条件",
      "definition_ja": "ある認識が、意識にとってどのような質的な体験として現れるかを問う視点。",
      "definition_en": "The qualitative character of how a cognition manifests as an experience for consciousness.",
      "levels": {
        "discovery": {
          "name_ja": "発見として",
          "definition_ja": "未知のものと出会う体験",
          "definition_en": "The experience of encountering the unknown"
        },
        "creation": {
          "name_ja": "創造として",
          "definition_ja": "新たなものを生み出す体験",
          "definition_en": "The experience of bringing something new into being"
        },
        "comprehension": {
          "name_ja": "了解として",
          "definition_ja": "存在の了解に至る体験（究極的理解）",
          "definition_en": "The experience of arriving at the comprehension of existence (ultimate understanding)"
        }
      }
    },
    "value_generation": {
      "name_ja": "価値生成の条件",
      "definition_ja": "ある存在や行為が、どのような基準で「価値」を持つとされるかを問う視点。実用性から始まり、感性、社会規範、そして根源的存在「ある」との共鳴に至る価値の階層性を問う。",
      "definition_en": "The criteria by which an entity or act is deemed to have value, progressing from utility through aesthetics and ethics to resonance with primordial existence.",
      "levels": {
        "functional": {
          "name_ja": "機能的",
          "definition_ja": "実用性・有用性に基づく価値",
          "definition_en": "Value based on utility and usefulness"
        },
        "aesthetic": {
          "name_ja": "美的",
          "definition_ja": "感性・美意識に基づく価値",
          "definition_en": "Value based on sensibility and aesthetic awareness"
        },
        "ethical": {
          "name_ja": "倫理的",
          "definition_ja": "社会規範・道義に基づく価値",
          "definition_en": "Value based on social norms and moral duty"
        },
        "sacred": {
          "name_ja": "聖性的",
          "definition_ja": "根源的存在「ある」との共鳴に基づく価値（Aruism 造語）",
          "definition_en": "Value based on resonance with primordial existence 'Aru' (Aruism neologism)"
        }
      }
    }
  }
}
```

### 2.3 atoms_v1.json（Pilot: EMO.like + EMO.dislike）

```json
{
  "version": "1.0",
  "atoms": {
    "EMO.like": {
      "category": "EMO",
      "en_label": "like",
      "kanji": "好",
      "kanji_semantic_field_en": [
        "favorable disposition, goodwill (好意)",
        "personal preference, taste (好み)",
        "desirability, agreeableness (好ましい)",
        "liking, fondness, being fond of (好き)",
        "favorable, positive, auspicious (好)"
      ],
      "short_definition_ja": "好意・愛着・親しみの位置を表す座標",
      "short_definition_en": "Coordinate representing favorable regard, attachment, and affinity",
      "symmetric_pair": "EMO.dislike"
    },
    "EMO.dislike": {
      "category": "EMO",
      "en_label": "dislike",
      "kanji": "嫌",
      "kanji_semantic_field_en": [
        "aversion, distaste (嫌悪)",
        "reluctance, unwillingness (嫌がる)",
        "repugnance, disgust (嫌気)",
        "disliking, being put off (嫌い)",
        "unpleasant, disagreeable (嫌な)"
      ],
      "short_definition_ja": "嫌悪・忌避・不快の位置を表す座標",
      "short_definition_en": "Coordinate representing aversion, avoidance, and displeasure",
      "symmetric_pair": "EMO.like"
    }
  }
}
```

---

## 3. Lexicon Master スキーマ

### 3.1 1 Atom の構造

```json
{
  "meta": {
    "version": "0.1.0",
    "generated_at": "2026-02-09T...",
    "pilot_atoms": ["EMO.like", "EMO.dislike"],
    "definition_pack_version": "1.0"
  },
  "EMO.like": {
    "slots": {
      "temporal.emergence": {
        "words": [
          {"w": "infatuation", "pos": "n", "reason": "sudden onset passion", "status": "core"},
          {"w": "crush", "pos": "n", "reason": "informal sudden attraction", "status": "core"},
          {"w": "attraction", "pos": "n", "reason": "initial pull toward", "status": "core"}
        ],
        "na": false,
        "evidence": {
          "infatuation::n": {
            "wordnet": {
              "synsets": ["infatuation.n.01"],
              "hypernym_chain": ["love", "emotion", "feeling", "state"],
              "antonyms": [],
              "entailments": [],
              "definition": "a foolish and usually extravagant passion or love or admiration"
            },
            "placement_reason": "sudden onset passion; defined with 'extravagant' suggesting temporal emergence",
            "corpus_collocations": ["develop", "sudden", "brief", "fade"]
          }
        },
        "source": "qwq_v1 + wordnet_validation + claude_review",
        "reviewed_by": null,
        "reviewed_at": null
      },
      "scale.stellar": {
        "words": [],
        "na": true,
        "na_reason": "Direct application not applicable; metaphorical use deferred to Phase 9 context analysis",
        "evidence": {},
        "source": "qwq_v1",
        "reviewed_by": null,
        "reviewed_at": null
      }
    }
  }
}
```

**N/A ルール（GPT 統一案）:**
- Pilot は **直義モード**: 一般的用法でノイズになるスロットは `na: true`
- 将来: 比喩モード（SF/宗教/神話等）で拡張。ドメイン別 Lexicon として Phase 9 側で管理
- N/A は「今は観測タグとして採用しない」宣言。概念の否定ではない

### 3.2 Runtime Index（Grounder 用逆引き）

```json
{
  "version": "1.0",
  "generated_from": "lexicon_master.json",
  "generated_at": "2026-02-09T...",
  "index": {
    "infatuation::n": [
      {"atom": "EMO.like", "axis": "temporal", "level": "emergence"}
    ],
    "crush::n": [
      {"atom": "EMO.like", "axis": "temporal", "level": "emergence"},
      {"atom": "EMO.like", "axis": "resonance", "level": "superficial"}
    ],
    "crush::v": [
      {"atom": "ACT.destroy", "axis": "symmetry", "level": "destructive"}
    ],
    "love::n": [
      {"atom": "EMO.like", "axis": "temporal", "level": "permanence"},
      {"atom": "EMO.like", "axis": "resonance", "level": "essential"},
      {"atom": "EMO.love", "axis": "resonance", "level": "existential"}
    ]
  }
}
```

1 lemma::pos が複数の (atom, axis, level) に出現しうる。これは正常。
同一 lemma でも POS が違えば別キー（crush::n ≠ crush::v）。
Grounder は全候補を返し、上流（Phase 8 等）が文脈で選択する。

> **Index 生成ルール（Final Patch B）**: `status == "core"` の語のみ Index に入れる。
> proposed / rejected は Master に保持するが Index には絶対に出さない。

---

## 4. LLM 候補生成仕様

### 4.1 QwQ 接続情報

```python
LLM_HOST = "http://100.107.6.119:8001/v1"
LLM_MODEL = "qwq32b_tp2_long32k_existing"
LLM_TIMEOUT = 180
LLM_MAX_TOKENS = 16000
LLM_TEMPERATURE = 0.3
```

### 4.2 QwQ Prompt テンプレート

```python
SYSTEM_PROMPT_LEXICON = """You are assisting with the construction of a Lexicon for ESDE
(Emergent Semantic Data Engine), a semantic coordinate system rooted in Aruism philosophy.

CRITICAL RULES:
1. Atoms are POSITIONS (coordinates), not meanings. You are mapping words to positions.
2. Propose English words (nouns, verbs, adjectives, adverbs) for each axis × level slot.
3. For each word, give a 1-sentence reason for that placement.
4. If a slot is not applicable to this Atom, write "N/A" with a brief reason.
5. Do NOT propose words that would better belong to the symmetric pair.
6. Stick to direct/literal usage. Metaphorical or domain-specific usage is out of scope.
7. Aim for 5-15 words per applicable slot.

Output format (strict JSON):
{
  "atom": "EMO.like",
  "slots": {
    "temporal.emergence": {
      "words": [
        {"word": "infatuation", "pos": "noun", "reason": "sudden onset of passionate liking"},
        ...
      ],
      "na": false
    },
    "temporal.indication": {
      "words": [],
      "na": true,
      "na_reason": "..."
    },
    ...all 48 slots...
  }
}

Think step by step. Output FINAL: followed by the JSON only."""

def build_user_prompt(atom_def: dict, axes_levels: dict, category_def: dict) -> str:
    """Build user prompt with full 4-layer context."""
    
    # Layer 1: Category
    cat = category_def
    cat_block = f"""### Category
Code: {atom_def['category']}
Name: {cat['name_en']} ({cat['name_ja']})
Description: {cat['description_en']} ({cat['description_ja']})"""

    # Layer 2: Atom + Kanji
    atom = atom_def
    kanji_field = "\n".join(f"  - {f}" for f in atom['kanji_semantic_field_en'])
    atom_block = f"""### Atom
ID: {atom['category']}.{atom['en_label']}
Kanji: {atom['kanji']}
Kanji semantic field (connotations in original Japanese):
{kanji_field}
Definition: {atom['short_definition_en']}

Symmetric pair: {atom['symmetric_pair']}
WARNING: Do NOT propose words that belong to {atom['symmetric_pair']} (the opposite coordinate)."""

    # Layer 3-4: All axes and levels
    axes_block = "### Axes and Levels (all 10 axes, 48 levels total)\n\n"
    slot_list = []
    for axis_id, axis_def in axes_levels['axes'].items():
        axes_block += f"**{axis_id}** — {axis_def['name_ja']} ({axis_def['definition_en']})\n"
        for level_id, level_def in axis_def['levels'].items():
            axes_block += f"  - {level_id}: {level_def['name_ja']} — {level_def['definition_en']}\n"
            slot_list.append(f"{axis_id}.{level_id}")
        axes_block += "\n"

    # Slots to fill
    slots_block = "### Slots to fill:\n\n"
    for slot in slot_list:
        slots_block += f"{slot}:\n"

    return f"""{cat_block}

{atom_block}

{axes_block}

{slots_block}

Provide words for EACH of the {len(slot_list)} slots above.
Mark inapplicable slots as na: true with a reason."""
```

### 4.3 QwQ 応答パース

```python
def parse_qwq_response(response: str) -> dict:
    """Extract JSON from QwQ response (after FINAL: marker)."""
    import re, json
    
    # QwQ outputs reasoning then FINAL:
    final_markers = ["FINAL:", "Final:", "final:"]
    for marker in final_markers:
        if marker in response:
            parts = response.split(marker, 1)
            if len(parts) > 1:
                response = parts[1].strip()
                break
    
    # Extract JSON
    # Remove markdown code fences if present
    response = re.sub(r'```json\s*', '', response)
    response = re.sub(r'```\s*', '', response)
    
    return json.loads(response)
```

---

## 5. WordNet 検証スクリプト仕様

### 5.1 wordnet_evidence_builder.py

```python
"""
WordNet Evidence Builder for Lexicon
Validates QwQ proposals and expands with synonyms.

Usage:
    python wordnet_evidence_builder.py \
        --candidates qwq_output.json \
        --atom EMO.like \
        --output evidence_report.json
"""

# 軸ごとの WordNet 信頼度（補遺 Section E.3 より）
AXIS_WORDNET_RELIABILITY = {
    "temporal":        "medium-low",
    "scale":           "medium",
    "epistemological":  "medium",
    "ontological":     "medium-low",
    "interconnection": "high",      # entailment, cause, verb_groups
    "resonance":       "low-medium",
    "symmetry":        "medium",    # antonyms
    "lawfulness":      "low",
    "experience":      "low",
    "value_generation": "low-medium",
}

def build_evidence(word: str, atom_category: str) -> dict:
    """Build WordNet evidence for a single word."""
    from nltk.corpus import wordnet as wn
    
    evidence = {
        "synsets": [],
        "hypernym_chain": [],
        "antonyms": [],
        "entailments": [],
        "causes": [],
        "verb_groups": [],
        "definition": "",
        "category_compatible": None,  # bool after check
        "synonym_expansion": [],
    }
    
    synsets = wn.synsets(word)
    if not synsets:
        return evidence
    
    # Primary synset
    primary = synsets[0]
    evidence["synsets"] = [s.name() for s in synsets]
    evidence["definition"] = primary.definition()
    
    # Hypernym chain (for category compatibility)
    if primary.hypernym_paths():
        chain = [s.name().split('.')[0] for s in primary.hypernym_paths()[0]]
        evidence["hypernym_chain"] = chain
    
    # Category compatibility check (GPT Patch #5: Pilot = EMO strict, others = null)
    # Only EMO has strict markers. Other categories return null (info only).
    STRICT_MARKERS = {
        "EMO": ["emotion", "feeling", "affection", "sentiment", "passion"],
    }
    markers = STRICT_MARKERS.get(atom_category)
    if markers:
        chain_str = " ".join(evidence["hypernym_chain"]).lower()
        evidence["category_compatible"] = any(m in chain_str for m in markers)
    else:
        evidence["category_compatible"] = None  # Not yet assessed for this category
    
    # Antonyms
    for syn in synsets:
        for lemma in syn.lemmas():
            for ant in lemma.antonyms():
                evidence["antonyms"].append(ant.name())
    
    # Entailments & causes (verbs)
    for syn in synsets:
        evidence["entailments"].extend([e.name() for e in syn.entailments()])
        evidence["causes"].extend([c.name() for c in syn.causes()])
        evidence["verb_groups"].extend([v.name() for v in syn.verb_groups()])
    
    # Synonym expansion (GPT Patch #6: always status:"proposed", never auto-core)
    seen = {word}
    for syn in synsets[:3]:  # Top 3 synsets only
        for lemma in syn.lemmas():
            name = lemma.name().replace("_", " ")
            pos = syn.pos()  # n/v/a/r
            pos_normalized = {"n": "n", "v": "v", "a": "adj", "r": "adv", "s": "adj"}.get(pos, pos)
            if name not in seen:
                evidence["synonym_expansion"].append({
                    "w": name,
                    "pos": pos_normalized,
                    "status": "proposed",
                    "source_synset": syn.name(),
                })
                seen.add(name)
    
    return evidence
```

> **重要（GPT Patch #6）**: `synonym_expansion` の語は必ず `status: "proposed"` で Master に入る。
> Taka 承認後に `status: "core"` に昇格。自動で Core に混入させない。

### 5.2 conflict_checker.py

```python
"""
Conflict Checker for Lexicon
Detects symmetric pair collisions and cross-category issues.

Checks:
1. Same lemma::pos in both sides of a symmetric pair (GPT Patch #4)
2. Word's WordNet hypernyms don't pass through expected category
3. Multi-sense words that span multiple atoms (disambiguation needed)

Usage:
    python conflict_checker.py \
        --master lexicon_master.json \
        --atoms atoms_v1.json \
        --output conflicts.json
"""

def _extract_lemma_pos_set(atom_data: dict) -> set:
    """Extract all (lemma, pos) pairs from an atom's slots."""
    pairs = set()
    for slot in atom_data["slots"].values():
        for w in slot.get("words", []):
            if isinstance(w, dict):
                pairs.add(f"{w['w']}::{w['pos']}")
    return pairs

def check_symmetric_collision(master: dict, atoms_def: dict) -> list:
    """Flag lemma::pos appearing in both sides of a symmetric pair (GPT Patch #4)."""
    conflicts = []
    checked = set()
    for atom_id, atom_data in master.items():
        sym_pair = atoms_def["atoms"][atom_id]["symmetric_pair"]
        if sym_pair not in master or frozenset({atom_id, sym_pair}) in checked:
            continue
        checked.add(frozenset({atom_id, sym_pair}))
        
        my_keys = _extract_lemma_pos_set(atom_data)
        sym_keys = _extract_lemma_pos_set(master[sym_pair])
        
        overlap = my_keys & sym_keys
        if overlap:
            conflicts.append({
                "type": "symmetric_collision",
                "atom": atom_id,
                "symmetric_pair": sym_pair,
                "overlapping_keys": sorted(overlap),
                "severity": "HIGH",
            })
    return conflicts

# Category check markers
# Pilot: only EMO is strict. Others report as INFO only. (GPT Patch #5)
STRICT_CATEGORY_MARKERS = {
    "EMO": {
        "hypernym_keywords": ["emotion", "feeling", "affection", "sentiment", "passion", "state"],
        "severity": "MEDIUM",
    },
}
DEFAULT_CATEGORY_SEVERITY = "INFO"  # Non-EMO categories: flag but don't block

def check_category_mismatch(master: dict, atom_id: str) -> list:
    """Flag words whose WordNet hypernyms don't pass through the expected category."""
    conflicts = []
    category = atom_id.split(".")[0]
    cat_config = STRICT_CATEGORY_MARKERS.get(category)
    
    for slot_key, slot_data in master[atom_id]["slots"].items():
        for w in slot_data.get("words", []):
            word = w["w"] if isinstance(w, dict) else w
            pos = w.get("pos", "?") if isinstance(w, dict) else "?"
            key = f"{word}::{pos}"
            
            evidence = slot_data.get("evidence", {}).get(key, {})
            wn_data = evidence.get("wordnet", {})
            
            if cat_config:
                # Strict check: hypernym path must contain expected keywords
                chain_str = " ".join(wn_data.get("hypernym_chain", [])).lower()
                compatible = any(m in chain_str for m in cat_config["hypernym_keywords"])
                if not compatible and wn_data.get("hypernym_chain"):
                    conflicts.append({
                        "type": "category_mismatch",
                        "atom": atom_id,
                        "slot": slot_key,
                        "word": key,
                        "expected_category": category,
                        "hypernym_chain": wn_data.get("hypernym_chain", []),
                        "severity": cat_config["severity"],
                    })
            else:
                # Non-strict: report as INFO if data available
                if wn_data.get("category_compatible") is False:
                    conflicts.append({
                        "type": "category_mismatch",
                        "atom": atom_id,
                        "slot": slot_key,
                        "word": key,
                        "expected_category": category,
                        "hypernym_chain": wn_data.get("hypernym_chain", []),
                        "severity": DEFAULT_CATEGORY_SEVERITY,
                        "note": "Non-strict category (Pilot mode). Info only.",
                    })
    return conflicts

def check_multi_atom_ambiguity(index: dict) -> list:
    """Flag lemma::pos keys that appear in multiple atoms (disambiguation needed)."""
    conflicts = []
    for key, placements in index.items():
        atoms = set(p["atom"] for p in placements)
        if len(atoms) > 1:
            conflicts.append({
                "type": "multi_atom_ambiguity",
                "key": key,
                "atoms": sorted(atoms),
                "placements": placements,
                "severity": "INFO",
                "note": "May be valid (polysemy). Flag for human review.",
            })
    return conflicts
```

---

## 6. WSD の位置づけ（最終決定）

GPT 統一案を採用:

| | 判定 |
|---|---|
| **禁止** | WordNet 側から「正しい意味を決めて Atom を曲げる」系の WSD |
| **許可** | Lexicon 内で lemma が複数 slot 候補に入る場合の **衝突解決支援** |
| **本質** | WSD は **判定器ではなく、監査資料の生成器** |

例: "crush" が emergence と dislike にまたがり得る場合、
Evidence として「どの synset がどう違う」を示し、最後は人間が判断。

---

## 7. Pilot 実行手順

### Phase A: 器と定義の固定（P0）

| Step | 作業 | 担当 | 成果物 |
|------|------|------|--------|
| A1 | Definition Pack 3 ファイル作成 | Claude | categories_v1.json, axes_levels_v1.json, atoms_v1.json |
| A2 | QwQ で kanji_semantic_field 検証 | QwQ | atoms_v1.json 更新 |
| A3 | Taka 承認 | Taka | atoms_v1.json 確定 |

### Phase B: L1 候補生成 + L2 WordNet 検証（P1）

| Step | 作業 | 担当 | 成果物 |
|------|------|------|--------|
| B1 | QwQ で 48 slot 候補生成 | QwQ | qwq_candidates_EMO.like.json |
| B2 | WordNet Evidence 付与 | Script | evidence_EMO.like.json |
| B3 | 自動フラグ検出 | Script | conflicts_EMO.like.json |
| B3a | 対称ペア衝突（EMO.dislike と重複チェック） | | |
| B3b | カテゴリ不整合（hypernym が emotion を通らない） | | |
| B3c | 多義語リスク（synset が多数分散） | | |
| B4 | EMO.dislike も同時に B1-B3 実行 | QwQ + Script | 対称ペアを同時構築 |

### Phase C: レビュー → 確定 → 検証（P1）

| Step | 作業 | 担当 | 成果物 |
|------|------|------|--------|
| C1 | レビューレポート生成（Flag 中心） | Script | pilot_report_EMO.like.md |
| C2 | Taka が Flag だけレビュー | Taka | 採否決定 |
| C3 | Lexicon Master 確定 | Claude | lexicon_master.json |
| C4 | Runtime Index 生成 | Script | lexicon_index.json |
| C5 | LexiconGrounder で mixed 再実行 | Claude | 結果比較レポート |
| C6 | v0.3.1（SynapseGrounder）と比較 | Claude | grounding_rate, misground, Phase9 分布 |

---

## 8. LexiconGrounder 仕様

```python
class LexiconGrounder:
    """
    SynapseGrounder の置き換え。
    Runtime Index（逆引き）のみを参照する軽量 Grounder。
    
    汎用 lookup: verb に限らず noun/adj/adv も対応（GPT Patch #3）。
    """
    
    def __init__(self, index_path: str):
        with open(index_path) as f:
            data = json.load(f)
        self.index = data["index"]  # "lemma::pos" → [(atom, axis, level), ...]
    
    def ground_token(self, lemma: str, pos: str) -> list:
        """
        汎用 lookup: lemma + pos → 候補リスト
        
        Args:
            lemma: 正規化済み lemma（lowercase, trimmed）
            pos: spaCy 準拠 POS（n/v/adj/adv）
        
        Returns:
            list of {"concept_id": str, "axis": str, "level": str, "source": str}
            Empty list if not in Lexicon (→ UNGROUNDED)
        """
        key = f"{lemma.strip().lower()}::{pos}"
        entries = self.index.get(key, [])
        
        return [
            {
                "concept_id": e["atom"],
                "axis": e["axis"],
                "level": e["level"],
                "source": "lexicon_v1",
            }
            for e in entries
        ]
    
    def ground_verb(self, lemma: str) -> list:
        """Backward-compatible wrapper for Relation Pipeline."""
        return self.ground_token(lemma, "v")
    
    def is_light_verb(self, lemma: str) -> bool:
        """Light verb check（既存ロジック維持）"""
        return lemma in LIGHT_VERB_STOPLIST
```

**SynapseGrounder との違い:**
- WordNet synset 展開なし
- embedding 計算なし
- POS は Index 側で解決済み（POS Guard 不要）
- Score Threshold 不要（登録されているかどうかの binary）
- **汎用 lookup**: verb 以外（noun, adj, adv）にも対応

---

## 9. v0.3.1 フリーズ

GPT 承認済み: SynapseGrounder v0.3.1（Conditional Primary-Lemma Guard, no penalty）で凍結。
Lexicon Pilot 完了後に LexiconGrounder と比較し、移行判断を行う。

---

## 10. 成功基準

Pilot（EMO.like + EMO.dislike）の成功基準:

### 10.1 Lexicon 品質

| 指標 | 基準 |
|------|------|
| 48 slot のうち有効 slot 数 | 10〜20（全部埋まる必要はない） |
| 対称ペア衝突（lemma::pos 単位） | 0 |
| カテゴリ不整合 flag（EMO strict） | Taka レビュー後に 0 |
| 全工程の所要時間 | 1 日以内 |

### 10.2 Grounder 比較（GPT Patch #7: 条件固定 + 指標分離）

**比較条件（全て固定）:**
- コーパス: mixed dataset（既存 Wikipedia articles）
- Runner: Observation C Pipeline（relation_logger.py）
- 対象記事集合: v0.3.1 で実行した同一集合
- 対象: EMO.like / EMO.dislike に関連する grounding のみ

**指標（2 本立て）:**

| 指標 | 定義 | 基準 |
|------|------|------|
| **coverage** | Lexicon に登録語が出現した時に候補を返せた割合 | 参考値（初期は低くて正常） |
| **precision** | 返した候補のうち misground が出なかった割合 | **1.0（目標）** |
| CONSISTENT_MISGROUND | 同一動詞が同一 Atom に繰り返し接続される異常 | 0 |

> Lexicon 方式は初期に coverage が下がるのは自然。
> **precision 重視の成功条件**に寄せる。これが ESDE の思想（記述の正確性 > 網羅性）と一致する。

### 10.3 将来の運用設計（推奨）

GPT 推奨事項:
- `lexicon_master.json` は将来 `master/atoms/EMO.like.json` のように Atom 単位ファイルに分割可能な設計にする
- `lexicon_conflicts.json` は累積 DB。`conflicts_EMO.like.json` は実行ログとして分離

---

*Document: Project Lexicon Unified Implementation Spec v1.1.1*  
*Sources: Gemini Design Decision + GPT Audit Integration + Claude Implementation*  
*GPT Audit Patches: #1 words dict型統一, #2 lemma::pos キー, #3 汎用Grounder, #4 衝突判定単位, #5 EMO strict/他INFO, #6 synonym status管理, #7 precision重視基準*  
*Final Patches: A(POS正規形4種), B(Index=core only), C(正規化2段階防御), D(Evidence/wordsキー整合), E(Masterメタ)*  
*Philosophy: Aruism — "Describe, but do not decide"*
