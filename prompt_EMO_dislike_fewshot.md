# SYSTEM PROMPT

You are a lexicographer for ESDE (Emergent Semantic Data Engine).

THE CORE PRINCIPLE: "Describe, but do not decide."

You will be given:
1. A COMPLETED EXAMPLE: EMO.like (好) — words assigned to each slot for the back 5 axes
2. A TARGET ATOM: EMO.dislike (嫌) — you must fill the same slots

RULES:
1. Study the example carefully. Notice HOW words relate to the atom AND the axis/level.
   - Example: EMO.like + resonance.superficial → "fancy, fling, casual, flirtation" 
     (surface-level liking that doesn't go deep)
   - So EMO.dislike + resonance.superficial → words for surface-level dislike that doesn't go deep
     (e.g., "annoyance, peeve, irritate" — mild aversion without depth)

2. EMO.dislike (嫌) means: aversion, distaste, reluctance, repugnance, displeasure.
   Words must express DISLIKE specifically — not just any negative emotion.
   "Sad" is NOT dislike. "Angry" is NOT dislike. "Repulsed" IS dislike.

3. Do NOT include words that belong to EMO.like (the symmetric pair).

4. Include mixed POS: nouns, verbs, adjectives, adverbs. POS must be one of: n, v, adj, adv.

5. For each word: provide a 1-sentence reason connecting it to BOTH the atom AND the slot.

6. If a slot was N/A for EMO.like, it may or may not be N/A for EMO.dislike — decide independently.

7. Aim for 5-8 words per applicable slot. Quality over quantity.

8. CRITICAL: Avoid "concept labels" that just name the axis.
   Bad: "superficial" in resonance.superficial (that's the axis label)
   Good: "peeve" in resonance.superficial (that's an actual EMO.dislike word at surface depth)

9. Output strict JSON only. Same format as the example.

---

# USER PROMPT

### Category
Code: EMO
Name: Emotion
Description: Emotional states and feelings

### Target Atom
ID: EMO.dislike
Kanji: 嫌
Kanji semantic field (authoritative):
  - aversion, distaste
  - reluctance, unwillingness
  - repugnance, disgust
  - disliking, being put off
  - unpleasant, disagreeable
Definition: Coordinate representing aversion, avoidance, and displeasure
Symmetric pair: EMO.like (好)

### Axes and Levels (20 slots)

**resonance** — Resonance Depth Condition (The qualitative depth to which a connection or relationship touches the essence.)
  - superficial: A surface-level, temporary connection
  - structural: Correspondence or similarity at the structural level
  - essential: A deep connection that touches the essence
  - existential: A fundamental connection involving existence itself

**symmetry** — Symmetry Relation Condition (The interactions produced by opposing or complementary symmetric forces.)
  - destructive: A relationship where symmetric forces negate or destroy each other
  - inclusive: A relationship where one subsumes or absorbs the other
  - transformative: A relationship where interaction between both produces qualitative transformation
  - generative: A relationship where something new is generated from symmetric forces
  - cyclical: A relationship where symmetric forces alternate cyclically

**lawfulness** — Lawfulness Condition (The nature of the rules governing a phenomenon: simple causality, complex emergence, or fundamental necessity.)
  - predictable: Laws predictable through simple causality
  - emergent: Laws that emerge from complex systems and are difficult to predict
  - contingent: Laws that arise from chance or conditional dependencies
  - necessary: Laws arising necessarily from the very foundation of existence

**experience** — Experiential Quality Condition (The qualitative character of how a cognition manifests as an experience for consciousness.)
  - discovery: The experience of encountering the unknown
  - creation: The experience of bringing something new into being
  - comprehension: The experience of arriving at the comprehension of existence (ultimate understanding)

**value_generation** — Value Generation Condition (The criteria by which an entity or act is deemed to have value, progressing from utility through aesthetics and ethics to resonance with primordial existence.)
  - functional: Value based on utility and usefulness
  - aesthetic: Value based on sensibility and aesthetic awareness
  - ethical: Value based on social norms and moral duty
  - sacred: Value based on resonance with primordial existence 'Aru' (Aruism neologism)



---

### COMPLETED EXAMPLE (EMO.like — study this carefully)

```json
{
  "atom": "EMO.like",
  "slots": {
    "resonance.superficial": {
      "na": false,
      "words": [
        {
          "w": "fancy",
          "pos": "v",
          "reason": "surface-level passing liking without deep connection"
        },
        {
          "w": "fling",
          "pos": "n",
          "reason": "brief superficial favorable regard"
        },
        {
          "w": "casual",
          "pos": "adj",
          "reason": "liking that remains at surface level"
        },
        {
          "w": "flirtation",
          "pos": "n",
          "reason": "surface-level expression of temporary favorable regard"
        },
        {
          "w": "whim",
          "pos": "n",
          "reason": "fleeting superficial preference or liking"
        },
        {
          "w": "amuse",
          "pos": "v",
          "reason": "generating surface-level pleasant liking"
        },
        {
          "w": "faddish",
          "pos": "adj",
          "reason": "liking based on temporary surface-level trends"
        },
        {
          "w": "pleasant",
          "pos": "adj",
          "reason": "mild liking without depth"
        }
      ]
    },
    "resonance.structural": {
      "na": false,
      "words": [
        {
          "w": "compatible",
          "pos": "adj",
          "reason": "liking grounded in structural correspondence"
        },
        {
          "w": "compatibility",
          "pos": "n",
          "reason": "structural match that supports sustained favorable regard"
        },
        {
          "w": "simpatico",
          "pos": "adj",
          "reason": "structural alignment producing natural liking"
        },
        {
          "w": "fit",
          "pos": "n",
          "reason": "structural correspondence enabling favorable regard"
        }
      ]
    },
    "resonance.essential": {
      "na": false,
      "words": [
        {
          "w": "cherish",
          "pos": "v",
          "reason": "liking that touches essential nature of what is valued"
        },
        {
          "w": "treasure",
          "pos": "v",
          "reason": "holding in fondness at the essential level"
        },
        {
          "w": "dear",
          "pos": "adj",
          "reason": "essential-level favorable regard for someone deeply valued"
        },
        {
          "w": "precious",
          "pos": "adj",
          "reason": "regarded with fondness touching essential worth"
        },
        {
          "w": "affinity",
          "pos": "n",
          "reason": "deep-seated sense of liking"
        },
        {
          "w": "kindred",
          "pos": "adj",
          "reason": "liking rooted in essential similarity"
        },
        {
          "w": "heartfelt",
          "pos": "adj",
          "reason": "favorable disposition originating from one's core"
        }
      ]
    },
    "resonance.existential": {
      "na": false,
      "words": [
        {
          "w": "belonging",
          "pos": "n",
          "reason": "existential state of being in a place one likes and is liked by"
        },
        {
          "w": "cherished",
          "pos": "adj",
          "reason": "liking tied to one's sense of existence"
        }
      ]
    },
    "symmetry.destructive": {
      "na": false,
      "words": [
        {
          "w": "possessive",
          "pos": "adj",
          "reason": "liking that negates the other's autonomy through consuming attachment"
        },
        {
          "w": "jealousy",
          "pos": "n",
          "reason": "favorable regard colliding with threat, producing destructive force"
        },
        {
          "w": "obsession",
          "pos": "n",
          "reason": "liking intensified to the point it destroys balanced interaction"
        },
        {
          "w": "smother",
          "pos": "v",
          "reason": "favorable regard so intense it negates the other's space"
        },
        {
          "w": "clingy",
          "pos": "adj",
          "reason": "liking that becomes destructive through excessive attachment"
        }
      ]
    },
    "symmetry.inclusive": {
      "na": false,
      "words": [
        {
          "w": "embrace",
          "pos": "v",
          "reason": "taking in another within one's favorable regard"
        },
        {
          "w": "accept",
          "pos": "v",
          "reason": "including another wholly within one's liking"
        },
        {
          "w": "welcome",
          "pos": "v",
          "reason": "absorbing another into sphere of goodwill"
        },
        {
          "w": "adopt",
          "pos": "v",
          "reason": "subsuming another into circle of fondness"
        },
        {
          "w": "inclusive",
          "pos": "adj",
          "reason": "favorable regard that absorbs and encompasses"
        },
        {
          "w": "accepting",
          "pos": "adj",
          "reason": "inclusive positive regard"
        }
      ]
    },
    "symmetry.transformative": {
      "na": false,
      "words": [
        {
          "w": "romance",
          "pos": "n",
          "reason": "mutual liking that transforms both parties qualitatively"
        },
        {
          "w": "courtship",
          "pos": "n",
          "reason": "interaction of mutual liking producing transformative change"
        },
        {
          "w": "kindle",
          "pos": "v",
          "reason": "interaction that transforms liking into something new"
        },
        {
          "w": "captivating",
          "pos": "adj",
          "reason": "favorable regard that transforms the perceiver's state"
        },
        {
          "w": "softened",
          "pos": "adj",
          "reason": "liking transforming relational stance"
        },
        {
          "w": "inspired",
          "pos": "adj",
          "reason": "changed for the better through experiencing liking"
        },
        {
          "w": "uplift",
          "pos": "v",
          "reason": "qualitatively transformed to more positive state by liking"
        }
      ]
    },
    "symmetry.generative": {
      "na": false,
      "words": [
        {
          "w": "romance",
          "pos": "n",
          "reason": "mutual liking generating new relationships and possibilities"
        },
        {
          "w": "partnership",
          "pos": "n",
          "reason": "favorable regard generating collaborative endeavors"
        },
        {
          "w": "inspire",
          "pos": "v",
          "reason": "liking generating new creative energy"
        },
        {
          "w": "fertile",
          "pos": "adj",
          "reason": "mutual favorable regard generating new growth"
        },
        {
          "w": "fruitful",
          "pos": "adj",
          "reason": "liking generating productive new developments"
        },
        {
          "w": "creative",
          "pos": "adj",
          "reason": "propelling generation of new things from liking"
        }
      ]
    },
    "symmetry.cyclical": {
      "na": false,
      "words": [
        {
          "w": "rekindle",
          "pos": "v",
          "reason": "favorable regard returning cyclically after dormancy"
        },
        {
          "w": "nostalgia",
          "pos": "n",
          "reason": "liking cycling back through memory of past fondness"
        },
        {
          "w": "reunion",
          "pos": "n",
          "reason": "cyclical return to mutual favorable regard after separation"
        },
        {
          "w": "reminisce",
          "pos": "v",
          "reason": "cyclically returning to favorable regard through recollection"
        },
        {
          "w": "seasonal",
          "pos": "adj",
          "reason": "liking that returns regularly with certain cycles"
        }
      ]
    },
    "lawfulness.predictable": {
      "na": false,
      "words": [
        {
          "w": "habitual",
          "pos": "adj",
          "reason": "liking following predictable causal patterns of familiarity"
        },
        {
          "w": "reliable",
          "pos": "adj",
          "reason": "favorable regard behaving according to predictable rules"
        },
        {
          "w": "comfort",
          "pos": "n",
          "reason": "liking arising predictably from familiar safe conditions"
        },
        {
          "w": "familiarity",
          "pos": "n",
          "reason": "predictable causal basis for developing favorable regard"
        },
        {
          "w": "popular",
          "pos": "adj",
          "reason": "liking following predictable patterns"
        }
      ]
    },
    "lawfulness.emergent": {
      "na": false,
      "words": [
        {
          "w": "chemistry",
          "pos": "n",
          "reason": "liking emerging unpredictably from complex interpersonal dynamics"
        },
        {
          "w": "inexplicable",
          "pos": "adj",
          "reason": "favorable regard arising from complex causes defying prediction"
        },
        {
          "w": "unexpected",
          "pos": "adj",
          "reason": "liking emerging from complex systems in surprising ways"
        },
        {
          "w": "serendipitous",
          "pos": "adj",
          "reason": "favorable regard from unpredictable convergence of factors"
        },
        {
          "w": "click",
          "pos": "v",
          "reason": "sudden emergent liking from complex interpersonal dynamics"
        },
        {
          "w": "vibe",
          "pos": "n",
          "reason": "hard-to-predict emergent feeling of liking for an environment"
        }
      ]
    },
    "lawfulness.contingent": {
      "na": false,
      "words": [
        {
          "w": "circumstantial",
          "pos": "adj",
          "reason": "liking dependent on particular contingent conditions"
        },
        {
          "w": "conditional",
          "pos": "adj",
          "reason": "favorable regard contingent on specific circumstances"
        },
        {
          "w": "situational",
          "pos": "adj",
          "reason": "liking arising from particular conditions"
        },
        {
          "w": "fleeting",
          "pos": "adj",
          "reason": "favorable regard dependent on passing conditions"
        },
        {
          "w": "opportunistic",
          "pos": "adj",
          "reason": "liking arising from contingent chance encounters"
        }
      ]
    },
    "lawfulness.necessary": {
      "na": false,
      "words": [
        {
          "w": "innate",
          "pos": "adj",
          "reason": "favorable regard arising necessarily from fundamental nature"
        },
        {
          "w": "instinctive",
          "pos": "adj",
          "reason": "liking arising necessarily from biological foundation"
        },
        {
          "w": "maternal",
          "pos": "adj",
          "reason": "fondness arising necessarily from bond of parenthood"
        },
        {
          "w": "inherent",
          "pos": "adj",
          "reason": "favorable regard existing as necessary part of being"
        },
        {
          "w": "primal",
          "pos": "adj",
          "reason": "liking rooted in most fundamental necessary drives"
        }
      ]
    },
    "experience.discovery": {
      "na": false,
      "words": [
        {
          "w": "smitten",
          "pos": "adj",
          "reason": "experiential quality of discovering unexpected favorable regard"
        },
        {
          "w": "fascinated",
          "pos": "adj",
          "reason": "experiencing discovery of something deeply likable"
        },
        {
          "w": "delighted",
          "pos": "adj",
          "reason": "experiential quality of discovering something that pleases"
        },
        {
          "w": "wonder",
          "pos": "n",
          "reason": "experience of encountering something surprisingly likable"
        },
        {
          "w": "enchanted",
          "pos": "adj",
          "reason": "experiential quality of discovering captivating favorable regard"
        },
        {
          "w": "fascination",
          "pos": "n",
          "reason": "experience of liking something new and unknown"
        },
        {
          "w": "enchanting",
          "pos": "adj",
          "reason": "liking characterized by wonder of a new discovery"
        }
      ]
    },
    "experience.creation": {
      "na": false,
      "words": [
        {
          "w": "passion",
          "pos": "n",
          "reason": "liking experienced as creative drive to bring something new"
        },
        {
          "w": "inspired",
          "pos": "adj",
          "reason": "experiencing fondness as generative creative force"
        },
        {
          "w": "enthusiasm",
          "pos": "n",
          "reason": "liking experienced as energy of creating new things"
        },
        {
          "w": "ardor",
          "pos": "n",
          "reason": "intense favorable regard experienced in the act of creating"
        }
      ]
    },
    "experience.comprehension": {
      "na": false,
      "words": [
        {
          "w": "gratitude",
          "pos": "n",
          "reason": "favorable regard through ultimate comprehension of what is valued"
        },
        {
          "w": "awe",
          "pos": "n",
          "reason": "favorable regard when fully comprehending something profound"
        },
        {
          "w": "appreciation",
          "pos": "n",
          "reason": "liking deepened to point of comprehensive understanding"
        },
        {
          "w": "admiration",
          "pos": "n",
          "reason": "liking arising from complete understanding of excellence"
        },
        {
          "w": "profoundly",
          "pos": "adv",
          "reason": "depth of liking when fully comprehending the subject"
        }
      ]
    },
    "value_generation.functional": {
      "na": false,
      "words": [
        {
          "w": "prefer",
          "pos": "v",
          "reason": "liking based on practical utility"
        },
        {
          "w": "handy",
          "pos": "adj",
          "reason": "liked for its functional usefulness"
        },
        {
          "w": "useful",
          "pos": "adj",
          "reason": "favorably regarded for practical value"
        },
        {
          "w": "convenient",
          "pos": "adj",
          "reason": "liked because it serves a functional purpose well"
        },
        {
          "w": "reliable",
          "pos": "adj",
          "reason": "favored for dependable functional performance"
        },
        {
          "w": "favorably",
          "pos": "adv",
          "reason": "regarding something positively based on utility"
        },
        {
          "w": "utilize",
          "pos": "v",
          "reason": "to employ something because of favorable utility"
        }
      ]
    },
    "value_generation.aesthetic": {
      "na": false,
      "words": [
        {
          "w": "beautiful",
          "pos": "adj",
          "reason": "liked for aesthetic and sensory appeal"
        },
        {
          "w": "admire",
          "pos": "v",
          "reason": "favorable regard grounded in aesthetic appreciation"
        },
        {
          "w": "lovely",
          "pos": "adj",
          "reason": "liked for its pleasing aesthetic qualities"
        },
        {
          "w": "gorgeous",
          "pos": "adj",
          "reason": "strongly liked for visual or sensory beauty"
        },
        {
          "w": "exquisite",
          "pos": "adj",
          "reason": "favorably regarded for refined aesthetic quality"
        },
        {
          "w": "elegant",
          "pos": "adj",
          "reason": "liked for graceful aesthetic sensibility"
        },
        {
          "w": "admiration",
          "pos": "n",
          "reason": "favorable regard based on aesthetic awareness"
        },
        {
          "w": "pleasing",
          "pos": "adj",
          "reason": "aesthetic-driven liking"
        }
      ]
    },
    "value_generation.ethical": {
      "na": false,
      "words": [
        {
          "w": "respect",
          "pos": "n",
          "reason": "favorable regard grounded in moral worth"
        },
        {
          "w": "esteem",
          "pos": "n",
          "reason": "liking based on perceived moral character"
        },
        {
          "w": "admirable",
          "pos": "adj",
          "reason": "liked because of upright ethical qualities"
        },
        {
          "w": "honorable",
          "pos": "adj",
          "reason": "favorably regarded for adherence to moral standards"
        },
        {
          "w": "trustworthy",
          "pos": "adj",
          "reason": "liked for reliability rooted in ethical character"
        },
        {
          "w": "upstanding",
          "pos": "adj",
          "reason": "favorably regarded for meeting moral norms"
        },
        {
          "w": "respectable",
          "pos": "adj",
          "reason": "liking based on moral evaluation"
        },
        {
          "w": "noble",
          "pos": "adj",
          "reason": "liking because it aligns with moral duty"
        }
      ]
    },
    "value_generation.sacred": {
      "na": false,
      "words": []
    }
  }
}
```

---

Now generate the same structure for **EMO.dislike (嫌)**.

Remember:
- EMO.dislike = aversion, distaste, reluctance, repugnance, displeasure
- Match the PATTERN of the example, not the words
- Each word must be specific to EMO.dislike AND to its slot
- Avoid axis labels as words
- POS: n/v/adj/adv only
- 5-8 words per slot

Output ONLY the JSON.