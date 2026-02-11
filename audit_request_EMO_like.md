# EMO.like Merged Lexicon — GPT Audit Request

## Context
Atom: EMO.like (好)
Kanji semantic field: favorable disposition, goodwill, personal preference, taste, desirability, agreeableness, liking, fondness
Symmetric pair: EMO.dislike (嫌)

## What this is
This is a MERGED lexicon from 3 independent AI generators (Claude, Gemini, GPT).
Each word has a consensus score (3/3 = all agreed, 1/3 = single source).

## Audit instructions
For each slot, check:
1. **AXIS LABEL TEST** — Is this word just naming the axis/level itself? (REJECT)
2. **ATOM SPECIFICITY** — Is this word specific to EMO.like, or could it go in ANY atom? (REJECT)
3. **SYMMETRIC PAIR** — Does this word belong to EMO.dislike instead? (REJECT)
4. **NATURAL TEXT** — Would you encounter this in a newspaper/novel? (REJECT if no)
5. **MISSING WORDS** — Are there obvious EMO.like words missing from this slot? (ADD)
6. **N/A ACCURACY** — Should this slot be N/A or APPLICABLE?

Output format per slot: list of ACCEPT/REJECT/ADD decisions, then overall PASS/REVISE.

---

## TEMPORAL

### temporal.emergence (11 words)
- **infatuation** [n] (2/3: Claude,GPT) — sudden onset of intense liking or attraction
- **smitten** [adj] (2/3: Claude,GPT) — struck with sudden favorable regard
- **spark** [n] (3/3: Claude,Gemini,GPT) — initial moment of attraction or interest appearing
- **captivate** [v] (2/3: Claude,GPT) — to seize attention and generate immediate liking
- **crush** [n] (1/3: Claude) — newly appeared intense liking for someone
- **fancy** [v] (2/3: Claude,Gemini) — to suddenly take a liking to something or someone
- **taken** [adj] (1/3: Claude) — captured by newly emerging favorable feeling
- **beguile** [v] (1/3: Claude) — to charm someone into newly formed favorable disposition
- **intrigued** [adj] (2/3: Gemini,GPT) — showing early-stage positive interest that may grow into liking
- **warm to** [v] (1/3: Gemini) — initial softening into a favorable feeling
- **appeal** [n] (1/3: Gemini) — initial attractive quality that gives rise to liking

### temporal.indication (11 words)
- **curious** [adj] (1/3: Claude) — showing early signs of interest that may develop into liking
- **warming** [adj] (1/3: Claude) — gradually showing signs of developing favorable regard
- **interest** [n] (1/3: Claude) — early indication of favorable attention before full liking
- **flirt** [v] (1/3: Claude) — signaling emerging attraction through tentative gestures
- **drawn** [adj] (1/3: Gemini) — observable pull toward someone or something
- **fondness** [n] (1/3: Gemini) — developing but not yet fully established liking
- **preference** [n] (1/3: Gemini) — emerging tendency to favor one option over others
- **lean toward** [v] (1/3: Gemini) — early directional inclination of liking
- **favorable** [adj] (1/3: Gemini) — positive orientation that hints at liking
- **glimmer** [n] (1/3: GPT) — faint sign that a feeling of liking is starting
- **leaning** [n] (1/3: GPT) — subtle tendency or inclination toward liking

### temporal.influence (11 words)
- **attract** [v] (1/3: Claude) — liking begins to exert pull on behavior and attention
- **charm** [v] (1/3: Claude) — favorable regard actively influencing another person
- **endear** [v] (2/3: Claude,GPT) — to cause fondness to develop in others through influence
- **appealing** [adj] (1/3: Claude) — exerting influence through desirability
- **alluring** [adj] (1/3: Claude) — drawing others in through attractive qualities
- **woo** [v] (1/3: Claude) — actively exerting liking to influence another's feelings
- **court** [v] (1/3: Claude) — pursuing someone under the influence of favorable regard
- **favor** [v] (1/3: Gemini) — preferential action driven by liking
- **enthusiasm** [n] (1/3: Gemini) — energetic effect liking has on engagement
- **swayed** [adj] (1/3: GPT) — being moved or affected by a growing favorable opinion
- **captivating** [adj] (1/3: GPT) — exerting a pull based on agreeable nature

### temporal.transformation (10 words)
- **enamored** [adj] (1/3: Claude) — liking has transformed into deeper attachment
- **devoted** [adj] (2/3: Claude,GPT) — fondness has undergone transformation into committed regard
- **besotted** [adj] (1/3: Claude) — liking has transformed the person's state of mind
- **fall** [v] (1/3: Claude) — undergoing qualitative shift from liking into love
- **deepen** [v] (1/3: Claude) — fondness undergoing qualitative change into stronger attachment
- **sweetheart** [n] (1/3: Claude) — person whose relationship was transformed by mutual liking
- **attachment** [n] (2/3: Gemini,GPT) — liking deepening into a more stable emotional bond
- **affection** [n] (1/3: Gemini) — qualitative change toward warmer liking
- **grow on** [v] (1/3: Gemini) — gradual transformation of feeling into liking
- **fondness** [n] (1/3: GPT) — developed state where casual liking changed into deeper warmth

### temporal.establishment (12 words)
- **fondness** [n] (1/3: Claude) — settled stable state of warm favorable regard
- **affection** [n] (1/3: Claude) — established and stable feeling of warm liking
- **fond** [adj] (2/3: Claude,Gemini) — settled disposition of warm liking
- **attachment** [n] (1/3: Claude) — stabilized bond of favorable regard
- **beloved** [adj] (1/3: Claude) — established as object of deep sustained liking
- **dear** [adj] (1/3: Claude) — stably held in favorable regard and affection
- **favorite** [n] (2/3: Claude,Gemini) — established preferred object of liking
- **prefer** [v] (1/3: Claude) — settled pattern of favorable disposition
- **preference** [n] (1/3: Gemini) — stabilized liking in choice patterns
- **trusted** [adj] (1/3: Gemini) — liking consolidated into confidence
- **valued** [adj] (1/3: Gemini) — established positive regard
- **bond** [n] (1/3: GPT) — stable connection built on mutual liking

### temporal.continuation (11 words)
- **cherish** [v] (1/3: Claude) — maintaining deep fondness over continuing time
- **loyal** [adj] (1/3: Claude) — persisting in favorable regard through ongoing attachment
- **adore** [v] (1/3: Claude) — continuing to hold in high affectionate regard
- **steadfast** [adj] (1/3: Claude) — unwavering in continued favorable disposition
- **devotion** [n] (1/3: Claude) — ongoing sustained commitment born of liking
- **faithfully** [adv] (1/3: Claude) — continuing to show favorable regard over time
- **treasure** [v] (1/3: Claude) — persistently valuing someone with fondness
- **abiding** [adj] (1/3: GPT) — liking that persists and stays for a long time
- **steadfastly** [adv] (1/3: GPT) — maintaining positive disposition firmly
- **affinity** [n] (1/3: Gemini) — ongoing liking that persists over time
- **fondly** [adv] (1/3: Gemini) — sustained positive feeling

### temporal.permanence (10 words)
- **eternal** [adj] (1/3: Claude) — liking that transcends temporal boundaries
- **undying** [adj] (2/3: Claude,GPT) — fondness that persists beyond all change
- **everlasting** [adj] (1/3: Claude) — permanent state of favorable regard
- **unconditional** [adj] (1/3: Claude) — liking so absolute it exists beyond circumstance
- **abiding** [adj] (1/3: Claude) — enduring fondness that has become permanent
- **cherished** [adj] (1/3: Gemini) — enduring and lasting liking
- **beloved** [adj] (1/3: Gemini) — liking perceived as long-lasting
- **enduring** [adj] (1/3: Gemini) — permanence of favorable regard
- **keepsake** [n] (1/3: Gemini) — object associated with lasting emotional liking
- **immortalize** [v] (1/3: GPT) — to make a state of liking last forever in memory

## SCALE

### scale.individual (11 words)
- **like** [v] (2/3: Claude,Gemini) — personal favorable disposition felt within one individual
- **prefer** [v] (2/3: Claude,Gemini) — individual-level expression of personal taste
- **enjoy** [v] (1/3: Claude) — personal experience of favorable regard
- **taste** [n] (3/3: Claude,Gemini,GPT) — individual preference and personal liking
- **fond** [adj] (2/3: Claude,Gemini) — personal warmth felt toward another
- **relish** [v] (1/3: Claude) — to personally take great pleasure in something
- **savor** [v] (1/3: Claude) — individual act of lingering in personal enjoyment
- **crush** [n] (1/3: Claude) — private individual feeling of strong liking
- **fancy** [n] (2/3: Claude,GPT) — individual whim of liking or personal inclination
- **enjoyment** [n] (1/3: Gemini) — individual experience of liking
- **partial** [adj] (1/3: GPT) — favoring one thing due to individual taste

### scale.community (11 words)
- **camaraderie** [n] (2/3: Claude,GPT) — mutual liking and warmth within a close group
- **neighborly** [adj] (1/3: Claude) — favorable disposition expressed at community scale
- **friendly** [adj] (1/3: Claude) — showing goodwill in direct interpersonal relationships
- **companionship** [n] (1/3: Claude) — mutual fondness shared between close associates
- **warmth** [n] (1/3: Claude) — favorable regard expressed within family or group
- **welcome** [v] (1/3: Claude) — community-level expression of favorable reception
- **kinship** [n] (1/3: Claude) — favorable bond felt among family or community
- **popular** [adj] (1/3: Gemini) — collective liking within a group
- **well-liked** [adj] (1/3: Gemini) — favorably regarded by others
- **beloved** [adj] (1/3: GPT) — held in high regard by a group
- **popularity** [n] (1/3: GPT) — state of being liked by a specific group

### scale.society (11 words)
- **popular** [adj] (2/3: Claude,Gemini) — widely liked across a society
- **popularity** [n] (1/3: Claude) — state of being broadly favored at societal scale
- **approval** [n] (2/3: Claude,Gemini) — society-wide favorable regard or endorsement
- **goodwill** [n] (1/3: Claude) — favorable disposition held collectively
- **favored** [adj] (1/3: Claude) — preferred or liked within broad social context
- **acclaim** [n] (2/3: Claude,GPT) — widespread expression of societal favorable regard
- **endorse** [v] (1/3: Claude) — to express institutional favorable disposition
- **mainstream** [adj] (2/3: Gemini,GPT) — representing broadly liked tastes of society
- **trending** [adj] (1/3: Gemini) — reflecting widespread current liking
- **accepted** [adj] (1/3: Gemini) — social liking or acceptance
- **reputation** [n] (1/3: GPT) — widespread favorable regard held by the public

### scale.ecosystem → N/A (3/3) Liking as a personal emotional state does not literally apply at ecosystem scale

### scale.stellar → N/A (3/3) Emotion of liking has no direct/literal application at stellar scale

### scale.cosmic → N/A (3/3) Emotion of liking has no direct/literal application at cosmic scale

## EPISTEMOLOGICAL

### epistemological.perception (9 words)
- **attractive** [adj] (2/3: Claude,GPT) — perceived through senses as evoking favorable regard
- **appealing** [adj] (3/3: Claude,Gemini,GPT) — sensory quality that registers as desirable
- **pleasant** [adj] (3/3: Claude,Gemini,GPT) — perceived favorably through immediate sensory apprehension
- **agreeable** [adj] (1/3: Claude) — sensed as immediately likable upon encounter
- **inviting** [adj] (1/3: Claude) — perceived quality that draws favorable attention
- **pleasing** [adj] (1/3: Claude) — registering as favorable through direct perception
- **nice** [adj] (1/3: Gemini) — simple perception of liking
- **enjoyable** [adj] (1/3: Gemini) — perceived pleasure
- **savory** [adj] (1/3: GPT) — sensory liking of taste or smell

### epistemological.identification (9 words)
- **favorite** [adj] (3/3: Claude,Gemini,GPT) — named and distinguished as preferred or most liked
- **preference** [n] (2/3: Claude,Gemini) — identified and labeled favorable disposition
- **pick** [n] (1/3: Claude) — identified choice reflecting recognized liking
- **type** [n] (1/3: Claude) — identified category of what one is disposed to like
- **choose** [v] (1/3: Claude) — act of distinguishing what one likes from alternatives
- **taste** [n] (2/3: Gemini,GPT) — identified personal liking category
- **likable** [adj] (1/3: Gemini) — categorized as easy to like
- **preferred** [adj] (1/3: Gemini) — identifies choice based on liking
- **predilection** [n] (1/3: GPT) — preference identified as a consistent pattern

### epistemological.understanding (9 words)
- **appreciate** [v] (3/3: Claude,Gemini,GPT) — to grasp the qualities that make something likable
- **appreciation** [n] (1/3: Claude) — comprehension of why one holds favorable regard
- **value** [v] (2/3: Claude,Gemini) — understanding the worth that grounds one's liking
- **esteem** [n] (1/3: Claude) — favorable regard grounded in understood merit
- **respect** [v] (2/3: Claude,Gemini) — favorable disposition based on comprehended qualities
- **recognition** [n] (1/3: Gemini) — awareness of liking reasons
- **regard** [n] (1/3: Gemini) — thoughtful positive evaluation
- **relate** [v] (1/3: GPT) — feeling liking because one understands the subject
- **sympathetic** [adj] (1/3: GPT) — liking based on shared understanding

### epistemological.experience (10 words)
- **savor** [v] (1/3: Claude) — integrating liking into lived experience beyond analysis
- **relish** [v] (2/3: Claude,GPT) — fully experiencing fondness as embodied enjoyment
- **delight** [n] (2/3: Claude,Gemini) — experienced pleasure of favorable regard
- **revel** [v] (1/3: Claude) — immersing oneself in the lived experience of liking
- **bask** [v] (1/3: Claude) — dwelling in the experiential warmth of favorable regard
- **enjoy** [v] (1/3: Gemini) — lived experience of liking
- **pleasure** [n] (1/3: Gemini) — experiential quality of liking
- **satisfaction** [n] (1/3: Gemini) — liking integrated into experience
- **enjoyment** [n] (1/3: GPT) — lived experience of pleasure and liking
- **delightful** [adj] (1/3: GPT) — causing high degree of experienced liking

### epistemological.creation (7 words)
- **inspire** [v] (1/3: Claude) — liking generates new creative cognition
- **muse** [n] (1/3: Claude) — object of fondness that sparks creative thought
- **passion** [n] (1/3: Claude) — intense liking that drives generation of new works
- **enthusiasm** [n] (1/3: Claude) — eager liking that fuels creative generation
- **cultivated** [v] (1/3: Gemini) — actively creating liking
- **idealize** [v] (1/3: GPT) — creating mental concept of something as most liked
- **affinity** [n] (1/3: GPT) — created sense of connection or inherent liking

## ONTOLOGICAL

### ontological.material (10 words)
- **caress** [v] (1/3: Claude) — physical expression of fondness through touch
- **hug** [v] (1/3: Claude) — bodily act expressing favorable regard and affection
- **kiss** [n] (1/3: Claude) — physical manifestation of liking through bodily contact
- **cuddle** [v] (1/3: Claude) — material/bodily expression of warm fondness
- **embrace** [n] (1/3: Claude) — physical act of holding that manifests favorable regard
- **nuzzle** [v] (1/3: Claude) — gentle physical gesture of fondness
- **comforting** [adj] (1/3: Gemini) — physical qualities that produce liking
- **fragrance** [n] (1/3: Gemini) — material stimulus that evokes liking
- **comfortable** [adj] (1/3: GPT) — liking based on physical material ease
- **tactile** [adj] (1/3: GPT) — liking for the physical feel or texture

### ontological.informational (10 words)
- **like** [n] (1/3: Claude) — digital signal of favorable regard (social media)
- **upvote** [n] (1/3: Claude) — data-encoded expression of approval and liking
- **rating** [n] (2/3: Claude,Gemini) — informational encoding of degree of favorable regard
- **recommendation** [n] (1/3: Claude) — informational expression of favorable disposition
- **review** [n] (2/3: Claude,Gemini) — encoded favorable assessment in data form
- **bookmark** [v] (1/3: Claude) — saving information one likes for future reference
- **subscribe** [v] (1/3: Claude) — informational action expressing ongoing liking
- **recommend** [v] (1/3: Gemini) — information-based expression of liking
- **interesting** [adj] (1/3: GPT) — liking directed toward information presented
- **curiosity** [n] (1/3: GPT) — positive disposition toward acquiring more information

### ontological.relational (10 words)
- **bond** [n] (1/3: Claude) — relational tie constituted by mutual favorable regard
- **rapport** [n] (2/3: Claude,GPT) — relational connection grounded in mutual liking
- **friendship** [n] (2/3: Claude,GPT) — relationship defined by mutual fondness
- **closeness** [n] (1/3: Claude) — relational quality arising from deep mutual liking
- **affinity** [n] (1/3: Claude) — natural relational pull of favorable disposition
- **ally** [n] (1/3: Claude) — entity connected through relational favorable regard
- **fond of** [adj] (1/3: Gemini) — liking within a relationship
- **affection** [n] (1/3: Gemini) — relational emotional liking
- **care for** [v] (1/3: Gemini) — liking directed toward another
- **cordial** [adj] (1/3: GPT) — characterized by warm and friendly regard

### ontological.structural (8 words)
- **preference** [n] (1/3: Claude) — structural pattern of consistently choosing what one likes
- **inclination** [n] (1/3: Claude) — structural tendency toward favorable regard
- **disposition** [n] (1/3: Claude) — underlying structural orientation toward liking
- **temperament** [n] (1/3: Claude) — structural personality pattern shaping what one likes
- **tendency** [n] (1/3: Claude) — patterned predisposition toward favorable regard
- **compatible** [adj] (1/3: Gemini) — structural fit that enables liking
- **well-designed** [adj] (1/3: Gemini) — structural qualities that cause liking
- **elegance** [n] (1/3: GPT) — liking for the way something is structured

### ontological.semantic (8 words)
- **lovable** [adj] (1/3: Claude) — carrying meaning of being worthy of favorable regard
- **likable** [adj] (1/3: Claude) — possessing quality of meaning-level desirability
- **desirable** [adj] (1/3: Claude) — bearing semantic value of being favorably regarded
- **worthwhile** [adj] (1/3: Claude) — carrying meaning that something merits liking
- **endearing** [adj] (1/3: Claude) — having qualities whose meaning evokes fondness
- **meaningful** [adj] (2/3: Gemini,GPT) — semantic value contributing to liking
- **valuable** [adj] (1/3: Gemini) — positive meaning associated with liking
- **significant** [adj] (1/3: Gemini) — semantic weight that draws liking

## INTERCONNECTION

### interconnection.independent (7 words)
- **self-love** [n] (1/3: Claude) — favorable regard directed inward independently
- **contentment** [n] (1/3: Claude) — self-contained favorable disposition
- **self-assured** [adj] (1/3: Claude) — favorable self-regard existing independently
- **preference** [n] (1/3: Gemini) — liking held without relational dependency
- **taste** [n] (1/3: Gemini) — individual liking existing on its own
- **content** [adj] (1/3: GPT) — internal state of liking one's situation independently
- **self-satisfied** [adj] (1/3: GPT) — liking or being pleased with oneself without external input

### interconnection.catalytic (9 words)
- **smitten** [adj] (1/3: Claude) — contact with another catalyzes sudden favorable regard
- **charmed** [adj] (1/3: Claude) — another's presence triggers emergence of liking
- **spark** [n] (1/3: Claude) — catalytic moment of contact initiating favorable feeling
- **click** [v] (1/3: Claude) — instant catalytic connection triggering mutual liking
- **enchant** [v] (1/3: Claude) — to catalyze favorable regard through encounter
- **attract** [v] (1/3: GPT) — one entity triggers liking in another upon contact
- **allure** [n] (1/3: GPT) — quality that triggers favorable disposition in others
- **inspiring** [adj] (1/3: Gemini) — liking that triggers further engagement
- **sparked** [v] (1/3: Gemini) — liking that initiates connection

### interconnection.chained (7 words)
- **contagious** [adj] (3/3: Claude,Gemini,GPT) — liking that spreads through a chain of social influence
- **viral** [adj] (1/3: Claude) — favorable regard propagating through chains of people
- **recommend** [v] (1/3: Claude) — passing favorable regard along a chain to others
- **referral** [n] (1/3: Claude) — chain-link transmission of favorable disposition
- **word-of-mouth** [n] (1/3: Claude) — favorable regard propagating through sequential social links
- **trend** [n] (1/3: GPT) — a chain of liking that moves through a population
- **caught on** [v] (1/3: Gemini) — liking propagated through sequence

### interconnection.synchronous (7 words)
- **mutual** [adj] (1/3: Claude) — two entities simultaneously experiencing favorable regard
- **reciprocal** [adj] (1/3: Claude) — liking expressed and returned in synchronous exchange
- **shared** [adj] (1/3: Claude) — favorable regard held simultaneously by multiple parties
- **together** [adv] (1/3: Claude) — experiencing fondness in synchronous connection
- **chemistry** [n] (1/3: Claude) — synchronous mutual favorable regard between people
- **congenial** [adj] (1/3: GPT) — shared simultaneous liking and agreement
- **harmonize** [v] (1/3: GPT) — multiple people simultaneously coming to like something

### interconnection.resonant (7 words)
- **soulmate** [n] (2/3: Claude,GPT) — deep resonant liking between two entities at core level
- **kindred** [adj] (2/3: Claude,GPT) — deeply resonant favorable regard based on shared nature
- **inseparable** [adj] (1/3: Claude) — liking so deeply resonant it defines the relationship
- **intimacy** [n] (1/3: Claude) — deeply resonant mutual fondness and closeness
- **devoted** [adj] (1/3: Claude) — favorable regard resonating at deepest relational level
- **deeply fond** [adj] (1/3: Gemini) — liking that strongly resonates between entities
- **mutual affection** [n] (1/3: Gemini) — reciprocal resonant liking

## RESONANCE

### resonance.superficial (8 words)
- **fancy** [v] (1/3: Claude) — surface-level passing liking without deep connection
- **fling** [n] (1/3: Claude) — brief superficial favorable regard
- **casual** [adj] (1/3: Claude) — liking that remains at surface level
- **flirtation** [n] (2/3: Claude,GPT) — surface-level expression of temporary favorable regard
- **whim** [n] (2/3: Claude,GPT) — fleeting superficial preference or liking
- **amuse** [v] (1/3: Claude) — generating surface-level pleasant liking
- **faddish** [adj] (1/3: GPT) — liking based on temporary surface-level trends
- **pleasant** [adj] (1/3: Gemini) — mild liking without depth

### resonance.structural (6 words)
- **compatible** [adj] (3/3: Claude,Gemini,GPT) — liking grounded in structural correspondence
- **compatibility** [n] (1/3: Claude) — structural match that supports sustained favorable regard
- **like-minded** [adj] (2/3: Claude,GPT) — favorable regard based on structural similarity of thought
- **simpatico** [adj] (1/3: Claude) — structural alignment producing natural liking
- **fit** [n] (1/3: Claude) — structural correspondence enabling favorable regard
- **well-matched** [adj] (1/3: Gemini) — fit-based liking

### resonance.essential (9 words)
- **cherish** [v] (1/3: Claude) — liking that touches essential nature of what is valued
- **adore** [v] (1/3: Claude) — deep favorable regard connecting to someone's core
- **treasure** [v] (1/3: Claude) — holding in fondness at the essential level
- **dear** [adj] (1/3: Claude) — essential-level favorable regard for someone deeply valued
- **precious** [adj] (1/3: Claude) — regarded with fondness touching essential worth
- **affinity** [n] (1/3: Gemini) — deep-seated sense of liking
- **kindred** [adj] (1/3: Gemini) — liking rooted in essential similarity
- **devotion** [n] (1/3: GPT) — deep liking reaching the core essence
- **heartfelt** [adj] (1/3: GPT) — favorable disposition originating from one's core

### resonance.existential (6 words)
- **love** [n] (1/3: Claude) — favorable regard so deep it becomes fundamental to existence
- **devotion** [n] (1/3: Claude) — fondness that defines existential orientation
- **unconditional** [adj] (2/3: Claude,GPT) — liking at existential level beyond all conditions
- **worship** [v] (1/3: Claude) — favorable regard involving one's entire being
- **belonging** [n] (1/3: GPT) — existential state of being in a place one likes and is liked by
- **cherished** [adj] (1/3: Gemini) — liking tied to one's sense of existence

## SYMMETRY

### symmetry.destructive (5 words)
*Note: GPT and Gemini marked N/A; Claude found EMO.like-specific destructive words. Kept as APPLICABLE for audit review.*
- **possessive** [adj] (1/3: Claude) — liking that negates the other's autonomy through consuming attachment
- **jealousy** [n] (1/3: Claude) — favorable regard colliding with threat, producing destructive force
- **obsession** [n] (1/3: Claude) — liking intensified to the point it destroys balanced interaction
- **smother** [v] (1/3: Claude) — favorable regard so intense it negates the other's space
- **clingy** [adj] (1/3: Claude) — liking that becomes destructive through excessive attachment

### symmetry.inclusive (6 words)
- **embrace** [v] (3/3: Claude,Gemini,GPT) — taking in another within one's favorable regard
- **accept** [v] (1/3: Claude) — including another wholly within one's liking
- **welcome** [v] (1/3: Claude) — absorbing another into sphere of goodwill
- **adopt** [v] (2/3: Claude,GPT) — subsuming another into circle of fondness
- **inclusive** [adj] (1/3: Claude) — favorable regard that absorbs and encompasses
- **accepting** [adj] (1/3: Gemini) — inclusive positive regard

### symmetry.transformative (8 words)
- **romance** [n] (1/3: Claude) — mutual liking that transforms both parties qualitatively
- **courtship** [n] (1/3: Claude) — interaction of mutual liking producing transformative change
- **kindle** [v] (1/3: Claude) — interaction that transforms liking into something new
- **captivating** [adj] (1/3: Claude) — favorable regard that transforms the perceiver's state
- **softened** [adj] (1/3: Gemini) — liking transforming relational stance
- **won over** [v] (1/3: Gemini) — liking produced through interaction
- **inspired** [adj] (1/3: GPT) — changed for the better through experiencing liking
- **uplift** [v] (1/3: GPT) — qualitatively transformed to more positive state by liking

### symmetry.generative (6 words)
- **romance** [n] (1/3: Claude) — mutual liking generating new relationships and possibilities
- **partnership** [n] (1/3: Claude) — favorable regard generating collaborative endeavors
- **inspire** [v] (1/3: Claude) — liking generating new creative energy
- **fertile** [adj] (1/3: Claude) — mutual favorable regard generating new growth
- **fruitful** [adj] (2/3: Claude,GPT) — liking generating productive new developments
- **creative** [adj] (1/3: GPT) — propelling generation of new things from liking

### symmetry.cyclical (6 words)
- **on-again** [adj] (1/3: Claude) — liking cycling between active and dormant phases
- **rekindle** [v] (1/3: Claude) — favorable regard returning cyclically after dormancy
- **nostalgia** [n] (2/3: Claude,GPT) — liking cycling back through memory of past fondness
- **reunion** [n] (1/3: Claude) — cyclical return to mutual favorable regard after separation
- **reminisce** [v] (1/3: Claude) — cyclically returning to favorable regard through recollection
- **seasonal** [adj] (1/3: GPT) — liking that returns regularly with certain cycles

## LAWFULNESS

### lawfulness.predictable (6 words)
- **habitual** [adj] (2/3: Claude,GPT) — liking following predictable causal patterns of familiarity
- **reliable** [adj] (1/3: Claude) — favorable regard behaving according to predictable rules
- **comfort** [n] (1/3: Claude) — liking arising predictably from familiar safe conditions
- **familiarity** [n] (1/3: Claude) — predictable causal basis for developing favorable regard
- **crowd-pleasing** [adj] (1/3: Gemini) — reliably liked by many
- **popular** [adj] (1/3: Gemini) — liking following predictable patterns

### lawfulness.emergent (7 words)
- **chemistry** [n] (2/3: Claude,GPT) — liking emerging unpredictably from complex interpersonal dynamics
- **inexplicable** [adj] (1/3: Claude) — favorable regard arising from complex causes defying prediction
- **unexpected** [adj] (1/3: Claude) — liking emerging from complex systems in surprising ways
- **serendipitous** [adj] (1/3: Claude) — favorable regard from unpredictable convergence of factors
- **click** [v] (1/3: Claude) — sudden emergent liking from complex interpersonal dynamics
- **vibe** [n] (1/3: GPT) — hard-to-predict emergent feeling of liking for an environment
- **grew on** [v] (1/3: Gemini) — liking emerging over time

### lawfulness.contingent (5 words)
- **circumstantial** [adj] (2/3: Claude,GPT) — liking dependent on particular contingent conditions
- **conditional** [adj] (1/3: Claude) — favorable regard contingent on specific circumstances
- **situational** [adj] (1/3: Claude) — liking arising from particular conditions
- **fleeting** [adj] (1/3: Claude) — favorable regard dependent on passing conditions
- **opportunistic** [adj] (2/3: Claude,GPT) — liking arising from contingent chance encounters

### lawfulness.necessary (5 words)
- **innate** [adj] (2/3: Claude,GPT) — favorable regard arising necessarily from fundamental nature
- **instinctive** [adj] (2/3: Claude,GPT) — liking arising necessarily from biological foundation
- **maternal** [adj] (1/3: Claude) — fondness arising necessarily from bond of parenthood
- **inherent** [adj] (1/3: Claude) — favorable regard existing as necessary part of being
- **primal** [adj] (1/3: Claude) — liking rooted in most fundamental necessary drives

## EXPERIENCE

### experience.discovery (7 words)
- **smitten** [adj] (1/3: Claude) — experiential quality of discovering unexpected favorable regard
- **fascinated** [adj] (1/3: Claude) — experiencing discovery of something deeply likable
- **delighted** [adj] (1/3: Claude) — experiential quality of discovering something that pleases
- **wonder** [n] (1/3: Claude) — experience of encountering something surprisingly likable
- **enchanted** [adj] (1/3: Claude) — experiential quality of discovering captivating favorable regard
- **fascination** [n] (1/3: GPT) — experience of liking something new and unknown
- **enchanting** [adj] (1/3: GPT) — liking characterized by wonder of a new discovery

### experience.creation (4 words)
- **passion** [n] (2/3: Claude,GPT) — liking experienced as creative drive to bring something new
- **inspired** [adj] (1/3: Claude) — experiencing fondness as generative creative force
- **enthusiasm** [n] (2/3: Claude,GPT) — liking experienced as energy of creating new things
- **ardor** [n] (1/3: Claude) — intense favorable regard experienced in the act of creating

### experience.comprehension (6 words)
- **gratitude** [n] (1/3: Claude) — favorable regard through ultimate comprehension of what is valued
- **reverence** [n] (1/3: Claude) — deep liking experienced upon comprehending fullness of worth
- **awe** [n] (1/3: Claude) — favorable regard when fully comprehending something profound
- **appreciation** [n] (2/3: Claude,Gemini) — liking deepened to point of comprehensive understanding
- **admiration** [n] (1/3: GPT) — liking arising from complete understanding of excellence
- **profoundly** [adv] (1/3: GPT) — depth of liking when fully comprehending the subject

## VALUE_GENERATION

### value_generation.functional (7 words)
- **prefer** [v] (1/3: Claude) — liking based on practical utility
- **handy** [adj] (2/3: Claude,GPT) — liked for its functional usefulness
- **useful** [adj] (3/3: Claude,Gemini,GPT) — favorably regarded for practical value
- **convenient** [adj] (1/3: Claude) — liked because it serves a functional purpose well
- **reliable** [adj] (2/3: Claude,Gemini) — favored for dependable functional performance
- **favorably** [adv] (1/3: Claude) — regarding something positively based on utility
- **utilize** [v] (1/3: GPT) — to employ something because of favorable utility

### value_generation.aesthetic (9 words)
- **beautiful** [adj] (3/3: Claude,Gemini,GPT) — liked for aesthetic and sensory appeal
- **admire** [v] (1/3: Claude) — favorable regard grounded in aesthetic appreciation
- **lovely** [adj] (1/3: Claude) — liked for its pleasing aesthetic qualities
- **gorgeous** [adj] (1/3: Claude) — strongly liked for visual or sensory beauty
- **exquisite** [adj] (2/3: Claude,GPT) — favorably regarded for refined aesthetic quality
- **elegant** [adj] (1/3: Claude) — liked for graceful aesthetic sensibility
- **admiration** [n] (1/3: Claude) — favorable regard based on aesthetic awareness
- **pleasing** [adj] (1/3: Gemini) — aesthetic-driven liking
- **graceful** [adj] (1/3: GPT) — favorable regard for aesthetic quality of form

### value_generation.ethical (9 words)
- **respect** [n] (1/3: Claude) — favorable regard grounded in moral worth
- **esteem** [n] (1/3: Claude) — liking based on perceived moral character
- **admirable** [adj] (3/3: Claude,Gemini,GPT) — liked because of upright ethical qualities
- **honorable** [adj] (1/3: Claude) — favorably regarded for adherence to moral standards
- **trustworthy** [adj] (1/3: Claude) — liked for reliability rooted in ethical character
- **upstanding** [adj] (1/3: Claude) — favorably regarded for meeting moral norms
- **respectable** [adj] (1/3: Gemini) — liking based on moral evaluation
- **noble** [adj] (1/3: GPT) — liking because it aligns with moral duty
- **virtuous** [adj] (1/3: GPT) — liking based on high moral standards

### value_generation.sacred (7 words)
- **revere** [v] (2/3: Claude,GPT) — favorable regard touching deepest level of existential value
- **devotion** [n] (1/3: Claude) — liking elevated to level of sacred commitment
- **worship** [v] (1/3: Claude) — favorable regard directed at what is primordially valuable
- **venerate** [v] (1/3: Claude) — liking expressed as sacred respect for deep worth
- **reverence** [n] (1/3: Claude) — profound favorable regard resonating with ultimate value
- **veneration** [n] (1/3: GPT) — deep sacred liking for a being or existence
- **hallowed** [adj] (1/3: GPT) — regarded as holy and held in highest favorable regard
