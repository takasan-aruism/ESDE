# ESDE Synapse v4 — Task 3: ObsC Grounding Comparison (v3 vs v4)

**Date**: 2026-03-01 19:28 UTC
**Phase**: Analysis-only (no permanent changes)
**Method**: Re-ground existing v3 edges using A1-derived v4 candidates

## Summary

| Metric | Synapse v3 | v4 Candidates | Delta |
|--------|-----------|---------------|-------|
| Grounded | 1944 | 4125 | +2181 |
| Ungrounded | 2649 | 468 | -2181 |
| Light verb (skipped) | 749 | 749 | 0 |
| **Grounding rate** | **42.3%** | **89.8%** | **+47.5pp** |
| Category mismatch | 0 | 0 | +0 |

## Agreement Analysis

| Metric | Count |
|--------|-------|
| Both grounded, same atom | 40 |
| Both grounded, different atom | 1781 |
| v4 newly grounded (v3 was ungrounded) | 2304 |
| v4 lost grounding (v3 was grounded) | 123 |
| **Agreement rate** (when both grounded) | **2.2%** |

## Category Coverage Comparison

| Category | v3 | v4 | Diff |
|----------|----|----|------|
| ACT | 339 | 1053 | +714 |
| PER | 24 | 263 | +239 |
| SOC | 264 | 501 | +237 |
| FND | 150 | 384 | +234 |
| CHG | 43 | 163 | +120 |
| STA | 33 | 152 | +119 |
| ECO | 90 | 192 | +102 |
| COG | 145 | 229 | +84 |
| BEI | 44 | 123 | +79 |
| EMO | 167 | 245 | +78 |
| LOG | 84 | 7 | -77 |
| ELM | 39 | 95 | +56 |
| COM | 177 | 228 | +51 |
| EXS | 4 | 50 | +46 |
| ABS | 66 | 104 | +38 |
| TIM | 5 | 37 | +32 |
| REL | 40 | 57 | +17 |
| WLD | 98 | 111 | +13 |
| SPC | 45 | 56 | +11 |
| BOD | 57 | 48 | -9 |
| VAL | 30 | 27 | -3 |

## Decision Rule Evaluation

✅ Grounding rate improved: 42.3% → 89.8%
✅ High-impact verb categories (ACT, CHG) improved

**Verdict**: v4 candidate approach shows viability for Synapse improvement.

## Sample Disagreements (v3 ≠ v4, both grounded)

| Verb | v3 Atom | v4 Atom | v4 cos | Sentence |
|------|---------|---------|--------|----------|
| surround | SPC.outside | SPC.inside | 0.7500 | Berlin is surrounded by the state of Brandenburg, and Brande... |
| leave | ACT.exit | PER.untouched | 0.9599 | After the Semnones left around 200 CE, the Burgundians follo... |
| reach | ACT.go | ACT.arrive | 0.9568 | In the 7th century Slavic tribes, the later known Hevelli an... |
| form | ACT.create | SPC.outside | 0.9291 | The two towns over time formed close economic and social tie... |
| form | ACT.create | SPC.outside | 0.9291 | In 1307 the two towns formed an alliance with a common exter... |
| start | ELM.morning | CHG.begin | 0.9094 | In 1443, Frederick II Irontooth started the construction of ... |
| succeed | EMO.pride | WLD.unskilled | 0.9142 | Frederick William, known as the "Great Elector", who had suc... |
| form | ACT.create | SPC.outside | 0.9291 | In 1701, the dual state formed the Kingdom of Prussia, as Fr... |
| transform | ACT.go | CHG.end | 0.7760 | The Industrial Revolution transformed Berlin during the 19th... |
| increase | SOC.work | CHG.advance | 0.8443 | The act increased the area of Berlin from 66 to 883 km2 (25 ... |
| experience | EMO.like | ACT.receive | 0.8669 | The metropolis experienced its heyday as a major world capit... |
| develop | ACT.obtain | COG.learn | 0.9022 | Adolf Hitler and Albert Speer developed architectural concep... |
| develop | ACT.obtain | COG.learn | 0.9022 | Adolf Hitler and Albert Speer developed architectural concep... |
| diminish | EMO.contempt | SOC.refuse | 0.9036 | NSDAP rule diminished Berlin's Jewish community from 160,000... |
| drop | ACT.fall | ACT.give | 0.9154 | The Allies dropped 67,607 tons of bombs on the city, destroy... |
| receive | ACT.receive | EMO.compassion | 0.8976 | After the end of World War II in Europe in May 1945, Berlin ... |
| divide | COG.separate | COM.conflict | 0.8971 | The victorious powers divided the city into four sectors, an... |
| divide | COG.separate | COM.conflict | 0.8971 | The victorious powers divided the city into four sectors, an... |
| form | ACT.create | SPC.outside | 0.9291 | The victorious powers divided the city into four sectors, an... |
| form | ACT.create | SPC.outside | 0.9291 | The victorious powers divided the city into four sectors, an... |
| form | ACT.create | SPC.outside | 0.9291 | The victorious powers divided the city into four sectors, an... |
| share | BOD.mouth | BEI.parent | 0.8475 | All four Allies of World War II shared administrative respon... |
| impose | ABS.responsibility | EMO.anger | 0.9469 | However, in 1948, when the Western Allies extended the curre... |
| increase | SOC.work | CHG.advance | 0.8443 | The founding of the two German states increased Cold War ten... |
| surround | SPC.outside | SPC.inside | 0.7500 | West Berlin was surrounded by East German territory, and Eas... |
| experience | EMO.like | ACT.receive | 0.8669 | After the fall of the Berlin Wall, the city experienced sign... |
| share | BOD.mouth | BEI.parent | 0.8475 | Both share a common history, dialect and culture and in 2020... |
| share | BOD.mouth | BEI.parent | 0.8475 | Both share a common history, dialect and culture and in 2020... |
| form | ACT.create | SPC.outside | 0.9291 | The Berliner Urstromtal (an ice age glacial valley), between... |
| follow | SPC.direction | COG.enlightenment | 0.9365 | The Spree follows this valley now.... |

## Sample v4 New Groundings (v3 was ungrounded)

| Verb | v4 Atom | v4 cos | Sentence |
|------|---------|--------|----------|
| include | ACT.receive | 0.8068 | The city includes lakes in the western and southeastern boro... |
| trigger | LOG.effect | 0.7613 | During the Gründerzeit, an industrialization-induced economi... |
| include | ACT.receive | 0.8068 | Significant industries include information technology, the h... |
| include | ACT.receive | 0.8068 | Other landmarks include the Brandenburg Gate, the Reichstag ... |
| include | ACT.receive | 0.8068 | Other landmarks include the Brandenburg Gate, the Reichstag ... |
| exhibit | EMO.manifest | 0.8837 | Berlin lies in northeastern Germany, in an area formerly set... |
| bear | STA.comfort | 0.9301 | Of Berlin's twelve boroughs, five bear a Slavic-derived name... |
| bear | STA.comfort | 0.9301 | Of Berlin's twelve boroughs, five bear a Slavic-derived name... |
| raid | ECO.waste | 0.8846 | In 1326 the territory of Berlin was raided by pagan Lithuani... |
| lose | CHG.retreat | 0.9567 | One third of its houses were damaged or destroyed, and the c... |
| initiate | COG.enlightenment | 0.9103 | Frederick William, known as the "Great Elector", who had suc... |
| offer | SOC.request | 0.9287 | With the Edict of Potsdam in 1685, Frederick William offered... |
| crown | EMO.respect | 0.8915 | In 1701, the dual state formed the Kingdom of Prussia, as Fr... |
| proclaim | FND.language | 0.9455 | At the end of World War I in 1918, a republic was proclaimed... |
| incorporate | CHG.grow | 0.8154 | In 1920, the Greater Berlin Act incorporated dozens of subur... |
| undergo | FND.transformation | 0.7737 | During the Weimar era, Berlin underwent political unrest due... |
| inspire | ACT.calm | 0.9132 | Hitler was inspired by the architecture he had experienced i... |
| host | ACT.receive | 0.8191 | Berlin hosted the 1936 Summer Olympics for which the Olympic... |
| extend | ACT.sit | 0.9025 | However, in 1948, when the Western Allies extended the curre... |
| overcome | SOC.awake | 0.8744 | The Berlin airlift, conducted by the three western Allies, o... |
