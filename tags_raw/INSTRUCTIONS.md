# Tagging Magic cards with synergy roles

Each input line is: `idx|name|mana cost|type line [power/toughness]|rules text` (" / " separates text lines, "//" separates faces).

For EVERY card, write exactly one output line:

`idx|PROVIDES|REWARDS|TRIBES`

* PROVIDES: codes for resources/things this card puts into the game for YOUR side (or simply *is*), that other cards could exploit.
* REWARDS: codes for things that make this card better, that it triggers off, counts, cares about, or needs.
* TRIBES: lowercase singular creature types this card specifically cares about (e.g. `elf` for "Elves you control"), else empty.
* Codes are comma-separated with no spaces. Leave a field empty if nothing applies. Tag only what is meaningful; most cards have 0-3 codes per field.
* Judge by what the card actually does (including new mechanics: read their reminder text).

## Codes (use the same list for PROVIDES and REWARDS)
TOK creature tokens | TRE treasure/clue/food/blood/other artifact tokens | ART artifacts (is an artifact or makes them) | ENC enchantments
SPL instant/sorcery spells (PROVIDES = is a cheap one; REWARDS = cares about casting them, prowess, magecraft)
NCS noncreature spells cast | CRE creatures entering/being cast (REWARDS = "whenever a creature enters/you cast a creature")
GOW many creatures (go-wide; REWARDS = anthems, "for each creature") | CTR +1/+1 counters | OCT other counters (oil, stun, lore, charge, proliferate-able)
GY puts cards into your own graveyard (mill, discard, surveil, self-sacrifice) | REC uses your graveyard (recursion, flashback, escape, delve, reanimate)
BIG large creature / high power (power 4+) | SAC sacrifice (PROVIDES = fodder or a free sac outlet; REWARDS = gains when things are sacrificed / die)
DIE death triggers | ETB valuable enters-the-battlefield effect (REWARDS = blink, flicker, ETB doubling)
LIF gain life | DRW extra card draw | DIS opponent discards / opponent mills | DMG noncombat damage or life loss to opponents
MAN extra mana / mana creatures / ramp | LND lands entering or extra land drops (REWARDS = landfall) | XSP big mana sinks, X spells, expensive payoffs
LEG legendary | EQA equipment or auras | VEH vehicles/crew | EVA evasive attacker | ATK attacking / combat damage triggers
TAP tapping/untapping your own permanents | CPY copying spells or permanents | CST cost reduction | FLS flash / instant-speed play
PSN poison/toxic/infect | NRG energy | HND hand size / cards in hand

## Examples
`12|Llanowar Elves|{G}|Creature — Elf Druid 1/1|{T}: Add {G}.` -> `12|MAN,CRE||elf`
`13|Hardened Scales|{G}|Enchantment|If one or more +1/+1 counters would be put on a creature you control, that many plus one ... are put on it instead.` -> `13|ENC|CTR|`
`14|Witch's Oven|{1}|Artifact|{T}, Sacrifice a creature: Create a Food token...` -> `14|ART,SAC,TRE|DIE|`
`15|Prowess-style card: Whenever you cast a noncreature spell, this creature gets +1/+1` -> `15|CRE|NCS,SPL|`

Output ONLY the tag lines, one per input card, in input order. Do not skip any card.
