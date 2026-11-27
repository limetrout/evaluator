"""Keyword -> reminder text dictionary, harvested from Scryfall Oracle text.

Many cards print a keyword's explanation in parentheses (e.g. "Earthbend 2 (Target land you control becomes a 0/0
creature ...)"). We collect, for every keyword in Scryfall's keyword list, the most common explanation that follows it,
with numbers normalised to N so "Earthbend 1" and "Earthbend 3" share one entry. Cards that use the keyword without
printing the explanation get it inserted (03_features.py), so the text models see what the keyword DOES.
Output: data/processed/reminders.json  {keyword_lower: reminder_text}
"""
import json, gzip, re, pathlib, collections
ROOT = pathlib.Path(__file__).resolve().parents[1]
cnt = collections.defaultdict(collections.Counter)
for l in gzip.open(ROOT / "data/raw/oracle_cards.jsonl.gz", "rt"):
    c = json.loads(l)
    kws = c.get("keywords") or []
    texts = [f.get("oracle_text", "") for f in (c.get("card_faces") or [c])]
    for t in texts:
        for k in kws:
            # keyword, optional cost/number/word(s), then a parenthesised explanation
            for m in re.finditer(rf"(?i)\b{re.escape(k)}\b[^\n(]{{0,40}}\(([^)]{{8,400}})\)", t):
                rem = re.sub(r"\b\d+\b", "N", m.group(1).strip())
                rem = re.sub(r"\b(one|two|three|four|five|six|seven|eight|nine|ten|X)\b", "N", rem)
                cnt[k.lower()][rem] += 1
rem = {k: v.most_common(1)[0][0] for k, v in cnt.items() if v}
(ROOT / "data/processed/reminders.json").write_text(json.dumps(rem, indent=1, ensure_ascii=False))
allk = collections.Counter()
for l in gzip.open(ROOT / "data/raw/oracle_cards.jsonl.gz", "rt"):
    for k in json.loads(l).get("keywords") or []: allk[k.lower()] += 1
missing = [(k, n) for k, n in allk.most_common() if k not in rem]
print(f"{len(rem)} keywords with harvested reminder text; {len(missing)} keywords without")
print("most common keywords WITHOUT reminder:", missing[:45])
for k in ["earthbend", "firebending", "airbend", "waterbend", "station", "warp", "lesson", "exhaust", "start your engines!", "max speed", "flashback", "prowess", "ward", "toxic"]:
    print(f"  {k}: {rem.get(k, '-')[:160]}")
