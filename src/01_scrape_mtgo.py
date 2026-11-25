"""Download MTGO tournament decklists from mtgo.com (the official Wizards site).

Walks the monthly archive pages (mtgo.com/decklists/YYYY/MM) and keeps competitive events:
  * Challenges / Qualifiers / Showcases / PTQs etc. in Standard, Pioneer, Modern, Legacy, Pauper, Vintage
  * Daily / Preliminary events (published winning-record decks) in Standard, Pioneer, Modern
Each event's embedded JSON (all decklists + standings) is saved to data/raw/mtgo/<slug>.json.
Resumable (skips files already on disk). Uses a small number of parallel connections.
"""
import json, re, sys, time, datetime, pathlib, requests, threading
from concurrent.futures import ThreadPoolExecutor

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "data/raw/mtgo"; OUT.mkdir(parents=True, exist_ok=True)
INDEX = ROOT / "data/raw/mtgo_index.json"
MAJOR = re.compile(r"^(standard|pioneer|modern|legacy|pauper|vintage)-(.*challenge|.*qualifier|.*showcase|.*champ|.*premier|.*ptq|.*playoff|.*finals?)", re.I)
MINOR = re.compile(r"^(standard|pioneer|modern)-(.*daily|.*preliminary|.*prelim)", re.I)
WORKERS = int(sys.argv[1]) if len(sys.argv) > 1 else 4
local = threading.local()

def get(url, must=None):
    if not hasattr(local, "s"):
        local.s = requests.Session(); local.s.headers["User-Agent"] = "Mozilla/5.0 (card-research; low-rate)"
    for i in range(5):
        try:
            r = local.s.get(url, timeout=90)
            if r.status_code == 200 and (must is None or must in r.text): return r.text
            if r.status_code == 404: return None
        except requests.RequestException: pass
        time.sleep(15 * (i + 1))
    return None

index = json.loads(INDEX.read_text()) if INDEX.exists() else {}
today = datetime.date.today()
months = [f"{y}/{m:02d}" for y in range(2016, today.year + 1) for m in range(1, 13) if (y, m) <= (today.year, today.month)]
need = [k for k in months if k not in index]   # delete a month from the index file to re-fetch it

def month(key):
    html = get(f"https://www.mtgo.com/decklists/{key}", must="/decklist/")
    if html is None: return key, None
    slugs = sorted(set(re.findall(r'/decklist/([a-z0-9-]+)', html)))
    return key, [s for s in slugs if MAJOR.match(s) or MINOR.match(s)]

with ThreadPoolExecutor(WORKERS) as ex:
    for key, kept in ex.map(month, need):
        if kept is None: continue
        index[key] = kept; INDEX.write_text(json.dumps(index, indent=0))
        print(key, len(kept), "kept", flush=True)

GH = ROOT / "data/raw/mtgo_gh"   # older events already fetched from the GitHub mirror (01b script)
todo = [s for k in sorted(index) for s in index[k] if (MAJOR.match(s) or MINOR.match(s))
        and not (OUT / f"{s}.json").exists() and not (GH / f"{s}.json").exists()
        and re.search(r"\d{4}-\d{2}-\d{2}", s).group(0) >= "2024-06-20"]
PRIO = {"standard": 0, "pioneer": 1, "modern": 2, "pauper": 3, "legacy": 4, "vintage": 5}
# Round-robin by month so every month gets coverage early: 1st event of every month for Std/Pio/Mod,
# then the 2nd, ... ; Pauper/Legacy/Vintage after that. Majors (challenges) before dailies/prelims.
from collections import defaultdict
seen = defaultdict(int); keyed = []
fmt_of = lambda s: s.split("-")[0]; date_of = lambda s: re.search(r"\d{4}-\d{2}-\d{2}", s).group(0)
for s_ in sorted(todo, key=lambda s: (bool(MINOR.match(s)), "challenge" not in s, date_of(s))):
    k = (fmt_of(s_), date_of(s_)[:7]); r = seen[k]; seen[k] += 1
    keyed.append(((PRIO[fmt_of(s_)] > 2, r, PRIO[fmt_of(s_)], date_of(s_)), s_))
todo = [s_ for _, s_ in sorted(keyed)]
print("events to download:", len(todo), flush=True)

# mtgo.com rate-limits bursts (serves an empty page): go one at a time, pause, back off when blocked
S1 = requests.Session(); S1.headers["User-Agent"] = "Mozilla/5.0 (card-research; low-rate)"
pause = 5
for i, slug in enumerate(todo):
    for attempt in range(6):
        try: html = S1.get(f"https://www.mtgo.com/decklist/{slug}", timeout=90).text
        except requests.RequestException: html = ""
        m = re.search(r'window\.MTGO\.decklists\.data\s*=\s*(\{.*?\});\s*\n', html, re.S)
        if m:
            (OUT / f"{slug}.json").write_text(m.group(1)); time.sleep(pause); break
        if "/decklist" not in html and len(html) > 1000 and attempt >= 2: break      # genuinely empty event page
        time.sleep(60 * (attempt + 1))
    if i % 25 == 0: print(f"{i}/{len(todo)} {slug}", flush=True)
print("done", flush=True)
