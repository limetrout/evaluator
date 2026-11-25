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
need = [k for k in months if k not in index or k == f"{today.year}/{today.month:02d}"]

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

todo = [s for k in sorted(index) for s in index[k] if (MAJOR.match(s) or MINOR.match(s)) and not (OUT / f"{s}.json").exists()]
todo.sort(key=lambda s: bool(MINOR.match(s)))   # challenges / qualifiers first
print("events to download:", len(todo), flush=True)

def event(slug):
    html = get(f"https://www.mtgo.com/decklist/{slug}", must="decklists.data")
    m = html and re.search(r'window\.MTGO\.decklists\.data\s*=\s*(\{.*?\});\s*\n', html, re.S)
    if m: (OUT / f"{slug}.json").write_text(m.group(1))
    return slug, bool(m)

with ThreadPoolExecutor(WORKERS) as ex:
    for i, (slug, ok) in enumerate(ex.map(event, todo)):
        if i % 100 == 0: print(f"{i}/{len(todo)} {slug} {ok}", flush=True)
print("done", flush=True)
