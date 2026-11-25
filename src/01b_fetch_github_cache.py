"""Fast path for older events: fetch the same mtgo.com events from the MTGODecklistCache archive on GitHub
(raw.githubusercontent.com), which mirrors mtgo.com up to mid-2024. Saved to data/raw/mtgo_gh/<slug>.json."""
import json, re, pathlib, requests
from concurrent.futures import ThreadPoolExecutor
ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "data/raw/mtgo_gh"; OUT.mkdir(parents=True, exist_ok=True)
idx = json.loads((ROOT / "data/raw/mtgo_index.json").read_text())
slugs = [s for k in sorted(idx) for s in idx[k]]
def date(s): return re.search(r"(\d{4})-(\d{2})-(\d{2})", s).groups()
todo = [s for s in slugs if "-".join(date(s)) < "2024-06-20" and not (OUT / f"{s}.json").exists() and not (ROOT / f"data/raw/mtgo/{s}.json").exists()]
print("to fetch:", len(todo), flush=True)
S = requests.Session()
def get(s):
    y, m, d = date(s)
    for base in ("Tournaments", "Tournaments-Archive"):
        try:
            r = S.get(f"https://raw.githubusercontent.com/Badaro/MTGODecklistCache/master/{base}/mtgo.com/{y}/{m}/{d}/{s}.json", timeout=60)
        except requests.RequestException: continue
        if r.status_code == 200: (OUT / f"{s}.json").write_bytes(r.content); return True
    return False
with ThreadPoolExecutor(8) as ex:
    res = list(ex.map(get, todo))
print("fetched", sum(res), "missing", len(res) - sum(res))
