"""NEURAL card-rating models, tested head-to-head with the gradient-boosted trees (05_train.py).

  mlp : neural net on [frozen language-model embedding of the card text (with reminder text) + all card features]
  ft  : fine-tuned language model (MiniLM-L6) reading "cost | type | P/T | rules text", its pooled output joined with
        the card features, trained end-to-end on how much the card saw play
Both predict all 6 formats at once (masked loss; Standard weighted x2). Target = log1p(100 x peak early play rate).
Forward test per year Y: train on cards whose window started before Y, predict cards first eligible in Y.
Usage: python src/24_card_nn.py mlp 2019,2020,2021,2022,2023
       python src/24_card_nn.py ft 2021,2023
Predictions saved to output/nn/<model>_<year>.parquet for the ensemble comparison (25_ensemble_eval.py).
"""
import os, sys, pathlib, importlib.util, time, numpy as np, pandas as pd, torch, torch.nn as nn
ROOT = pathlib.Path(__file__).resolve().parents[1]; PROC, OUT = ROOT / "data/processed", ROOT / "output"; NO = OUT / "nn"; NO.mkdir(exist_ok=True)
spec = importlib.util.spec_from_file_location("t5", ROOT / "src/05_train.py"); T = importlib.util.module_from_spec(spec); spec.loader.exec_module(T)
torch.set_num_threads(4); torch.manual_seed(0); np.random.seed(0)
FMTS = T.FORMATS; FW = torch.tensor([2.0 if f == "standard" else 1.0 for f in FMTS])
df = T.df
cards = pd.read_parquet(PROC / "cards.parquet").set_index("oracle_id")
feat = pd.read_parquet(PROC / "features.parquet").set_index("oracle_id")
NUMC = [c for c in feat.select_dtypes("number").columns]
NUM = feat[NUMC].fillna(0).astype(np.float32); NUM = ((NUM - NUM.mean()) / (NUM.std() + 1e-6)).clip(-5, 5)
E = np.load(PROC / "embeddings.npy"); EID = pd.read_csv(PROC / "embeddings_ids.csv").oracle_id; EPOS = {o: i for i, o in enumerate(EID)}
def doc(o):
    c = cards.loc[o]; pt = f" | {c.power}/{c.toughness}" if isinstance(c.power, str) else ""
    return f"{c.mana_cost or ''} | {feat.type_line[o]}{pt} | {feat.clean_text[o]}"

def targets(rows):
    """card-level matrix of targets (6 formats) + mask"""
    piv = rows.pivot_table(index="oracle_id", columns="format", values="y", aggfunc="mean").reindex(columns=FMTS)
    return piv.index.values, torch.tensor(piv.fillna(0).values, dtype=torch.float32), torch.tensor(piv.notna().values, dtype=torch.float32)

class MLP(nn.Module):
    def __init__(s, d):
        super().__init__(); s.net = nn.Sequential(nn.Linear(d, 512), nn.GELU(), nn.Dropout(0.3), nn.Linear(512, 128), nn.GELU(), nn.Dropout(0.2), nn.Linear(128, len(FMTS)))
    def forward(s, x): return s.net(x)

def masked_loss(pred, y, m): return (((pred - y) ** 2) * m * FW).sum() / (m * FW).sum()

def run_mlp(train_rows, test_ids):
    ids, Y, Mk = targets(train_rows)
    X = lambda o_: torch.tensor(np.hstack([E[[EPOS[o] for o in o_]], NUM.loc[o_].values]), dtype=torch.float32)
    Xtr = X(ids); net = MLP(Xtr.shape[1]); opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
    for ep in range(40):
        net.train(); perm = torch.randperm(len(ids))
        for s0 in range(0, len(ids), 256):
            j = perm[s0:s0 + 256]; loss = masked_loss(net(Xtr[j]), Y[j], Mk[j]); opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad(): return net(X(test_ids)).numpy()

class FT(nn.Module):
    def __init__(s, d_num):
        super().__init__()
        from transformers import AutoModel
        s.lm = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        s.num = nn.Sequential(nn.Linear(d_num, 128), nn.GELU(), nn.Dropout(0.2))
        s.head = nn.Sequential(nn.Linear(384 + 128, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, len(FMTS)))
    def forward(s, enc, xnum):
        h = s.lm(**enc).last_hidden_state; m = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
        return s.head(torch.cat([pooled, s.num(xnum)], 1))

def run_ft(train_rows, test_ids, epochs=int(os.environ.get("FT_EPOCHS", "3"))):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    ids, Y, Mk = targets(train_rows)
    net = FT(NUM.shape[1])
    opt = torch.optim.AdamW([{"params": net.lm.parameters(), "lr": 4e-5}, {"params": list(net.num.parameters()) + list(net.head.parameters()), "lr": 1e-3}], weight_decay=1e-2)
    docs = {o: doc(o) for o in set(ids) | set(test_ids)}
    def batch(o_):
        enc = tok([docs[o] for o in o_], padding=True, truncation=True, max_length=160, return_tensors="pt")
        return enc, torch.tensor(NUM.loc[o_].values, dtype=torch.float32)
    steps = epochs * ((len(ids) + 31) // 32); sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[4e-5, 1e-3], total_steps=steps, pct_start=0.1)
    t0 = time.time()
    for ep in range(epochs):
        net.train(); perm = np.random.permutation(len(ids)); tot = 0
        for s0 in range(0, len(ids), 32):
            j = perm[s0:s0 + 32]; enc, xn = batch(list(ids[j]))
            loss = masked_loss(net(enc, xn), Y[j], Mk[j]); opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += loss.item() * len(j)
        print(f"     epoch {ep + 1}/{epochs} loss {tot / len(ids):.3f}  ({(time.time() - t0) / 60:.0f} min)", flush=True)
    net.eval(); out = []
    with torch.no_grad():
        for s0 in range(0, len(test_ids), 128):
            enc, xn = batch(list(test_ids[s0:s0 + 128])); out.append(net(enc, xn).numpy())
    return np.vstack(out)

if __name__ == "__main__" and sys.argv[1] in ("mlp", "ft"):
    kind = sys.argv[1]; years = [int(y) for y in sys.argv[2].split(",")]
    for Y in years:
        lo, hi = f"{Y}-01", f"{Y + 1}-01"
        tr = df[df.window_start < lo]; te = df[(df.window_start >= lo) & (df.window_start < hi)].reset_index(drop=True)
        tids = te.oracle_id.unique()
        t0 = time.time()
        P = run_mlp(tr, tids) if kind == "mlp" else run_ft(tr, tids)
        pc = pd.DataFrame(P, index=tids, columns=FMTS)
        te["pred"] = [pc.at[o, f] for o, f in zip(te.oracle_id, te.format)]
        te[["oracle_id", "name", "format", "rarity", "target", "y", "pred"]].to_parquet(NO / f"{kind}_{Y}.parquet")
        r = T.report(kind, te, te.pred.values)
        print(f"{kind} {Y}: " + str({k: v for k, v in r.items() if k.startswith("standard")}) + f"  ({(time.time() - t0) / 60:.1f} min)", flush=True)

# ------------------------------------------------------------------ final: train on everything, score the post-rotation pool
if __name__ == "__main__" and sys.argv[1:2] == ["final"]:
    pool = [o for o in pd.read_csv(PROC / "future_pool.csv").oracle_id if o in feat.index and o in cards.index]
    # blind, like 06_score_pool.py: the pool cards' own results are never in training
    P = run_ft(df[~df.oracle_id.isin(set(pool))].reset_index(drop=True), np.array(pool))
    out = pd.DataFrame(P, index=pool, columns=FMTS)[["standard"]].rename(columns={"standard": "ft_pred"})
    out.index.name = "oracle_id"; out.reset_index().to_parquet(NO / "ft_pool.parquet"); print("scored", len(out), "pool cards")
