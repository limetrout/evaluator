"""NEURAL PAIR-SYNERGY MODEL: learns WHY two cards go together.

Each card -> neural encoder (inputs: language-model text embedding, card stats/types/regex roles, Claude role tags)
         -> GIVE vector g (K channels, >= 0): what the card supplies
         -> WANT vector w (K channels, >= 0): what the card benefits from
         -> STYLE vector s (KS channels, >= 0): "both are this kind of card"
synergy(A,B) = sum_k [g_A,k * w_B,k + g_B,k * w_A,k]  +  sum_j s_A,j * s_B,j        <- interpretable "why"
logit = synergy + linear(explicit pair facts: colour overlap, curve, shared tribe, mentions, partner popularity)
Trained on: for every card released in a past year, every card it could have paired with in its format; target =
strong partner (1.0) / mild partner (0.4) / not (0). Channels are sparse (L1) so each card uses few of them.

Modes:  python src/18_neural_synergy.py backtest   -> forward test 2019-2024, per-card recall@10 vs v3 ranker
        python src/18_neural_synergy.py final      -> train on all years, name the channels, explain + score future pool
"""
import os, sys, pathlib, importlib.util, numpy as np, pandas as pd, torch, torch.nn as nn
ROOT = pathlib.Path(__file__).resolve().parents[1]; PROC, OUT = ROOT / "data/processed", ROOT / "output"; NO = OUT / "neural"
spec = importlib.util.spec_from_file_location("r14", ROOT / "src/14_synergy_ranker.py"); R = importlib.util.module_from_spec(spec); spec.loader.exec_module(R)
M = R.M
torch.set_num_threads(4); torch.manual_seed(0); np.random.seed(0)
K, KS = 24, 8
EXPL = ["colors_shared", "colors_union", "same_colors", "any_colorless", "mv_min", "mv_diff", "shared_subtype", "mention_max",
        "tag_tribe", "tag_shared_tribe_care", "cand_pop_prev", "fmt_standard", "fmt_pioneer", "fmt_modern",
        ] + (["map_cos", "map_cos_textonly", "map_n_real"] if os.environ.get("NN_MAP") == "1" else [])   # deck-map shortcut OFF by default

# ---------------------------------------------------------------- per-card input matrix (static, printed-card info only)
ids = list(M.feat.index); CI = {o: i for i, o in enumerate(ids)}
num = M.feat.select_dtypes("number").drop(columns=[c for c in ["release_year"] if c in M.feat.columns]).fillna(0).astype(np.float32)
num = (num - num.mean()) / (num.std() + 1e-6)
emb = M.E[[M.epos[o] for o in ids]].astype(np.float32)
roles = np.hstack([M.Smat, M.Pmat]).astype(np.float32)
tg = pd.read_parquet(PROC / "card_tags.parquet").set_index("oracle_id").reindex(ids)
TAGCOLS = [c for c in tg.columns if c != "tribes"]
tags = tg[TAGCOLS].fillna(0).values.astype(np.float32)
XC = torch.tensor(np.hstack([emb, num.values.clip(-5, 5) * 0.5, roles, tags]))
print("card input dims:", XC.shape, flush=True)

class PairNet(nn.Module):
    def __init__(s, d_in, n_expl):
        super().__init__()
        s.enc = nn.Sequential(nn.Linear(d_in, 384), nn.ReLU(), nn.Dropout(0.2), nn.Linear(384, 192), nn.ReLU())
        s.give, s.want, s.style = nn.Linear(192, K), nn.Linear(192, K), nn.Linear(192, KS)
        s.hub = nn.Linear(192, 1)          # general flexibility ("goes in lots of decks") - NOT part of the explanation
        s.lin = nn.Linear(n_expl, 1)
    def card(s, x):
        h = s.enc(x); sp = nn.functional.softplus
        off = float(os.environ.get("NN_OFF", "0.5"))
        return sp(s.give(h) - off), sp(s.want(h) - off), sp(s.style(h) - off)
    def channels(s, xa, xb):
        ga, wa, sa = s.card(xa); gb, wb, sb = s.card(xb)
        return ga * wb + gb * wa, sa * sb, (ga, wa, sa, gb, wb, sb)
    def hubness(s, x): return s.hub(s.enc(x)).squeeze(1)
    def forward(s, xa, xb, e):
        ch, st, acts = s.channels(xa, xb)
        return ch.sum(1) + st.sum(1) + s.lin(e).squeeze(1) + s.hubness(xa) + s.hubness(xb), acts

def rows_for(years, cut_q=None, subsample=True, rng=np.random.default_rng(0)):
    out = []
    for Yp in years:
        d = R.build_year(Yp); l = R.labels(d, cut_q)
        d = d[["focal", "cand", "format", "fy"] + [c for c in EXPL if c in d.columns]].copy()
        for f in ["standard", "pioneer", "modern"]: d[f"fmt_{f}"] = (d.format == f).astype(float)
        d["lab"] = l
        if subsample: keep = (l > 0) | (rng.random(len(d)) < 0.3); d = d[keep]
        out.append(d)
    return pd.concat(out, ignore_index=True)

EMU, ESD = None, None
def tensors(d):
    global EMU, ESD
    E_ = d[EXPL].fillna(-1).values.astype(np.float32)
    if EMU is None: EMU, ESD = E_.mean(0), E_.std(0) + 1e-6
    a = torch.tensor(d.focal.map(CI).values); b = torch.tensor(d.cand.map(CI).values)
    y = torch.tensor(np.select([d.lab.values == 2, d.lab.values == 1], [1.0, 0.4], 0.0).astype(np.float32))
    return a, b, torch.tensor((E_ - EMU) / ESD), y

def train(d, epochs=int(os.environ.get("NN_EPOCHS", "6")), l1=float(os.environ.get("NN_L1", "1e-4"))):
    global EMU, ESD
    EMU = ESD = None
    a, b, e, y = tensors(d)
    net = PairNet(XC.shape[1], len(EXPL))
    with torch.no_grad():   # start the baseline at the real partner rate so channels aren't the only way to lower scores
        p0 = float(y.mean().clamp(1e-3, 0.5)); net.lin.bias.fill_(np.log(p0 / (1 - p0)) - 1.0); net.lin.weight.zero_()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    pos_w = torch.tensor(float((y == 0).sum() / max(1, (y > 0).sum())) ** float(os.environ.get("NN_POSPOW", "1.0")))
    n = len(y)
    for ep in range(epochs):
        net.train(); perm = torch.randperm(n); tot = 0
        for s0 in range(0, n, 4096):
            idx = perm[s0:s0 + 4096]
            # random swap of A/B so the model stays symmetric in practice
            xa, xb = XC[a[idx]], XC[b[idx]]
            logit, (ga, wa, sa, gb, wb, sb) = net(xa, xb, e[idx])
            w = torch.where(y[idx] > 0, pos_w, torch.tensor(1.0))
            loss = (nn.functional.binary_cross_entropy_with_logits(logit, y[idx], reduction="none") * w).mean()
            loss = loss + l1 * (ga.mean() + wa.mean() + gb.mean() + wb.mean() + sa.mean() + sb.mean())
            # diversity: penalise channels that fire on the same cards (off-diagonal correlation across the batch)
            def offcorr(A):
                A = A - A.mean(0, keepdim=True); A = A / (A.norm(dim=0, keepdim=True) + 1e-6); C = A.T @ A
                return (C - torch.diag(torch.diag(C))).pow(2).sum() / (C.shape[0] * (C.shape[0] - 1))
            lam = float(os.environ.get("NN_DIV", "0.5"))
            loss = loss + lam * (offcorr(torch.cat([ga, gb])) + offcorr(torch.cat([wa, wb])) + offcorr(torch.cat([sa, sb])))
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(idx)
        with torch.no_grad():   # health check: how much of the score comes from the interpretable channels?
            j = perm[:20000]; ch, st, _ = net.channels(XC[a[j]], XC[b[j]]); lin = net.lin(e[j]).squeeze(1) + net.hubness(XC[a[j]]) + net.hubness(XC[b[j]])
            syn = ch.sum(1) + st.sum(1)
            share = float(syn.std() / (syn.std() + lin.std() + 1e-9)); alive = int(((ch.mean(0)) > 0.01).sum())
            gg = net.card(XC[a[j]])[0]; gg = gg - gg.mean(0); gg = gg / (gg.norm(dim=0) + 1e-6); cc = (gg.T @ gg).abs()
            redund = float((cc.sum() - cc.diag().sum()) / (K * (K - 1)))
        print(f"   epoch {ep + 1}: loss {tot / n:.4f} | channel share of score spread {share:.2f} | live channels {alive}/{K} | avg channel overlap {redund:.2f}", flush=True)
    net.eval(); return net

@torch.no_grad()
def score(net, d):
    a, b, e, _ = tensors(d); out = []
    for s0 in range(0, len(a), 50000):
        out.append(net(XC[a[s0:s0 + 50000]], XC[b[s0:s0 + 50000]], e[s0:s0 + 50000])[0])
    return torch.cat(out).numpy()

if __name__ == "__main__" and sys.argv[1:2] == ["backtest"]:
    V3 = {2019: .175, 2020: .169, 2021: .185, 2022: .215, 2023: .241, 2024: .228}
    res = []
    for Y in ([int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else range(2019, 2025)):
        cut = Y * 10 + 1
        tr = rows_for([y for y in R.YEARS if y < Y], cut)
        net = train(tr)
        te = rows_for([Y], None, subsample=False)
        rec, n = R.recall_at10(te, score(net, te), te.lab.values)
        r = {"year": Y, "neural_recall@10": round(rec, 3), "v3_ranker_recall@10": V3[Y], "graded": n}; res.append(r); print(r, flush=True)
    Rr = pd.DataFrame(res).set_index("year"); Rr.to_csv(NO / "neural_backtest.csv"); print(Rr.T.to_string()); print(Rr.mean().round(3).to_dict())

# ====================================================================================== FINAL: train, interpret, project
@torch.no_grad()
def card_acts(net, oids):
    g, w, s = net.card(XC[[CI[o] for o in oids]]); return g.numpy(), w.numpy(), s.numpy()

def channel_report(net, ref_ids):
    """For each channel: top givers / wanters and which Claude tags + words are over-represented among them."""
    g, w, s = card_acts(net, ref_ids); name = M.cards.name
    T = tg.reindex(ref_ids)[TAGCOLS].fillna(0); base = T.mean() + 0.01
    words = M.feat.clean_text.reindex(ref_ids).fillna("")
    lines = []
    def enrich(top):
        r = (T.iloc[top].mean() / base).sort_values(ascending=False); return ", ".join(f"{k}x{v:.1f}" for k, v in r.head(5).items())
    def topwords(top):
        from collections import Counter
        cnt = Counter(x for t in words.iloc[top] for x in set(t.split()) if len(x) > 3)
        allc = Counter(x for t in words for x in set(t.split()) if len(x) > 3)
        sc = {k: v / (allc[k] / len(words) * len(top) + 1) for k, v in cnt.items() if v >= 5}
        return ", ".join(k for k, _ in sorted(sc.items(), key=lambda kv: -kv[1])[:8])
    for kind, A in [("GIVE/WANT", None), ("STYLE", s)]:
        rng_ = range(K) if kind == "GIVE/WANT" else range(KS)
        for k in rng_:
            if kind == "GIVE/WANT":
                gs = g[:, k] * g[:, k] / (g.sum(1) + 1e-6); ws = w[:, k] * w[:, k] / (w.sum(1) + 1e-6)   # strength x share
                tg_, tw_ = np.argsort(-gs)[:60], np.argsort(-ws)[:60]
                strength = float(np.sort(g[:, k])[-60:].mean() * np.sort(w[:, k])[-60:].mean())
                lines.append(f"\n=== channel {k}  (strength {strength:.3f})")
                lines.append("  GIVERS: " + "; ".join(name[ref_ids[i]] for i in tg_[:14]))
                lines.append(f"     tags: {enrich(tg_)} | words: {topwords(tg_)}")
                lines.append("  WANTERS: " + "; ".join(name[ref_ids[i]] for i in tw_[:14]))
                lines.append(f"     tags: {enrich(tw_)} | words: {topwords(tw_)}")
            else:
                ts = np.argsort(-(A[:, k] * A[:, k] / (A.sum(1) + 1e-6)))[:60]
                lines.append(f"\n=== style {k}  (strength {float(np.sort(A[:, k])[-60:].mean()**2):.3f})")
                lines.append("  CARDS: " + "; ".join(name[ref_ids[i]] for i in ts[:16]))
                lines.append(f"     tags: {enrich(ts)} | words: {topwords(ts)}")
    return "\n".join(lines)

def explain_pair(net, a, b, names_=None, top=3):
    g, w, s = card_acts(net, [a, b]); name = M.cards.name
    contrib = []
    for k in range(K):
        if g[0, k] * w[1, k] > 0.02: contrib.append((g[0, k] * w[1, k], f"{name[a]} gives [{(names_ or {}).get(str(k), f'ch{k}')}] -> {name[b]} wants it"))
        if g[1, k] * w[0, k] > 0.02: contrib.append((g[1, k] * w[0, k], f"{name[b]} gives [{(names_ or {}).get(str(k), f'ch{k}')}] -> {name[a]} wants it"))
    for j in range(KS):
        v = s[0, j] * s[1, j]
        if v > 0.02: contrib.append((v, f"both are [{(names_ or {}).get('s' + str(j), f'style{j}')}]"))
    contrib.sort(reverse=True)
    total = sum(c for c, _ in contrib)
    return total, contrib[:top]

if __name__ == "__main__" and sys.argv[1:2] == ["final"]:
    tr = rows_for(list(R.YEARS), None)
    net = train(tr, epochs=8)
    torch.save(net.state_dict(), NO / "pairnet.pt"); pd.to_pickle((EMU, ESD), NO / "pairnet_norm.pkl")
    ref = [o for o in tg.index[tg[TAGCOLS[0]].notna()] if o in CI]          # played + post-rotation cards
    (NO / "channel_report.txt").write_text(channel_report(net, ref))
    print("channel report written:", NO / "channel_report.txt")

# ====================================================================================== PROJECT onto post-rotation Standard
if __name__ == "__main__" and sys.argv[1:2] == ["project"]:
    import json
    net = PairNet(XC.shape[1], len(EXPL)); net.load_state_dict(torch.load(NO / "pairnet.pt")); net.eval()
    NAMES = json.loads((NO / "channel_names.json").read_text()) if (NO / "channel_names.json").exists() else {}
    cname = lambda k: NAMES.get(str(k), f"channel {k}"); sname = lambda j: NAMES.get(f"s{j}", f"style {j}")
    name = M.cards.name
    single = pd.read_csv(OUT / "post_rotation_scores.csv"); single["oid"] = single.name.map(M.feat.reset_index().drop_duplicates("name").set_index("name").oracle_id)
    single = single.drop_duplicates("oid").set_index("oid")
    pool = [o for o in pd.read_csv(PROC / "future_pool.csv").oracle_id if o in CI and M.feat.type_land.get(o, 1) == 0 and o in single.index]
    tier = single.tier.astype(str).str[:2].str.strip(); GOOD = {"A", "B+", "B", "C"}
    G_, W_, S_ = card_acts(net, pool)
    cm = np.array([M.CM[M.idx[o]] for o in pool])
    RAW = G_ @ W_.T + W_ @ G_.T + S_ @ S_.T
    bad = M.POP[cm[:, None] | cm[None, :]] > 3
    # SPECIFIC synergy: how much better this pair is than each card's typical pairing (like lift) -> hubs don't dominate
    mu = np.where(bad, np.nan, RAW); np.fill_diagonal(mu, np.nan); mu = np.nanmean(mu, 1)
    sd = np.where(bad, np.nan, RAW); np.fill_diagonal(sd, np.nan); sd = np.nanstd(sd, 1)
    sd = np.maximum(sd, np.nanmedian(sd))      # floor: low-variance cards must not get inflated "specific" scores
    SYN = (RAW - mu[:, None]) / sd[:, None] / 2 + (RAW - mu[None, :]) / sd[None, :] / 2
    SYN[bad] = -9; np.fill_diagonal(SYN, -9)
    # recent co-play
    dk = R.S.decks[(R.S.decks.format == "standard") & (R.S.decks.date >= "2025-01-01")]; NREC = dk.deck.nunique(); DS = dk.groupby("oracle_id").deck.agg(set)
    def together(a, b):
        A_, B_ = DS.get(a, set()), DS.get(b, set()); return len(A_ & B_)
    # explanations use ABOVE-AVERAGE activity only: a channel is a reason when both cards are unusually strong on it
    Gc, Wc, Sc = np.clip(G_ - G_.mean(0), 0, None), np.clip(W_ - W_.mean(0), 0, None), np.clip(S_ - S_.mean(0), 0, None)
    def why(i, j, top=3):
        c = []
        for k in range(K):
            if Gc[i, k] * Wc[j, k] > 0.02: c.append((Gc[i, k] * Wc[j, k], f"{name[pool[i]]} gives {cname(k)} -> {name[pool[j]]} wants it"))
            if Gc[j, k] * Wc[i, k] > 0.02: c.append((Gc[j, k] * Wc[i, k], f"{name[pool[j]]} gives {cname(k)} -> {name[pool[i]]} wants it"))
        for t in range(KS):
            if Sc[i, t] * Sc[j, t] > 0.02: c.append((Sc[i, t] * Sc[j, t], f"both are {sname(t)}"))
        ia_, ib_ = M.idx[pool[i]], M.idx[pool[j]]
        tr = (M.TRIBES[ia_] & M.SUBT[ib_]) | (M.TRIBES[ib_] & M.SUBT[ia_])
        if tr: c.append((9, f"tribal: {'/'.join(sorted(tr))}"))
        c.sort(reverse=True); return " | ".join(x for _, x in c[:top])
    iu = np.triu_indices(len(pool), 1); vals = SYN[iu]
    order = np.argsort(-vals)[:20000]
    rows = []
    for o in order:
        i, j = iu[0][o], iu[1][o]; a, b = pool[i], pool[j]
        rows.append(dict(card_a=name[a], card_b=name[b], specific_synergy=round(float(vals[o]), 2), raw_synergy=round(float(RAW[i, j]), 2), tier_a=tier.get(a), tier_b=tier.get(b),
                         set_a=single.pool_set.get(a), set_b=single.pool_set.get(b), decks_together_now=together(a, b), why=why(i, j)))
    TP = pd.DataFrame(rows); TP.to_csv(NO / "future_pairs_explained.csv", index=False)
    # per-card best partners (raw synergy)
    pr = []
    for i, o in enumerate(pool):
        for j in np.argsort(-SYN[i])[:10]:
            pr.append(dict(card=name[o], partner=name[pool[j]], specific_synergy=round(float(SYN[i, j]), 2), raw_synergy=round(float(RAW[i, j]), 2), partner_tier=tier.get(pool[j]),
                           decks_together_now=together(o, pool[j]), why=why(i, j, 2)))
    pd.DataFrame(pr).to_csv(NO / "future_card_partners.csv", index=False)
    # mechanic outlook: support of each channel in future Standard vs. historical Standard years
    def support(ids_):
        ids_ = [o for o in ids_ if o in CI]
        if len(ids_) < 50: return np.full(K, np.nan)
        g, w, _ = card_acts(net, ids_)
        return np.sort(g, 0)[-8:].mean(0) * np.sort(w, 0)[-8:].mean(0)
    rel = M.cards.first_release; st = M.cards.first_set_type
    hist = {}
    for Y in range(2017, 2026):
        legal = [o for o in M.feat.index if isinstance(rel.get(o), str) and st.get(o) in ("expansion", "core") and f"{Y - 2}-10-01" <= rel[o] < f"{Y}-07-01" and M.feat.type_land[o] == 0]
        hist[Y] = support(legal)
    H = np.nanmean(np.stack(list(hist.values())), 0); F = support(pool)
    playable = [i for i, o in enumerate(pool) if tier.get(o) in GOOD or single.current_std_play_pct.get(o, 0) >= 2]
    out = []
    for k in np.argsort(-F):
        gi = sorted(playable, key=lambda i: -G_[i, k])[:6]; wi = sorted(playable, key=lambda i: -W_[i, k])[:6]
        out.append(dict(channel=cname(k), future_support=round(float(F[k]), 3), historical_avg=round(float(H[k]), 3),
                        vs_history=f"{F[k] / max(H[k], 1e-6):.1f}x", top_enablers=", ".join(name[pool[i]] for i in gi), top_payoffs=", ".join(name[pool[i]] for i in wi)))
    OUTL = pd.DataFrame(out); OUTL.to_csv(NO / "mechanic_outlook.csv", index=False)
    # print summary
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 200)
    print("MECHANIC OUTLOOK (future Standard vs. average Standard 2017-2025):"); print(OUTL.to_string(index=False))
    gb = TP[TP.tier_a.isin(GOOD) & TP.tier_b.isin(GOOD)]
    print("\nTOP RAW-SYNERGY PAIRS (both tier C+):"); print(gb.head(30)[["card_a", "card_b", "specific_synergy", "decks_together_now", "why"]].to_string(index=False))
    print("\nTOP HIDDEN (both tier C+, never together in recent decks):"); print(gb[gb.decks_together_now == 0].head(25)[["card_a", "card_b", "specific_synergy", "why"]].to_string(index=False))
    # known-pair sanity + new cards
    nm2 = {v: k for k, v in name.items()}
    def explain_any(x, y):
        a, b = nm2.get(x), nm2.get(y)
        if a is None or b is None: return "n/a"
        g, w, s = card_acts(net, [a, b]); c = []
        g, w, s = np.clip(g - G_.mean(0), 0, None), np.clip(w - W_.mean(0), 0, None), np.clip(s - S_.mean(0), 0, None)
        for k in range(K):
            if g[0, k] * w[1, k] > .02: c.append((g[0, k] * w[1, k], f"{x} gives {cname(k)} -> {y} wants it"))
            if g[1, k] * w[0, k] > .02: c.append((g[1, k] * w[0, k], f"{y} gives {cname(k)} -> {x} wants it"))
        for t in range(KS):
            if s[0, t] * s[1, t] > .02: c.append((s[0, t] * s[1, t], f"both are {sname(t)}"))
        tr = (M.TRIBES[M.idx[a]] & M.SUBT[M.idx[b]]) | (M.TRIBES[M.idx[b]] & M.SUBT[M.idx[a]])
        if tr: c.append((9, f"tribal: {'/'.join(sorted(tr))}"))
        c.sort(reverse=True); tot = float((g[0] * w[1] + g[1] * w[0]).sum() + (s[0] * s[1]).sum())
        return " | ".join(z for _, z in c[:3]) or "(no learned mechanic - unique combo or generic)"
    print("\nSANITY - known synergies:")
    for x, y in [("Hardened Scales", "Walking Ballista"), ("Witch's Oven", "Cauldron Familiar"), ("Chief of the Edge", "Bloodsoaked Champion"),
                 ("Badgermole Cub", "Llanowar Elves"), ("Monastery Swiftspear", "Lightning Bolt"), ("Saheeli Rai", "Felidar Guardian"),
                 ("Lightning Bolt", "Counterspell")]:
        print(f"  {x} + {y}: {explain_any(x, y)}")
    print("\nNEW CARDS (Nauctis / Darkhold):")
    for i, o in enumerate(pool):
        if single.pool_set.get(o) not in ("NAU", "MDD"): continue
        gives = ", ".join(f"{cname(k)}" for k in np.argsort(-Gc[i])[:2] if Gc[i, k] > .05)
        wants = ", ".join(f"{cname(k)}" for k in np.argsort(-Wc[i])[:2] if Wc[i, k] > .05)
        best = "; ".join(name[pool[j]] for j in np.argsort(-SYN[i])[:5])
        print(f"  {name[o]} [{tier.get(o)}] gives: {gives or '-'} | wants: {wants or '-'} | best partners: {best}")
