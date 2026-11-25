"""Quick comparison of model settings on the temporal holdout."""
import sys, importlib.util, pathlib, pandas as pd
ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("train", ROOT / "src/05_train.py"); T = importlib.util.module_from_spec(spec); spec.loader.exec_module(T)
cut = sys.argv[1]
train = T.df[T.df.window_start < cut].reset_index(drop=True); test = T.df[T.df.window_start >= cut].reset_index(drop=True)
base = dict(T.PARAMS); rows = []
for name, upd, rounds in [("default", {}, 1500), ("huber", {"objective": "huber", "alpha": 1.0}, 1500),
                          ("tweedie", {"objective": "tweedie", "tweedie_variance_power": 1.3}, 1500),
                          ("bigger trees", {"num_leaves": 63, "min_data_in_leaf": 20}, 1000),
                          ("slower", {"learning_rate": 0.015}, 3000)]:
    T.PARAMS = {**base, **upd}
    pred, *_ = T.fit_predict(train, test, rounds)
    r = T.report(name, test, pred); rows.append(r); print(name, {k: v for k, v in r.items() if "standard" in k}, flush=True)
pd.DataFrame(rows).set_index("model").T.to_csv(ROOT / "output/tuning.csv")
