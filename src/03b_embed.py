"""Semantic embedding of each card (sentence-transformers, all-mpnet-base-v2): cards that DO similar things end
up close together even if worded differently. Used for (a) direct features and (b) 'historical analog' features."""
import pathlib, numpy as np, pandas as pd
from sentence_transformers import SentenceTransformer
ROOT = pathlib.Path(__file__).resolve().parents[1]; PROC = ROOT / "data/processed"
feat = pd.read_parquet(PROC / "features.parquet"); cards = pd.read_parquet(PROC / "cards.parquet").set_index("oracle_id")
def doc(r):
    c = cards.loc[r.oracle_id]; pt = f" {c.power}/{c.toughness}" if isinstance(c.power, str) else ""
    return f"{c.mana_cost} {r.type_line}{pt}. {r.clean_text}"
docs = [doc(r) for r in feat.itertuples()]
m = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
E = m.encode(docs, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
np.save(PROC / "embeddings.npy", E.astype(np.float32)); pd.Series(feat.oracle_id).to_csv(PROC / "embeddings_ids.csv", index=False)
print(E.shape)
