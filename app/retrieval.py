import json
import re

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

INDEX_DIR = "index"
EMBED_MODEL = "allenai/specter2_base"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

STOPWORDS = set(
    "a an the of to in on for and or is are was were be been being by with as at "
    "from that this these those it its how what which who whom why when where "
    "does do did can could would should may might will shall used use using into "
    "than then so such not no also their there they them we our us you your i he "
    "she his her has have had".split()
)


def tokenize(text, drop_stop=False):
    toks = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in toks if t not in STOPWORDS] if drop_stop else toks


class Retriever:
    def __init__(self, load_reranker=True):
        with open(f"{INDEX_DIR}/chunks.json", encoding="utf-8") as f:
            self.chunks = json.load(f)
        self.index = faiss.read_index(f"{INDEX_DIR}/faiss.index")
        # BM25 is rebuilt at startup (about a second), so no file to save
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in self.chunks])
        self.bm25_stop = BM25Okapi([tokenize(c["text"], True) for c in self.chunks])
        self.use_stop = False
        self.embedder = SentenceTransformer(EMBED_MODEL, device="cpu")
        self.reranker = (
            CrossEncoder(RERANK_MODEL, device="cpu") if load_reranker else None
        )

    def _hit(self, i, score):
        return {**self.chunks[i], "score": float(score)}

    def bm25_search(self, query, k=5):
        bm25 = self.bm25_stop if self.use_stop else self.bm25
        scores = bm25.get_scores(tokenize(query, self.use_stop))
        top = np.argsort(scores)[::-1][:k]
        return [self._hit(i, scores[i]) for i in top]

    def dense_search(self, query, k=5):
        vec = self.embedder.encode([query], normalize_embeddings=True)
        scores, ids = self.index.search(vec.astype("float32"), k)
        return [self._hit(i, s) for i, s in zip(ids[0], scores[0]) if i != -1]

    def hybrid_search(self, query, k=5, fetch_k=20,
                      w_dense=0.6, w_sparse=0.4, rrf_k=60):
        dense = self.dense_search(query, fetch_k)
        sparse = self.bm25_search(query, fetch_k)
        fused = {}
        for rank, h in enumerate(dense):
            fused[h["id"]] = fused.get(h["id"], 0) + w_dense / (rrf_k + rank + 1)
        for rank, h in enumerate(sparse):
            fused[h["id"]] = fused.get(h["id"], 0) + w_sparse / (rrf_k + rank + 1)
        top = sorted(fused, key=fused.get, reverse=True)[:k]
        return [self._hit(i, fused[i]) for i in top]

    def rerank_search(self, query, k=5, fetch_k=20, pool="hybrid"):
        if pool == "union":
            seen, candidates = set(), []
            for h in self.dense_search(query, fetch_k) + self.bm25_search(query, fetch_k):
                if h["id"] not in seen:
                    seen.add(h["id"])
                    candidates.append(h)
        else:
            candidates = self.hybrid_search(query, k=fetch_k, fetch_k=fetch_k)
        pairs = [[query, c["text"]] for c in candidates]
        scores = self.reranker.predict(pairs, show_progress_bar=False)
        order = np.argsort(scores)[::-1][:k]
        return [{**candidates[i], "score": float(scores[i])} for i in order]


    def relevance(self, query, hits):
        """Best cross-encoder score among hits (higher = more relevant)."""
        if not hits or self.reranker is None:
            return None
        pairs = [[query, h["text"]] for h in hits]
        return float(max(self.reranker.predict(pairs, show_progress_bar=False)))

    def search(self, query, mode="rerank", k=5):
        if mode == "bm25":
            return self.bm25_search(query, k)
        if mode == "dense":
            return self.dense_search(query, k)
        if mode == "hybrid":
            return self.hybrid_search(query, k)
        if mode == "rerank":
            return self.rerank_search(query, k)
        if mode == "rerank_union":
            return self.rerank_search(query, k, pool="union")
        raise ValueError(f"Unknown mode: {mode}")