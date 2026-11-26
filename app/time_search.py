import time
from retrieval import Retriever

r = Retriever()
qs = ["How does FedAvg combine client updates on the server?",
      "What does the proximal term in FedProx do?",
      "How is answer relevance computed in RAGAS?"]
r.search(qs[0], mode="rerank")  # warm-up
for mode in ["rerank", "rerank_union"]:
    t = time.time()
    for q in qs:
        r.search(q, mode=mode, k=5)
    print(mode, round((time.time() - t) / len(qs), 2), "s per query")