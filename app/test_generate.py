from retrieval import Retriever
from generate import answer

r = Retriever()

tests = [
    "What is the Population Stability Index used for in Clust-PSI-PFL?",
    "How does FedAvg combine client updates on the server?",
    "What is the capital of France?",  # NOT in the documents on purpose
]

for q in tests:
    hits = r.search(q, mode="rerank", k=5)
    print("\n" + "=" * 80)
    print("Q:", q)
    print("rerank scores:", [round(h["score"], 2) for h in hits])
    print("\nANSWER:", answer(q, hits))
    print("\nSOURCES:")
    for i, h in enumerate(hits):
        print(f"  [{i + 1}] {h['source']} p.{h['page']}")