from retrieval import Retriever

r = Retriever()

questions = [
    "What is the Population Stability Index used for?",     # keyword-heavy
    "How do clients' model updates get combined on the server?",  # paraphrased
]

for q in questions:
    print("\n" + "=" * 80)
    print("Q:", q)
    for mode in ["bm25", "dense", "hybrid", "rerank"]:
        print(f"\n--- {mode} ---")
        for h in r.search(q, mode=mode, k=3):
            snippet = h["text"][:110].replace("\n", " ")
            print(f"{h['source']} p.{h['page']}  score={h['score']:.3f}  {snippet}")