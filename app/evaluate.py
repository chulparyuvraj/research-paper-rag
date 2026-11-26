import json
from collections import defaultdict

from retrieval import Retriever

# (label, search mode, drop stopwords in BM25?)
CONFIGS = [
    ("bm25", "bm25", False),
    ("dense", "dense", False),
    ("hybrid", "hybrid", False),
    ("rerank", "rerank", False),
    ("rerank_union", "rerank_union", False),
    ("bm25+stop", "bm25", True),
    ("hybrid+stop", "hybrid", True),
    ("rerank_union+stop", "rerank_union", True),
]
GATE_CONFIGS = [
    ("rerank", "rerank", False),
    ("rerank_union", "rerank_union", False),
    ("rerank_union+stop", "rerank_union", True),
]
MIN_RELEVANCE = -6.0
TOP = 10

with open("data/eval_questions.json", encoding="utf-8") as f:
    data = json.load(f)

r = Retriever()


def first_rank(hits, expected):
    for i, h in enumerate(hits):
        if h["source"] in expected:
            return i + 1
    return None


def summarize(rks):
    n = len(rks)
    r1 = sum(1 for x in rks if x == 1) / n
    r5 = sum(1 for x in rks if x is not None and x <= 5) / n
    mrr = sum(1 / x for x in rks if x) / n
    return r1, r5, mrr


ranks = defaultdict(list)
per_q = defaultdict(dict)

for label, mode, stop in CONFIGS:
    r.use_stop = stop
    for item in data["questions"]:
        hits = r.search(item["question"], mode=mode, k=TOP)
        rk = first_rank(hits, item["expected"])
        per_q[item["id"]][label] = rk
        ranks[(label, "all")].append(rk)
        ranks[(label, item["type"])].append(rk)

print("\nRetrieval (a hit = a chunk from the expected paper)")
print(f"{'config':18} {'set':4} {'R@1':>6} {'R@5':>6} {'MRR@10':>8}")
results = {}
for label, _, _ in CONFIGS:
    for group in ("all", "K", "P"):
        r1, r5, mrr = summarize(ranks[(label, group)])
        results[f"{label}/{group}"] = {"R@1": r1, "R@5": r5, "MRR": mrr}
        print(f"{label:18} {group:4} {r1:6.3f} {r5:6.3f} {mrr:8.3f}")


def show_changes(a, b):
    print(f"\nRank changes {a} -> {b} (None = not found in top {TOP}):")
    changed = False
    for item in data["questions"]:
        ra, rb = per_q[item["id"]][a], per_q[item["id"]][b]
        if ra != rb:
            changed = True
            print(f"  #{item['id']} [{item['type']}] {ra} -> {rb}  {item['question'][:70]}")
    if not changed:
        print("  none")


show_changes("rerank", "rerank_union")
show_changes("rerank_union", "rerank_union+stop")
show_changes("bm25", "bm25+stop")

print("\nStill missing from the top 5 with rerank_union+stop:")
for item in data["questions"]:
    rk = per_q[item["id"]]["rerank_union+stop"]
    if rk is None or rk > 5:
        print(f"  #{item['id']} {item['question']} -> {rk}")


def gate_scores(mode, stop):
    r.use_stop = stop

    def best(q):
        return r.relevance(q, r.search(q, mode=mode, k=5))

    return ([best(it["question"]) for it in data["questions"]],
            [best(q) for q in data["off_topic"]])


print(f"\nRelevance gate (cutoff {MIN_RELEVANCE})")
results["gate"] = {}
for label, mode, stop in GATE_CONFIGS:
    ins, offs = gate_scores(mode, stop)
    refused = [it["id"] for it, s in zip(data["questions"], ins) if s < MIN_RELEVANCE]
    ok = sum(1 for s in offs if s < MIN_RELEVANCE)
    print(f"{label:18} in-scope wrongly refused: {len(refused)}/30 {refused}  "
          f"off-topic refused: {ok}/5  off-topic scores: {[round(s, 1) for s in offs]}")
    results["gate"][label] = {"false_refusals": len(refused), "correct_refusals": ok}

with open("data/eval_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
print("\nSaved data/eval_results.json")