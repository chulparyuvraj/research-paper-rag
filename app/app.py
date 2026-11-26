try:
    import spaces
    gpu = spaces.GPU(duration=30)
except ImportError:  # running locally without the package
    def gpu(fn):
        return fn

import gradio as gr
import json
import os

from generate import answer
from retrieval import Retriever

MIN_RELEVANCE = -6.0  # below this, we say "not found" without calling the LLM
MODES = ["rerank_union", "rerank", "hybrid", "bm25", "dense"]

print("Loading models (about 30 seconds)...")
retriever = Retriever()


def format_hits(hits, max_chars=400):
    parts = []
    for i, h in enumerate(hits):
        snippet = h["text"][:max_chars].replace("\n", " ")
        parts.append(
            f"**[{i + 1}] {h['source']} p.{h['page']}** "
            f"(score {h['score']:.2f})\n\n> {snippet}...\n"
        )
    return "\n".join(parts)


@gpu
def retrieve(question, mode, k):
    hits = retriever.search(question, mode=mode, k=k)
    rel = retriever.relevance(question, hits)
    return hits, rel


@gpu
def compare_search(question):
    return [retriever.search(question, mode=m, k=3)
            for m in ["bm25", "dense", "hybrid", "rerank_union"]]


def ask(question, mode, k):
    question = (question or "").strip()
    if not question:
        return "Please type a question.", ""
    hits, rel = retrieve(question, mode, int(k))
    if rel is not None and rel < MIN_RELEVANCE:
        msg = (
            "I could not find this in the provided documents.\n\n"
            f"*(Relevance check: best score {rel:.1f} is below {MIN_RELEVANCE})*"
        )
        return msg, format_hits(hits)
    return answer(question, hits), format_hits(hits)


def compare(question):
    question = (question or "").strip()
    if not question:
        return ["Please type a question."] * 4
    return [format_hits(h, max_chars=250) for h in compare_search(question)]

def eval_markdown():
    path = "data/eval_results.json"
    if not os.path.exists(path):
        return "Evaluation results not found."
    with open(path, encoding="utf-8") as f:
        res = json.load(f)
    rows = ["| Pipeline | R@1 | R@5 | MRR@10 | R@1 keyword (K) | R@1 paraphrased (P) |",
            "|---|---|---|---|---|---|"]
    names = [("bm25", "BM25"), ("dense", "Dense (SPECTER2)"),
             ("hybrid", "Hybrid (RRF)"), ("rerank", "Hybrid + rerank"),
             ("rerank_union", "BM25 ∪ dense + rerank (default)")]
    for key, label in names:
        a, k, p = res[f"{key}/all"], res[f"{key}/K"], res[f"{key}/P"]
        rows.append(f"| {label} | {a['R@1']:.2f} | {a['R@5']:.2f} | {a['MRR']:.2f} "
                    f"| {k['R@1']:.2f} | {p['R@1']:.2f} |")
    notes = (
        "\n\n30 labelled questions (17 keyword-style, 13 paraphrased). A hit means a "
        "retrieved chunk from the expected paper, so this is paper-level and lenient. "
        "One question is 3.3 points, so gaps under about 10 points are noise. "
        "The candidate-pool fix was developed on this same set, so the numbers are "
        "somewhat optimistic."
    )
    return "\n".join(rows) + notes

with gr.Blocks(title="Research Paper RAG") as demo:
    
    gr.Markdown("Same question, four retrieval methods. No LLM involved. "
                "Note: scores are NOT comparable across methods "
                "(BM25, cosine, rank-fusion and reranker use different scales).")

    with gr.Tab("Ask"):
        q = gr.Textbox(label="Your question",
                       placeholder="e.g. What is the Population Stability Index used for?")
        with gr.Row():
            mode = gr.Dropdown(MODES, value="rerank_union", label="Retrieval mode")
            k = gr.Slider(1, 8, value=5, step=1, label="Passages to retrieve")
        btn = gr.Button("Ask", variant="primary")
        out = gr.Markdown()
        with gr.Accordion("Retrieved passages", open=True):
            src = gr.Markdown()
        gr.Examples(
            [["What is the Population Stability Index used for in Clust-PSI-PFL?"],
             ["How does Retrieval-Augmented Generation combine retrieval and generation?"],
             ["What does RAFT do to train models for domain-specific RAG?"]],
            inputs=q,
        )
        btn.click(ask, [q, mode, k], [out, src])
        q.submit(ask, [q, mode, k], [out, src])

    with gr.Tab("Compare retrieval"):
        gr.Markdown("Same question, four retrieval methods. No LLM involved. "
                    "Scores are NOT comparable across methods "
                    "(BM25, cosine, rank-fusion and reranker use different scales).")
        cq = gr.Textbox(label="Your question")
        cbtn = gr.Button("Compare", variant="primary")
        titles = ["BM25 (keywords)", "Dense (SPECTER2)", "Hybrid (RRF)",
                  "Rerank (BM25 ∪ dense)"]
        with gr.Row():
            cols = []
            for t in titles:
                with gr.Column():
                    gr.Markdown(f"**{t}**")
                    cols.append(gr.Markdown())
        cbtn.click(compare, [cq], cols)
        cq.submit(compare, [cq], cols)

    with gr.Tab("Evaluation"):
        gr.Markdown(eval_markdown())

demo.launch()