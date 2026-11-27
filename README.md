---
title: Research Paper RAG
emoji: 📚
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.32.0
app_file: app/app.py
pinned: false
---

# Research Paper RAG

Ask questions about 10 papers on federated learning and RAG; every answer cites its source file and page.

**Live demo:** https://huggingface.co/spaces/YuvrajChulpar/research-paper-rag

## Pipeline

```
PDFs → page text → 512-char chunks (64 overlap)
                      ├─ BM25 (keywords)
                      └─ SPECTER2 embeddings → FAISS (meaning)
 question → BM25 + dense → RRF fusion → cross-encoder rerank (top 5)
          → relevance gate (skip LLM if best rerank score < -6)
          → LLM answer with [n] citations (Groq, with model fallback)
```

## Design decisions

- **Four retrieval modes** (BM25, dense, hybrid, rerank) are exposed in the UI, and a Compare tab runs all four on one question so differences are visible.
- **Relevance gate:** in manual tests, in-scope questions scored about +3 to +7 on the reranker and an off-topic one scored about -11, so low-scoring queries return "not found" without calling the LLM.
- **Grounded prompt:** the LLM may use only the numbered passages and must cite them or refuse.
- **Model fallback:** if one LLM is unavailable or rate-limited, the next is tried.
- **Deployment:** free Hugging Face Gradio Spaces run on ZeroGPU, so the search step is wrapped in `@spaces.GPU`. The models themselves run on CPU.

## Evaluation

30 labelled questions (17 keyword-style, 13 paraphrased) over the 10 papers. A hit means a retrieved chunk from the expected paper (paper-level, so lenient). One question is 3.3 points, so gaps under about 10 points are within noise.

| Pipeline | R@1 | R@5 | MRR@10 |
|---|---|---|---|
| BM25 | 0.83 | 0.93 | 0.87 |
| Dense (SPECTER2) | 0.60 | 0.87 | 0.69 |
| Hybrid (RRF) | 0.67 | 0.93 | 0.77 |
| Hybrid + rerank | 0.83 | 0.97 | 0.88 |
| BM25 ∪ dense + rerank (default) | 0.90 | 1.00 | 0.93 |

- SPECTER2 is clearly the weakest retriever for question-to-chunk search.
- With 0.6/0.4 RRF weights, BM25's best chunk was sometimes cut before the reranker saw it. Giving the reranker the union of both candidate lists fixed that failure, and R@5 reached 1.0. The R@1 gain (+2 questions) is within noise.
- BM25 stopword removal was tested and gave no net gain, so it wasn't adopted.
- - Relevance gate (cutoff -6): 1 of 30 in-scope questions wrongly refused, 4 of 5 off-topic questions refused. The one near-topic question that passed the gate (Llama 3 accuracy on MMLU) was refused by the LLM's grounded prompt ("I could not find this in the provided documents"). That is a single check, not a measured rate.

Caveats: labels were drafted with an AI assistant and reviewed only on failures, and the pool fix was developed on this same question set, so numbers are somewhat optimistic. A held-out set is the next step.

## Limitations

Text-only PDFs (no OCR, tables or figures), a small 10-paper corpus, and answers can still be wrong, so check the cited passage.

## Run locally

```
pip install -r requirements.txt
python app/download_papers.py
python app/ingest.py
# add GROQ_API_KEY to .env
python app/app.py
```

## Repo layout

`app/` code · `index/` prebuilt indexes · `experiments/` earlier notebooks