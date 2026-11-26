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

## Observations (informal, a handful of manual queries, not a benchmark)

- SPECTER2 is trained for paper-to-paper similarity and sometimes returns off-topic chunks for short questions.
- With 0.6/0.4 RRF weights, hybrid results mostly mirror dense results.
- BM25 without stopword removal can match generic words ("how", "combine").
- Name variants hurt retrieval: "FedAvg" vs "FederatedAveraging".
- A proper evaluation on a labelled question set is in progress.

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