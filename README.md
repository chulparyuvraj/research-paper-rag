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

Ask questions about 10 papers on federated learning and RAG. Answers cite the source file and page.

**Pipeline:** PDF chunking → BM25 + SPECTER2/FAISS → RRF fusion → cross-encoder rerank → LLM answer (Groq) with relevance gate.

Run locally: `pip install -r requirements.txt`, add `GROQ_API_KEY` to `.env`, then `python app/app.py`.