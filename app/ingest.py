import os, json
import faiss, numpy as np
import pymupdf as fitz
from sentence_transformers import SentenceTransformer

PAPER_DIR = "data/papers"
INDEX_DIR = "index"
MODEL_NAME = "allenai/specter2_base"


def load_pdfs(folder, min_chars=50):
    pages = []
    for fn in sorted(os.listdir(folder)):
        if not fn.endswith(".pdf"):
            continue
        doc = fitz.open(os.path.join(folder, fn))
        for i, page in enumerate(doc):
            text = page.get_text("text").strip()
            if len(text) >= min_chars:
                pages.append({"source": fn, "page": i + 1, "text": text})
        doc.close()
    return pages


def chunk_text(text, size=512, overlap=64):
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):  # try to cut at a sentence/line end
            cut = max(text.rfind(". ", start, end), text.rfind("\n", start, end))
            if cut > start + size // 2:
                end = cut + 1
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def main():
    os.makedirs(INDEX_DIR, exist_ok=True)
    pages = load_pdfs(PAPER_DIR)
    print(f"{len(pages)} pages loaded")

    chunks = []
    for p in pages:
        for piece in chunk_text(p["text"]):
            chunks.append({"id": len(chunks), "text": piece,
                           "source": p["source"], "page": p["page"]})
    print(f"{len(chunks)} chunks created")

    with open(f"{INDEX_DIR}/chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f)

    print("Embedding (a few minutes on CPU)...")
    model = SentenceTransformer(MODEL_NAME, device="cpu")
    emb = model.encode([c["text"] for c in chunks], batch_size=32,
                       normalize_embeddings=True, show_progress_bar=True)
    emb = np.asarray(emb, dtype="float32")

    index = faiss.IndexFlatIP(emb.shape[1])  # inner product = cosine (vectors normalized)
    index.add(emb)
    faiss.write_index(index, f"{INDEX_DIR}/faiss.index")
    print(f"Saved FAISS index with {index.ntotal} vectors")


if __name__ == "__main__":
    main()