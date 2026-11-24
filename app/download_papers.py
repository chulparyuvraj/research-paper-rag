import os, time, urllib.request
import fitz

# arXiv ID -> what we INTEND it to be (we verify below)
PAPERS = {
    "2512.20363": "Clust-PSI-PFL (your paper)",
    "1602.05629": "FedAvg",
    "1812.06127": "FedProx",
    "2003.00295": "FedOpt / Adaptive Federated Optimization",
    "1910.01991": "Clustered Federated Learning",
    "2005.11401": "RAG (Lewis et al.)",
    "2312.10997": "RAG Survey (Gao et al.)",
    "2403.10131": "RAFT",
    "2309.15217": "RAGAS",
    "2004.07180": "SPECTER",
}

os.makedirs("data/papers", exist_ok=True)

for aid, name in PAPERS.items():
    out = f"data/papers/{aid}.pdf"
    try:
        if not os.path.exists(out):
            req = urllib.request.Request(
                f"https://arxiv.org/pdf/{aid}",
                headers={"User-Agent": "Mozilla/5.0"},
            )
            with urllib.request.urlopen(req, timeout=60) as r, open(out, "wb") as f:
                f.write(r.read())
            time.sleep(3)  # be polite to arXiv
        doc = fitz.open(out)
        first_line = doc[0].get_text().strip().split("\n")[0][:70]
        print(f"{aid} | wanted: {name} | got: {first_line} | {len(doc)} pages")
        doc.close()
    except Exception as e:
        print(f"{aid} FAILED: {e}")