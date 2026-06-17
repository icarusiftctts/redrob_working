# precompute.py
"""
Run once: encodes all 100K candidates with SBERT and BM25.
Saves precomputed/ folder consumed by rank.py.

Usage:
    python precompute.py
    python precompute.py --candidates candidates.jsonl --out precomputed
"""

import argparse
import gzip
import json
import pickle
import re
import time
from pathlib import Path

import numpy as np

# ─── JD text for semantic matching ───────────────────────────────────────────
JD_TEXT = """
senior ai engineer production embeddings retrieval ranking nlp vector search
elasticsearch faiss pinecone qdrant weaviate python machine learning llm
evaluation framework ndcg mrr recommendation systems startup founding team
semantic search sentence transformers fine-tuning information retrieval
learning to rank product company deployed at scale
"""


def load_candidates(path: str) -> list:
    path = str(path)
    if path.endswith('.gz'):
        opener = lambda: gzip.open(path, 'rt', encoding='utf-8')
    else:
        opener = lambda: open(path, 'r', encoding='utf-8')
    candidates = []
    with opener() as f:
        for line in f:
            if line.strip():
                candidates.append(json.loads(line))
    return candidates


def build_career_text(c: dict) -> str:
    """Concatenate candidate profile text for semantic matching."""
    p = c['profile']
    parts = [
        p.get('headline', ''),
        p.get('current_title', ''),
        p.get('summary', '')[:200],
    ]
    for role in c.get('career_history', [])[:5]:
        parts.append(role.get('title', ''))
        parts.append(role.get('description', '')[:250])
    skills = [s['name'] for s in c.get('skills', [])[:20]]
    parts.append(' '.join(skills))
    return ' '.join(filter(None, parts))


def tokenize(text: str) -> list:
    return re.findall(r'\b[a-z]+\b', text.lower())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidates', default='candidates.jsonl')
    parser.add_argument('--out', default='precomputed')
    args = parser.parse_args()

    # Fallback checking for candidates file path (gzip vs raw)
    candidates_path = Path(args.candidates)
    if not candidates_path.exists():
        if args.candidates.endswith('.gz') and Path(args.candidates[:-3]).exists():
            candidates_path = Path(args.candidates[:-3])
        elif not args.candidates.endswith('.gz') and Path(args.candidates + '.gz').exists():
            candidates_path = Path(args.candidates + '.gz')
        else:
            raise FileNotFoundError(f"Could not find candidates file: {args.candidates}")

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)
    t0 = time.time()

    # ── Load candidates ───────────────────────────────────────────────────────
    print(f'Loading candidates from {candidates_path}...')
    candidates = load_candidates(candidates_path)
    print(f'  {len(candidates)} candidates loaded in {time.time()-t0:.1f}s')

    # ── Save id → index mapping ───────────────────────────────────────────────
    id_to_idx = {c['candidate_id']: i for i, c in enumerate(candidates)}
    with open(out_dir / 'id_to_idx.pkl', 'wb') as f:
        pickle.dump(id_to_idx, f)
    print('  id_to_idx.pkl saved')

    # ── Extract career texts ──────────────────────────────────────────────────
    print('Extracting career texts...')
    texts = [build_career_text(c) for c in candidates]

    # ── SBERT embeddings ──────────────────────────────────────────────────────
    print('Computing SBERT embeddings (~20 min on CPU)...')
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer('all-MiniLM-L6-v2')   # 22 MB, downloads once
    embeddings = model.encode(
        texts,
        batch_size=512,
        show_progress_bar=True,
        normalize_embeddings=True,    # L2-normalize: cosine = dot product
    )
    jd_emb  = model.encode([JD_TEXT], normalize_embeddings=True)[0]
    raw     = embeddings @ jd_emb                      # shape (n,)
    sbert_sc = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)

    np.save(out_dir / 'embeddings.npy',    embeddings.astype(np.float32))
    np.save(out_dir / 'sbert_scores.npy',  sbert_sc.astype(np.float32))
    print(f'  SBERT done. Range: [{sbert_sc.min():.3f}, {sbert_sc.max():.3f}]')

    # Quick sanity check — print top-5 by SBERT score
    top5_idx = sbert_sc.argsort()[-5:][::-1]
    print('  Top 5 by SBERT:')
    for idx in top5_idx:
        c = candidates[idx]
        print(f'    {sbert_sc[idx]:.3f}  {c["profile"]["current_title"]}'
              f'  |  {c["profile"]["location"]}')

    # ── BM25 scores ───────────────────────────────────────────────────────────
    print('Computing BM25 scores...')
    from rank_bm25 import BM25Okapi

    corpus      = [tokenize(t) for t in texts]
    jd_tokens   = tokenize(JD_TEXT)
    bm25        = BM25Okapi(corpus)
    bm25_raw    = np.array(bm25.get_scores(jd_tokens), dtype=np.float32)
    bm25_sc     = (bm25_raw - bm25_raw.min()) / (bm25_raw.max() - bm25_raw.min() + 1e-9)

    np.save(out_dir / 'bm25_scores.npy', bm25_sc)
    print(f'  BM25 done.  Range: [{bm25_sc.min():.3f}, {bm25_sc.max():.3f}]')

    elapsed = time.time() - t0
    print(f'\nPrecompute complete in {elapsed/60:.1f} min.')
    print(f'Files saved to {out_dir}/')
    print(f'  embeddings.npy   {embeddings.nbytes/1e6:.0f} MB')
    print(f'  sbert_scores.npy  bm25_scores.npy  id_to_idx.pkl')


if __name__ == '__main__':
    main()
