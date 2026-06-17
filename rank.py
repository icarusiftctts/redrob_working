# rank.py
"""
Main ranking pipeline. Produces submission.csv.

Usage:
    python rank.py
    python rank.py --candidates candidates.jsonl --out submission.csv
"""

import argparse
import csv
import gzip
import json
import pickle
import time
from pathlib import Path

import numpy as np

from features import (
    is_honeypot,
    score_candidate,
    generate_reasoning,
    f_skill_core_score,
    f_title_current,
    f_title_avg,
    f_consulting_penalty,
)


# ─── Loaders ──────────────────────────────────────────────────────────────────

def load_candidates(path: Path) -> list:
    path_str = str(path)
    candidates = []
    if path_str.endswith('.gz'):
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    candidates.append(json.loads(line))
    else:
        # Check if JSONL or regular JSON list
        try:
            with open(path, 'r', encoding='utf-8') as f:
                # Try JSON list first
                data = json.load(f)
                candidates = data if isinstance(data, list) else [data]
        except json.JSONDecodeError:
            # Fallback to JSONL
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        candidates.append(json.loads(line))
    return candidates


def load_precomputed(pre_dir: Path, n: int):
    """Load cached scores. Returns (id_to_idx, sbert_scores, bm25_scores)."""
    if not pre_dir.exists():
        print(f'  WARNING: {pre_dir}/ not found — semantic features = 0.')
        return {}, np.zeros(n, dtype=np.float32), np.zeros(n, dtype=np.float32)

    id_to_idx    = pickle.load(open(pre_dir / 'id_to_idx.pkl', 'rb'))
    sbert_scores = np.load(pre_dir / 'sbert_scores.npy')
    bm25_scores  = np.load(pre_dir / 'bm25_scores.npy')
    print(f'  Precomputed scores loaded from {pre_dir}/')
    return id_to_idx, sbert_scores, bm25_scores


# ─── Submission writer ────────────────────────────────────────────────────────

def write_submission(top100: list, out_path: str) -> None:
    """
    Write submission CSV.
    Rows sorted by score descending; tie-break by candidate_id ascending
    (required by validate_submission.py).
    """
    # Re-sort to guarantee tie-break rule
    top100_sorted = sorted(top100, key=lambda x: (-x[1], x[0]['candidate_id']))

    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['candidate_id', 'rank', 'score', 'reasoning'])
        for rank, (c, score, feats) in enumerate(top100_sorted, 1):
            writer.writerow([
                c['candidate_id'],
                rank,
                score,
                generate_reasoning(c, feats),
            ])
    print(f'  Wrote {out_path}')


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidates',  default='candidates.jsonl')
    parser.add_argument('--out',         default='submission.csv')
    parser.add_argument('--precomputed', default='precomputed')
    parser.add_argument('--coarse-k',   type=int, default=5000,
                        help='Candidates kept after coarse filter')
    args = parser.parse_args()

    t0 = time.time()

    # Fallback checking for candidates file path (gzip vs raw)
    candidates_path = Path(args.candidates)
    if not candidates_path.exists():
        if args.candidates.endswith('.gz') and Path(args.candidates[:-3]).exists():
            candidates_path = Path(args.candidates[:-3])
        elif not args.candidates.endswith('.gz') and Path(args.candidates + '.gz').exists():
            candidates_path = Path(args.candidates + '.gz')
        else:
            raise FileNotFoundError(f"Could not find candidates file: {args.candidates}")

    # ── Step 1: Load ──────────────────────────────────────────────────────────
    print(f'Loading candidates from {candidates_path}...')
    candidates = load_candidates(candidates_path)
    print(f'  {len(candidates)} candidates loaded in {time.time()-t0:.1f}s')

    id_to_idx, sbert_scores, bm25_scores = load_precomputed(
        Path(args.precomputed), len(candidates)
    )

    # ── Step 2: Hard filter + coarse score → top K ────────────────────────────
    print(f'Stage 1: coarse filter (target top {args.coarse_k})...')
    coarse = []
    honeypot_count = 0

    for i, c in enumerate(candidates):
        if is_honeypot(c):
            honeypot_count += 1
            continue

        idx = id_to_idx.get(c['candidate_id'], i)
        # Handle cases where indexing could be mismatched or candidate pool size changes
        sb  = float(sbert_scores[min(idx, len(sbert_scores) - 1)]) if len(sbert_scores) > 0 else 0.0
        bm  = float(bm25_scores[min(idx, len(bm25_scores) - 1)]) if len(bm25_scores) > 0 else 0.0

        sk  = c.get('skills', [])
        ca  = c.get('career_history', [])
        p   = c['profile']

        # 6-feature coarse score: fast, no function-call overhead
        coarse_score = (
            0.30 * f_skill_core_score(sk)
          + 0.20 * f_title_current(p)
          + 0.15 * f_title_avg(ca)
          + 0.15 * sb
          + 0.10 * bm
          + 0.10 * f_consulting_penalty(ca)
        )
        coarse.append((c, coarse_score, sb, bm))

    coarse.sort(key=lambda x: -x[1])
    top_k = coarse[:args.coarse_k]
    print(f'  Honeypots filtered: {honeypot_count}')
    print(f'  Remaining: {len(coarse)} -> kept top {len(top_k)}')

    # ── Step 3: Full 19-feature scoring on top K ──────────────────────────────
    print('Stage 2: full feature scoring...')
    full = []
    for (c, _, sb, bm) in top_k:
        score, feats = score_candidate(c, sb, bm)
        if score > 0:   # skip any honeypots caught by full scorer
            full.append((c, score, feats))

    # Sort: score descending, candidate_id ascending for ties
    full.sort(key=lambda x: (-x[1], x[0]['candidate_id']))
    top100 = full[:100]
    print(f'  Full scoring done.')
    print(f'  Top score: {top100[0][1]:.4f}   Rank-100 score: {top100[-1][1]:.4f}')

    # ── Step 4: Write CSV ─────────────────────────────────────────────────────
    write_submission(top100, args.out)

    elapsed = time.time() - t0
    print(f'\nDone in {elapsed:.1f}s -> {args.out}')


if __name__ == '__main__':
    main()
