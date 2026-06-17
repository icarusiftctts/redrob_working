# label.py
"""
Interactive labeling tool.
Labels: 3=definitely hire  2=probably hire  1=borderline  0=reject

Usage:
    python label.py                              # labels sample_candidates.json
    python label.py --extra candidates.jsonl     # also labels from full pool
"""

import argparse
import gzip
import json
from pathlib import Path


def load_sample(path='sample_candidates.json') -> list:
    with open(path) as f:
        return json.load(f)


def load_extra_from_jsonl(path: str, skip_ids: set, limit: int = 300) -> list:
    extra = []
    path_str = str(path)
    if path_str.endswith('.gz'):
        opener = lambda: gzip.open(path_str, 'rt', encoding='utf-8')
    else:
        opener = lambda: open(path_str, 'r', encoding='utf-8')
    with opener() as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            c = json.loads(line.strip())
            if c['candidate_id'] not in skip_ids:
                extra.append(c)
    return extra


def show_candidate(c: dict) -> None:
    p   = c['profile']
    sig = c['redrob_signals']
    ca  = c.get('career_history', [])
    sk  = c.get('skills', [])

    print(f"\n{'='*60}")
    print(f"ID: {c['candidate_id']}")
    print(f"Title:  {p['current_title']}  |  {p['years_of_experience']:.0f} yr  |  {p['location']}")
    print(f"OTW:    {sig['open_to_work_flag']}  |  "
          f"Notice: {sig['notice_period_days']}d  |  "
          f"Response: {sig['recruiter_response_rate']:.0%}  |  "
          f"Active: {sig['last_active_date']}")
    print(f"Skills: {[s['name'] for s in sk[:8]]}")
    print("Career:")
    for r in ca[:4]:
        print(f"  {r['title']:35s}  @  {r['company']:20s}  ({r['duration_months']}mo)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--extra', default=None,
                        help='Path to candidates.jsonl/candidates.jsonl.gz for extra candidates')
    parser.add_argument('--limit', type=int, default=300,
                        help='Max candidates to load from --extra')
    args = parser.parse_args()

    label_file = Path('labeled_candidates.json')
    labels: dict = {}
    if label_file.exists():
        labels = json.load(open(label_file))
        print(f'Resuming: {len(labels)} already labeled.')

    # Build candidate list
    candidates = load_sample()
    if args.extra:
        extra_path = Path(args.extra)
        if not extra_path.exists():
            if args.extra.endswith('.gz') and Path(args.extra[:-3]).exists():
                extra_path = Path(args.extra[:-3])
            elif not args.extra.endswith('.gz') and Path(args.extra + '.gz').exists():
                extra_path = Path(args.extra + '.gz')
            else:
                print(f"WARNING: Extra candidates file {args.extra} not found. Skipping.")
                extra_path = None
        
        if extra_path:
            extra = load_extra_from_jsonl(extra_path, set(labels), args.limit)
            candidates = candidates + extra
            print(f'Total candidates to label: {len(candidates)}')

    labeled_this_session = 0
    for c in candidates:
        cid = c['candidate_id']
        if cid in labels:
            continue

        show_candidate(c)
        label = input('\n  Label  3=hire  2=maybe  1=weak  0=reject  s=skip  q=quit: ').strip()

        if label == 'q':
            break
        if label == 's':
            continue
        if label in ('0', '1', '2', '3'):
            labels[cid] = int(label)
            json.dump(labels, open(label_file, 'w'))
            labeled_this_session += 1

    print(f'\nSession: labeled {labeled_this_session} new candidates.')
    print(f'Total labeled: {len(labels)}  →  labeled_candidates.json')

    # Show distribution
    from collections import Counter
    dist = Counter(labels.values())
    for lbl in sorted(dist):
        names = {3: 'hire', 2: 'maybe', 1: 'weak', 0: 'reject'}
        print(f"  Label {lbl} ({names[lbl]:6s}): {dist[lbl]}")


if __name__ == '__main__':
    main()
