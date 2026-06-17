# app.py
"""
HuggingFace Spaces sandbox.
Push this file + features.py + requirements.txt to your HF Space repo.

The app accepts a JSON array of candidate profiles and returns a ranked CSV.
Designed with a premium theme and Outfit typography.
"""

import csv
import io
import json

import gradio as gr

from features import is_honeypot, score_candidate, generate_reasoning


def rank_candidates(json_input: str) -> str:
    """
    Input:  JSON string — either a list or a single candidate dict.
    Output: CSV string with columns candidate_id, rank, score, reasoning.
    """
    try:
        data = json.loads(json_input)
    except json.JSONDecodeError as e:
        return f'JSON parse error: {e}'

    candidates = data if isinstance(data, list) else [data]

    scored = []
    for c in candidates:
        if not isinstance(c, dict):
            continue
        if is_honeypot(c):
            continue
        # No precomputed scores in sandbox — semantic features default to 0
        score, feats = score_candidate(c, sbert_score=0.0, bm25_score=0.0)
        if score > 0:
            scored.append((c, score, feats))

    # Sort: score descending, candidate_id ascending (tie-break)
    scored.sort(key=lambda x: (-x[1], x[0].get('candidate_id', '')))
    top = scored[:100]

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(['candidate_id', 'rank', 'score', 'reasoning'])
    for rank, (c, score, feats) in enumerate(top, 1):
        writer.writerow([
            c.get('candidate_id', f'UNKNOWN_{rank}'),
            rank,
            score,
            generate_reasoning(c, feats),
        ])
    return buf.getvalue()


# Premium soft modern theme using Outfit typography and Indigo/Slate palettes
premium_theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="slate",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Outfit"), "sans-serif"]
)

demo = gr.Interface(
    fn=rank_candidates,
    inputs=gr.Textbox(
        lines=20,
        placeholder='Paste a JSON array of candidate objects here (e.g., from sample_candidates.json)...',
        label='Candidate JSON input',
    ),
    outputs=gr.Textbox(
        lines=15,
        label='Ranked CSV output',
        show_copy_button=True
    ),
    title='Redrob Candidate Ranker — Hosted Sandbox',
    description=(
        '⚡ Paste a list of candidates in JSON format to rank them against the Senior AI Engineer role. '
        'Honeypots are filtered out automatically using logic validation rules. '
        'Offline semantic scores (SBERT, BM25) default to 0 in this lightweight demo.'
    ),
    theme=premium_theme,
    allow_flagging='never',
)

if __name__ == '__main__':
    demo.launch()
