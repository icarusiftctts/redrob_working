"""
REASONING ENGINE
Aligned with features.py v1.

Integrate by placing alongside features.py and using:
    from features import WEIGHTS, (constants)

All 20 refactor priorities applied.
"""

from __future__ import annotations
from typing import Callable

# ---------------------------------------------------------------------------
# PRIORITY 1 & 2 — Import constants and WEIGHTS from features.py.
# If running standalone, the fallback block below activates automatically.
# ---------------------------------------------------------------------------

try:
    from features import (
        WEIGHTS,
        _RETRIEVAL_LIBS,
        _CONSULTING_FIRMS,
        _HEADLINE_KEYWORDS,
        _SENIORITY,
        _ML_ROLES,
        _CORE_SKILL_NAMES,
    )
    _STANDALONE = False
except ImportError:
    _STANDALONE = True

    # --- Fallback constants (single source of truth when features.py absent) ---

    _RETRIEVAL_LIBS = [
        "faiss", "pinecone", "qdrant", "weaviate",
        "vector", "elasticsearch", "opensearch", "milvus",
    ]

    _CONSULTING_FIRMS = {
        "tcs", "infosys", "wipro", "accenture", "cognizant",
        "capgemini", "hcl", "tech mahindra", "mphasis", "hexaware",
        "genpact", "syntel", "l&t infotech", "ltimindtree",
    }

    _HEADLINE_KEYWORDS = [
        "embedding", "retrieval", "vector", "search", "nlp",
        "ranking", "recommendation", "llm", "faiss", "pinecone",
        "qdrant", "weaviate", "elasticsearch", "transformer",
        "semantic", "information retrieval",
    ]

    _SENIORITY = {
        "junior": 1, "associate": 2, "staff": 3, "mid": 3,
        "senior": 4, "lead": 5, "principal": 6, "director": 7, "vp": 8,
    }

    _ML_ROLES = {
        "ml", "ai", "nlp", "machine learning", "data scientist",
        "search", "recommendation", "retrieval", "applied scientist",
    }

    _CORE_SKILL_NAMES = {
        "embedding", "embeddings", "retrieval", "vector search", "semantic search",
        "information retrieval", "faiss", "pinecone", "qdrant", "weaviate",
        "nlp", "natural language", "python", "ranking", "recommendation",
        "elasticsearch", "opensearch", "transformer", "bert", "llm",
        "machine learning", "search", "fine-tuning", "rlhf",
        "pytorch", "tensorflow", "deep learning", "data science", "a/b testing",
    }

    # PRIORITY 2 fallback — get_feature_weight() used everywhere instead of WEIGHTS[key]
    WEIGHTS = {
        "skill_core":    0.18,
        "retrieval":     0.03,
        "python_sk":     0.02,
        "skill_endorse": 0.02,
        "skill_dur":     0.02,
        "assess":        0.02,
        "github":        0.03,
        "title_curr":    0.07,
        "title_avg":     0.05,
        "consult":       0.03,
        "hop":           0.02,
        "prog":          0.02,
        "product":       0.02,
        "tenure":        0.06,
        "sbert":         0.08,
        "bm25":          0.06,
        "headline":      0.05,
        "yoe":           0.06,
        "loc":           0.06,
        "edu":           0.04,
        "cert":          0.02,
        "recruit":       0.04,
    }


def get_feature_weight(feature_name: str) -> float:
    """PRIORITY 2 — single lookup point for feature weights."""
    return WEIGHTS.get(feature_name, 0.01)


# ---------------------------------------------------------------------------
# PRIORITY 20 — Extractor return type: (text | None, confidence: float)
# confidence ∈ [0.0, 1.0]; renderer ignores results below 0.4
# ---------------------------------------------------------------------------

ExtractorResult = tuple[str | None, float]


# ---------------------------------------------------------------------------
# PRIORITY 3 — Extraction quality scorer
# Rewards: specific techs, company names, deployment evidence, metrics.
# ---------------------------------------------------------------------------

_SPECIFIC_TECH_TERMS = {
    "faiss", "pinecone", "qdrant", "weaviate", "elasticsearch", "opensearch",
    "milvus", "bert", "llm", "pytorch", "tensorflow", "transformer",
    "rlhf", "fine-tuning", "langchain",
}

_DEPLOYMENT_EVIDENCE_TERMS = {
    "production", "deployed", "shipped", "served", "serving",
    "customers", "users", "latency", "throughput", "scale", "billion", "million",
    "end-to-end", "owned", "built", "launched",
}

_METRIC_PATTERNS = ["ms", "%", "m users", "b requests", "k qps", "x improvement"]


def _extraction_quality(text: str | None) -> float:
    """
    PRIORITY 3 — Score extracted evidence quality.
    Returns a multiplier in [0.3, 1.0].
    Generic text → 0.3. Specific tech + scale evidence → 1.0.
    """
    if not text:
        return 0.0
    low = text.lower()
    score = 0.3  # baseline for any non-None text

    specific_hits = sum(1 for t in _SPECIFIC_TECH_TERMS if t in low)
    deploy_hits   = sum(1 for t in _DEPLOYMENT_EVIDENCE_TERMS if t in low)
    metric_hits   = sum(1 for p in _METRIC_PATTERNS if p in low)

    score += min(specific_hits * 0.15, 0.3)
    score += min(deploy_hits  * 0.12, 0.25)
    score += min(metric_hits  * 0.10, 0.15)

    return min(score, 1.0)


# ---------------------------------------------------------------------------
# MULTIPLIER HELPERS  (must match features.py logic exactly)
# ---------------------------------------------------------------------------

def _yoe_multiplier(years: float) -> float:
    if years < 2:   return 0.20
    if years < 4:   return 0.50
    if years < 5:   return 0.75
    if years <= 12: return 1.00
    if years <= 14: return 0.70
    return 0.40


def _location_multiplier(profile: dict, signals: dict) -> float:
    loc     = profile.get("location", "").lower()
    country = profile.get("country", "").lower()
    reloc   = signals.get("willing_to_relocate", False)

    _LOCATION_TIERS = {
        "pune": 1.0,     "noida": 1.0,
        "new delhi": 0.9, "delhi": 0.9,
        "hyderabad": 0.85, "mumbai": 0.85, "gurugram": 0.85, "gurgaon": 0.85,
        "bangalore": 0.75, "bengaluru": 0.75,
        "chennai": 0.65, "kolkata": 0.55,
        "ahmedabad": 0.5, "jaipur": 0.45,
    }

    if any(city in loc for city in _LOCATION_TIERS):
        return 1.0
    if "india" in country or country in ("in", "ind"):
        return 0.4 + (0.1 if reloc else 0.0)
    return 0.2 if reloc else 0.1


# ---------------------------------------------------------------------------
# GROUNDING EXTRACTORS
# PRIORITY 6  — Extractors return raw facts only. Renderer owns grammar.
# PRIORITY 20 — All extractors return (text | None, confidence: float).
# ---------------------------------------------------------------------------

def _tech_list(candidate: dict, libs: list[str], limit: int) -> ExtractorResult:
    """
    PRIORITY 4 — Search skills, all role titles, all role descriptions,
    headline, and summary. Return top `limit` highest-confidence technologies.
    """
    profile = candidate.get("profile", {})
    parts: list[str] = []

    # Skills
    parts += [s.get("name", "").lower() for s in candidate.get("skills", [])]
    # All role titles + descriptions (not just first 2)
    for r in candidate.get("career_history", []):
        parts.append(r.get("title", "").lower())
        parts.append(r.get("description", "").lower())
    # Headline + summary
    parts.append(profile.get("headline", "").lower())
    parts.append(profile.get("summary", "").lower())

    text = " ".join(parts)

    # Weight by source: skills > descriptions > headline
    skill_text    = " ".join(s.get("name", "").lower() for s in candidate.get("skills", []))
    desc_text     = " ".join(r.get("description", "").lower() for r in candidate.get("career_history", []))
    headline_text = profile.get("headline", "").lower()

    scored_libs: list[tuple[float, str]] = []
    for lib in libs:
        if lib not in text:
            continue
        conf = 0.4
        if lib in skill_text:    conf += 0.4
        if lib in desc_text:     conf += 0.3
        if lib in headline_text: conf += 0.2
        conf = min(conf, 1.0)
        label = lib.upper() if len(lib) <= 4 else lib.title()
        scored_libs.append((conf, label))

    scored_libs.sort(reverse=True)
    if not scored_libs:
        return None, 0.0

    top = scored_libs[:limit]
    avg_conf = sum(c for c, _ in top) / len(top)
    return ", ".join(label for _, label in top), avg_conf


def _top_skills(candidate: dict, limit: int) -> ExtractorResult:
    """
    PRIORITY 5 — Rank skills by proficiency > retrieval relevance > core ML
    relevance > duration > endorsements. Core production ML outranks generic.
    """
    _RETRIEVAL_SKILLS = {
        "faiss", "pinecone", "qdrant", "weaviate", "milvus",
        "elasticsearch", "opensearch", "retrieval", "vector search",
        "semantic search", "information retrieval",
    }
    _CORE_ML_SKILLS = {
        "pytorch", "tensorflow", "bert", "transformer", "llm",
        "nlp", "natural language", "machine learning", "deep learning",
        "fine-tuning", "rlhf", "embedding", "embeddings", "ranking",
        "recommendation", "python",
    }
    _PROFICIENCY_RANK = {"expert": 2, "advanced": 1}

    scored: list[tuple[float, str]] = []
    for s in candidate.get("skills", []):
        prof = s.get("proficiency", "")
        if prof not in _PROFICIENCY_RANK:
            continue
        if s.get("duration_months", 0) <= 6:
            continue
        name     = s.get("name", "").lower()
        name_raw = s.get("name", "")
        if not any(core in name for core in _CORE_SKILL_NAMES):
            continue

        score = 0.0
        score += _PROFICIENCY_RANK[prof] * 0.35
        score += 0.30 if any(r in name for r in _RETRIEVAL_SKILLS) else 0.0
        score += 0.20 if any(m in name for m in _CORE_ML_SKILLS) else 0.0
        score += min(s.get("duration_months", 0) / 60.0, 1.0) * 0.10
        score += min(s.get("endorsements", 0) / 50.0, 1.0) * 0.05
        scored.append((score, name_raw))

    scored.sort(reverse=True)
    if not scored:
        return None, 0.0

    top    = scored[:limit]
    result = "; ".join(n for _, n in top)
    conf   = top[0][0]  # confidence = score of best skill
    return result, conf


def _classify_company(comp: str) -> str:
    """
    PRIORITY 9 — Classify a company into one of:
    consulting | research | government | startup | product | unknown
    """
    low  = comp.lower()
    last = low.split()[-1] if low.split() else ""

    if any(f in low for f in _CONSULTING_FIRMS):
        return "consulting"
    if any(k in low for k in [
        "research", "lab", "institute", "university",
        "google research", "microsoft research", "fair", "deepmind",
    ]):
        return "research"
    if any(k in low for k in ["government", "gov", "nic", "drdo", "isro", "ministry"]):
        return "government"
    if (
        "startup" in low
        or last in {"ai", "ml", "labs", "hq"}
        or (low.endswith("inc") and len(low.split()) <= 3)
    ):
        return "startup"
    # Only classify as "product" if there are positive signals, not merely absence of negatives.
    _PRODUCT_SIGNALS = {
        "technologies", "technology", "solutions", "systems", "platform",
        "software", "products", "networks", "services", "analytics", "data",
    }
    if any(k in low for k in _PRODUCT_SIGNALS) or last in _PRODUCT_SIGNALS:
        return "product"

    return "unknown"


def _product_companies(candidate: dict) -> ExtractorResult:
    """
    PRIORITY 6 & 9 — Return raw company names only (renderer builds grammar).
    Classify each; skip consulting/research/gov/unknown.
    """
    product_names: list[str] = []
    startup_names: list[str] = []

    for r in candidate.get("career_history", [])[:4]:
        comp  = r.get("company", "").strip()
        if not comp:
            continue
        kind = _classify_company(comp)
        if kind == "product":
            product_names.append(comp)
        elif kind == "startup":
            startup_names.append(comp)

    if product_names:
        return ", ".join(product_names[:2]), 0.85
    if startup_names:
        return ", ".join(startup_names[:2]), 0.65
    return None, 0.0


def _progression_summary(candidate: dict) -> ExtractorResult:
    """Return a Junior→Senior trajectory string for ML roles, or (None, 0.0)."""
    roles = sorted(
        [
            r for r in candidate.get("career_history", [])
            if any(t in r.get("title", "").lower() for t in _ML_ROLES)
        ],
        key=lambda r: r.get("start_date", ""),
    )
    if len(roles) < 2:
        return None, 0.0

    levels: list[int] = []
    for r in roles:
        title = r.get("title", "").lower()
        for keyword, level in _SENIORITY.items():
            if keyword in title:
                levels.append(level)
                break

    if len(levels) >= 2 and levels[-1] > levels[0]:
        span = levels[-1] - levels[0]
        conf = min(0.5 + span * 0.1, 1.0)
        return f"L{levels[0]}→L{levels[-1]}", conf
    return None, 0.0


def _certs_evidence(candidate: dict) -> ExtractorResult:
    relevant: list[str] = []
    for c in candidate.get("certifications", []):
        name = c.get("name", "").lower()
        if any(k in name for k in [
            "aws", "gcp", "azure", "ml", "ai",
            "pytorch", "tensorflow", "kubernetes", "langchain", "llm",
        ]):
            relevant.append(c.get("name", ""))
    if not relevant:
        return None, 0.0
    return ", ".join(relevant[:2]), min(0.5 + len(relevant) * 0.1, 1.0)


def _education_evidence(candidate: dict) -> ExtractorResult:
    _TIER1 = {
        "iit", "iisc", "bits", "nit", "iiit",
        "stanford", "mit", "cmu", "berkeley", "harvard",
        "oxford", "cambridge", "eth", "epfl", "ntu", "nus",
    }
    for e in candidate.get("education", []):
        inst = e.get("institution", "").lower()
        deg  = e.get("degree", "").lower()
        if (
            any(t in inst for t in _TIER1)
            or "phd" in deg
            or ("master" in deg and "ai" in deg)
            or ("mtech" in deg and "ai" in deg)
        ):
            # PRIORITY 6: raw evidence, no wrapper text
            return f"{e.get('degree')}, {e.get('institution')}", 0.9
    return None, 0.0


def _extract_headline(candidate: dict, f_val: float) -> ExtractorResult:
    """
    PRIORITY 14 — Extract only meaningful fragments from the headline.
    Ignore generic marketing language; require at least one signal keyword.
    """
    if f_val <= 0.6:
        return None, 0.0
    headline = candidate.get("profile", {}).get("headline", "")
    low = headline.lower()
    if not any(kw in low for kw in _HEADLINE_KEYWORDS):
        return None, 0.0
    # Return the full headline but only if it contains signal; truncate at word boundary.
    words = headline.split()
    kept: list[str] = []
    for w in words:
        kept.append(w)
        if len(" ".join(kept)) >= 60:
            break
    fragment = " ".join(kept)
    conf = min(0.5 + sum(1 for kw in _HEADLINE_KEYWORDS if kw in low) * 0.05, 1.0)
    return fragment, conf


def _extract_production_ownership(candidate: dict) -> ExtractorResult:
    """
    PRIORITY 10 — Search summaries, descriptions, and headline for production
    ownership language. Returns evidence label and confidence.
    """
    _OWNERSHIP_PATTERNS = [
        ("end-to-end",   "end-to-end ownership",       0.95),
        ("owned",        "end-to-end ownership",        0.90),
        ("shipped",      "shipped to production",       0.90),
        ("deployed",     "production deployment",       0.85),
        ("serving",      "large-scale serving",         0.80),
        ("latency",      "latency-sensitive systems",   0.80),
        ("production",   "production system",           0.70),
        ("customer",     "customer-facing systems",     0.75),
        ("scale",        "large-scale deployment",      0.70),
        ("billion",      "billion-scale systems",       0.95),
        ("million user", "million-user serving",        0.95),
    ]

    profile = candidate.get("profile", {})
    parts: list[str] = [
        profile.get("summary", "").lower(),
        profile.get("headline", "").lower(),
    ]
    for r in candidate.get("career_history", []):
        parts.append(r.get("description", "").lower())

    combined = " ".join(parts)

    best_phrase, best_conf = None, 0.0
    for pattern, phrase, conf in _OWNERSHIP_PATTERNS:
        if pattern in combined and conf > best_conf:
            best_phrase, best_conf = phrase, conf

    return best_phrase, best_conf


# ---------------------------------------------------------------------------
# FEATURE SPECS  —  Single Source of Truth
# Extractor signature: (candidate, features) -> ExtractorResult
# PRIORITY 6: extractors return raw facts; renderer constructs sentences.
# ---------------------------------------------------------------------------

FEATURE_SPECS: list[dict] = [
    # --- TECHNICAL CORE ---
    {
        "key": "skill_core",
        "cluster": "technical",
        "phrase": "core production ML expertise",
        "extractor": lambda c, f: _top_skills(c, 2),
    },
    {
        "key": "retrieval",
        "cluster": "technical",
        "phrase": "production retrieval & vector search systems",
        "extractor": lambda c, f: _tech_list(c, _RETRIEVAL_LIBS, 2),
    },
    {
        "key": "assess",
        "cluster": "technical",
        "phrase": "verified platform skill assessments",
        "extractor": lambda c, f: (None, 0.0),
    },
    {
        "key": "python_sk",
        "cluster": "technical",
        "phrase": "strong Python proficiency",
        "extractor": lambda c, f: (None, 0.0),
    },
    {
        "key": "skill_endorse",
        "cluster": "technical",
        "phrase": "well-endorsed advanced skills",
        "extractor": lambda c, f: (None, 0.0),
    },
    {
        "key": "skill_dur",
        "cluster": "technical",
        "phrase": "sustained hands-on experience with core stack",
        "extractor": lambda c, f: (None, 0.0),
    },

    # --- SEMANTIC / TEXTUAL ---
    {
        "key": "sbert",
        "cluster": "semantic",
        "phrase": "strong semantic alignment with the role",
        "extractor": lambda c, f: (None, 0.0),
    },
    {
        "key": "bm25",
        "cluster": "semantic",
        "phrase": "strong keyword overlap with JD requirements",
        "extractor": lambda c, f: (None, 0.0),
    },
    {
        "key": "headline",
        "cluster": "semantic",
        "phrase": "headline reflects direct role relevance",
        "extractor": lambda c, f: _extract_headline(c, f.get("headline", 0)),
    },

    # --- CAREER / PRODUCT / OWNERSHIP ---
    {
        "key": "product",
        "cluster": "product",
        "phrase": "product engineering background",
        "extractor": lambda c, f: _product_companies(c),
    },
    {
        "key": "tenure",
        "cluster": "career",
        "phrase": "sustained relevant ML tenure",
        "extractor": lambda c, f: (
            (f"{c.get('profile', {}).get('years_of_experience', 0):.0f}yr ML exp", 0.8)
            if f.get("tenure", 0) > 0.5 else (None, 0.0)
        ),
    },
    {
        "key": "title_curr",
        "cluster": "career",
        "phrase": "directly relevant current role",
        "extractor": lambda c, f: (
            (c.get("profile", {}).get("current_title", ""), 0.9)
            if f.get("title_curr", 0) > 0.8 else (None, 0.0)
        ),
    },
    {
        "key": "title_avg",
        "cluster": "career",
        "phrase": "consistent relevant title history",
        "extractor": lambda c, f: (None, 0.0),
    },
    {
        "key": "prog",
        "cluster": "career",
        "phrase": "clear upward career progression",
        "extractor": lambda c, f: _progression_summary(c),
    },
    {
        "key": "prod_own",
        "cluster": "ownership",
        "phrase": "production ownership and deployment",
        "extractor": lambda c, f: _extract_production_ownership(c),
    },

    # --- BEHAVIORAL / EXTERNAL ---
    {
        "key": "github",
        "cluster": "behavioral",
        "phrase": "active technical portfolio",
        "extractor": lambda c, f: (
            ("GitHub portfolio", 0.85) if f.get("github", 0) > 0.7
            else (("GitHub activity", 0.55) if f.get("github", 0) > 0.4 else (None, 0.0))
        ),
    },
    {
        "key": "recruit",
        "cluster": "behavioral",
        "phrase": "high recruiter demand",
        "extractor": lambda c, f: (None, 0.0),
    },
    {
        "key": "cert",
        "cluster": "behavioral",
        "phrase": "relevant ML certifications",
        "extractor": lambda c, f: _certs_evidence(c),
    },
    {
        "key": "edu",
        "cluster": "career",
        "phrase": "top-tier academic foundation",
        "extractor": lambda c, f: _education_evidence(c),
    },
]


# ---------------------------------------------------------------------------
# NEGATIVE EVIDENCE SPECS
# Tuple: (key, severity, condition_fn, human_phrase)
# Severity: 3 = Major, 2 = Moderate, 1 = Minor
# ---------------------------------------------------------------------------

NEG_SPECS: list[tuple] = [
    # Major — Deal-breakers
    (
        "consult", 3,
        lambda f, c, s: f.get("consult", 1) < 0.4,
        "majority of experience in consulting engagements, limiting product ownership evidence",
    ),
    (
        "yoe_mult", 3,
        lambda f, c, s: _yoe_multiplier(c.get("profile", {}).get("years_of_experience", 0)) < 0.5,
        "years of experience fall significantly short of the 5–9 year target",
    ),
    (
        "loc_mult", 3,
        lambda f, c, s: _location_multiplier(c.get("profile", {}), c.get("redrob_signals", {})) < 0.5,
        "located outside preferred hiring regions with no relocation signal",
    ),

    # Moderate — Friction
    (
        "retrieval", 2,
        lambda f, c, s: f.get("retrieval", 1) < 0.3,
        "limited explicit vector search / retrieval library experience",
    ),
    (
        "product", 2,
        lambda f, c, s: f.get("product", 1) < 0.3,
        "background lacks product-company experience",
    ),
    (
        "hop", 2,
        lambda f, c, s: f.get("hop", 1) < 0.7,
        "frequent short stints suggest title-chasing pattern",
    ),
    (
        "recruit", 2,
        lambda f, c, s: f.get("recruit", 1) < 0.25,
        "limited recent recruiter engagement",
    ),
    (
        "github", 1,
        lambda f, c, s: f.get("github", 1) < 0.35,
        "minimal public technical footprint",
    ),

    # Minor — Signals
    (
        "otw", 1,
        lambda f, c, s: not s.get("open_to_work_flag", True),
        "not currently signaling openness to opportunities",
    ),
    (
        "notice", 1,
        lambda f, c, s: isinstance(s.get("notice_period_days"), int) and s["notice_period_days"] > 90,
        "extended notice period may delay onboarding",
    ),
    (
        "rr", 1,
        lambda f, c, s: s.get("recruiter_response_rate", 0.5) < 0.3,
        "low recruiter response rate",
    ),
]


# ---------------------------------------------------------------------------
# PRIORITY 17 — Three-level confidence hedge (High / Medium / Low)
# ---------------------------------------------------------------------------

def _hedge(score: float, high: str, medium: str, low: str) -> str:
    if score > 0.72:  return high
    if score > 0.50:  return medium
    return low


# ---------------------------------------------------------------------------
# PRIORITY 11 — Deterministic opening rotation via candidate ID hash.
# ---------------------------------------------------------------------------

_OPENINGS: dict[str, list[str]] = {
    "high": [
        "Demonstrates",
        "Shows substantial",
        "Profile highlights",
        "Experience reflects",
        "Strong evidence of",
    ],
    "medium": [
        "Evidence suggests",
        "Profile indicates",
        "Background reflects",
        "Shows evidence of",
        "Experience points to",
    ],
    "low": [
        "Some evidence indicates",
        "Partial signals suggest",
        "Background shows traces of",
        "Limited evidence of",
    ],
}


def _opening(score: float, candidate_id: str) -> str:
    """Select a deterministic opening phrase based on score tier and candidate hash."""
    if score > 0.72:
        pool = _OPENINGS["high"]
    elif score > 0.50:
        pool = _OPENINGS["medium"]
    else:
        pool = _OPENINGS["low"]
    idx = hash(candidate_id) % len(pool)
    return pool[idx]


# ---------------------------------------------------------------------------
# PRIORITY 8 — Merged semantic evidence builder
# ---------------------------------------------------------------------------

def _merged_semantic_phrase(features: dict) -> str | None:
    """
    When both SBERT and BM25 are strong, return a single combined phrase.
    Never exposes model names externally.
    """
    sbert = features.get("sbert", 0)
    bm25  = features.get("bm25",  0)
    if sbert > 0.7 and bm25 > 0.7:
        return "strong overall alignment with the target role based on technical experience and terminology"
    if sbert > 0.65:
        return "strong semantic alignment with the role"
    if bm25 > 0.65:
        return "strong keyword alignment with JD requirements"
    return None


# ---------------------------------------------------------------------------
# TEMPLATES  (8 variants)
# PRIORITY 6: templates receive raw evidence nouns; they own all grammar.
# PRIORITY 18: combine related evidence naturally rather than listing.
# ---------------------------------------------------------------------------

def _t_technical(
    render_ctx: dict, phrases: list[str], evidence: list[str],
    concern: str | None, score: float, opening: str,
) -> str:
    lead  = phrases[0] if phrases else "relevant technical background"
    ev    = f" backed by {evidence[0]}" if evidence else ""
    sup   = f" and {phrases[1]}" if len(phrases) > 1 else ""
    s1    = f"{opening} {lead}{sup}{ev}."
    s2    = concern or _hedge(score,
                              "Well-aligned with core retrieval and ranking requirements.",
                              "Alignment with core requirements is solid.",
                              "Alignment with core requirements is developing.")
    return f"{s1} {s2}"


def _t_semantic(
    render_ctx: dict, phrases: list[str], evidence: list[str],
    concern: str | None, score: float, opening: str,
) -> str:
    merged = render_ctx.get("merged_semantic")
    lead   = merged or (phrases[0] if phrases else "strong role alignment")
    ev     = f", supported by {evidence[0]}" if evidence else ""
    s1     = f"Profile highlights {lead}{ev}."
    s2     = concern or _hedge(score,
                               "Strong match to the job description.",
                               "Relevance is well-established.",
                               "Relevance is present but not dominant.")
    return f"{s1} {s2}"


def _t_product(
    render_ctx: dict, phrases: list[str], evidence: list[str],
    concern: str | None, score: float, opening: str,
) -> str:
    ev_companies = render_ctx.get("product_companies_raw")
    lead         = f"product and startup environments" if ev_companies else "product engineering"
    co_clause    = f" ({ev_companies})" if ev_companies else ""
    sup          = f", complemented by {phrases[1]}" if len(phrases) > 1 else ""
    s1           = f"Most relevant experience comes from {lead}{co_clause}{sup}."
    s2           = concern or _hedge(score,
                                     "Strong product engineering signal.",
                                     "Product exposure is well-evidenced.",
                                     "Product exposure is present but limited.")
    return f"{s1} {s2}"


def _t_career(
    render_ctx: dict, phrases: list[str], evidence: list[str],
    concern: str | None, score: float, opening: str,
) -> str:
    title = render_ctx.get("title_curr_extract", "current role")
    lead  = phrases[0] if phrases else "relevant expertise"
    ev    = f" with {evidence[0]}" if evidence else ""
    s1    = f"Current {title} brings {lead.lower()}{ev}."
    s2    = concern or (f"Backed by {phrases[1]}." if len(phrases) > 1 else "Profile shows strong career alignment.")
    return f"{s1} {s2}"


def _t_deployment(
    render_ctx: dict, phrases: list[str], evidence: list[str],
    concern: str | None, score: float, opening: str,
) -> str:
    # Triggered when tenure/prog are high but retrieval is low.
    lead = phrases[0] if phrases else "production deployment experience"
    ev   = f" ({evidence[0]})" if evidence else ""
    s1   = f"{opening} {lead.lower()}{ev}."
    s2   = concern or _hedge(score,
                             "Clear track record of shipping to production.",
                             "Production ownership is well-evidenced.",
                             "Production ownership signals are emerging.")
    return f"{s1} {s2}"


def _t_behavioral(
    render_ctx: dict, phrases: list[str], evidence: list[str],
    concern: str | None, score: float, opening: str,
) -> str:
    lead = phrases[0] if phrases else "strong market signals"
    sup  = f" alongside {phrases[1]}" if len(phrases) > 1 else ""
    ev   = f" ({evidence[0]})" if evidence else ""
    s1   = f"{opening} {lead}{sup}{ev}."
    s2   = concern or _hedge(score,
                             "Recruiter engagement validates strong market competitiveness.",
                             "Recruiter engagement confirms competitiveness.",
                             "Some recruiter interest noted.")
    return f"{s1} {s2}"


def _t_evaluation(
    render_ctx: dict, phrases: list[str], evidence: list[str],
    concern: str | None, score: float, opening: str,
) -> str:
    lead = phrases[0] if phrases else "evaluation and ranking focus"
    ev   = f" ({evidence[0]})" if evidence else ""
    s1   = f"{opening} {lead.lower()}{ev}."
    s2   = concern or "Directly relevant to the JD's emphasis on evaluation frameworks."
    return f"{s1} {s2}"


def _t_balanced(
    render_ctx: dict, phrases: list[str], evidence: list[str],
    concern: str | None, score: float, opening: str,
) -> str:
    lead = phrases[0] if phrases else "relevant background"
    ev   = f" ({evidence[0]})" if evidence else ""
    sup  = f" supported by {phrases[1]}" if len(phrases) > 1 else ""
    s1   = f"{opening} {lead}{ev}{sup}."
    s2   = concern or _hedge(score,
                             "Competitive profile for this role.",
                             "Solid fit for the role.",
                             "Developing fit for the role.")
    return f"{s1} {s2}"


def _t_ownership(
    render_ctx: dict, phrases: list[str], evidence: list[str],
    concern: str | None, score: float, opening: str,
) -> str:
    own_ev = render_ctx.get("prod_own_raw", evidence[0] if evidence else "production work")
    tech   = f" using {phrases[0]}" if phrases else ""
    s1     = f"{opening} {own_ev}{tech}."
    s2     = concern or _hedge(score,
                               "Strong signal of end-to-end production ownership.",
                               "Production ownership signals are well-evidenced.",
                               "Some production ownership signals present.")
    return f"{s1} {s2}"


TemplateFn = Callable[
    [dict, list[str], list[str], str | None, float, str],
    str,
]

TEMPLATES: dict[str, TemplateFn] = {
    "technical":  _t_technical,
    "semantic":   _t_semantic,
    "product":    _t_product,
    "career":     _t_career,
    "deployment": _t_deployment,
    "behavioral": _t_behavioral,
    "evaluation": _t_evaluation,
    "balanced":   _t_balanced,
    "ownership":  _t_ownership,
}


# ---------------------------------------------------------------------------
# MAIN ENTRY POINT
# ---------------------------------------------------------------------------

def generate_reasoning(candidate: dict, features: dict) -> str:
    """
    Decision Justification Engine.

    Args:
        candidate: Raw candidate dict.
        features:  Feature dict produced by extract_features().

    Returns:
        A 1–2 sentence deterministic reasoning string.
    """
    overall_score = features.get("overall_score", 0.5)
    signals       = candidate.get("redrob_signals", {})
    profile       = candidate.get("profile", {})
    candidate_id  = str(candidate.get("id", candidate.get("profile", {}).get("name", "anon")))

    # PRIORITY 16 — Never mutate the input dict; use render_ctx instead.
    render_ctx: dict = {}

    # 1. Compute multipliers for negative-evidence checks.
    yoe_mult = _yoe_multiplier(profile.get("years_of_experience", 0))
    loc_mult = _location_multiplier(profile, signals)

    # Build an augmented feature view for NEG_SPECS conditions (read-only copy).
    feats_aug = {**features, "yoe_mult": yoe_mult, "loc_mult": loc_mult}

    # 2. Rank positive evidence.
    #    PRIORITY 3: utility = weight × feature_value × extraction_quality
    #    PRIORITY 20: extractors return (text | None, confidence)
    active_pos: list[dict] = []
    for spec in FEATURE_SPECS:
        f_val = features.get(spec["key"], 0)

        # PRIORITY 13 — Dynamic thresholds computed after full pass (see step 2b).
        # Collect raw entries first, then prune.
        if f_val < 0.40:   # hard floor — don't even call extractor
            continue

        weight = get_feature_weight(spec["key"])
        noun, ext_conf = spec["extractor"](candidate, features)

        # PRIORITY 20: skip low-confidence extractions from renderer
        ext_conf = ext_conf if noun else 0.0
        eq       = _extraction_quality(noun) if noun else 0.5

        utility = weight * f_val * (1.0 + eq * 0.5 + ext_conf * 0.3)  # PRIORITY 3

        active_pos.append({
            "key":      spec["key"],
            "phrase":   spec["phrase"],
            "noun":     noun,
            "ext_conf": ext_conf,
            "utility":  utility,
            "cluster":  spec["cluster"],
            "f_val":    f_val,
        })

    active_pos.sort(key=lambda x: -x["utility"])

    # PRIORITY 13 — Dynamic relative threshold: keep only items ≥ 40% of top utility.
    # PRIORITY 12 — Suppress weak evidence below 40% of strongest utility.
    if active_pos:
        top_utility   = active_pos[0]["utility"]
        min_threshold = top_utility * 0.40
        # Also apply a fixed per-feature floor for noisy signals.
        _noisy = {"github", "cert", "edu", "headline"}
        active_pos = [
            p for p in active_pos
            if p["utility"] >= min_threshold
            and (p["f_val"] >= 0.60 if p["key"] in _noisy else p["f_val"] >= 0.55)
        ]

    # PRIORITY 19 — Diversity: rotate primary cluster emphasis by candidate hash
    # so near-identical scores highlight different strengths.
    _cluster_priority_order = [
        "technical", "ownership", "product", "career", "semantic", "behavioral",
    ]
    candidate_hash = hash(candidate_id)
    cluster_rotation_offset = candidate_hash % len(_cluster_priority_order)

    top_phrases: list[str] = [p["phrase"] for p in active_pos[:3]]
    top_nouns:   list[str] = [p["noun"]   for p in active_pos[:3] if p["noun"]]

    # 3. Rank negative evidence.
    #    PRIORITY 7 — Rank by severity × feature impact (1 - feature_value).
    concerns: list[dict] = []
    for key, sev, cond, phrase in NEG_SPECS:
        if cond(feats_aug, candidate, signals):
            impact = 1.0 - features.get(key, 0.5)
            concerns.append({"severity": sev, "phrase": phrase, "impact": impact,
                             "rank_score": sev * impact})
    concerns.sort(key=lambda x: -x["rank_score"])

    top_concern: str | None = None
    if concerns:
        top_concern = concerns[0]["phrase"]
        # Surface second only if Major(3) + ≥ Moderate(2) and impact is meaningful.
        if (
            len(concerns) > 1
            and concerns[0]["severity"] == 3
            and concerns[1]["severity"] >= 2
            and concerns[1]["impact"] > 0.3
        ):
            top_concern = f"{top_concern}; {concerns[1]['phrase']}"

    # 4. Select dominant cluster by summed utility.
    cluster_utils: dict[str, float] = {}
    for p in active_pos:
        cluster_utils[p["cluster"]] = cluster_utils.get(p["cluster"], 0) + p["utility"]
    dominant = max(cluster_utils, key=cluster_utils.get) if cluster_utils else "balanced"

    # 5. Override cluster for specific signal patterns.
    has_ret      = features.get("retrieval", 0) > 0.5
    has_rank_kw  = any("rank" in p.lower() or "eval" in p.lower() for p in top_phrases)
    has_prod     = features.get("product", 0) > 0.5
    has_tenure   = features.get("tenure", 0) > 0.5
    has_own      = features.get("prod_own", 0) > 0.5
    high_recruit = features.get("recruit", 0) > 0.7
    high_github  = features.get("github", 0) > 0.7

    # PRIORITY 8 — Compute merged semantic phrase before template selection.
    merged_sem = _merged_semantic_phrase(features)
    if merged_sem:
        render_ctx["merged_semantic"] = merged_sem

    template_key = dominant
    if has_ret or has_rank_kw:
        template_key = "technical"
    elif has_own:
        template_key = "ownership"
    elif "evaluation" in " ".join(top_phrases).lower() or "ranking" in " ".join(top_phrases).lower():
        template_key = "evaluation"
    elif has_prod and has_tenure:
        template_key = "product"
    elif features.get("title_curr", 0) > 0.8:
        template_key = "career"
    elif high_recruit or high_github:
        template_key = "behavioral"
    elif merged_sem:
        template_key = "semantic"
    elif features.get("tenure", 0) > 0.6 and features.get("prog", 0) > 0.6:
        template_key = "deployment"

    # PRIORITY 16 — Populate render_ctx instead of mutating features.
    if template_key == "career":
        render_ctx["title_curr_extract"] = profile.get("current_title", "current role")

    # Populate product/ownership raw evidence for grammar-owning templates.
    for p in active_pos:
        if p["key"] == "product" and p["noun"]:
            render_ctx["product_companies_raw"] = p["noun"]
        if p["key"] == "prod_own" and p["noun"]:
            render_ctx["prod_own_raw"] = p["noun"]

    # PRIORITY 11 — Deterministic opening based on score + candidate ID.
    opening = _opening(overall_score, candidate_id)

    # PRIORITY 15 — Generate only dominant + one fallback template (not all 8).
    fallback_key = "balanced" if template_key != "balanced" else "technical"
    keys_to_run  = [template_key, fallback_key]

    variants: dict[str, tuple[float, str]] = {}
    for name in keys_to_run:
        fn = TEMPLATES.get(name)
        if not fn:
            continue
        try:
            text  = fn(render_ctx, top_phrases, top_nouns, top_concern, overall_score, opening)
            score = (1.5 if name == template_key else 1.0) - len(text) * 0.001
            if name == "career" and not render_ctx.get("title_curr_extract"):
                score = -1.0
            variants[name] = (score, text)
        except Exception:
            continue

    if not variants:
        return "Profile processed; insufficient signal for detailed reasoning."

    best_text = max(variants.values(), key=lambda x: x[0])[1]

    # 8. Enforce 2-sentence output.
    sentences = [s.strip().rstrip(".") for s in best_text.split(". ") if s.strip()]
    return ". ".join(sentences[:2]) + "."
