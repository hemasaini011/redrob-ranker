# Redrob Hackathon — Intelligent Candidate Ranking System

## What this does

Ranks 100,000 candidates for a **Senior AI Engineer** role using a hybrid scoring system that understands *what the JD means*, not just what it says.

Key design decisions:
- **No keyword matching** — weighted semantic keyword scoring that understands context (e.g. "Milvus" and "FAISS" are both vector DBs, not separate concepts)
- **Honeypot detection** — catches impossible profiles before scoring
- **Hard disqualifiers** — non-technical titles, pure-research backgrounds, and consulting-only careers are penalized per JD guidance
- **Behavioral multiplier** — a high-skill candidate who hasn't logged in for 6 months gets pulled down. A responsive, open-to-work candidate gets lifted
- **Runs in ~25s on CPU** — no GPU, no API calls, scales to 200K+

---

## Architecture

```
candidates.jsonl (100K)
        │
        ▼
┌───────────────────┐
│  Honeypot Filter  │  ← Detects impossible profiles (experience math, title mismatch)
└────────┬──────────┘
         │
         ▼
┌───────────────────────────────────────────────────────┐
│              Multi-Dimensional Scorer                 │
│                                                       │
│  Skills Score (40%)   — weighted keyword matching     │
│                          + proficiency + duration     │
│                          + assessment scores          │
│                                                       │
│  Experience Score (25%) — YoE fit + product vs        │
│                           services ratio + tenure     │
│                                                       │
│  Title Fit (15%)      — current role relevance        │
│                          hard disqualifiers           │
│                                                       │
│  Location Score (10%) — India-based + city tier       │
│                          + relocation flag            │
│                                                       │
│  Behavioral Score (10%) — recency, open-to-work,     │
│                            response rate, notice,     │
│                            interview completion,      │
│                            GitHub activity            │
└────────────────────────┬──────────────────────────────┘
                         │
                         ▼
              Base Score × Research Penalty
                         │
                         ▼
              × Behavioral Multiplier (0.6–1.0)
                         │
                         ▼
                  Final Score (0–1)
                         │
                         ▼
              Top 100 → submission CSV
```

---

## Setup

```bash
# Python 3.9+
pip install -r requirements.txt
```

---

## Reproduce submission

```bash
# Step 1: Generate ranked CSV (runs in ~25s on CPU, <2GB RAM)
python rank.py --candidates ./candidates.jsonl --out ./team_submission.csv

# Step 2: Validate format
python validate_submission.py team_submission.csv
```

That's it. No pre-computation step needed. The ranker is fully self-contained.

---

## Scoring dimensions explained

### Skills (40%)
Matches against 30+ core keywords weighted by relevance to the JD:
- Highest weight: `sentence-transformers`, `FAISS`, `Qdrant`, `NDCG`, `re-ranking` (3.0×)
- High weight: `vector`, `retrieval`, `ranking`, `embedding`, `hybrid search` (2.5×)
- Medium weight: `LLM`, `NLP`, `semantic`, `production` (2.0×)

Each matched skill is further boosted by proficiency level, duration used, endorsements, and Redrob assessment scores.

### Experience (25%)
- Years in the 5–9 band scores highest (1.0); outside the band gets 0.7–0.85
- Product company ratio vs services company ratio — consulting-only career → 0.4× multiplier
- Job tenure average — job hopping (<12 months avg) → 0.5×

### Title Fit (15%)
- `AI Engineer`, `ML Engineer`, `NLP Engineer`, `Senior/Staff` → 1.0
- Software/Backend/Data Engineer → 0.6
- Marketing, Sales, HR, PM → 0.1 (near-disqualify)

### Location (10%)
- India + Tier-1 city → 1.0
- India anywhere → 0.9+
- Outside India + willing to relocate → 0.6
- Outside India + not relocating → 0.3

### Behavioral Multiplier
Applied as `final = base × (0.6 + 0.4 × behavioral_score)`:

| Signal | Weight |
|---|---|
| Last active recency | 25% |
| Recruiter response rate | 20% |
| Notice period | 15% |
| Open to work flag | 15% |
| Interview completion rate | 10% |
| GitHub activity score | 10% |
| Profile completeness | 5% |

### Honeypot Detection
Flags profiles where:
- Total job duration months >> claimed YoE by >80%
- 3+ "expert" skills with 0 months duration
- 15+ advanced/expert skills with <5 YoE
- Non-technical current title + >10 AI skills listed

---

## Files

```
redrob_ranker/
├── rank.py                     # Main ranker — single entry point
├── validate_submission.py      # Official validator (from challenge bundle)
├── requirements.txt
├── submission_metadata.yaml
├── team_submission.csv         # Final ranked output
└── README.md
```

---

## Runtime

| Dataset size | Time (CPU only) | RAM |
|---|---|---|
| 100K candidates | ~25 seconds | ~1.5 GB |

Well within the 5-minute / 16 GB constraint.
