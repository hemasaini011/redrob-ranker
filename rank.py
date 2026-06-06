"""
Redrob Hackathon — Intelligent Candidate Ranking System
Architecture: Hybrid Scoring = Semantic Profile Score + Rule-Based JD Fit + Behavioral Signal Multiplier
Author: Redrob Ranker
"""

import json
import csv
import math
import argparse
import sys
from datetime import datetime, date
from pathlib import Path


# ─────────────────────────────────────────────
# JD UNDERSTANDING — structured signal extraction
# ─────────────────────────────────────────────

JD_SIGNALS = {
    # Hard requirements (disqualifiers if missing)
    "required_skills": [
        "embeddings", "vector", "retrieval", "ranking", "search",
        "sentence-transformers", "openai embeddings", "bge", "e5",
        "pinecone", "weaviate", "qdrant", "milvus", "faiss", "opensearch",
        "elasticsearch", "weaviate", "hybrid search", "dense retrieval",
        "python", "production", "embedding", "ndcg", "mrr", "map", "a/b",
        "llm", "reranking", "re-ranking", "vector database", "nlp",
        "information retrieval", "recommendation", "semantic search",
    ],
    # Nice to have
    "preferred_skills": [
        "lora", "qlora", "peft", "fine-tuning", "fine tuning",
        "learning to rank", "xgboost", "langchain",
        "distributed systems", "inference optimization",
        "open source", "rag", "transformers", "huggingface",
    ],
    # Strongly positive job titles / roles
    "positive_titles": [
        "ai engineer", "ml engineer", "machine learning engineer",
        "applied scientist", "research engineer", "nlp engineer",
        "search engineer", "ranking engineer", "data scientist",
        "senior engineer", "staff engineer", "founding engineer",
    ],
    # Disqualifying patterns
    "disqualify_titles": [
        "marketing", "sales", "hr ", "human resource", "finance",
        "accountant", "graphic design", "ui designer", "ux designer",
        "content writer", "seo", "business analyst", "project manager",
        "product manager", "scrum master", "recruiter",
    ],
    # Companies explicitly flagged as "bad fit" by the JD
    "services_companies": [
        "tcs", "tata consultancy", "infosys", "wipro", "accenture",
        "cognizant", "capgemini", "hcl", "tech mahindra", "mphasis",
        "hexaware", "l&t infotech", "ltimindtree", "mindtree",
        "persistent", "niit", "mastech", "zensar",
    ],
    # Location signals (India-based preferred)
    "preferred_locations": [
        "noida", "pune", "delhi", "ncr", "gurgaon", "gurugram",
        "bengaluru", "bangalore", "mumbai", "hyderabad", "chennai",
        "india",
    ],
    # Experience range
    "exp_min": 4,
    "exp_max": 10,
    "exp_ideal_min": 5,
    "exp_ideal_max": 9,
}

# Skills that are CORE to the role — weighted higher
CORE_SKILL_KEYWORDS = {
    "embedding": 3.0, "embeddings": 3.0, "vector": 2.5,
    "retrieval": 2.5, "ranking": 2.5, "search": 2.0,
    "faiss": 2.5, "pinecone": 2.5, "qdrant": 2.5, "milvus": 2.5,
    "weaviate": 2.5, "opensearch": 2.0, "elasticsearch": 2.0,
    "sentence-transformer": 3.0, "sentence_transformer": 3.0,
    "bge": 2.5, "e5": 2.0, "bi-encoder": 2.5, "cross-encoder": 2.5,
    "llm": 2.0, "nlp": 2.0, "transformer": 2.0,
    "rag": 2.0, "rerank": 2.5, "re-rank": 2.5,
    "ndcg": 3.0, "mrr": 2.5, "map": 2.0, "a/b test": 2.0,
    "python": 1.5, "production": 2.0, "deployed": 2.0,
    "fine-tun": 1.5, "fine_tun": 1.5, "lora": 1.5, "qlora": 1.5,
    "recommendation": 1.5, "semantic": 2.0, "hybrid search": 2.5,
    "information retrieval": 2.5, "ir ": 2.0,
}


# ─────────────────────────────────────────────
# HONEYPOT DETECTION
# ─────────────────────────────────────────────

def detect_honeypot(candidate: dict) -> bool:
    """
    Detect candidates with impossible/fabricated profiles.
    Returns True if the candidate is likely a honeypot.
    """
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])

    # Check: years of experience vs company founding dates
    yoe = profile.get("years_of_experience", 0)

    for job in career:
        start = job.get("start_date", "")
        if start:
            try:
                start_year = int(start[:4])
                if start_year < 2000 and yoe < 10:
                    return True
            except:
                pass

    # Check: too many "expert" skills with 0 duration
    expert_zero = sum(
        1 for s in skills
        if s.get("proficiency") == "expert" and s.get("duration_months", 1) == 0
    )
    if expert_zero >= 3:
        return True

    # Check: impossible skill count with high proficiency
    advanced_expert = sum(
        1 for s in skills
        if s.get("proficiency") in ("advanced", "expert")
    )
    if advanced_expert >= 15 and yoe < 5:
        return True

    # Check: title mismatch — non-technical title with heavy AI skill claims
    title = (profile.get("current_title") or "").lower()
    headline = (profile.get("headline") or "").lower()
    is_nontechnical = any(
        kw in title for kw in JD_SIGNALS["disqualify_titles"]
    )
    has_ai_skills = any(
        kw in (s.get("name") or "").lower()
        for s in skills
        for kw in ["embedding", "llm", "vector", "rag", "nlp", "transformer"]
    )
    if is_nontechnical and has_ai_skills and len(skills) > 10:
        return True

    # Check: job duration math doesn't add up (flag extreme cases)
    total_job_months = sum(j.get("duration_months", 0) for j in career if not j.get("is_current"))
    if career and yoe > 0:
        claimed_months = yoe * 12
        # If total past jobs exceed claimed experience by huge margin → fabricated
        if total_job_months > claimed_months * 1.8:
            return True

    return False


# ─────────────────────────────────────────────
# FEATURE EXTRACTION
# ─────────────────────────────────────────────

def extract_text_blob(candidate: dict) -> str:
    """Combine all text fields into one blob for keyword scoring."""
    parts = []
    profile = candidate.get("profile", {})
    parts.append(profile.get("headline", "") or "")
    parts.append(profile.get("summary", "") or "")
    parts.append(profile.get("current_title", "") or "")

    for job in candidate.get("career_history", []):
        parts.append(job.get("title", "") or "")
        parts.append(job.get("description", "") or "")
        parts.append(job.get("company", "") or "")

    for skill in candidate.get("skills", []):
        parts.append(skill.get("name", "") or "")

    for cert in candidate.get("certifications", []):
        parts.append(cert.get("name", "") or "")

    return " ".join(parts).lower()


def score_skills(candidate: dict) -> float:
    """
    Score candidates on relevant skills with weighted matching.
    Returns 0.0 – 1.0
    """
    skills = candidate.get("skills", [])
    signals = candidate.get("redrob_signals", {})
    assessment_scores = signals.get("skill_assessment_scores", {}) or {}

    total_weight = 0.0
    matched_weight = 0.0

    blob = extract_text_blob(candidate)

    for keyword, weight in CORE_SKILL_KEYWORDS.items():
        total_weight += weight
        if keyword in blob:
            # Extra credit: assess score confirms proficiency
            matched_weight += weight

    # Bonus: actual skill entries with high proficiency for relevant skills
    for skill in skills:
        name = (skill.get("name") or "").lower()
        proficiency = skill.get("proficiency") or ""
        duration = skill.get("duration_months") or 0
        endorsements = skill.get("endorsements") or 0

        # Check if this skill is relevant
        is_relevant = any(kw in name for kw in CORE_SKILL_KEYWORDS)
        if not is_relevant:
            continue

        # Proficiency multiplier
        prof_mult = {"beginner": 0.3, "intermediate": 0.6, "advanced": 0.85, "expert": 1.0}.get(proficiency, 0.5)

        # Duration bonus (capped at 36 months)
        duration_bonus = min(duration / 36.0, 1.0) * 0.2

        # Endorsement bonus (capped at 50)
        endorse_bonus = min(endorsements / 50.0, 1.0) * 0.1

        # Assessment score bonus
        assess_key = skill.get("name", "")
        if assess_key in assessment_scores:
            assess_bonus = (assessment_scores[assess_key] / 100.0) * 0.15
        else:
            assess_bonus = 0

        matched_weight += 0.5 * (prof_mult + duration_bonus + endorse_bonus + assess_bonus)

    if total_weight == 0:
        return 0.0

    return min(matched_weight / total_weight, 1.0)


def score_experience(candidate: dict) -> float:
    """
    Score based on experience quality — product vs services, relevance, tenure.
    Returns 0.0 – 1.0
    """
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])

    yoe = profile.get("years_of_experience") or 0

    # Experience range fit
    if yoe < JD_SIGNALS["exp_min"]:
        exp_range_score = yoe / JD_SIGNALS["exp_min"] * 0.5
    elif yoe <= JD_SIGNALS["exp_ideal_max"]:
        exp_range_score = 1.0
    elif yoe <= JD_SIGNALS["exp_max"]:
        exp_range_score = 0.85
    else:
        # Over-experienced — can be fine but slight penalty
        exp_range_score = 0.7

    # Product company vs services company history
    product_months = 0
    services_months = 0
    total_months = 0

    for job in career:
        company = (job.get("company") or "").lower()
        duration = job.get("duration_months") or 0
        total_months += duration

        is_services = any(sc in company for sc in JD_SIGNALS["services_companies"])
        # Also check company size as proxy (10001+ is likely services/enterprise)
        company_size = job.get("company_size") or ""

        if is_services:
            services_months += duration
        else:
            product_months += duration

    product_ratio = product_months / max(total_months, 1)
    # JD explicitly says consulting-only career is disqualifying
    services_ratio = services_months / max(total_months, 1)

    if services_ratio > 0.9:
        services_penalty = 0.4  # strong penalty but not zero
    elif services_ratio > 0.6:
        services_penalty = 0.7
    else:
        services_penalty = 1.0

    # Tenure score — JD wants 3+ year commitment, penalize job hopping
    if len(career) > 1:
        avg_tenure = total_months / len(career)
        if avg_tenure < 12:
            tenure_score = 0.5  # job hopper
        elif avg_tenure < 18:
            tenure_score = 0.75
        else:
            tenure_score = 1.0
    else:
        tenure_score = 0.9  # single job, can't tell

    return (exp_range_score * 0.5 + product_ratio * 0.3 + tenure_score * 0.2) * services_penalty


def score_location(candidate: dict) -> float:
    """Score based on location fit and relocation willingness."""
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})

    location = (profile.get("location") or "").lower()
    country = (profile.get("country") or "").lower()
    willing_to_relocate = signals.get("willing_to_relocate", False)

    # India-based preferred
    if country == "india" or "india" in location:
        loc_base = 1.0
    elif willing_to_relocate:
        loc_base = 0.6
    else:
        loc_base = 0.3  # Outside India, won't relocate

    # Tier-1 city bonus
    preferred_city = any(city in location for city in JD_SIGNALS["preferred_locations"])
    if preferred_city:
        loc_base = min(loc_base * 1.1, 1.0)

    return loc_base


def score_behavioral(candidate: dict) -> float:
    """
    Score behavioral signals — availability, engagement, responsiveness.
    This acts as a MULTIPLIER on skill fit.
    Returns 0.0 – 1.0
    """
    signals = candidate.get("redrob_signals", {})

    # Recency: last active date
    last_active = signals.get("last_active_date")
    if last_active:
        try:
            last_dt = datetime.strptime(last_active, "%Y-%m-%d").date()
            today = date(2026, 6, 5)
            days_inactive = (today - last_dt).days
            if days_inactive <= 7:
                recency_score = 1.0
            elif days_inactive <= 30:
                recency_score = 0.9
            elif days_inactive <= 90:
                recency_score = 0.7
            elif days_inactive <= 180:
                recency_score = 0.4
            else:
                recency_score = 0.2
        except:
            recency_score = 0.5
    else:
        recency_score = 0.5

    # Open to work
    open_to_work = signals.get("open_to_work_flag", False)
    otw_score = 1.0 if open_to_work else 0.6

    # Recruiter response rate — critical for hireability
    response_rate = signals.get("recruiter_response_rate") or 0.0
    if response_rate >= 0.7:
        response_score = 1.0
    elif response_rate >= 0.4:
        response_score = 0.75
    elif response_rate >= 0.2:
        response_score = 0.5
    else:
        response_score = 0.2

    # Notice period
    notice = signals.get("notice_period_days") or 90
    if notice <= 30:
        notice_score = 1.0
    elif notice <= 60:
        notice_score = 0.85
    elif notice <= 90:
        notice_score = 0.65
    else:
        notice_score = 0.4

    # Interview completion rate
    icr = signals.get("interview_completion_rate") or 0.5
    icr_score = icr

    # Profile completeness
    completeness = (signals.get("profile_completeness_score") or 50) / 100.0

    # GitHub activity — relevant for this role
    github = signals.get("github_activity_score") or -1
    if github == -1:
        github_score = 0.4  # no github linked — minor negative
    elif github >= 70:
        github_score = 1.0
    elif github >= 40:
        github_score = 0.75
    elif github >= 10:
        github_score = 0.5
    else:
        github_score = 0.3

    # Weighted behavioral score
    behavioral = (
        recency_score * 0.25 +
        otw_score * 0.15 +
        response_score * 0.20 +
        notice_score * 0.15 +
        icr_score * 0.10 +
        completeness * 0.05 +
        github_score * 0.10
    )

    return behavioral


def score_title_fit(candidate: dict) -> float:
    """Check if current title/role is relevant to Senior AI Engineer."""
    profile = candidate.get("profile", {})
    title = (profile.get("current_title") or "").lower()
    headline = (profile.get("headline") or "").lower()
    combined = title + " " + headline

    # Hard disqualifier — non-technical role
    if any(kw in combined for kw in JD_SIGNALS["disqualify_titles"]):
        return 0.1

    # Strong positive signals
    if any(kw in combined for kw in JD_SIGNALS["positive_titles"]):
        return 1.0

    # Technical but less aligned
    technical_adjacent = [
        "software engineer", "developer", "data engineer", "backend",
        "fullstack", "platform", "sre", "devops", "cloud"
    ]
    if any(kw in combined for kw in technical_adjacent):
        return 0.6

    return 0.4


def score_purely_research(candidate: dict) -> float:
    """
    Penalize pure researchers without production experience.
    JD explicitly disqualifies academic-only profiles.
    Returns penalty multiplier 0.3 – 1.0
    """
    career = candidate.get("career_history", [])
    profile = candidate.get("profile", {})
    summary = (profile.get("summary") or "").lower()

    research_signals = ["phd", "research scientist", "researcher", "academia",
                        "professor", "postdoc", "post-doc", "lab ", "university research"]
    production_signals = ["production", "deployed", "launched", "shipped",
                          "users", "scale", "million", "latency", "a/b test",
                          "serving", "inference", "api", "product"]

    has_research = any(kw in summary for kw in research_signals)
    has_production = any(kw in summary for kw in production_signals)

    for job in career:
        desc = (job.get("description") or "").lower()
        title = (job.get("title") or "").lower()
        company = (job.get("company") or "").lower()

        if any(kw in title or kw in company for kw in ["research", "lab", "university"]):
            has_research = True
        if any(kw in desc for kw in production_signals):
            has_production = True

    if has_research and not has_production:
        return 0.4  # Pure researcher — penalized per JD
    return 1.0


# ─────────────────────────────────────────────
# COMPOSITE SCORE
# ─────────────────────────────────────────────

def compute_score(candidate: dict) -> tuple[float, str]:
    """
    Compute the final composite score for a candidate.
    Returns (score: float 0-1, reasoning: str)
    """
    # Step 1: Honeypot detection
    if detect_honeypot(candidate):
        return 0.0, "Flagged as honeypot — profile contains internally inconsistent signals."

    profile = candidate.get("profile", {})

    # Step 2: Hard disqualifier — non-technical current title
    title = (profile.get("current_title") or "").lower()
    if any(kw in title for kw in JD_SIGNALS["disqualify_titles"]):
        return 0.05, f"Disqualified: current title '{profile.get('current_title')}' is non-technical and incompatible with Senior AI Engineer role."

    # Step 3: Individual dimension scores
    s_skills = score_skills(candidate)
    s_exp = score_experience(candidate)
    s_location = score_location(candidate)
    s_behavioral = score_behavioral(candidate)
    s_title = score_title_fit(candidate)
    research_penalty = score_purely_research(candidate)

    # Step 4: Weighted composite
    # Skills + Experience are the core; behavioral is a multiplier
    base_score = (
        s_skills * 0.40 +
        s_exp * 0.25 +
        s_title * 0.15 +
        s_location * 0.10 +
        s_behavioral * 0.10
    )

    # Apply research penalty
    base_score *= research_penalty

    # Apply behavioral as a multiplier on upper range
    # (A high-skill candidate who is inactive gets pulled down)
    final_score = base_score * (0.6 + 0.4 * s_behavioral)

    final_score = max(0.0, min(1.0, final_score))

    # Build reasoning
    reasoning = build_reasoning(candidate, s_skills, s_exp, s_location, s_behavioral, s_title, final_score)

    return final_score, reasoning


def build_reasoning(candidate, s_skills, s_exp, s_location, s_behavioral, s_title, final_score):
    """Generate honest, specific reasoning for each candidate."""
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})

    yoe = profile.get("years_of_experience") or 0
    title = profile.get("current_title") or "Unknown"
    location = profile.get("location") or "Unknown"
    notice = signals.get("notice_period_days") or 90
    response_rate = signals.get("recruiter_response_rate") or 0.0
    last_active = signals.get("last_active_date") or "unknown"
    open_to_work = signals.get("open_to_work_flag", False)

    skills = candidate.get("skills", [])
    relevant_skills = [
        s["name"] for s in skills
        if any(kw in (s.get("name") or "").lower() for kw in CORE_SKILL_KEYWORDS)
        and s.get("proficiency") in ("advanced", "expert", "intermediate")
    ][:4]

    parts = []

    if final_score >= 0.75:
        parts.append(f"{yoe:.1f}yr {title} with strong AI/ML production background.")
    elif final_score >= 0.55:
        parts.append(f"{yoe:.1f}yr {title} — solid technical fit with some gaps.")
    else:
        parts.append(f"{yoe:.1f}yr {title} — partial fit only.")

    if relevant_skills:
        parts.append(f"Relevant skills: {', '.join(relevant_skills)}.")

    # Location note
    country = (profile.get("country") or "").lower()
    if "india" in country or any(c in location.lower() for c in JD_SIGNALS["preferred_locations"]):
        parts.append(f"Based in {location} (India-preferred role).")
    else:
        willing = signals.get("willing_to_relocate", False)
        parts.append(f"Located in {location}; {'willing' if willing else 'not willing'} to relocate.")

    # Behavioral note
    if response_rate < 0.3:
        parts.append(f"Low recruiter response rate ({response_rate:.0%}) — availability concern.")
    if notice > 90:
        parts.append(f"Long notice period ({notice}d) may delay onboarding.")
    if not open_to_work:
        parts.append("Not marked open to work.")

    return " ".join(parts)[:300]  # Keep under 300 chars


# ─────────────────────────────────────────────
# MAIN RANKING PIPELINE
# ─────────────────────────────────────────────

def rank_candidates(candidates_path: str, output_path: str):
    print(f"Loading candidates from {candidates_path}...")
    candidates = []
    with open(candidates_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))

    print(f"Loaded {len(candidates)} candidates. Scoring...")

    scored = []
    for i, c in enumerate(candidates):
        if i % 10000 == 0:
            print(f"  Processed {i}/{len(candidates)}...")
        score, reasoning = compute_score(c)
        scored.append({
            "candidate_id": c["candidate_id"],
            "score": score,
            "reasoning": reasoning,
        })

    # Sort by score descending; tie-break: candidate_id ascending (per spec)
    scored.sort(key=lambda x: (-x["score"], x["candidate_id"]))

    # Take top 100
    top100 = scored[:100]

    # Assign ranks 1–100, ensure scores are non-increasing
    # (handle floating point ties)
    prev_score = top100[0]["score"]
    for rank_idx, item in enumerate(top100):
        if item["score"] > prev_score:
            item["score"] = prev_score  # enforce non-increasing
        prev_score = item["score"]
        item["rank"] = rank_idx + 1

    # Write output CSV
    print(f"Writing top 100 to {output_path}...")
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        writer.writeheader()
        for item in top100:
            writer.writerow({
                "candidate_id": item["candidate_id"],
                "rank": item["rank"],
                "score": round(item["score"], 4),
                "reasoning": item["reasoning"],
            })

    print(f"Done. Top candidate: {top100[0]['candidate_id']} (score={top100[0]['score']:.4f})")
    print(f"Rank 100 candidate: {top100[99]['candidate_id']} (score={top100[99]['score']:.4f})")
    return top100


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Redrob Hackathon — Candidate Ranker")
    parser.add_argument("--candidates", default="candidates.jsonl", help="Path to candidates.jsonl")
    parser.add_argument("--out", default="submission.csv", help="Output CSV path")
    args = parser.parse_args()

    rank_candidates(args.candidates, args.out)
