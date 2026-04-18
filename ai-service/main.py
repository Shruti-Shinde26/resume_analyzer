import re
import io
import requests
import pdfplumber
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel

# ── Constants ──────────────────────────────────────────────────────────────────

STOP_WORDS = {
    "a", "an", "the", "with", "and", "or", "for", "to", "of", "in",
    "looking", "experience", "required", "candidate", "role",
    "work", "skills", "knowledge", "ability"
}

from dotenv import load_dotenv
import os
load_dotenv()
HF_API_KEY = os.getenv("HF_API_KEY")

skills_list = [
    "python", "java", "sql", "docker", "aws", "mongodb", "react",
    "javascript", "typescript", "nodejs", "django", "flask", "fastapi",
    "kubernetes", "git", "linux", "html", "css", "machine learning",
    "deep learning", "tensorflow", "pytorch", "pandas", "numpy"
]

app = FastAPI()

# ── Text Cleaning ──────────────────────────────────────────────────────────────

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

# ── PDF Text Extraction ────────────────────────────────────────────────────────

def extract_text_from_pdf(file_bytes: bytes) -> str:
    text = ""
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()

# ── Skill Extraction ───────────────────────────────────────────────────────────

def extract_skills(text):
    text = clean_text(text)
    found   = [skill for skill in skills_list if skill in text]
    missing = [skill for skill in skills_list if skill not in text]
    return found, missing

# ── Education Detection ────────────────────────────────────────────────────────

def extract_education(text):
    text = text.lower()
    if any(word in text for word in ["b.tech", "btech", "bachelor", "b.e", "be "]):
        return "undergraduate", 20
    elif any(word in text for word in ["m.tech", "mtech", "master", "mba", "m.e"]):
        return "postgraduate", 20
    elif any(word in text for word in ["phd", "doctorate"]):
        return "phd", 20
    elif any(word in text for word in ["diploma", "12th", "hsc"]):
        return "diploma", 10
    else:
        return "not found", 0

# ── Experience Detection ───────────────────────────────────────────────────────

def extract_experience(text):
    text = text.lower()
    match = re.search(r'(\d+)\+?\s*year', text)
    if match:
        years = int(match.group(1))
        if years >= 5:
            return years, 20
        elif years >= 2:
            return years, 15
        else:
            return years, 10
    if any(word in text for word in ["fresher", "intern", "trainee", "student"]):
        return 0, 10
    return 0, 5

# ── Real ATS Score ─────────────────────────────────────────────────────────────

def calculate_real_ats_score(found, resume_text, job_match=None):
    # Component 1 — Skills Match (40%)
    skill_score = int((len(found) / len(skills_list)) * 40)

    # Component 2 — Education (20%)
    _, edu_score = extract_education(resume_text)

    # Component 3 — Experience (20%)
    _, exp_score = extract_experience(resume_text)

    # Component 4 — Keyword/JD Match (20%)
    if job_match is not None:
        keyword_score = int((job_match / 100) * 20)
    else:
        keyword_score = 10  # neutral if no JD provided

    total = skill_score + edu_score + exp_score + keyword_score
    return min(total, 100)  # cap at 100

# ── Hugging Face Summarization ─────────────────────────────────────────────────

def summarize(text):
    API_URL = "https://router.huggingface.co/hf-inference/models/facebook/bart-large-cnn"
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}

    trimmed = " ".join(text.split()[:800])

    try:
        response = requests.post(
            API_URL,
            headers=headers,
            json={
                "inputs": trimmed,
                "options": {"wait_for_model": True}
            },
            timeout=30
        )

        print("HF Status:", response.status_code)
        print("HF Response:", response.text[:200])

        if response.status_code != 200:
            return f"AI unavailable (status {response.status_code})"

        result = response.json()

        if isinstance(result, list) and "summary_text" in result[0]:
            return result[0]["summary_text"]
        elif isinstance(result, dict) and "error" in result:
            return f"AI busy: {result['error']}"
        else:
            return "Summary not available"

    except requests.exceptions.Timeout:
        return "AI timeout, try again"
    except Exception as e:
        return f"Summary error: {str(e)}"

# ── Job Match Score ────────────────────────────────────────────────────────────

def match_score(resume_text, job_desc):
    resume_words = set(clean_text(resume_text).split()) - STOP_WORDS
    job_words    = set(clean_text(job_desc).split())    - STOP_WORDS
    common       = resume_words.intersection(job_words)
    score        = int((len(common) / len(job_words)) * 100) if job_words else 0
    return score, list(common)

# ── Missing Keywords from JD ───────────────────────────────────────────────────

def get_missing_from_jd(resume_text, job_desc):
    resume_words = set(clean_text(resume_text).split()) - STOP_WORDS
    job_words    = set(clean_text(job_desc).split())    - STOP_WORDS
    missing      = job_words - resume_words
    return list(missing)

# ── Request Model ──────────────────────────────────────────────────────────────

class ResumeRequest(BaseModel):
    text: str
    job_description: Optional[str] = ""

# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/")
def home():
    return {"message": "Resume Analyzer AI Service Running"}

@app.get("/health")
def health():
    return {"status": "ok", "service": "Resume Analyzer"}

# ── Text Analyze Endpoint ──────────────────────────────────────────────────────

@app.post("/analyze")
def analyze(data: ResumeRequest):
    found, missing  = extract_skills(data.text)
    summary         = summarize(data.text)
    edu_level, _    = extract_education(data.text)
    exp_years, _    = extract_experience(data.text)

    if data.job_description:
        match, matched_words = match_score(data.text, data.job_description)
        jd_missing           = get_missing_from_jd(data.text, data.job_description)
        score                = calculate_real_ats_score(found, data.text, job_match=match)
    else:
        match, matched_words, jd_missing = 0, [], []
        score                = calculate_real_ats_score(found, data.text)

    return {
        "skills":              found,
        "missing_skills":      jd_missing if data.job_description else missing,  # ← option B
        "ats_score":           score,
        "score_breakdown": {
            "skills_score":    int((len(found) / len(skills_list)) * 40),
            "education_score": extract_education(data.text)[1],
            "experience_score":extract_experience(data.text)[1],
            "keyword_score":   int((match / 100) * 20) if match else 10
        },
        "education_detected":  edu_level,
        "experience_detected": f"{exp_years} years" if exp_years else "fresher/not mentioned",
        "summary":             summary,
        "job_match_score":     match,
        "matched_keywords":    matched_words,
        "jd_missing_keywords": jd_missing,
        "suggestions": (
            f"Add these skills from job description: {', '.join(jd_missing[:5])}"
            if jd_missing else "Add more relevant skills to improve your profile"
        )
    }

# ── PDF Analyze Endpoint ───────────────────────────────────────────────────────

@app.post("/analyze-pdf")
async def analyze_pdf(
    file: UploadFile = File(...),
    job_description: str = Form(default="")
):
    if not file.filename.endswith(".pdf"):
        return {"error": "Only PDF files are supported"}

    file_bytes  = await file.read()
    resume_text = extract_text_from_pdf(file_bytes)

    if not resume_text:
        return {"error": "Could not extract text from PDF. Make sure it's not a scanned image PDF."}

    found, missing  = extract_skills(resume_text)
    summary         = summarize(resume_text)
    edu_level, _    = extract_education(resume_text)
    exp_years, _    = extract_experience(resume_text)

    if job_description:
        match, matched_words = match_score(resume_text, job_description)
        jd_missing           = get_missing_from_jd(resume_text, job_description)
        score                = calculate_real_ats_score(found, resume_text, job_match=match)
    else:
        match, matched_words, jd_missing = 0, [], []
        score                = calculate_real_ats_score(found, resume_text)

    return {
        "filename":            file.filename,
        "extracted_preview":   resume_text[:300] + "...",
        "skills":              found,
        "missing_skills":      jd_missing if job_description else missing,  # ← option B
        "ats_score":           score,
        "score_breakdown": {
            "skills_score":    int((len(found) / len(skills_list)) * 40),
            "education_score": extract_education(resume_text)[1],
            "experience_score":extract_experience(resume_text)[1],
            "keyword_score":   int((match / 100) * 20) if match else 10
        },
        "education_detected":  edu_level,
        "experience_detected": f"{exp_years} years" if exp_years else "fresher/not mentioned",
        "summary":             summary,
        "job_match_score":     match,
        "matched_keywords":    matched_words,
        "jd_missing_keywords": jd_missing,
        "suggestions": (
            f"Add these skills from job description: {', '.join(jd_missing[:5])}"
            if jd_missing else "Add more relevant skills to improve your profile"
        )
    }