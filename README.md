# Smart Resume Analyzer

A cloud-based resume analysis tool built with FastAPI and Hugging Face.

## What it does
- Takes a resume (PDF or text) as input
- Extracts skills automatically using rule-based NLP
- Calculates an ATS score based on skills, education, experience and job match
- Summarizes resume content using Hugging Face AI models
- Compares resume against a job description and shows missing keywords

## Tech Stack
- FastAPI — Python backend
- Hugging Face — AI summarization
- pdfplumber — PDF text extraction
- Rule-based NLP — skill and keyword matching

## To run
pip install fastapi uvicorn pdfplumber transformers
uvicorn main:app --reload
