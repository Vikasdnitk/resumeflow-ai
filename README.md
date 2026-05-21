# resumeflow.ai

A lightweight web app where job seekers can:

- Paste a job description from LinkedIn, Naukri, or any portal
- Upload their latest resume (`.txt`, `.pdf`, `.docx`)
- Generate a structured ATS-oriented resume draft with keyword alignment
- See estimated ATS match score and missing keywords
- Download generated resume as PDF, DOCX, or TXT
- Choose from 3 PDF templates before downloading:
  - Template 1: Horizontal-line separated sections
  - Template 2: Two-column partition layout
  - Template 3: Classic single-column ATS layout
- Optional strict mode: preserve uploaded resume content as-is and only append ATS keywords at the end
- Local SQLite database integration (`ats_resume.db`) to store recent resume generations
- Refreshing home starts a new on-screen session; previous versions can be reopened from history
- Post-generation modification box allows user-requested updates to create a new version
- AI Resume Assistant chat box for role-specific rewrite suggestions
- One-click "Apply Last AI Suggestion" to create and save a new resume version

The app updates your uploaded resume content with job-description keywords while preserving the core section flow. Exact pixel-perfect editing of original PDF layout is not guaranteed.

## Tech stack

- Python
- Flask
- Basic NLP-style keyword extraction (heuristic)

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Set your OpenAI key before running chat features:

```powershell
$env:OPENAI_API_KEY="your_openai_api_key"
```

Open: `http://127.0.0.1:5000`

## Notes

- This app creates a strong first draft for ATS matching; users should still verify correctness and truthfulness.
- For production-grade AI rewriting, you can integrate an LLM API in `app.py` by replacing `build_tailored_resume` logic.
