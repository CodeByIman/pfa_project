# syntax=docker/dockerfile:1
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System deps for pdfplumber, spacy models
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    poppler-utils \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY deployment/requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

# Pre-download small models to speed up first run (optional)
RUN python - <<'PY'
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
PY

# You may optionally download spaCy models at build time
# RUN python -m spacy download en_core_web_sm
# RUN python -m spacy download fr_core_news_sm

COPY . /app

EXPOSE 8000

CMD ["uvicorn", "ai_research_agent.src.api.main:app", "--host", "0.0.0.0", "--port", "8000"] 