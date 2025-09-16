# AI Research Agent (RAG)

An AI academic research assistant that performs query understanding (multilingual), arXiv retrieval, pre-ranking using extractive summaries + semantic similarity, PDF processing, and final summarization (extractive + abstractive). Exposes a FastAPI backend and a Streamlit frontend.

## Quickstart

1. Create a Python 3.10 environment (Conda recommended)

```bash
conda create -n ai_ra python=3.10 -y
conda activate ai_ra
pip install -r deployment/requirements.txt
# Optional small models
python -m spacy download en_core_web_sm || true
python -m spacy download fr_core_news_sm || true
```

2. Run API

```bash
uvicorn ai_research_agent.src.api.main:app --reload --port 8000
```

3. Run Streamlit UI

```bash
streamlit run ai_research_agent/src/frontend/app.py
```

## Example queries
- "État de l’art apprentissage profond diagnostic médical 2023"
- "Give me recent papers about medical AI diagnosis"

## Project layout
See `ai_research_agent/src/core/*` for modules:
- query_understanding: language, intent, entities, expansion
- retrieval: arXiv client
- ranking: embeddings + pre-ranking
- pdf_processing: download + text extraction
- generation: TF-IDF, LSA, abstractive summarizers
- evaluation: ROUGE, BERTScore
- agent: orchestration + CLI

## Tests
```bash
pytest -q
```

## Notes
- GPU will be used automatically for abstractive summarization if CUDA is available.
- If some heavy models cannot be downloaded in your environment, fallbacks are provided to keep the pipeline functional. 