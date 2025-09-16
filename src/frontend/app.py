import sys
from pathlib import Path

# Ensure project root is on sys.path so `ai_research_agent` can be imported
_CURRENT_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _CURRENT_FILE.parents[3]
if str(_PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

from ai_research_agent.src.core.agent.orchestrator import run_pipeline

st.set_page_config(page_title="AI Research Agent", page_icon="📚", layout="wide")

st.title("AI Academic Research Assistant (RAG)")
st.write("Type a query (English or French), e.g., \"État de l'art en résumé automatique avec LLMs\"")

query = st.text_input("Your research query", value="")
col1, col2, col3, col4 = st.columns([1,1,1,3])
with col1:
	top_k = st.number_input("Top K", value=3, min_value=1, max_value=10)
with col2:
	max_pdfs = st.number_input("Max PDFs", value=3, min_value=1, max_value=20)
with col3:
	api = st.selectbox("API", ["arxiv", "semantic_scholar", "pubmed", "crossref"], index=0)

# Performance options
st.sidebar.header("Performance Options")
use_pdfs = st.sidebar.checkbox("Download & Process PDFs (slower but more detailed)", value=False)
if use_pdfs:
	st.sidebar.warning("PDF processing will take 2-5x longer!")

if st.button("Search"):
	if not query.strip():
		st.warning("Please enter a query.")
	else:
		with st.spinner("Searching and summarizing..."):
			result = run_pipeline(query, top_k=int(top_k), max_pdfs=int(max_pdfs), api=api, use_pdfs=use_pdfs)
		st.subheader("Results")
		st.caption(f"Detected language: {result['query_language']} | Intent: {result['intent']} | API: {result['api_used']} | Mode: {result['processing_mode']}")
		for i, item in enumerate(result['results'], start=1):
			st.markdown(f"**{i}. {item['title']}**")
			st.write(f"Authors: {', '.join(item['authors'][:6])}")
			st.write(f"Year: {item['year']} | Score: {item['score']:.3f}")
			st.write(f"Link: [PDF/Entry]({item['link']})")
			with st.expander("Abstractive summary"):
				st.write(item['abstractive_summary'])
			with st.expander("Extractive (TF-IDF + LSA)"):
				st.write(item['abstract_summary'])
			st.markdown("---") 