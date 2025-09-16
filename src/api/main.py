from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Any, Dict, List
from pathlib import Path

from ..core.agent.orchestrator import run_pipeline, get_data_dir
from ..core.evaluation.metrics import log_human_feedback
from ..core.generation.structured_pdf_summarizer import download_and_process_pdf, StructuredSummary


class SearchRequest(BaseModel):
	query: str
	top_k: int = 5
	max_results: int = 20
	max_pdfs: int = 3
	api: str = 'arxiv'
	use_pdfs: bool = False
	summary_mode: str = "auto"  # "tfidf", "lsa", "abstractive", "mistral", "auto"


class PDFSummaryRequest(BaseModel):
	paper_id: Optional[str] = None
	pdf_url: Optional[str] = None
	title: str
	authors: List[str]
	year: str


class PDFSummaryResponse(BaseModel):
	short_summary: str
	long_summary: Dict[str, str]
	abstractive_summary: str
	status: str


class FeedbackRequest(BaseModel):
	query: str
	paper_id: str
	relevant: bool
	notes: Optional[str] = ''


app = FastAPI(title='AI Research Agent', version='0.1.0')

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get('/health')
def health() -> Dict[str, Any]:
	return {'status': 'ok'}


@app.post('/search')
def search(req: SearchRequest) -> Dict[str, Any]:
	result = run_pipeline(
		query=req.query,
		max_results=req.max_results,
		max_pdfs=req.max_pdfs,
		top_k=req.top_k,
		api=req.api,
		use_pdfs=req.use_pdfs,
		summary_mode=req.summary_mode,
		use_mistral_final_response=False,  # Disabled - using Ollama instead
		use_ollama_fast=True  # Enable new Smart Extractive + Ollama pipeline
	)
	return result


@app.post('/summarize_pdf')
def summarize_pdf(req: PDFSummaryRequest) -> PDFSummaryResponse:
	"""
	Generate structured PDF summary with extractive sections and abstractive rewrite.
	"""
	try:
		# Validate input
		if not req.pdf_url and not req.paper_id:
			raise HTTPException(status_code=400, detail="Either pdf_url or paper_id must be provided")
		
		if not req.title or not req.authors:
			raise HTTPException(status_code=400, detail="Title and authors are required")
		
		# Use PDF URL directly or construct from paper_id
		pdf_url = req.pdf_url
		if not pdf_url and req.paper_id:
			# Extract ArXiv ID from various formats
			paper_id = req.paper_id
			
			# Handle different ArXiv ID formats
			if 'arxiv.org/abs/' in paper_id:
				# Extract from full URL: http://arxiv.org/abs/2204.09719
				arxiv_id = paper_id.split('/abs/')[-1]
			elif 'arxiv:' in paper_id.lower():
				# Extract from arxiv:2204.09719 format
				arxiv_id = paper_id.replace('arxiv:', '').replace('ArXiv:', '')
			else:
				# Assume it's already just the ID
				arxiv_id = paper_id
			
			# Clean up version numbers
			arxiv_id = arxiv_id.replace('v1', '').replace('v2', '').replace('v3', '').replace('v4', '').strip()
			
			# Construct proper PDF URL
			pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
		
		# Process the PDF
		cache_dir = get_data_dir() / "cache" / "pdfs"
		structured_summary = download_and_process_pdf(
			pdf_url=pdf_url,
			title=req.title,
			authors=req.authors,
			year=req.year,
			cache_dir=cache_dir
		)
		
		# Format response
		return PDFSummaryResponse(
			short_summary=structured_summary.short_overview,
			long_summary={
				"contributions": structured_summary.contributions,
				"methodology": structured_summary.methodology,
				"results": structured_summary.results,
				"limitations": structured_summary.limitations,
				"future_work": structured_summary.future_work
			},
			abstractive_summary=structured_summary.abstractive_summary,
			status="success"
		)
		
	except Exception as e:
		# Return error response
		return PDFSummaryResponse(
			short_summary=f"Error processing PDF: {str(e)}",
			long_summary={
				"contributions": "Error occurred",
				"methodology": "Error occurred", 
				"results": "Error occurred",
				"limitations": "Error occurred",
				"future_work": "Error occurred"
			},
			abstractive_summary="PDF processing failed",
			status="error"
		)


@app.post('/feedback')
def feedback(req: FeedbackRequest) -> Dict[str, Any]:
	log_human_feedback(get_data_dir(), req.query, req.paper_id, req.relevant, req.notes or '')
	return {'status': 'recorded'} 