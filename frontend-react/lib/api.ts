import axios from 'axios'

const API_BASE_URL = process.env.API_BASE_URL || 'http://localhost:8000'

export interface SearchRequest {
  query: string
  top_k?: number
  max_results?: number
  max_pdfs?: number
  api?: string
  use_pdfs?: boolean
  use_abstractive?: boolean  // New parameter for ultra-fast mode
}

export interface SearchResponse {
  query_language: string
  intent: string
  entities: {
    domain: string[]
    methods: string[]
    datasets: string[]
    metrics: string[]
    keywords: string[]
  }
  expanded_query: string
  api_used: string
  processing_mode: string
  timestamp: string
  results: Array<{
    paper_id: string
    title: string
    link: string
    year: number | null
    authors: string[]
    score: number
    abstract_summary: string
    abstractive_summary: string
  }>
}

export interface PDFSummaryRequest {
  paper_id?: string
  pdf_url?: string
  title: string
  authors: string[]
  year: string
}

export interface PDFSummaryResponse {
  short_summary: string
  long_summary: {
    contributions: string
    methodology: string
    results: string
    limitations: string
    future_work: string
  }
  abstractive_summary: string
  status: string
}

export interface FeedbackRequest {
  query: string
  paper_id: string
  relevant: boolean
  notes?: string
}

class ApiClient {
  private client = axios.create({
    baseURL: API_BASE_URL,
    timeout: 900000, // Increased to 2 minutes for complex queries
    headers: {
      'Content-Type': 'application/json',
    },
  })

  async search(params: SearchRequest): Promise<SearchResponse> {
    try {
      const response = await this.client.post('/search', params)
      return response.data
    } catch (error) {
      console.error('Search API error:', error)
      throw new Error('Erreur lors de la recherche. Veuillez réessayer.')
    }
  }

  async sendFeedback(params: FeedbackRequest): Promise<{ status: string }> {
    try {
      const response = await this.client.post('/feedback', params)
      return response.data
    } catch (error) {
      console.error('Feedback API error:', error)
      throw new Error('Erreur lors de l\'envoi du feedback.')
    }
  }

  async summarizePDF(params: PDFSummaryRequest): Promise<PDFSummaryResponse> {
    try {
      const response = await this.client.post('/summarize_pdf', params)
      return response.data
    } catch (error) {
      console.error('PDF Summary API error:', error)
      throw new Error('Erreur lors du résumé PDF. Veuillez réessayer.')
    }
  }

  async healthCheck(): Promise<{ status: string }> {
    try {
      const response = await this.client.get('/health')
      return response.data
    } catch (error) {
      console.error('Health check error:', error)
      throw new Error('API non disponible')
    }
  }
}

export const apiClient = new ApiClient()

