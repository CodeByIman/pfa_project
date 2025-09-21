'use client'

import React from 'react'
import { motion } from 'framer-motion'
import { User, Bot, ExternalLink, Calendar, Users, Star, ChevronDown, ChevronRight, Zap, Brain, Target, Tag, FileText, Download } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import { apiClient, PDFSummaryRequest, PDFSummaryResponse, NewOllamaSummaryRequest, NewOllamaSummaryResponse } from '../lib/api'

interface ChatMessageProps {
  message: string
  isUser: boolean
  timestamp?: string
  searchResults?: any
}

const ChatMessage: React.FC<ChatMessageProps> = ({ 
  message, 
  isUser, 
  timestamp, 
  searchResults 
}) => {
  const [expandedSummaries, setExpandedSummaries] = React.useState<{[key: string]: boolean}>({})
  const [expandedMetadata, setExpandedMetadata] = React.useState<{[key: string]: boolean}>({})
  const [pdfSummaries, setPdfSummaries] = React.useState<{[key: string]: PDFSummaryResponse}>({})
  const [loadingPdf, setLoadingPdf] = React.useState<{[key: string]: boolean}>({})
  const [showFinal, setShowFinal] = React.useState<{[key: string]: boolean}>({})
  const [newOllamaLoading, setNewOllamaLoading] = React.useState<{[key: string]: boolean}>({})
  const [newOllamaResults, setNewOllamaResults] = React.useState<{[key: string]: NewOllamaSummaryResponse}>({})

  // Disable auto-fetch of PDF summaries to avoid long processing; summaries will be fetched on-demand via button
  // React.useEffect(() => {}, [searchResults])

  const toggleSummary = (paperId: string, summaryType: string) => {
    const key = `${paperId}-${summaryType}`
    setExpandedSummaries(prev => ({
      ...prev,
      [key]: !prev[key]
    }))
  }

  const toggleMetadata = (paperId: string) => {
    setExpandedMetadata(prev => ({
      ...prev,
      [paperId]: !prev[paperId]
    }))
  }

  const formatTimestamp = (ts: string) => {
    return new Date(ts).toLocaleTimeString('fr-FR', {
      hour: '2-digit',
      minute: '2-digit'
    })
  }

  const getSummaryIcon = (type: string) => {
    switch (type) {
      case 'ollama': return '🦙'
      case 'abstractive': return '🔗'
      case 'tfidf': return '📊'
      case 'lsa': return '🔍'
      case 'extractive': return '📝'
      case 'combined': return '🔗'
      default: return '📝'
    }
  }

  // New: Call backend to summarize the extractive text with the fixed prompt
  const handleNewOllamaSummary = async (result: any) => {
    const paperId = result.paper_id
    setNewOllamaLoading(prev => ({ ...prev, [paperId]: true }))
    try {
      const text = result.extractive_summary || result.source_text || result.original_abstract || ''
      if (!text || text.trim().length < 30) {
        setNewOllamaResults(prev => ({ ...prev, [paperId]: { summary: 'Aucun texte extractif disponible pour ce papier.', status: 'error' } }))
        return
      }
      const req: NewOllamaSummaryRequest = {
        text,
        title: result.title,
        authors: result.authors,
        year: (result.year && String(result.year)) || undefined,
        model: 'mistral'
      }
      const resp = await apiClient.newOllamaSummary(req)
      setNewOllamaResults(prev => ({ ...prev, [paperId]: resp }))
    } catch (e) {
      setNewOllamaResults(prev => ({ ...prev, [paperId]: { summary: 'Erreur lors de la génération du nouveau résumé Ollama.', status: 'error' } }))
    } finally {
      setNewOllamaLoading(prev => ({ ...prev, [paperId]: false }))
    }
  }

  const getSummaryLabel = (type: string) => {
    switch (type) {
      case 'ollama': return 'Résumé Ollama (IA Avancée)'
      case 'abstractive': return 'Overview'
      case 'tfidf': return 'Résumé TF-IDF'
      case 'lsa': return 'Résumé LSA'
      case 'extractive': return 'Résumé Extractif (utilisé par Ollama)'
      case 'combined': return 'Résumé Combiné'
      default: return 'Résumé'
    }
  }

  const getSummaryPriority = (type: string) => {
    // Higher numbers = higher priority
    switch (type) {
      case 'ollama': return 10
      case 'abstractive': return 8
      case 'combined': return 6
      case 'tfidf': return 4
      case 'lsa': return 2
      default: return 1
    }
  }

  const handlePdfSummary = async (result: any) => {
    const paperId = result.paper_id
    setLoadingPdf(prev => ({ ...prev, [paperId]: true }))
    
    try {
      const request: PDFSummaryRequest = {
        paper_id: paperId,
        title: result.title,
        authors: result.authors,
        year: result.year?.toString() || '2024'
      }
      
      const response = await apiClient.summarizePDF(request)
      setPdfSummaries(prev => ({ ...prev, [paperId]: response }))
    } catch (error) {
      console.error('PDF summary error:', error)
      // Set error state
      setPdfSummaries(prev => ({ 
        ...prev, 
        [paperId]: {
          short_summary: 'Erreur lors du traitement du PDF',
          long_summary: {
            contributions: 'Erreur',
            methodology: 'Erreur',
            results: 'Erreur',
            limitations: 'Erreur',
            future_work: 'Erreur'
          },
          abstractive_summary: 'Traitement PDF échoué',
          status: 'error'
        }
      }))
    } finally {
      setLoadingPdf(prev => ({ ...prev, [paperId]: false }))
    }
  }

  const isOllamaMode = (searchResults: any) => {
    return searchResults?.processing_mode?.includes('fast_extractive_ollama') || 
           searchResults?.processing_mode?.includes('ollama') ||
           searchResults?.performance?.fast_mode_used
  }

  if (isUser) {
    return (
      <motion.div
        initial={{ opacity: 0, x: 20 }}
        animate={{ opacity: 1, x: 0 }}
        className="flex justify-end mb-6"
      >
        <div className="flex items-start space-x-3 max-w-3xl">
          <div className="chat-message user-message">
            <div className="flex items-center space-x-2 mb-2">
              <User className="w-4 h-4" />
              <span className="text-sm font-medium">Vous</span>
              {timestamp && (
                <span className="text-xs opacity-75">
                  {formatTimestamp(timestamp)}
                </span>
              )}
            </div>
            <p className="text-sm">{message}</p>
          </div>
        </div>
      </motion.div>
    )
  }

  return (
    <motion.div
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      className="flex justify-start mb-6"
    >
      <div className="flex items-start space-x-3 max-w-4xl">
        <div className="flex-shrink-0">
          <div className="w-8 h-8 bg-primary-100 rounded-full flex items-center justify-center">
            <Bot className="w-4 h-4 text-primary-600" />
          </div>
        </div>
        <div className="chat-message bot-message flex-1">
          <div className="flex items-center space-x-2 mb-3">
            <span className="text-sm font-medium text-gray-700">Assistant IA</span>
            {timestamp && (
              <span className="text-xs text-gray-500">
                {formatTimestamp(timestamp)}
              </span>
            )}
            {searchResults && isOllamaMode(searchResults) && (
              <span className="inline-flex items-center px-2 py-1 rounded-full text-xs bg-green-100 text-green-800">
                <Zap className="w-3 h-3 mr-1" />
                Mode Ollama Rapide
              </span>
            )}
          </div>
          
          <div className="prose prose-sm max-w-none">
            <ReactMarkdown>{message}</ReactMarkdown>
          </div>

          {searchResults && (
            <div className="mt-4 space-y-4">
              <div className="text-sm text-gray-600 bg-gray-50 p-3 rounded-lg">
                <p><strong>🔍 Recherche :</strong> {searchResults.expanded_query}</p>
                <div className="flex items-center space-x-4 mt-2">
                  <p><strong>🌐 API :</strong> {searchResults.api_used}</p>
                  <p><strong>⚙️ Mode :</strong> {searchResults.processing_mode}</p>
                  {searchResults.performance?.ollama_available && (
                    <span className="inline-flex items-center px-2 py-1 rounded-full text-xs bg-blue-100 text-blue-800">
                      🦙 Ollama Disponible
                    </span>
                  )}
                </div>
              </div>

              {searchResults.results && searchResults.results.length > 0 && (
                <div className="space-y-4">
                  <h4 className="font-semibold text-gray-800">
                    📚 {searchResults.results.length} article(s) trouvé(s)
                  </h4>
                  
                  {searchResults.results.map((result: any, index: number) => (
                    <div key={result.paper_id} className="paper-card">
                      <div className="flex items-start justify-between mb-3">
                        <h5 className="font-semibold text-lg text-gray-900 line-clamp-2">
                          {index + 1}. {result.title}
                        </h5>
                        <a
                          href={result.link}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-primary-600 hover:text-primary-700 ml-2"
                        >
                          <ExternalLink className="w-4 h-4" />
                        </a>
                      </div>

                      <div className="flex items-center space-x-4 text-sm text-gray-600 mb-3">
                        <div className="flex items-center space-x-1">
                          <Users className="w-4 h-4" />
                          <span>{result.authors.slice(0, 2).join(', ')}</span>
                          {result.authors.length > 2 && <span>+{result.authors.length - 2}</span>}
                        </div>
                        {result.year && (
                          <div className="flex items-center space-x-1">
                            <Calendar className="w-4 h-4" />
                            <span>{result.year}</span>
                          </div>
                        )}
                        <div className="flex items-center space-x-1">
                          <Star className="w-4 h-4" />
                          <span>{result.score.toFixed(3)}</span>
                        </div>
                      </div>

                      {/* Ollama Metadata (if available) */}
                      {(result.paper_focus || result.contribution || result.key_terms) && (
                        <div className="mb-3">
                          <button
                            onClick={() => toggleMetadata(result.paper_id)}
                            className="flex items-center space-x-2 text-sm font-medium text-emerald-600 hover:text-emerald-800 transition-colors"
                          >
                            {expandedMetadata[result.paper_id] ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
                            <Brain className="w-4 h-4" />
                            <span>Analyse Ollama</span>
                          </button>
                          
                          {expandedMetadata[result.paper_id] && (
                            <motion.div
                              initial={{ opacity: 0, height: 0 }}
                              animate={{ opacity: 1, height: 'auto' }}
                              exit={{ opacity: 0, height: 0 }}
                              className="mt-2 grid grid-cols-1 md:grid-cols-2 gap-3"
                            >
                              {result.paper_focus && (
                                <div className="bg-emerald-50 p-3 rounded-lg border-l-4 border-emerald-400">
                                  <div className="flex items-center space-x-2 mb-1">
                                    <Target className="w-4 h-4 text-emerald-600" />
                                    <span className="text-sm font-medium text-emerald-800">Focus</span>
                                  </div>
                                  <p className="text-sm text-emerald-700">{result.paper_focus}</p>
                                </div>
                              )}
                              
                              {result.contribution && (
                                <div className="bg-blue-50 p-3 rounded-lg border-l-4 border-blue-400">
                                  <div className="flex items-center space-x-2 mb-1">
                                    <Star className="w-4 h-4 text-blue-600" />
                                    <span className="text-sm font-medium text-blue-800">Contribution</span>
                                  </div>
                                  <p className="text-sm text-blue-700">{result.contribution}</p>
                                </div>
                              )}
                              
                              {result.key_terms && result.key_terms.length > 0 && (
                                <div className="bg-purple-50 p-3 rounded-lg border-l-4 border-purple-400 md:col-span-2">
                                  <div className="flex items-center space-x-2 mb-2">
                                    <Tag className="w-4 h-4 text-purple-600" />
                                    <span className="text-sm font-medium text-purple-800">Termes Clés</span>
                                  </div>
                                  <div className="flex flex-wrap gap-1">
                                    {result.key_terms.map((term: string, idx: number) => (
                                      <span 
                                        key={idx} 
                                        className="inline-flex items-center px-2 py-1 rounded-full text-xs bg-purple-100 text-purple-700"
                                      >
                                        {term}
                                      </span>
                                    ))}
                                  </div>
                                </div>
                              )}
                            </motion.div>
                          )}
                        </div>
                      )}

                      <div className="space-y-3">
                        {/* Final summary on demand: only show when user clicks */}
                        <div className="flex items-center gap-2">
                          <button
                            onClick={() => setShowFinal(prev => ({ ...prev, [result.paper_id]: !prev[result.paper_id] }))}
                            className="px-4 py-2 bg-emerald-600 text-white rounded-lg hover:bg-emerald-700 transition-colors text-sm"
                          >
                            {showFinal[result.paper_id] ? 'Hide overview' : 'See overview'}
                          </button>
                          {loadingPdf[result.paper_id] && (
                            <span className="text-xs text-gray-500">(Analyse PDF en cours...)</span>
                          )}
                          <button
                            onClick={() => handleNewOllamaSummary(result)}
                            disabled={newOllamaLoading[result.paper_id]}
                            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors text-sm"
                          >
                            {newOllamaLoading[result.paper_id] ? '✅ Génération...' : '✅ Résumé de la feuille de recherche'}
                          </button>
                        </div>

                        {showFinal[result.paper_id] && (
                          <div className="mt-3">
                            <h6 className="font-medium text-gray-800 mb-2 flex items-center">
                              ✨ Résumé Final (Mistral)
                              <span className="ml-2 inline-flex items-center px-2 py-1 rounded-full text-xs bg-green-100 text-green-800">
                                IA Avancée
                              </span>
                            </h6>
                            <p className={`text-sm text-gray-700 p-3 rounded-lg border-l-4 bg-green-50 border-green-400`}>
                              {result.final_response && result.final_response.trim() !== 'Final response not available'
                                ? result.final_response
                                : ( result.abstractive_summary || 'Résumé indisponible')}
                            </p>
                          </div>
                        )}

                        {/* All available summaries (hide technical ones by default) */}
                        {result.summaries && Object.keys(result.summaries).length > 0 && (
                          <div className="border-t pt-3">
                            <h6 className="font-medium text-gray-800 mb-3 flex items-center">
                              plus d'options
                            </h6>
                            
                            {Object.entries(result.summaries)
                              .filter(([type, summaryText]: [string, any]) => {
                                if (!summaryText || (typeof summaryText === 'string' && summaryText.trim() === '')) return false
                                // Hide technical summaries (tfidf, lsa, combined)
                                return !['tfidf', 'lsa', 'combined'].includes(type)
                              })
                              .sort(([typeA], [typeB]) => getSummaryPriority(typeB) - getSummaryPriority(typeA))
                              .map(([summaryType, summaryText]: [string, any]) => {
                                const key = `${result.paper_id}-${summaryType}`
                                const isExpanded = expandedSummaries[key]
                                const isMainSummary = false
                                
                                // Skip if this is the main summary already displayed
                                if (isMainSummary) return null
                                
                                return (
                                  <div key={summaryType} className="mb-3">
                                    <button
                                      onClick={() => toggleSummary(result.paper_id, summaryType)}
                                      className="flex items-center space-x-2 text-sm font-medium text-gray-600 hover:text-gray-800 transition-colors"
                                    >
                                      {isExpanded ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
                                      <span>{getSummaryIcon(summaryType)} {getSummaryLabel(summaryType)}</span>
                                    </button>
                                    
                                    {isExpanded && (
                                      <motion.div
                                        initial={{ opacity: 0, height: 0 }}
                                        animate={{ opacity: 1, height: 'auto' }}
                                        exit={{ opacity: 0, height: 0 }}
                                        className={`mt-2 text-sm text-gray-700 p-3 rounded-lg border-l-4 ${
                                          summaryType === 'ollama' ? 'bg-green-50 border-green-300' :
                                          summaryType === 'abstractive' ? 'bg-blue-50 border-blue-300' :
                                          summaryType === 'extractive' ? 'bg-yellow-50 border-yellow-300' :
                                          'bg-gray-50 border-gray-300'
                                        }`}
                                      >
                                        {summaryText}
                                      </motion.div>
                                    )}
                                  </div>
                                )
                              })}
                          </div>
                        )}

                        {/* Full PDF Summary Button
                        <div className="border-t pt-3 mt-4">
                          <button
                            onClick={() => handlePdfSummary(result)}
                            disabled={loadingPdf[result.paper_id]}
                            className="flex items-center space-x-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                          >
                            {loadingPdf[result.paper_id] ? (
                              <>
                                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                                <span>Traitement PDF...</span>
                              </>
                            ) : (
                              <>
                                <FileText className="w-4 h-4" />
                                <span>Résumé Complet PDF</span>
                                <Download className="w-4 h-4" />
                              </>
                            )}
                          </button>
                        </div> */}

                        {/* PDF Summary Results */}
                        {pdfSummaries[result.paper_id] && (
                          <div className="mt-4 border-t pt-4">
                            <h6 className="font-semibold text-gray-800 mb-3 flex items-center">
                              <FileText className="w-5 h-5 mr-2 text-blue-600" />
                              Analyse Complète du PDF
                            </h6>

                            {/* Only show the long Ollama-generated summary or error */}
                            <div className="p-4 bg-green-50 rounded-lg border-l-4 border-green-500">
                              <div className="font-semibold text-green-900 mb-2">🦙 Résumé du PDF (Ollama)</div>
                              <p className="text-sm text-green-800 whitespace-pre-wrap">
                                {pdfSummaries[result.paper_id].status === 'error'
                                  ? 'Erreur lors de la génération du résumé Ollama pour le PDF.'
                                  : (pdfSummaries[result.paper_id].abstractive_summary || 'Résumé Ollama indisponible')}
                              </p>
                            </div>
                          </div>
                        )}

                        {/* Résumé de la feuille de recherche from extractive summary */}
                        {newOllamaResults[result.paper_id] && (
                          <div className="mt-4 border-t pt-4">
                            <h6 className="font-semibold text-gray-800 mb-3 flex items-center">
                              <span className="mr-2">✅</span> Résumé de la feuille de recherche
                            </h6>
                            <div className={`p-4 rounded-lg border-l-4 ${newOllamaResults[result.paper_id].status === 'error' ? 'bg-red-50 border-red-500' : 'bg-green-50 border-green-500'}`}>
                              <p className={`text-sm ${newOllamaResults[result.paper_id].status === 'error' ? 'text-red-800' : 'text-green-800'} whitespace-pre-wrap`}>
                                {newOllamaResults[result.paper_id].summary}
                              </p>
                            </div>
                          </div>
                        )}

                        {/* Final response if available and different from main summary */}
                        {result.final_response && 
                         result.final_response !== result.ollama_summary && 
                         result.final_response !== result.abstractive_summary && 
                         result.final_response.trim() !== 'Final response not available' && (
                          <details className="group border-t pt-3">
                            <summary className="cursor-pointer text-sm font-medium text-gray-600 hover:text-gray-800 flex items-center">
                              ✨ Réponse Finale Meta-Processeur
                            </summary>
                            <div className="mt-2 text-sm text-gray-700 bg-indigo-50 p-3 rounded-lg border-l-4 border-indigo-300">
                              {result.final_response}
                            </div>
                          </details>
                        )}

                        {/* Legacy abstract summary (if different from others) */}
                        {result.abstract_summary && 
                         result.abstract_summary !== result.ollama_summary &&
                         result.abstract_summary !== result.abstractive_summary && 
                         (!result.summaries || !Object.values(result.summaries).includes(result.abstract_summary)) && (
                          <details className="group border-t pt-3">
                            <summary className="cursor-pointer text-sm font-medium text-gray-600 hover:text-gray-800 flex items-center">
                              📄 Abstract Original
                            </summary>
                            <div className="mt-2 text-sm text-gray-700 bg-yellow-50 p-3 rounded-lg border-l-4 border-yellow-300">
                              {result.abstract_summary}
                            </div>
                          </details>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </motion.div>
  )
}

export default ChatMessage