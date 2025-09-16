'use client'

import React from 'react'
import { motion } from 'framer-motion'
import { User, Bot, ExternalLink, Calendar, Users, Star, ChevronDown, ChevronRight, Zap, Brain, Target, Tag, FileText, Download } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import { apiClient, PDFSummaryRequest, PDFSummaryResponse } from '../lib/api'

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
      case 'abstractive': return '🤖'
      case 'tfidf': return '📊'
      case 'lsa': return '🔍'
      case 'combined': return '🔗'
      default: return '📝'
    }
  }

  const getSummaryLabel = (type: string) => {
    switch (type) {
      case 'ollama': return 'Résumé Ollama (IA Avancée)'
      case 'abstractive': return 'Résumé Abstractif (IA)'
      case 'tfidf': return 'Résumé TF-IDF'
      case 'lsa': return 'Résumé LSA'
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
                        {/* Primary summary - prioritize Ollama if available */}
                        {(result.ollama_summary || result.abstractive_summary) && (
                          <div>
                            <h6 className="font-medium text-gray-800 mb-2 flex items-center">
                              {result.ollama_summary ? (
                                <>
                                  🦙 Résumé Ollama (Principal)
                                  <span className="ml-2 inline-flex items-center px-2 py-1 rounded-full text-xs bg-green-100 text-green-800">
                                    IA Avancée
                                  </span>
                                </>
                              ) : (
                                '🤖 Résumé Abstractif (IA) - Principal'
                              )}
                            </h6>
                            <p className="text-sm text-gray-700 bg-green-50 p-3 rounded-lg border-l-4 border-green-400">
                              {result.ollama_summary || result.abstractive_summary}
                            </p>
                          </div>
                        )}

                        {/* All available summaries */}
                        {result.summaries && Object.keys(result.summaries).length > 0 && (
                          <div className="border-t pt-3">
                            <h6 className="font-medium text-gray-800 mb-3 flex items-center">
                              📋 Tous les résumés disponibles
                            </h6>
                            
                            {Object.entries(result.summaries)
                              .filter(([_, summaryText]: [string, any]) => summaryText && summaryText.trim() !== '')
                              .sort(([typeA], [typeB]) => getSummaryPriority(typeB) - getSummaryPriority(typeA))
                              .map(([summaryType, summaryText]: [string, any]) => {
                                const key = `${result.paper_id}-${summaryType}`
                                const isExpanded = expandedSummaries[key]
                                const isMainSummary = summaryType === 'ollama' && result.ollama_summary
                                
                                // Skip if this is the main summary already displayed
                                if (isMainSummary && result.ollama_summary) return null
                                
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

                        {/* Full PDF Summary Button */}
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
                        </div>

                        {/* PDF Summary Results */}
                        {pdfSummaries[result.paper_id] && (
                          <div className="mt-4 border-t pt-4">
                            <h6 className="font-semibold text-gray-800 mb-3 flex items-center">
                              <FileText className="w-5 h-5 mr-2 text-blue-600" />
                              Analyse Complète du PDF
                            </h6>
                            
                            {/* Short Overview */}
                            <div className="mb-4 p-3 bg-blue-50 rounded-lg border-l-4 border-blue-400">
                              <div className="font-medium text-blue-800 mb-2">📋 Résumé Court</div>
                              <p className="text-sm text-blue-700">{pdfSummaries[result.paper_id].short_summary}</p>
                            </div>

                            {/* Structured Long Summary */}
                            <div className="space-y-3 mb-4">
                              <div className="font-medium text-gray-800">📚 Analyse Structurée</div>
                              
                              {Object.entries(pdfSummaries[result.paper_id].long_summary).map(([section, content]) => (
                                <div key={section} className="p-3 bg-gray-50 rounded-lg border-l-4 border-gray-300">
                                  <div className="font-medium text-gray-700 mb-1 capitalize">
                                    {section === 'contributions' && '🎯 Contributions'}
                                    {section === 'methodology' && '🔬 Méthodologie'}
                                    {section === 'results' && '📊 Résultats'}
                                    {section === 'limitations' && '⚠️ Limitations'}
                                    {section === 'future_work' && '🚀 Travaux Futurs'}
                                  </div>
                                  <p className="text-sm text-gray-600">{content}</p>
                                </div>
                              ))}
                            </div>

                            {/* Abstractive Summary */}
                            <div className="p-3 bg-green-50 rounded-lg border-l-4 border-green-400">
                              <div className="font-medium text-green-800 mb-2">🦙 Résumé Abstractif (Ollama)</div>
                              <p className="text-sm text-green-700">{pdfSummaries[result.paper_id].abstractive_summary}</p>
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