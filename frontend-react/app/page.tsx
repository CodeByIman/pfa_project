'use client'

import React, { useState, useRef, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Send, Trash2, MessageSquare, Sparkles } from 'lucide-react'
import ChatMessage from '@/components/ChatMessage'
import LoadingSpinner from '@/components/LoadingSpinner'
import SettingsPanel from '@/components/SettingsPanel'
import { apiClient, SearchRequest, SearchResponse } from '@/lib/api'

interface ChatMessage {
  id: string
  message: string
  isUser: boolean
  timestamp: string
  searchResults?: SearchResponse
}

export default function Home() {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [inputMessage, setInputMessage] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [apiChoice, setApiChoice] = useState('arxiv')
  const [usePdfs, setUsePdfs] = useState(false)
  const [useAbstractive, setUseAbstractive] = useState(true)
  const [topK, setTopK] = useState(3)
  const [maxPdfs, setMaxPdfs] = useState(3)
  const [showSettings, setShowSettings] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const formatSearchResponse = (response: SearchResponse): string => {
    let formatted = `🔍 **Recherche effectuée avec succès !**\n\n`
    formatted += `📊 **Détails de la recherche :**\n`
    formatted += `• Langue détectée : ${response.query_language}\n`
    formatted += `• Intent : ${response.intent}\n`
    formatted += `• API utilisée : ${response.api_used}\n`
    formatted += `• Mode de traitement : ${response.processing_mode}\n\n`

    if (response.results && response.results.length > 0) {
      formatted += `📚 **${response.results.length} article(s) trouvé(s) :**\n\n`
      
      response.results.forEach((result, index) => {
        formatted += `**${index + 1}. ${result.title}**\n`
        formatted += `👥 Auteurs : ${result.authors.slice(0, 3).join(', ')}${result.authors.length > 3 ? '...' : ''}\n`
        formatted += `📅 Année : ${result.year || 'N/A'} | ⭐ Score : ${result.score.toFixed(3)}\n`
        formatted += `🔗 [Lire l'article](${result.link})\n\n`
        formatted += `📝 **Résumé :**\n${result.abstractive_summary}\n\n`
        formatted += `---\n\n`
      })
    } else {
      formatted += `❌ Aucun résultat trouvé pour cette requête.\n`
      formatted += `Essayez de reformuler votre question ou d'utiliser des mots-clés différents.`
    }

    return formatted
  }

  const handleSendMessage = async () => {
    if (!inputMessage.trim() || isLoading) return

    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      message: inputMessage,
      isUser: true,
      timestamp: new Date().toISOString(),
    }

    setMessages(prev => [...prev, userMessage])
    setInputMessage('')
    setIsLoading(true)

    try {
      const searchRequest: SearchRequest = {
        query: inputMessage,
        top_k: topK,
        max_results: 20,
        max_pdfs: maxPdfs,
        api: apiChoice,
        use_pdfs: usePdfs,
        use_abstractive: useAbstractive,
      }

      const response = await apiClient.search(searchRequest)
      
      const botMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        message: formatSearchResponse(response),
        isUser: false,
        timestamp: new Date().toISOString(),
        searchResults: response,
      }

      setMessages(prev => [...prev, botMessage])
    } catch (error) {
      const errorMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        message: `❌ **Erreur lors de la recherche**\n\n${error instanceof Error ? error.message : 'Une erreur inattendue s\'est produite.'}\n\nVeuillez réessayer ou vérifier que l'API est bien démarrée.`,
        isUser: false,
        timestamp: new Date().toISOString(),
      }
      setMessages(prev => [...prev, errorMessage])
    } finally {
      setIsLoading(false)
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSendMessage()
    }
  }

  const clearChat = () => {
    setMessages([])
  }

  const exampleQueries = [
    "État de l'art en apprentissage profond pour le diagnostic médical",
    "Recent advances in natural language processing with transformers",
    "Machine learning applications in healthcare 2023",
    "Deep learning for computer vision: latest developments",
  ]

  return (
    <div className="min-h-screen flex flex-col">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center space-x-3">
              <div className="w-8 h-8 bg-gradient-to-r from-primary-500 to-primary-700 rounded-lg flex items-center justify-center">
                <Sparkles className="w-5 h-5 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-gradient">AI Research Agent</h1>
                <p className="text-sm text-gray-600">Assistant de recherche académique intelligent</p>
              </div>
            </div>
            
            <div className="flex items-center space-x-4">
              <button
                onClick={() => setShowSettings(!showSettings)}
                className="btn-secondary"
              >
                Paramètres
              </button>
              <button
                onClick={clearChat}
                className="btn-secondary"
                disabled={messages.length === 0}
              >
                <Trash2 className="w-4 h-4 mr-2" />
                Effacer
              </button>
            </div>
          </div>
        </div>
      </header>

      <div className="flex-1 flex max-w-7xl mx-auto w-full">
        {/* Main Chat Area */}
        <div className="flex-1 flex flex-col">
          {/* Chat Messages */}
          <div className="flex-1 overflow-y-auto p-6">
            {messages.length === 0 ? (
              <div className="flex flex-col items-center justify-center h-full text-center">
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="max-w-2xl"
                >
                  <div className="w-16 h-16 bg-primary-100 rounded-full flex items-center justify-center mx-auto mb-6">
                    <MessageSquare className="w-8 h-8 text-primary-600" />
                  </div>
                  <h2 className="text-2xl font-bold text-gray-900 mb-4">
                    Bienvenue dans l'Assistant de Recherche IA
                  </h2>
                  <p className="text-gray-600 mb-8">
                    Posez vos questions de recherche et obtenez des résumés d'articles scientifiques pertinents.
                  </p>
                  
                  <div className="space-y-3">
                    <h3 className="text-lg font-semibold text-gray-800">Exemples de questions :</h3>
                    {exampleQueries.map((query, index) => (
                      <motion.button
                        key={index}
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: index * 0.1 }}
                        onClick={() => setInputMessage(query)}
                        className="block w-full p-4 text-left bg-white border border-gray-200 rounded-lg hover:border-primary-300 hover:shadow-sm transition-all"
                      >
                        <span className="text-sm text-gray-700">{query}</span>
                      </motion.button>
                    ))}
                  </div>
                </motion.div>
              </div>
            ) : (
              <div className="space-y-4">
                <AnimatePresence>
                  {messages.map((message) => (
                    <ChatMessage
                      key={message.id}
                      message={message.message}
                      isUser={message.isUser}
                      timestamp={message.timestamp}
                      searchResults={message.searchResults}
                    />
                  ))}
                </AnimatePresence>
                
                {isLoading && <LoadingSpinner />}
                <div ref={messagesEndRef} />
              </div>
            )}
          </div>

          {/* Input Area */}
          <div className="border-t border-gray-200 bg-white p-6">
            <div className="flex space-x-4">
              <div className="flex-1">
                <textarea
                  value={inputMessage}
                  onChange={(e) => setInputMessage(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder="Tapez votre question de recherche ici... (Appuyez sur Entrée pour envoyer)"
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent resize-none"
                  rows={2}
                  disabled={isLoading}
                />
              </div>
              <button
                onClick={handleSendMessage}
                disabled={!inputMessage.trim() || isLoading}
                className="btn-primary flex items-center space-x-2 px-6"
              >
                <Send className="w-4 h-4" />
                <span>Envoyer</span>
              </button>
            </div>
          </div>
        </div>

        {/* Settings Panel */}
        <AnimatePresence>
          {showSettings && (
            <motion.div
              initial={{ opacity: 0, x: 300 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 300 }}
              className="w-80 border-l border-gray-200 bg-gray-50 p-6 overflow-y-auto"
            >
              <SettingsPanel
                apiChoice={apiChoice}
                setApiChoice={setApiChoice}
                usePdfs={usePdfs}
                setUsePdfs={setUsePdfs}
                useAbstractive={useAbstractive}
                setUseAbstractive={setUseAbstractive}
                topK={topK}
                setTopK={setTopK}
                maxPdfs={maxPdfs}
                setMaxPdfs={setMaxPdfs}
              />
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  )
}

