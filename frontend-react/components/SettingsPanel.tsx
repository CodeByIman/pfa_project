'use client'

import React from 'react'
import { motion } from 'framer-motion'
import { Settings, Database, FileText, Info } from 'lucide-react'

interface SettingsPanelProps {
  apiChoice: string
  setApiChoice: (api: string) => void
  usePdfs: boolean
  setUsePdfs: (use: boolean) => void
  useAbstractive: boolean
  setUseAbstractive: (use: boolean) => void
  topK: number
  setTopK: (k: number) => void
  maxPdfs: number
  setMaxPdfs: (max: number) => void
}

const SettingsPanel: React.FC<SettingsPanelProps> = ({
  apiChoice,
  setApiChoice,
  usePdfs,
  setUsePdfs,
  useAbstractive,
  setUseAbstractive,
  topK,
  setTopK,
  maxPdfs,
  setMaxPdfs,
}) => {
  const apiOptions = [
    { value: 'arxiv', label: 'arXiv', description: 'Articles scientifiques gratuits' },
    { value: 'semantic_scholar', label: 'Semantic Scholar', description: 'Base de données académique' },
    { value: 'pubmed', label: 'PubMed', description: 'Articles médicaux et biologiques' },
    { value: 'crossref', label: 'CrossRef', description: 'Métadonnées d\'articles' },
  ]

  return (
    <motion.div
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      className="bg-white rounded-lg shadow-soft p-6"
    >
      <div className="flex items-center space-x-2 mb-6">
        <Settings className="w-5 h-5 text-primary-600" />
        <h3 className="text-lg font-semibold text-gray-900">Paramètres de recherche</h3>
      </div>

      <div className="space-y-6">
        {/* API Selection */}
        <div>
          <label className="flex items-center space-x-2 mb-3">
            <Database className="w-4 h-4 text-gray-600" />
            <span className="text-sm font-medium text-gray-700">Source de données</span>
          </label>
          <div className="grid grid-cols-1 gap-2">
            {apiOptions.map((option) => (
              <label
                key={option.value}
                className={`flex items-center p-3 rounded-lg border cursor-pointer transition-colors ${
                  apiChoice === option.value
                    ? 'border-primary-500 bg-primary-50'
                    : 'border-gray-200 hover:border-gray-300'
                }`}
              >
                <input
                  type="radio"
                  name="api"
                  value={option.value}
                  checked={apiChoice === option.value}
                  onChange={(e) => setApiChoice(e.target.value)}
                  className="sr-only"
                />
                <div className="flex-1">
                  <div className="font-medium text-sm text-gray-900">
                    {option.label}
                  </div>
                  <div className="text-xs text-gray-600">
                    {option.description}
                  </div>
                </div>
              </label>
            ))}
          </div>
        </div>

        {/* PDF Processing */}
        <div>
          <label className="flex items-center space-x-2 mb-3">
            <FileText className="w-4 h-4 text-gray-600" />
            <span className="text-sm font-medium text-gray-700">Traitement des PDFs</span>
          </label>
          <div className="flex items-center space-x-3">
            <input
              type="checkbox"
              id="usePdfs"
              checked={usePdfs}
              onChange={(e) => setUsePdfs(e.target.checked)}
              className="w-4 h-4 text-primary-600 border-gray-300 rounded focus:ring-primary-500"
            />
            <label htmlFor="usePdfs" className="text-sm text-gray-700">
              Télécharger et analyser les PDFs complets
            </label>
          </div>
          {usePdfs && (
            <div className="mt-2 p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
              <div className="flex items-start space-x-2">
                <Info className="w-4 h-4 text-yellow-600 mt-0.5" />
                <div className="text-xs text-yellow-800">
                  <p className="font-medium">Mode détaillé activé</p>
                  <p>Le traitement sera 2-5x plus lent mais fournira des résumés plus complets.</p>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Abstractive Summarization Toggle */}
        <div className="space-y-3">
          <label className="block">
            <span className="text-sm font-medium text-gray-700">Résumé abstrait</span>
          </label>
          <div className="flex items-center space-x-3">
            <input
              type="checkbox"
              id="useAbstractive"
              checked={useAbstractive}
              onChange={(e) => setUseAbstractive(e.target.checked)}
              className="w-4 h-4 text-primary-600 border-gray-300 rounded focus:ring-primary-500"
            />
            <label htmlFor="useAbstractive" className="text-sm text-gray-700">
              Utiliser l'IA pour les résumés (plus lent mais meilleur)
            </label>
          </div>
          {!useAbstractive && (
            <div className="mt-2 p-3 bg-blue-50 border border-blue-200 rounded-lg">
              <div className="flex items-start space-x-2">
                <Info className="w-4 h-4 text-blue-600 mt-0.5" />
                <div className="text-xs text-blue-800">
                  <p className="font-medium">Mode ultra-rapide activé</p>
                  <p>Seuls les résumés TF-IDF et LSA seront utilisés</p>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Number of Results */}
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Nombre de résultats
            </label>
            <input
              type="number"
              min="1"
              max="10"
              value={topK}
              onChange={(e) => setTopK(parseInt(e.target.value) || 3)}
              className="input-field"
            />
          </div>
          
          {usePdfs && (
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Max PDFs à traiter
              </label>
              <input
                type="number"
                min="1"
                max="20"
                value={maxPdfs}
                onChange={(e) => setMaxPdfs(parseInt(e.target.value) || 3)}
                className="input-field"
              />
            </div>
          )}
        </div>

        {/* Tips */}
        <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg">
          <h4 className="text-sm font-medium text-blue-900 mb-2">💡 Conseils</h4>
          <ul className="text-xs text-blue-800 space-y-1">
            <li>• Utilisez des mots-clés spécifiques pour de meilleurs résultats</li>
            <li>• arXiv est recommandé pour les articles récents</li>
            <li>• PubMed est idéal pour la recherche médicale</li>
            <li>• Désactivez le traitement PDF pour des résultats plus rapides</li>
          </ul>
        </div>
      </div>
    </motion.div>
  )
}

export default SettingsPanel

