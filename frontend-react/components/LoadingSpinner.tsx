'use client'

import React from 'react'
import { motion } from 'framer-motion'
import { Search, FileText, Brain, Zap } from 'lucide-react'

interface LoadingSpinnerProps {
  message?: string
  showSteps?: boolean
}

const LoadingSpinner: React.FC<LoadingSpinnerProps> = ({ 
  message = "Recherche en cours...", 
  showSteps = true 
}) => {
  const steps = [
    { icon: Search, text: "Analyse de la requête", delay: 0 },
    { icon: FileText, text: "Recherche d'articles", delay: 0.5 },
    { icon: Brain, text: "Génération de résumés", delay: 1 },
    { icon: Zap, text: "Finalisation", delay: 1.5 },
  ]

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="flex flex-col items-center justify-center p-8"
    >
      <div className="relative">
        <div className="w-16 h-16 border-4 border-primary-200 border-t-primary-600 rounded-full animate-spin"></div>
        <div className="absolute inset-0 flex items-center justify-center">
          <Search className="w-6 h-6 text-primary-600 animate-pulse" />
        </div>
      </div>

      <motion.p
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.3 }}
        className="mt-4 text-lg font-medium text-gray-700"
      >
        {message}
      </motion.p>

      {showSteps && (
        <div className="mt-6 space-y-3 w-full max-w-md">
          {steps.map((step, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: step.delay }}
              className="flex items-center space-x-3 p-3 bg-white rounded-lg shadow-sm"
            >
              <div className="flex-shrink-0">
                <step.icon className="w-5 h-5 text-primary-600" />
              </div>
              <span className="text-sm text-gray-700">{step.text}</span>
              <div className="ml-auto">
                <div className="w-2 h-2 bg-primary-600 rounded-full animate-pulse"></div>
              </div>
            </motion.div>
          ))}
        </div>
      )}

      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 2 }}
        className="mt-4 text-xs text-gray-500"
      >
        Cela peut prendre quelques secondes...
      </motion.div>
    </motion.div>
  )
}

export default LoadingSpinner

