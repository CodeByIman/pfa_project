# AI Research Agent - Frontend React

Interface React moderne pour l'Assistant de Recherche Académique IA utilisant Next.js, TypeScript et Tailwind CSS.

## 🚀 Fonctionnalités

- **💬 Interface Chat** - Style ChatGPT avec bulles de conversation
- **🎨 Design Moderne** - Interface responsive avec animations
- **⚙️ Paramètres Avancés** - Choix d'API, mode PDF, nombre de résultats
- **📱 Responsive** - Optimisé pour desktop et mobile
- **🔄 Temps Réel** - Connexion directe à l'API FastAPI
- **📊 Résultats Riches** - Affichage structuré des articles et résumés

## 🛠️ Technologies

- **Next.js 14** - Framework React avec App Router
- **TypeScript** - Typage statique
- **Tailwind CSS** - Framework CSS utilitaire
- **Framer Motion** - Animations fluides
- **Axios** - Client HTTP
- **React Markdown** - Rendu Markdown
- **Lucide React** - Icônes modernes

## 📦 Installation

```bash
# Aller dans le dossier frontend
cd ai_research_agent/frontend-react

# Installer les dépendances
npm install

# Copier le fichier d'environnement
cp .env.example .env.local

# Modifier l'URL de l'API si nécessaire
# Dans .env.local :
# API_BASE_URL=http://localhost:8000
```

## 🚀 Démarrage

```bash
# Mode développement
npm run dev

# Build de production
npm run build
npm start
```

L'application sera disponible sur `http://localhost:3000`

## 🔧 Configuration

### Variables d'environnement

Créer un fichier `.env.local` :

```env
# URL de l'API FastAPI
API_BASE_URL=http://localhost:8000

# Configuration de l'app
NEXT_PUBLIC_APP_NAME=AI Research Agent
NEXT_PUBLIC_APP_VERSION=1.0.0
```

### API Backend

Assurez-vous que l'API FastAPI est démarrée :

```bash
# Dans le dossier principal
uvicorn ai_research_agent.src.api.main:app --reload --port 8000
```

## 🎨 Personnalisation

### Couleurs

Modifier `tailwind.config.js` pour changer le thème :

```javascript
theme: {
  extend: {
    colors: {
      primary: {
        // Vos couleurs personnalisées
      }
    }
  }
}
```

### Composants

- `components/ChatMessage.tsx` - Messages de chat
- `components/LoadingSpinner.tsx` - Animation de chargement
- `components/SettingsPanel.tsx` - Panneau de paramètres
- `lib/api.ts` - Client API

## 📱 Responsive Design

L'interface s'adapte automatiquement :
- **Desktop** : Layout en colonnes avec panneau de paramètres
- **Tablet** : Layout adaptatif
- **Mobile** : Interface optimisée tactile

## 🔌 Intégration API

Le frontend communique avec l'API FastAPI via :

- `POST /search` - Recherche d'articles
- `POST /feedback` - Envoi de feedback
- `GET /health` - Vérification de santé

## 🚀 Déploiement

### Vercel (Recommandé)

```bash
# Installer Vercel CLI
npm i -g vercel

# Déployer
vercel
```

### Docker

```dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
RUN npm run build
EXPOSE 3000
CMD ["npm", "start"]
```

## 🧪 Tests

```bash
# Tests unitaires
npm test

# Tests E2E
npm run test:e2e

# Linting
npm run lint
```

## 📊 Performance

- **Lazy Loading** - Composants chargés à la demande
- **Code Splitting** - Bundles optimisés
- **Image Optimization** - Images optimisées automatiquement
- **Caching** - Cache intelligent des requêtes API

## 🔒 Sécurité

- **CORS** - Configuration sécurisée
- **XSS Protection** - Sanitisation des entrées
- **CSRF Protection** - Protection contre les attaques CSRF
- **Environment Variables** - Variables sensibles sécurisées

## 📈 Monitoring

- **Error Tracking** - Gestion des erreurs
- **Performance Monitoring** - Métriques de performance
- **Analytics** - Suivi d'utilisation (optionnel)

## 🤝 Contribution

1. Fork le projet
2. Créer une branche feature
3. Commit les changements
4. Push vers la branche
5. Ouvrir une Pull Request

## 📄 Licence

MIT License - Voir le fichier LICENSE pour plus de détails.

