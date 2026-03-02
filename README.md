# 🤖 AI Agent Server - Phase 1

Agent IA intelligent pour l'automatisation des tests unitaires Android.

## 📋 Vue d'ensemble

Cet agent IA analyse les échecs de tests unitaires Android (Kotlin + JUnit + MockK) et génère automatiquement des corrections.

**Technologies:**
- **Framework:** FastAPI (Python)
- **IA:** Groq Llama 3.3 70B Versatile
- **Cache:** TTLCache (in-memory)
- **Logging:** Loguru

## 🚀 Installation Rapide

### 1. Prérequis

```bash
# Python 3.11 ou supérieur
python --version

# pip
pip --version
```

### 2. Installation

```bash
# Créer un environnement virtuel
python -m venv venv

# Activer l'environnement virtuel
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Installer les dépendances
pip install -r requirements.txt
```

### 3. Configuration

```bash
# Copier le fichier d'exemple
cp .env.example .env

# Éditer .env et ajouter votre clé API Groq
# Obtenir une clé API: https://console.groq.com/keys
```

**Variables critiques dans `.env`:**
```env
GROQ_API_KEY=gsk_your_api_key_here
GIT_REPO_PATH=C:/Users/Stayha/Desktop/pfe/SmartTalk-Android
```

### 4. Créer le dossier logs

```bash
mkdir logs
```

## ▶️ Démarrage

```bash
# Activer l'environnement virtuel (si pas déjà fait)
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Démarrer le serveur
python main.py
```

Le serveur démarre sur: **http://localhost:8000**

**URLs importantes:**
- 📚 Documentation API: http://localhost:8000/docs
- 🏥 Health check: http://localhost:8000/health
- 📊 Statistiques: http://localhost:8000/stats

## 📡 API Endpoints

### 1. Analyser un échec de test

**POST** `/analyze-failure`

**Body:**
```json
{
  "test_file": "app/src/test/.../UserManagerTest.kt",
  "test_name": "testUserLogin",
  "test_code": "@Test\nfun testUserLogin() {\n    val user = userManager.login(\"test@example.com\", \"password\")\n    assertEquals(\"test@example.com\", user.email)\n}",
  "error_logs": "java.lang.NullPointerException: userManager is null\n    at com.smarttalk.UserManagerTest.testUserLogin(UserManagerTest.kt:25)"
}
```

**Response:**
```json
{
  "success": true,
  "correction_id": "corr_20260218_143022",
  "corrected_code": "@Test\nfun testUserLogin() {\n    mockkStatic(UserManager::class)\n    every { UserManager.getInstance() } returns mockUserManager\n    val user = userManager.login(\"test@example.com\", \"password\")\n    assertEquals(\"test@example.com\", user.email)\n}",
  "confidence": 0.85,
  "tokens_used": 450
}
```

### 2. Approuver/Rejeter une correction

**POST** `/approve-correction`

**Body:**
```json
{
  "correction_id": "corr_20260218_143022",
  "approved": true,
  "feedback": "Looks good!"
}
```

### 3. Obtenir les statistiques

**GET** `/stats`

**Response:**
```json
{
  "groq_usage": {
    "tokens_used_day": 1250,
    "rate_limit_day": 14400,
    "percentage_used_day": 8.68
  },
  "cache": {
    "size": 3,
    "enabled": true
  }
}
```

## 🧪 Test Manuel

### Tester avec curl

```bash
# Test simple
curl -X POST "http://localhost:8000/analyze-failure" \
  -H "Content-Type: application/json" \
  -d '{
    "test_file": "UserManagerTest.kt",
    "test_name": "testUserLogin",
    "test_code": "@Test\nfun testUserLogin() {\n    val user = userManager.login(\"test\", \"pass\")\n    assertNotNull(user)\n}",
    "error_logs": "NullPointerException: userManager is null"
  }'
```

### Tester avec Python

```python
import requests

response = requests.post(
    "http://localhost:8000/analyze-failure",
    json={
        "test_file": "UserManagerTest.kt",
        "test_name": "testUserLogin",
        "test_code": "@Test\nfun testUserLogin() { ... }",
        "error_logs": "NullPointerException: userManager is null"
    }
)

result = response.json()
print("Success:", result["success"])
print("Corrected code:", result["corrected_code"])
print("Confidence:", result["confidence"])
```

## 📊 Monitoring

### Vérifier la santé du serveur

```bash
curl http://localhost:8000/health
```

### Voir les logs

```bash
# Logs en temps réel
tail -f logs/agent_ia.log

# Windows
type logs\agent_ia.log
```

## 🔧 Configuration Avancée

### Ajuster les limites de rate

Dans `.env`:
```env
RATE_LIMIT_TOKENS_PER_MINUTE=6000  # Free tier Groq
RATE_LIMIT_TOKENS_PER_DAY=14400    # Free tier Groq
```

### Désactiver le cache

Dans `.env`:
```env
CACHE_ENABLED=false
```

### Changer le niveau de logs

Dans `.env`:
```env
LOG_LEVEL=DEBUG  # DEBUG, INFO, WARNING, ERROR
```

## ⚠️ Limites Actuelles (Phase 1)

- ✅ Analyse et correction de tests via API
- ✅ Rate limiting et cache
- ❌ Pas encore d'intégration Jenkins (Phase 3)
- ❌ Pas encore de système RAG (Phase 2)
- ❌ Pas encore d'application automatique Git (Phase 3)
- ❌ Pas encore d'interface UI (Phase 4)

## 🐛 Troubleshooting

### Erreur: "GROQ_API_KEY must be set"

**Solution:** Vérifiez que votre `.env` contient une clé API valide:
```env
GROQ_API_KEY=gsk_xxxxxxxxxxxxx
```

### Erreur: "Git repository not found"

**Solution:** Vérifiez le chemin dans `.env`:
```env
GIT_REPO_PATH=C:/Users/Stayha/Desktop/pfe/SmartTalk-Android
```

### Erreur: "Rate limit exceeded"

**Solution:** Attendez que le compteur se réinitialise (1 minute ou 1 jour selon le type de limite).

Vérifiez l'utilisation:
```bash
curl http://localhost:8000/stats
```

### Le serveur ne démarre pas

**Vérifications:**
1. L'environnement virtuel est activé?
2. Les dépendances sont installées? `pip install -r requirements.txt`
3. Le dossier `logs/` existe?
4. Le port 8000 est libre? Essayez un autre port dans `.env`

## 📈 Prochaines Étapes

### Phase 2: Système RAG
- Indexation du code source
- Recherche de corrections similaires
- Amélioration de la qualité des corrections

### Phase 3: Intégration Jenkins
- Webhook automatique
- Application Git automatique
- Re-tests automatiques

### Phase 4: Interface Développeur
- Dashboard web
- Révision des corrections
- Visualisation des diffs

## 📝 Notes de Développement

### Architecture

```
┌─────────────┐
│   Jenkins   │
└──────┬──────┘
       │ (Phase 3)
       ↓
┌─────────────┐
│  FastAPI    │ ← PHASE 1 (ACTUELLE)
│   Server    │
└──────┬──────┘
       │
       ↓
┌─────────────┐
│  Groq API   │
│ Llama 3.3   │
└─────────────┘
```

### Structure du Code

```
agent-ia-server/
├── main.py              # FastAPI app + endpoints
├── groq_client.py       # Client Groq avec rate limiting
├── config.py            # Configuration
├── requirements.txt     # Dépendances
└── .env                 # Variables d'environnement (ne pas commit!)
```

## 🎯 Métriques de Succès Phase 1

- ✅ Serveur FastAPI fonctionnel
- ✅ Intégration Groq API
- ✅ Rate limiting actif
- ✅ Cache fonctionnel
- ✅ Génération de corrections pour tests MockK
- ✅ API documentée (Swagger)

## 📞 Support

Pour toute question ou problème, consultez:
- Documentation Groq: https://console.groq.com/docs
- Documentation FastAPI: https://fastapi.tiangolo.com/
- Logs du serveur: `logs/agent_ia.log`

---

**Version:** 1.0.0 - Phase 1  
**Dernière mise à jour:** 18 Février 2026
