"""
Application FastAPI principale pour le service NLP de détection de spam
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from app.routes import nlp
from app.services.nlp_service import nlp_service

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestion du cycle de vie de l'application (startup/shutdown)"""
    # Startup : chargement du modèle
    logger.info("=" * 60)
    logger.info("🚀 Démarrage de l'application NLP Spam Detector")
    logger.info("=" * 60)
    
    try:
        nlp_service.load_model()
        logger.info("✅ Modèle chargé et prêt à servir des prédictions")
        logger.info("=" * 60)
    except Exception as e:
        logger.error(f"❌ ERREUR CRITIQUE : Échec du chargement du modèle")
        logger.error(f"   Détails : {e}")
        logger.error("=" * 60)
        raise
    
    yield
    
    # Shutdown
    logger.info("🛑 Arrêt de l'application")


# Initialisation de l'application FastAPI
app = FastAPI(
    title="API NLP - Détection de Spam",
    description="""
    API REST pour la classification automatique de textes spam/ham.
    
    ## Fonctionnalités
    
    * **Prédiction** : Classification de texte en spam ou ham
    * **Probabilités** : Scores de confiance pour chaque classe
    * **Métriques** : Accuracy du modèle entraîné
    
    ## Modèle
    
    Pipeline scikit-learn : TFIDF + Multinomial Naive Bayes
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configuration CORS (si nécessaire pour un frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # À restreindre en production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inclusion des routes
app.include_router(nlp.router)


@app.get(
    "/",
    tags=["Système"],
    summary="Endpoint racine",
    description="Retourne les informations de base sur l'API"
)
async def root():
    """Page d'accueil de l'API"""
    return {
        "message": "API NLP - Détection de Spam",
        "version": "1.0.0",
        "status": "running",
        "documentation": "/docs",
        "endpoints": {
            "predict": "/api/nlp/predict",
            "model_info": "/api/nlp/model-info",
            "health": "/health"
        }
    }


@app.get(
    "/health",
    tags=["Système"],
    summary="Health check",
    description="Vérifie l'état de santé de l'application et du modèle"
)
async def health_check():
    """Endpoint de vérification de santé"""
    model_info = nlp_service.get_model_info()
    
    return {
        "status": "healthy" if model_info["model_loaded"] else "unhealthy",
        "model_loaded": model_info["model_loaded"],
        "accuracy": model_info["accuracy"],
        "model_type": model_info["model_type"]
    }