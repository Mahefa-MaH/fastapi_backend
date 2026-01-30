"""
Routes NLP : endpoints pour la prédiction spam/ham
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Optional
import logging

from app.services.nlp_service import nlp_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/nlp", tags=["NLP"])


# Modèles Pydantic
class PredictionRequest(BaseModel):
    """Requête de prédiction"""
    text: str = Field(
        ..., 
        min_length=1,
        description="Texte à classifier",
        example="Félicitations! Vous avez gagné 1000€. Cliquez ici maintenant!"
    )


class PredictionResponse(BaseModel):
    """Réponse de prédiction"""
    label: str = Field(..., description="Classification : 'spam' ou 'ham'")
    accuracy: Optional[float] = Field(None, description="Précision du modèle")
    confidence: Optional[float] = Field(None, description="Confiance de la prédiction (0-1)")
    probabilities: Optional[Dict[str, float]] = Field(
        None, 
        description="Probabilités pour chaque classe"
    )
    text_length: int = Field(..., description="Longueur du texte analysé")
    raw_prediction: Optional[str] = Field(None, description="Prédiction brute du modèle (debug)")
    
    
class ModelInfoResponse(BaseModel):
    """Informations sur le modèle"""
    model_loaded: bool = Field(..., description="Statut du chargement du modèle")
    accuracy: Optional[float] = Field(None, description="Précision du modèle")
    model_type: Optional[str] = Field(None, description="Type du modèle")


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Classifier un texte",
    description="Analyse un texte et retourne la classification spam/ham avec les probabilités"
)
async def predict(request: PredictionRequest):
    """
    Endpoint principal de prédiction
    
    Args:
        request: Objet JSON contenant le texte à classifier
        
    Returns:
        JSON avec label, accuracy et probabilités
        
    Raises:
        HTTPException 422: Si le texte est invalide
        HTTPException 500: En cas d'erreur interne
    """
    try:
        # Prédiction via le service
        result = nlp_service.predict(request.text)
        
        # Enrichissement de la réponse
        result["text_length"] = len(request.text)
        
        return PredictionResponse(**result)
        
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=422, detail=str(e))
        
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction : {e}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"Erreur interne du serveur : {str(e)}"
        )


@router.get(
    "/model-info",
    response_model=ModelInfoResponse,
    summary="Informations sur le modèle",
    description="Retourne les métadonnées et le statut du modèle chargé"
)
async def get_model_info():
    """
    Endpoint pour récupérer les informations du modèle
    
    Returns:
        JSON avec les métadonnées du modèle
    """
    try:
        info = nlp_service.get_model_info()
        return ModelInfoResponse(**info)
        
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des infos : {e}")
        raise HTTPException(status_code=500, detail=str(e))