import json
import joblib
from pathlib import Path
from typing import Dict, Optional, Union
import logging
import numpy as np

logger = logging.getLogger(__name__)


class NLPService:
    """Service pour charger et utiliser le modèle de classification spam/ham"""
    
    def __init__(self):
        self.model = None
        self.accuracy: Optional[float] = None
        self.classes_: Optional[np.ndarray] = None
        self.model_path = Path(__file__).parent.parent / "models" / "spam_pipeline_fr_nb.joblib"
        self.metrics_path = Path(__file__).parent.parent / "models" / "metrics.json"
    
    def load_model(self) -> None:
        try:
            # Vérification de l'existence des fichiers
            if not self.model_path.exists():
                raise FileNotFoundError(
                    f"Modèle introuvable : {self.model_path.absolute()}"
                )
            
            if not self.metrics_path.exists():
                raise FileNotFoundError(
                    f"Fichier metrics.json introuvable : {self.metrics_path.absolute()}"
                )
            
            # Chargement du modèle
            logger.info(f"Chargement du modèle depuis {self.model_path}")
            self.model = joblib.load(self.model_path)
            logger.info("✓ Modèle chargé avec succès")
            
            # Récupération des classes
            if hasattr(self.model, 'classes_'):
                self.classes_ = self.model.classes_
                logger.info(f"✓ Classes du modèle : {self.classes_}")
                logger.info(f"✓ Type des classes : {type(self.classes_[0])}")
            else:
                logger.warning("⚠️  Le modèle n'a pas d'attribut 'classes_'")
            
            # Chargement de l'accuracy
            with open(self.metrics_path, 'r', encoding='utf-8') as f:
                metrics = json.load(f)
                self.accuracy = metrics.get("accuracy")
                
            if self.accuracy is None:
                logger.warning("Accuracy non trouvée dans metrics.json")
            else:
                logger.info(f"✓ Accuracy chargée : {self.accuracy:.4f}")
                
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle : {e}")
            raise
    
    def predict(self, text: str) -> Dict[str, any]:
     
        if not text or not text.strip():
            raise ValueError("Le texte ne peut pas être vide")
        
        if self.model is None:
            raise ValueError("Le modèle n'est pas chargé")
        
        # Prédiction brute
        raw_prediction = self.model.predict([text])[0]
        
        # Initialisation des variables
        label = None
        probabilities = None
        confidence = None
        
        # Calcul des probabilités
        if hasattr(self.model, 'predict_proba'):
            try:
                probas = self.model.predict_proba([text])[0]
                
                # ✅ GESTION DES CLASSES STRING OU INT
                if self.classes_ is not None:
                    # Créer un mapping classe -> probabilité
                    class_proba_map = {}
                    
                    for i, cls in enumerate(self.classes_):
                        # Convertir la classe en string si nécessaire
                        class_key = str(cls).lower()
                        class_proba_map[class_key] = float(probas[i])
                    
                    logger.debug(f"Mapping classes->probas : {class_proba_map}")
                    
                    # Extraire les probabilités ham/spam
                    ham_proba = class_proba_map.get('ham', 0.0)
                    spam_proba = class_proba_map.get('spam', 0.0)
                    
                    # Si les clés ne correspondent pas, essayer avec '0' et '1'
                    if ham_proba == 0.0 and spam_proba == 0.0:
                        ham_proba = class_proba_map.get('0', probas[0])
                        spam_proba = class_proba_map.get('1', probas[1])
                    
                    probabilities = {
                        "ham": ham_proba,
                        "spam": spam_proba
                    }
                    
                    # ✅ Déterminer le label par la plus haute probabilité
                    if spam_proba > ham_proba:
                        label = "spam"
                        confidence = spam_proba
                    else:
                        label = "ham"
                        confidence = ham_proba
                    
                    logger.info(
                        f"Prédiction : Ham={ham_proba:.4f}, Spam={spam_proba:.4f} "
                        f"-> Label={label} (confiance={confidence:.4f})"
                    )
                    
                else:
                    # Fallback si classes_ n'est pas disponible
                    probabilities = {
                        "ham": float(probas[0]),
                        "spam": float(probas[1])
                    }
                    label = "spam" if probas[1] > probas[0] else "ham"
                    confidence = float(max(probas))
                    
            except Exception as e:
                logger.error(f"Erreur lors du calcul des probabilités : {e}", exc_info=True)
                # Fallback sur la prédiction brute
                label = str(raw_prediction).lower()
                if label not in ['ham', 'spam']:
                    label = "spam" if raw_prediction == 1 else "ham"
        else:
            # Pas de predict_proba disponible
            label = str(raw_prediction).lower()
            if label not in ['ham', 'spam']:
                label = "spam" if raw_prediction == 1 else "ham"
        
        # Construction du résultat
        result = {
            "label": label,
            "accuracy": self.accuracy,
            "confidence": confidence,
            "raw_prediction": str(raw_prediction)
        }
        
        if probabilities:
            result["probabilities"] = probabilities
        
        return result
    
    def get_model_info(self) -> Dict[str, any]:
        """
        Retourne les informations sur le modèle chargé
        
        Returns:
            Dict avec les métadonnées du modèle
        """
        classes_info = None
        if self.classes_ is not None:
            classes_info = [str(c) for c in self.classes_]
        
        return {
            "model_loaded": self.model is not None,
            "accuracy": self.accuracy,
            "model_path": str(self.model_path),
            "model_type": str(type(self.model).__name__) if self.model else None,
            "classes": classes_info,
            "classes_type": str(type(self.classes_[0])) if self.classes_ is not None and len(self.classes_) > 0 else None
        }


# Instance singleton du service
nlp_service = NLPService()
