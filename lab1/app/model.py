"""
ML Model wrapper for movie rating prediction.
"""

import pickle
import logging
from typing import List, Tuple, Optional

from app.config import MODEL_PATH

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MovieRatingModel:
    """
    Wrapper class for the movie rating prediction model.

    This class handles:
    - Loading the trained model from disk
    - Making single predictions
    - Making batch predictions
    """

    def __init__(self, model_path: str = MODEL_PATH):
        """
        Initialize the model wrapper.

        Args:
            model_path: Path to the saved model file (.pkl)
        """
        self.model_path = model_path
        self.model = None
        self._load_model()

    # =========================================================================
    # TODO 1: _load_model
    # =========================================================================
    def _load_model(self) -> None:
        """Load the trained model from disk."""
        try:
            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)
            logger.info(f"Model loaded successfully from {self.model_path}")
        except FileNotFoundError:
            logger.error(f"Model file not found: {self.model_path}")
            raise
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    # =========================================================================
    # TODO 2: predict
    # =========================================================================
    def predict(self, user_id: str, movie_id: str) -> float:
        """
        Predict rating for a single user-movie pair.

        Args:
            user_id: User ID (string)
            movie_id: Movie ID (string)

        Returns:
            Predicted rating (float between 1.0 and 5.0)
        """
        if self.model is None:
            raise RuntimeError("Model is not loaded")

        prediction = self.model.predict(user_id, movie_id)
        # Clamp the estimate to the valid rating domain [1.0, 5.0]
        # ("closest legal value" defensive technique, Code Complete p.917).
        # SVD.est is unbounded; without this an out-of-range estimate would
        # trip the response_model ge/le constraint and surface as an HTTP 500
        # on a perfectly valid request.
        est = min(5.0, max(1.0, float(prediction.est)))
        return round(est, 2)

    # =========================================================================
    # TODO 3: predict_batch
    # =========================================================================
    def predict_batch(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """
        Predict ratings for multiple user-movie pairs.

        Args:
            pairs: List of (user_id, movie_id) tuples

        Returns:
            List of predicted ratings
        """
        return [self.predict(user_id, movie_id) for user_id, movie_id in pairs]

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None


# =============================================================================
# Singleton instance (optional pattern)
# =============================================================================
_model_instance: Optional[MovieRatingModel] = None


def get_model() -> MovieRatingModel:
    """Get or create the model singleton instance."""
    global _model_instance
    if _model_instance is None:
        _model_instance = MovieRatingModel()
    return _model_instance
