"""Utilities for loading and caching transformer models."""

import os
from typing import Tuple, Optional
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from loguru import logger

from config.settings import MODEL_CACHE_DIR


class ModelLoader:
    """Loader and cache manager for transformer models."""

    def __init__(self, cache_dir: str = None):
        """
        Initialize model loader.

        Args:
            cache_dir: Directory to cache models (defaults to config setting)
        """
        self.cache_dir = cache_dir or MODEL_CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)

        # Determine device (GPU if available, else CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Cache for loaded models
        self._model_cache = {}
        self._tokenizer_cache = {}

    def load_model_and_tokenizer(
        self,
        model_name: str,
        model_type: str = "sequence_classification"
    ) -> Tuple[AutoModel, AutoTokenizer]:
        """
        Load a pretrained model and tokenizer with caching.

        Args:
            model_name: Hugging Face model identifier
            model_type: Type of model ('sequence_classification' or 'base')

        Returns:
            Tuple of (model, tokenizer)
        """
        # Check if already in cache
        cache_key = f"{model_name}_{model_type}"
        if cache_key in self._model_cache:
            logger.debug(f"Using cached model: {model_name}")
            return self._model_cache[cache_key], self._tokenizer_cache[model_name]

        logger.info(f"Loading model: {model_name}")

        try:
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=self.cache_dir
            )

            # Load model based on type
            if model_type == "sequence_classification":
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    cache_dir=self.cache_dir
                )
            else:
                model = AutoModel.from_pretrained(
                    model_name,
                    cache_dir=self.cache_dir
                )

            # Move model to device
            model = model.to(self.device)
            model.eval()  # Set to evaluation mode

            # Cache the loaded models
            self._model_cache[cache_key] = model
            self._tokenizer_cache[model_name] = tokenizer

            logger.success(f"Successfully loaded model: {model_name}")

            return model, tokenizer

        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            raise

    def get_device(self) -> torch.device:
        """Get the device being used (CPU or GPU)."""
        return self.device

    def clear_cache(self):
        """Clear the model cache to free memory."""
        self._model_cache.clear()
        self._tokenizer_cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Model cache cleared")

    def get_model_info(self, model_name: str) -> dict:
        """
        Get information about a model.

        Args:
            model_name: Hugging Face model identifier

        Returns:
            Dictionary with model information
        """
        cache_key = f"{model_name}_sequence_classification"
        is_cached = cache_key in self._model_cache

        info = {
            'model_name': model_name,
            'is_cached': is_cached,
            'device': str(self.device),
            'cache_dir': self.cache_dir
        }

        return info


# Global model loader instance
_global_loader = None


def get_model_loader() -> ModelLoader:
    """
    Get the global model loader instance (singleton pattern).

    Returns:
        ModelLoader instance
    """
    global _global_loader
    if _global_loader is None:
        _global_loader = ModelLoader()
    return _global_loader


def load_sentiment_model(model_name: str) -> Tuple[AutoModel, AutoTokenizer]:
    """
    Convenience function to load a sentiment analysis model.

    Args:
        model_name: Hugging Face model identifier

    Returns:
        Tuple of (model, tokenizer)
    """
    loader = get_model_loader()
    return loader.load_model_and_tokenizer(model_name, model_type="sequence_classification")
