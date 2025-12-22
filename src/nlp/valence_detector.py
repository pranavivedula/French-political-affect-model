"""Valence (sentiment) detection for French political texts."""

import numpy as np
import torch
from typing import List, Union, Dict
from loguru import logger

from config.settings import VALENCE_MODEL, MAX_SEQUENCE_LENGTH
from src.nlp.model_loader import get_model_loader
from src.nlp.preprocessing import TextPreprocessor


class ValenceDetector:
    """
    Detector for valence (positive/negative sentiment) in text.

    Valence represents the pleasantness dimension of affect:
    - Negative valence (-1): Unpleasant, negative sentiment
    - Neutral valence (0): Neutral sentiment
    - Positive valence (+1): Pleasant, positive sentiment
    """

    def __init__(self, model_name: str = None):
        """
        Initialize valence detector.

        Args:
            model_name: Hugging Face model identifier (defaults to config setting)
        """
        self.model_name = model_name or VALENCE_MODEL
        self.preprocessor = TextPreprocessor()
        self.model_loader = get_model_loader()

        # Load model and tokenizer
        logger.info(f"Initializing ValenceDetector with model: {self.model_name}")
        self.model, self.tokenizer = self.model_loader.load_model_and_tokenizer(
            self.model_name,
            model_type="sequence_classification"
        )
        self.device = self.model_loader.get_device()

        # Get number of labels from model config
        self.num_labels = self.model.config.num_labels
        logger.info(f"Model has {self.num_labels} output labels")

    def _logits_to_valence(self, logits: torch.Tensor) -> float:
        """
        Convert model logits to valence score.

        For 5-class sentiment models (1-5 stars):
        - Maps 1-5 star rating to -1 to +1 valence scale

        Args:
            logits: Model output logits

        Returns:
            Valence score from -1 (negative) to +1 (positive)
        """
        # Get predicted class
        probs = torch.softmax(logits, dim=-1)
        predicted_class = torch.argmax(probs, dim=-1).item()

        if self.num_labels == 5:
            # 5-class model (1-5 stars)
            # Map to valence: 1 star = -1, 3 stars = 0, 5 stars = +1
            valence = (predicted_class - 2) / 2.0
        elif self.num_labels == 3:
            # 3-class model (negative, neutral, positive)
            # Map to valence: 0 = -1, 1 = 0, 2 = +1
            valence = (predicted_class - 1) * 1.0
        else:
            # For other models, use weighted average
            # Assume classes are ordered from negative to positive
            class_weights = np.linspace(-1, 1, self.num_labels)
            valence = float(np.dot(probs.cpu().numpy()[0], class_weights))

        return valence

    def predict_valence(self, text: str) -> float:
        """
        Predict valence score for a single text.

        Args:
            text: Input text

        Returns:
            Valence score from -1 (negative) to +1 (positive)
        """
        if not text or len(text.strip()) == 0:
            logger.warning("Empty text provided, returning neutral valence")
            return 0.0

        try:
            # Tokenize input
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_SEQUENCE_LENGTH,
                padding=True
            )

            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Get model prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits

            # Convert to valence score
            valence = self._logits_to_valence(logits)

            return float(valence)

        except Exception as e:
            logger.error(f"Error predicting valence: {e}")
            return 0.0

    def predict_batch(self, texts: List[str], batch_size: int = 8) -> List[float]:
        """
        Predict valence scores for multiple texts efficiently.

        Args:
            texts: List of input texts
            batch_size: Number of texts to process at once

        Returns:
            List of valence scores
        """
        if not texts:
            return []

        valences = []

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            try:
                # Tokenize batch
                inputs = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    truncation=True,
                    max_length=MAX_SEQUENCE_LENGTH,
                    padding=True
                )

                # Move to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Get predictions
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits

                # Convert each output to valence
                for j in range(len(batch)):
                    valence = self._logits_to_valence(logits[j:j+1])
                    valences.append(float(valence))

            except Exception as e:
                logger.error(f"Error in batch prediction: {e}")
                # Return neutral for failed batch
                valences.extend([0.0] * len(batch))

        return valences

    def analyze_document(
        self,
        text: str,
        return_sentences: bool = True
    ) -> Dict[str, any]:
        """
        Analyze a complete document and return detailed valence information.

        Args:
            text: Document text
            return_sentences: Whether to return sentence-level scores

        Returns:
            Dictionary with valence analysis results
        """
        # Preprocess document
        processed = self.preprocessor.preprocess_document(text)

        if not processed:
            logger.warning("Document preprocessing failed")
            return {
                'document_valence': 0.0,
                'num_sentences': 0,
                'sentences': []
            }

        sentences = processed['sentences']

        # Predict valence for each sentence
        sentence_valences = self.predict_batch(sentences)

        # Calculate document-level valence (weighted average by sentence length)
        sentence_lengths = [len(s.split()) for s in sentences]
        total_words = sum(sentence_lengths)

        if total_words > 0:
            weighted_valence = sum(
                v * length for v, length in zip(sentence_valences, sentence_lengths)
            ) / total_words
        else:
            weighted_valence = 0.0

        result = {
            'document_valence': float(weighted_valence),
            'num_sentences': len(sentences),
            'valence_std': float(np.std(sentence_valences)) if sentence_valences else 0.0,
            'valence_min': float(min(sentence_valences)) if sentence_valences else 0.0,
            'valence_max': float(max(sentence_valences)) if sentence_valences else 0.0,
        }

        if return_sentences:
            result['sentences'] = [
                {
                    'text': sent,
                    'valence': val,
                    'word_count': length
                }
                for sent, val, length in zip(sentences, sentence_valences, sentence_lengths)
            ]

        return result


def predict_valence(text: str, model_name: str = None) -> float:
    """
    Convenience function to predict valence for text.

    Args:
        text: Input text
        model_name: Optional model name (uses default if not provided)

    Returns:
        Valence score from -1 to +1
    """
    detector = ValenceDetector(model_name=model_name)
    return detector.predict_valence(text)
