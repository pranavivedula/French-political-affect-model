"""Arousal (emotional activation) detection for French political texts."""

import re
import numpy as np
import torch
from typing import List, Dict
from loguru import logger

from config.settings import AROUSAL_MODEL, MAX_SEQUENCE_LENGTH
from src.nlp.model_loader import get_model_loader
from src.nlp.preprocessing import TextPreprocessor


class ArousalDetector:
    """
    Detector for arousal (emotional activation) in text.

    Arousal represents the activation dimension of affect:
    - Low arousal (0): Calm, passive, low energy (e.g., sadness, contentment)
    - High arousal (1): Excited, active, high energy (e.g., anger, joy, fear)

    Note: Arousal is harder to detect than valence. This implementation uses:
    1. A sentiment model as a proxy (stronger emotions = higher arousal)
    2. Linguistic features (exclamation marks, caps, intensifiers)
    3. Future: Fine-tuned model on arousal-labeled data
    """

    def __init__(self, model_name: str = None):
        """
        Initialize arousal detector.

        Args:
            model_name: Hugging Face model identifier (defaults to config setting)
        """
        self.model_name = model_name or AROUSAL_MODEL
        self.preprocessor = TextPreprocessor()
        self.model_loader = get_model_loader()

        # Load model (using sentiment model as proxy for now)
        logger.info(f"Initializing ArousalDetector with model: {self.model_name}")
        self.model, self.tokenizer = self.model_loader.load_model_and_tokenizer(
            self.model_name,
            model_type="sequence_classification"
        )
        self.device = self.model_loader.get_device()

        # French intensifier words (increase arousal)
        self.intensifiers = {
            'très', 'vraiment', 'tellement', 'extrêmement', 'absolument',
            'totalement', 'complètement', 'particulièrement', 'incroyablement',
            'jamais', 'toujours', 'urgent', 'grave', 'crise', 'danger'
        }

        # High-arousal emotion words
        self.high_arousal_words = {
            'colère', 'furieux', 'outragé', 'scandale', 'révoltant', 'urgent',
            'catastrophe', 'crise', 'alerte', 'combat', 'lutte', 'bataille',
            'victoire', 'triomphe', 'excitant', 'passionnant', 'enthousiaste'
        }

    def _extract_linguistic_features(self, text: str) -> Dict[str, float]:
        """
        Extract linguistic features that correlate with arousal.

        Args:
            text: Input text

        Returns:
            Dictionary of feature scores
        """
        if not text:
            return {
                'exclamation_ratio': 0.0,
                'caps_ratio': 0.0,
                'intensifier_ratio': 0.0,
                'high_arousal_word_ratio': 0.0
            }

        # Count exclamation marks
        exclamation_count = text.count('!')
        sentence_count = max(1, text.count('.') + text.count('!') + text.count('?'))
        exclamation_ratio = exclamation_count / sentence_count

        # Count capital letters (excluding first letter of sentences)
        # Remove sentence starts
        text_no_starts = re.sub(r'(?:^|[.!?]\s+)([A-ZÀÂÄÉÈÊËÏÎÔÙÛÜŸÇ])', r' \1', text)
        caps_count = sum(1 for c in text_no_starts if c.isupper())
        alpha_count = sum(1 for c in text if c.isalpha())
        caps_ratio = caps_count / max(1, alpha_count)

        # Count intensifiers
        words = text.lower().split()
        intensifier_count = sum(1 for word in words if word in self.intensifiers)
        intensifier_ratio = intensifier_count / max(1, len(words))

        # Count high-arousal emotion words
        high_arousal_count = sum(1 for word in words if word in self.high_arousal_words)
        high_arousal_ratio = high_arousal_count / max(1, len(words))

        return {
            'exclamation_ratio': min(1.0, exclamation_ratio),
            'caps_ratio': min(1.0, caps_ratio * 10),  # Scale up
            'intensifier_ratio': min(1.0, intensifier_ratio * 20),  # Scale up
            'high_arousal_word_ratio': min(1.0, high_arousal_ratio * 15)  # Scale up
        }

    def _model_based_arousal(self, text: str) -> float:
        """
        Estimate arousal using the sentiment model.

        Strategy: Use the confidence/extremity of sentiment as proxy for arousal.
        Strong positive or negative sentiment typically indicates high arousal.

        Args:
            text: Input text

        Returns:
            Arousal score from 0 (low) to 1 (high)
        """
        if not text or len(text.strip()) == 0:
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

            # Get probabilities
            probs = torch.softmax(logits, dim=-1)

            # Calculate arousal from prediction confidence
            # High confidence (extreme predictions) = high arousal
            max_prob = torch.max(probs).item()

            # Also consider distance from neutral
            # For 5-class model, neutral is class 2 (index 2)
            predicted_class = torch.argmax(probs).item()
            num_classes = probs.shape[1]
            neutral_class = num_classes // 2

            # Distance from neutral (normalized)
            distance_from_neutral = abs(predicted_class - neutral_class) / (num_classes / 2)

            # Combine confidence and extremity
            arousal = (max_prob * 0.5) + (distance_from_neutral * 0.5)

            return float(arousal)

        except Exception as e:
            logger.error(f"Error predicting arousal: {e}")
            return 0.0

    def predict_arousal(self, text: str) -> float:
        """
        Predict arousal score for a single text.

        Combines model-based and linguistic feature-based approaches.

        Args:
            text: Input text

        Returns:
            Arousal score from 0 (low activation) to 1 (high activation)
        """
        if not text or len(text.strip()) == 0:
            logger.warning("Empty text provided, returning low arousal")
            return 0.0

        # Get model-based arousal
        model_arousal = self._model_based_arousal(text)

        # Get linguistic features
        features = self._extract_linguistic_features(text)

        # Combine features (weighted average)
        linguistic_arousal = (
            features['exclamation_ratio'] * 0.3 +
            features['caps_ratio'] * 0.2 +
            features['intensifier_ratio'] * 0.3 +
            features['high_arousal_word_ratio'] * 0.2
        )

        # Final arousal: weighted combination of model and linguistic features
        final_arousal = (model_arousal * 0.6) + (linguistic_arousal * 0.4)

        return float(min(1.0, final_arousal))

    def predict_batch(self, texts: List[str], batch_size: int = 8) -> List[float]:
        """
        Predict arousal scores for multiple texts efficiently.

        Args:
            texts: List of input texts
            batch_size: Number of texts to process at once

        Returns:
            List of arousal scores
        """
        if not texts:
            return []

        arousals = []

        for text in texts:
            arousal = self.predict_arousal(text)
            arousals.append(arousal)

        return arousals

    def analyze_document(
        self,
        text: str,
        return_sentences: bool = True
    ) -> Dict[str, any]:
        """
        Analyze a complete document and return detailed arousal information.

        Args:
            text: Document text
            return_sentences: Whether to return sentence-level scores

        Returns:
            Dictionary with arousal analysis results
        """
        # Preprocess document
        processed = self.preprocessor.preprocess_document(text)

        if not processed:
            logger.warning("Document preprocessing failed")
            return {
                'document_arousal': 0.0,
                'num_sentences': 0,
                'sentences': []
            }

        sentences = processed['sentences']

        # Predict arousal for each sentence
        sentence_arousals = self.predict_batch(sentences)

        # Calculate document-level arousal (weighted average by sentence length)
        sentence_lengths = [len(s.split()) for s in sentences]
        total_words = sum(sentence_lengths)

        if total_words > 0:
            weighted_arousal = sum(
                a * length for a, length in zip(sentence_arousals, sentence_lengths)
            ) / total_words
        else:
            weighted_arousal = 0.0

        result = {
            'document_arousal': float(weighted_arousal),
            'num_sentences': len(sentences),
            'arousal_std': float(np.std(sentence_arousals)) if sentence_arousals else 0.0,
            'arousal_min': float(min(sentence_arousals)) if sentence_arousals else 0.0,
            'arousal_max': float(max(sentence_arousals)) if sentence_arousals else 0.0,
        }

        if return_sentences:
            result['sentences'] = [
                {
                    'text': sent,
                    'arousal': arou,
                    'word_count': length
                }
                for sent, arou, length in zip(sentences, sentence_arousals, sentence_lengths)
            ]

        return result


def predict_arousal(text: str, model_name: str = None) -> float:
    """
    Convenience function to predict arousal for text.

    Args:
        text: Input text
        model_name: Optional model name (uses default if not provided)

    Returns:
        Arousal score from 0 to 1
    """
    detector = ArousalDetector(model_name=model_name)
    return detector.predict_arousal(text)
