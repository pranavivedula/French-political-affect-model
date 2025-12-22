"""Mapping affect scores to the Circumplex Model of Affect."""

import math
from typing import Dict, Tuple, List
import numpy as np


class CircumplexMapper:
    """
    Mapper for the Circumplex Model of Affect (Russell, 1980).

    The circumplex model represents emotions in a 2D space:
    - X-axis (Valence): Negative (-1) to Positive (+1)
    - Y-axis (Arousal): Low (0) to High (1)

    Quadrants:
    - Q1 (High Arousal, Positive Valence): Excited, Alert, Elated
    - Q2 (High Arousal, Negative Valence): Stressed, Upset, Tense
    - Q3 (Low Arousal, Negative Valence): Sad, Depressed, Bored
    - Q4 (Low Arousal, Positive Valence): Calm, Relaxed, Content
    """

    # Emotion labels for different regions of the circumplex
    EMOTION_LABELS = {
        'excited': (0.7, 0.7),      # High positive valence, high arousal
        'alert': (0.5, 0.8),
        'elated': (0.8, 0.6),
        'happy': (0.7, 0.5),
        'content': (0.6, 0.3),
        'calm': (0.5, 0.2),
        'relaxed': (0.4, 0.2),
        'serene': (0.3, 0.1),
        'bored': (-0.1, 0.2),
        'depressed': (-0.6, 0.2),
        'sad': (-0.7, 0.3),
        'upset': (-0.6, 0.6),
        'stressed': (-0.5, 0.8),
        'tense': (-0.4, 0.7),
        'nervous': (-0.3, 0.8),
        'angry': (-0.7, 0.8),
    }

    def __init__(self):
        """Initialize the circumplex mapper."""
        pass

    def get_quadrant(self, valence: float, arousal: float) -> int:
        """
        Get the quadrant number for a valence/arousal pair.

        Args:
            valence: Valence score (-1 to +1)
            arousal: Arousal score (0 to 1)

        Returns:
            Quadrant number (1-4)
        """
        # Convert arousal to -1 to +1 scale for quadrant calculation
        arousal_centered = arousal * 2 - 1  # 0-1 -> -1 to +1

        if valence >= 0 and arousal_centered >= 0:
            return 1  # High arousal, positive valence
        elif valence < 0 and arousal_centered >= 0:
            return 2  # High arousal, negative valence
        elif valence < 0 and arousal_centered < 0:
            return 3  # Low arousal, negative valence
        else:
            return 4  # Low arousal, positive valence

    def get_quadrant_label(self, quadrant: int) -> str:
        """
        Get descriptive label for a quadrant.

        Args:
            quadrant: Quadrant number (1-4)

        Returns:
            Quadrant label
        """
        labels = {
            1: "High Activation / Positive",
            2: "High Activation / Negative",
            3: "Low Activation / Negative",
            4: "Low Activation / Positive"
        }
        return labels.get(quadrant, "Unknown")

    def get_nearest_emotion(self, valence: float, arousal: float) -> Tuple[str, float]:
        """
        Find the nearest emotion label in the circumplex.

        Args:
            valence: Valence score
            arousal: Arousal score

        Returns:
            Tuple of (emotion_name, distance)
        """
        min_distance = float('inf')
        nearest_emotion = 'neutral'

        for emotion, (em_val, em_ar) in self.EMOTION_LABELS.items():
            # Euclidean distance
            distance = math.sqrt(
                (valence - em_val) ** 2 +
                (arousal - em_ar) ** 2
            )

            if distance < min_distance:
                min_distance = distance
                nearest_emotion = emotion

        return nearest_emotion, min_distance

    def calculate_distance(
        self,
        valence1: float,
        arousal1: float,
        valence2: float,
        arousal2: float
    ) -> float:
        """
        Calculate Euclidean distance between two points in circumplex space.

        Args:
            valence1, arousal1: First point
            valence2, arousal2: Second point

        Returns:
            Euclidean distance
        """
        return math.sqrt(
            (valence2 - valence1) ** 2 +
            (arousal2 - arousal1) ** 2
        )

    def calculate_angle(self, valence: float, arousal: float) -> float:
        """
        Calculate angle (in degrees) from origin in circumplex space.

        Args:
            valence: Valence score
            arousal: Arousal score (will be centered: 0.5 -> 0)

        Returns:
            Angle in degrees (0-360)
        """
        # Center arousal at 0
        arousal_centered = arousal * 2 - 1

        # Calculate angle using arctangent
        angle_rad = math.atan2(arousal_centered, valence)

        # Convert to degrees (0-360)
        angle_deg = math.degrees(angle_rad)
        if angle_deg < 0:
            angle_deg += 360

        return angle_deg

    def calculate_magnitude(self, valence: float, arousal: float) -> float:
        """
        Calculate magnitude (distance from origin) in circumplex space.

        Args:
            valence: Valence score
            arousal: Arousal score

        Returns:
            Magnitude
        """
        arousal_centered = arousal * 2 - 1
        return math.sqrt(valence ** 2 + arousal_centered ** 2)

    def map_to_circumplex(
        self,
        valence: float,
        arousal: float
    ) -> Dict[str, any]:
        """
        Map valence and arousal scores to circumplex space with interpretations.

        Args:
            valence: Valence score (-1 to +1)
            arousal: Arousal score (0 to 1)

        Returns:
            Dictionary with circumplex information
        """
        quadrant = self.get_quadrant(valence, arousal)
        quadrant_label = self.get_quadrant_label(quadrant)
        nearest_emotion, emotion_distance = self.get_nearest_emotion(valence, arousal)
        angle = self.calculate_angle(valence, arousal)
        magnitude = self.calculate_magnitude(valence, arousal)

        return {
            'valence': valence,
            'arousal': arousal,
            'quadrant': quadrant,
            'quadrant_label': quadrant_label,
            'nearest_emotion': nearest_emotion,
            'emotion_distance': emotion_distance,
            'angle': angle,
            'magnitude': magnitude,
            'coordinates': (valence, arousal)
        }

    def compare_positions(
        self,
        party1_valence: float,
        party1_arousal: float,
        party2_valence: float,
        party2_arousal: float
    ) -> Dict[str, any]:
        """
        Compare positions of two parties in circumplex space.

        Args:
            party1_valence, party1_arousal: First party scores
            party2_valence, party2_arousal: Second party scores

        Returns:
            Dictionary with comparison metrics
        """
        distance = self.calculate_distance(
            party1_valence, party1_arousal,
            party2_valence, party2_arousal
        )

        valence_diff = party2_valence - party1_valence
        arousal_diff = party2_arousal - party1_arousal

        # Determine which party is more extreme in each dimension
        more_positive = "party2" if valence_diff > 0 else "party1"
        more_aroused = "party2" if arousal_diff > 0 else "party1"

        return {
            'distance': distance,
            'valence_difference': valence_diff,
            'arousal_difference': arousal_diff,
            'more_positive': more_positive,
            'more_aroused': more_aroused,
            'similar': distance < 0.3  # Threshold for similarity
        }

    def get_regional_interpretation(
        self,
        valence: float,
        arousal: float
    ) -> str:
        """
        Get a textual interpretation of a position in circumplex space.

        Args:
            valence: Valence score
            arousal: Arousal score

        Returns:
            Textual interpretation
        """
        quadrant = self.get_quadrant(valence, arousal)
        nearest_emotion, _ = self.get_nearest_emotion(valence, arousal)

        # Valence interpretation
        if valence > 0.5:
            val_desc = "highly positive"
        elif valence > 0.2:
            val_desc = "moderately positive"
        elif valence > -0.2:
            val_desc = "neutral"
        elif valence > -0.5:
            val_desc = "moderately negative"
        else:
            val_desc = "highly negative"

        # Arousal interpretation
        if arousal > 0.7:
            ar_desc = "very high activation"
        elif arousal > 0.5:
            ar_desc = "high activation"
        elif arousal > 0.3:
            ar_desc = "moderate activation"
        else:
            ar_desc = "low activation"

        interpretation = (
            f"This position shows {val_desc} valence with {ar_desc}, "
            f"closest to the emotion '{nearest_emotion}'. "
            f"Located in {self.get_quadrant_label(quadrant)}."
        )

        return interpretation


def map_party_to_circumplex(valence: float, arousal: float) -> Dict[str, any]:
    """
    Convenience function to map party scores to circumplex.

    Args:
        valence: Valence score
        arousal: Arousal score

    Returns:
        Circumplex mapping information
    """
    mapper = CircumplexMapper()
    return mapper.map_to_circumplex(valence, arousal)
