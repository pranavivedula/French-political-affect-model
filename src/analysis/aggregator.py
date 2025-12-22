"""Aggregation logic for computing party-level affect scores."""

import math
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import numpy as np
from loguru import logger

from config.settings import (
    RECENCY_HALFLIFE_DAYS,
    MAX_WORD_COUNT_FOR_WEIGHTING,
    RECENCY_WEIGHT,
    LENGTH_WEIGHT,
    MIN_DOCUMENTS_FOR_AGGREGATION,
    CONFIDENCE_LEVEL
)
from src.analysis.statistics import (
    weighted_mean,
    weighted_std,
    calculate_confidence_interval,
    bootstrap_confidence_interval,
    calculate_summary_statistics
)


class PartyAffectAggregator:
    """Aggregator for computing party-level affect scores from documents."""

    def __init__(
        self,
        recency_halflife: int = RECENCY_HALFLIFE_DAYS,
        max_word_count: int = MAX_WORD_COUNT_FOR_WEIGHTING,
        recency_weight_factor: float = RECENCY_WEIGHT,
        length_weight_factor: float = LENGTH_WEIGHT
    ):
        """
        Initialize aggregator.

        Args:
            recency_halflife: Half-life for recency decay in days
            max_word_count: Maximum word count for weighting (cap)
            recency_weight_factor: Weight for recency component (0-1)
            length_weight_factor: Weight for length component (0-1)
        """
        self.recency_halflife = recency_halflife
        self.max_word_count = max_word_count
        self.recency_weight_factor = recency_weight_factor
        self.length_weight_factor = length_weight_factor

        # Ensure weights sum to 1
        total = recency_weight_factor + length_weight_factor
        self.recency_weight_factor = recency_weight_factor / total
        self.length_weight_factor = length_weight_factor / total

    def calculate_recency_weight(
        self,
        document_date: datetime,
        reference_date: datetime = None
    ) -> float:
        """
        Calculate recency weight using exponential decay.

        Args:
            document_date: Date of the document
            reference_date: Reference date (defaults to now)

        Returns:
            Recency weight (0-1)
        """
        if not document_date:
            return 0.5  # Neutral weight for unknown dates

        if reference_date is None:
            reference_date = datetime.utcnow()

        # Calculate days difference
        days_diff = (reference_date - document_date).days

        if days_diff < 0:
            days_diff = 0  # Future dates get max weight

        # Exponential decay: weight = 0.5^(days / halflife)
        weight = math.pow(0.5, days_diff / self.recency_halflife)

        return float(weight)

    def calculate_length_weight(self, word_count: int) -> float:
        """
        Calculate length weight (normalized by word count).

        Args:
            word_count: Number of words in document

        Returns:
            Length weight (0-1)
        """
        if word_count <= 0:
            return 0.0

        # Normalize and cap at max_word_count
        weight = min(word_count, self.max_word_count) / self.max_word_count

        return float(weight)

    def calculate_combined_weight(
        self,
        document_date: datetime,
        word_count: int,
        reference_date: datetime = None
    ) -> float:
        """
        Calculate combined weight from recency and length.

        Args:
            document_date: Date of the document
            word_count: Number of words
            reference_date: Reference date for recency calculation

        Returns:
            Combined weight
        """
        recency_w = self.calculate_recency_weight(document_date, reference_date)
        length_w = self.calculate_length_weight(word_count)

        combined = (
            recency_w * self.recency_weight_factor +
            length_w * self.length_weight_factor
        )

        return float(combined)

    def aggregate_documents(
        self,
        documents: List[Dict],
        method: str = 'weighted',
        reference_date: datetime = None
    ) -> Dict[str, any]:
        """
        Aggregate document scores to party-level scores.

        Args:
            documents: List of document dictionaries with keys:
                       - valence: float
                       - arousal: float
                       - date_published: datetime
                       - word_count: int
            method: Aggregation method ('weighted', 'mean', 'median')
            reference_date: Reference date for recency weighting

        Returns:
            Dictionary with aggregated scores and statistics
        """
        if not documents:
            logger.warning("No documents provided for aggregation")
            return self._empty_result()

        # Extract values
        valences = [d['valence'] for d in documents if d.get('valence') is not None]
        arousals = [d['arousal'] for d in documents if d.get('arousal') is not None]

        if not valences or not arousals:
            logger.warning("No valid valence/arousal scores in documents")
            return self._empty_result()

        if len(documents) < MIN_DOCUMENTS_FOR_AGGREGATION:
            logger.warning(
                f"Only {len(documents)} documents (minimum: {MIN_DOCUMENTS_FOR_AGGREGATION})"
            )

        # Calculate weights for each document
        weights = []
        for doc in documents:
            if method == 'weighted':
                weight = self.calculate_combined_weight(
                    doc.get('date_published'),
                    doc.get('word_count', 0),
                    reference_date
                )
            else:
                weight = 1.0  # Equal weights for mean/median

            weights.append(weight)

        # Aggregate scores
        if method == 'median':
            agg_valence = float(np.median(valences))
            agg_arousal = float(np.median(arousals))
        elif method == 'weighted' or method == 'mean':
            agg_valence = weighted_mean(valences, weights)
            agg_arousal = weighted_mean(arousals, weights)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

        # Calculate statistics
        valence_std = weighted_std(valences, weights) if method == 'weighted' else np.std(valences)
        arousal_std = weighted_std(arousals, weights) if method == 'weighted' else np.std(arousals)

        # Calculate confidence intervals
        valence_ci_lower, valence_ci_upper = calculate_confidence_interval(valences, CONFIDENCE_LEVEL)
        arousal_ci_lower, arousal_ci_upper = calculate_confidence_interval(arousals, CONFIDENCE_LEVEL)

        valence_ci_width = valence_ci_upper - valence_ci_lower
        arousal_ci_width = arousal_ci_upper - arousal_ci_lower

        # Summary statistics
        valence_stats = calculate_summary_statistics(valences)
        arousal_stats = calculate_summary_statistics(arousals)

        return {
            'valence': agg_valence,
            'arousal': agg_arousal,
            'valence_std': float(valence_std),
            'arousal_std': float(arousal_std),
            'valence_ci': valence_ci_width,
            'arousal_ci': arousal_ci_width,
            'valence_ci_lower': valence_ci_lower,
            'valence_ci_upper': valence_ci_upper,
            'arousal_ci_lower': arousal_ci_lower,
            'arousal_ci_upper': arousal_ci_upper,
            'num_documents': len(documents),
            'aggregation_method': method,
            'valence_stats': valence_stats,
            'arousal_stats': arousal_stats,
            'weights_sum': sum(weights),
            'weights_mean': np.mean(weights) if weights else 0.0
        }

    def aggregate_temporal(
        self,
        documents: List[Dict],
        period_start: datetime,
        period_end: datetime,
        method: str = 'weighted'
    ) -> Dict[str, any]:
        """
        Aggregate documents for a specific time period.

        Args:
            documents: List of documents
            period_start: Start of period
            period_end: End of period
            method: Aggregation method

        Returns:
            Aggregated scores for the period
        """
        # Filter documents within period
        period_docs = [
            d for d in documents
            if d.get('date_published') and
            period_start <= d['date_published'] <= period_end
        ]

        if not period_docs:
            logger.warning(f"No documents in period {period_start} to {period_end}")
            return self._empty_result()

        # Aggregate using period end as reference date
        result = self.aggregate_documents(
            period_docs,
            method=method,
            reference_date=period_end
        )

        # Add period information
        result['period_start'] = period_start
        result['period_end'] = period_end
        result['period_days'] = (period_end - period_start).days

        return result

    def aggregate_monthly(
        self,
        documents: List[Dict],
        start_date: datetime = None,
        end_date: datetime = None,
        method: str = 'weighted'
    ) -> List[Dict[str, any]]:
        """
        Aggregate documents into monthly snapshots.

        Args:
            documents: List of documents
            start_date: Start date (defaults to earliest document)
            end_date: End date (defaults to latest document)
            method: Aggregation method

        Returns:
            List of monthly aggregations
        """
        if not documents:
            return []

        # Determine date range
        dates = [d['date_published'] for d in documents if d.get('date_published')]
        if not dates:
            return []

        if start_date is None:
            start_date = min(dates)
        if end_date is None:
            end_date = max(dates)

        # Generate monthly periods
        monthly_results = []
        current = start_date.replace(day=1)

        while current <= end_date:
            # Calculate period end (last day of month)
            if current.month == 12:
                next_month = current.replace(year=current.year + 1, month=1)
            else:
                next_month = current.replace(month=current.month + 1)

            period_end = next_month - timedelta(days=1)

            # Aggregate for this month
            result = self.aggregate_temporal(
                documents,
                period_start=current,
                period_end=period_end,
                method=method
            )

            if result['num_documents'] > 0:
                result['snapshot_date'] = period_end
                monthly_results.append(result)

            # Move to next month
            current = next_month

        return monthly_results

    def _empty_result(self) -> Dict[str, any]:
        """Return empty result structure."""
        return {
            'valence': 0.0,
            'arousal': 0.0,
            'valence_std': 0.0,
            'arousal_std': 0.0,
            'valence_ci': 0.0,
            'arousal_ci': 0.0,
            'num_documents': 0,
            'aggregation_method': 'none'
        }


def aggregate_party_scores(
    documents: List[Dict],
    method: str = 'weighted'
) -> Dict[str, any]:
    """
    Convenience function to aggregate party scores.

    Args:
        documents: List of document dictionaries
        method: Aggregation method

    Returns:
        Aggregated scores
    """
    aggregator = PartyAffectAggregator()
    return aggregator.aggregate_documents(documents, method=method)
