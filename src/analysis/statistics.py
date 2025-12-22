"""Statistical utilities for affect analysis."""

import numpy as np
from typing import List, Tuple, Dict
from scipy import stats
from loguru import logger

from config.settings import CONFIDENCE_LEVEL


def calculate_confidence_interval(
    values: List[float],
    confidence: float = CONFIDENCE_LEVEL
) -> Tuple[float, float]:
    """
    Calculate confidence interval for a list of values.

    Args:
        values: List of numeric values
        confidence: Confidence level (default: 0.95 for 95% CI)

    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    if not values or len(values) < 2:
        return (0.0, 0.0)

    values_array = np.array(values)
    n = len(values_array)
    mean = np.mean(values_array)
    std_err = stats.sem(values_array)

    # Calculate margin of error using t-distribution
    margin = std_err * stats.t.ppf((1 + confidence) / 2, n - 1)

    return (mean - margin, mean + margin)


def calculate_ci_width(values: List[float], confidence: float = CONFIDENCE_LEVEL) -> float:
    """
    Calculate the width of the confidence interval.

    Useful for assessing uncertainty in measurements.

    Args:
        values: List of numeric values
        confidence: Confidence level

    Returns:
        Width of confidence interval
    """
    lower, upper = calculate_confidence_interval(values, confidence)
    return upper - lower


def weighted_mean(values: List[float], weights: List[float]) -> float:
    """
    Calculate weighted mean.

    Args:
        values: List of values
        weights: List of weights (same length as values)

    Returns:
        Weighted mean
    """
    if not values or not weights or len(values) != len(weights):
        return 0.0

    values_array = np.array(values)
    weights_array = np.array(weights)

    # Normalize weights
    weights_sum = np.sum(weights_array)
    if weights_sum == 0:
        return np.mean(values_array)

    normalized_weights = weights_array / weights_sum

    return float(np.dot(values_array, normalized_weights))


def weighted_std(values: List[float], weights: List[float]) -> float:
    """
    Calculate weighted standard deviation.

    Args:
        values: List of values
        weights: List of weights

    Returns:
        Weighted standard deviation
    """
    if not values or not weights or len(values) != len(weights):
        return 0.0

    values_array = np.array(values)
    weights_array = np.array(weights)

    # Normalize weights
    weights_sum = np.sum(weights_array)
    if weights_sum == 0:
        return np.std(values_array)

    normalized_weights = weights_array / weights_sum

    # Calculate weighted mean
    mean = np.dot(values_array, normalized_weights)

    # Calculate weighted variance
    variance = np.dot(normalized_weights, (values_array - mean) ** 2)

    return float(np.sqrt(variance))


def bootstrap_confidence_interval(
    values: List[float],
    n_iterations: int = 1000,
    confidence: float = CONFIDENCE_LEVEL
) -> Tuple[float, float]:
    """
    Calculate confidence interval using bootstrap resampling.

    More robust for small sample sizes or non-normal distributions.

    Args:
        values: List of values
        n_iterations: Number of bootstrap iterations
        confidence: Confidence level

    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    if not values or len(values) < 2:
        return (0.0, 0.0)

    values_array = np.array(values)
    n = len(values_array)

    # Bootstrap resampling
    bootstrap_means = []
    for _ in range(n_iterations):
        sample = np.random.choice(values_array, size=n, replace=True)
        bootstrap_means.append(np.mean(sample))

    # Calculate percentile-based confidence interval
    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    lower = np.percentile(bootstrap_means, lower_percentile)
    upper = np.percentile(bootstrap_means, upper_percentile)

    return (float(lower), float(upper))


def calculate_effect_size(
    group1_values: List[float],
    group2_values: List[float]
) -> float:
    """
    Calculate Cohen's d effect size between two groups.

    Useful for comparing party positions.

    Args:
        group1_values: Values from first group
        group2_values: Values from second group

    Returns:
        Cohen's d effect size
    """
    if not group1_values or not group2_values:
        return 0.0

    g1 = np.array(group1_values)
    g2 = np.array(group2_values)

    # Calculate means
    mean1 = np.mean(g1)
    mean2 = np.mean(g2)

    # Calculate pooled standard deviation
    n1, n2 = len(g1), len(g2)
    var1, var2 = np.var(g1, ddof=1), np.var(g2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    # Cohen's d
    d = (mean1 - mean2) / pooled_std

    return float(d)


def detect_outliers(
    values: List[float],
    method: str = 'iqr',
    threshold: float = 1.5
) -> List[int]:
    """
    Detect outliers in a list of values.

    Args:
        values: List of values
        method: Detection method ('iqr' or 'zscore')
        threshold: Threshold for outlier detection

    Returns:
        List of indices of outliers
    """
    if not values or len(values) < 3:
        return []

    values_array = np.array(values)
    outlier_indices = []

    if method == 'iqr':
        # Interquartile range method
        q1 = np.percentile(values_array, 25)
        q3 = np.percentile(values_array, 75)
        iqr = q3 - q1

        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr

        outlier_indices = [
            i for i, v in enumerate(values_array)
            if v < lower_bound or v > upper_bound
        ]

    elif method == 'zscore':
        # Z-score method
        mean = np.mean(values_array)
        std = np.std(values_array)

        if std > 0:
            z_scores = np.abs((values_array - mean) / std)
            outlier_indices = [
                i for i, z in enumerate(z_scores)
                if z > threshold
            ]

    return outlier_indices


def calculate_summary_statistics(values: List[float]) -> Dict[str, float]:
    """
    Calculate comprehensive summary statistics.

    Args:
        values: List of values

    Returns:
        Dictionary of statistics
    """
    if not values:
        return {
            'mean': 0.0,
            'median': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0,
            'q25': 0.0,
            'q75': 0.0,
            'count': 0
        }

    values_array = np.array(values)

    return {
        'mean': float(np.mean(values_array)),
        'median': float(np.median(values_array)),
        'std': float(np.std(values_array)),
        'min': float(np.min(values_array)),
        'max': float(np.max(values_array)),
        'q25': float(np.percentile(values_array, 25)),
        'q75': float(np.percentile(values_array, 75)),
        'count': len(values_array)
    }


def compare_distributions(
    values1: List[float],
    values2: List[float],
    test: str = 'mannwhitneyu'
) -> Dict[str, any]:
    """
    Compare two distributions statistically.

    Args:
        values1: First set of values
        values2: Second set of values
        test: Statistical test ('mannwhitneyu', 'ttest', 'ks')

    Returns:
        Dictionary with test results
    """
    if not values1 or not values2:
        return {'statistic': 0.0, 'p_value': 1.0, 'significant': False}

    v1 = np.array(values1)
    v2 = np.array(values2)

    if test == 'mannwhitneyu':
        # Mann-Whitney U test (non-parametric)
        statistic, p_value = stats.mannwhitneyu(v1, v2, alternative='two-sided')
    elif test == 'ttest':
        # Independent t-test
        statistic, p_value = stats.ttest_ind(v1, v2)
    elif test == 'ks':
        # Kolmogorov-Smirnov test
        statistic, p_value = stats.ks_2samp(v1, v2)
    else:
        raise ValueError(f"Unknown test: {test}")

    return {
        'statistic': float(statistic),
        'p_value': float(p_value),
        'significant': p_value < 0.05,
        'test': test
    }
