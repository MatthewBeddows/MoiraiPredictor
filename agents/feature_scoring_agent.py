"""
Feature Scoring Agent - Multi-dimensional similarity computation

Deterministic agent that computes similarity across multiple feature dimensions.
This is what the LLM was pretending to do - we make it explicit and mathematical.
"""

import numpy as np
from typing import Dict, List, Tuple


class FeatureScoringAgent:
    """
    Computes multi-dimensional similarity scores
    Pure mathematical - deterministic and fast
    """

    def __init__(self,
                 shape_weight: float = 0.35,
                 pattern_weight: float = 0.20,
                 temporal_weight: float = 0.20,
                 farm_weight: float = 0.15,
                 volatility_weight: float = 0.10):
        """
        Initialize with feature weights

        Args:
            shape_weight: Weight for curve shape similarity
            pattern_weight: Weight for growth pattern match
            temporal_weight: Weight for temporal proximity
            farm_weight: Weight for farm match
            volatility_weight: Weight for volatility similarity
        """
        self.weights = {
            'shape': shape_weight,
            'pattern': pattern_weight,
            'temporal': temporal_weight,
            'farm': farm_weight,
            'volatility': volatility_weight
        }

        # Normalize weights to sum to 1
        total = sum(self.weights.values())
        self.weights = {k: v/total for k, v in self.weights.items()}

    def compute_similarity(self,
                          target_metadata: Dict,
                          candidate_metadata: Dict) -> Dict[str, float]:
        """
        Compute similarity scores across all dimensions

        Args:
            target_metadata: Target plot metadata
            candidate_metadata: Candidate plot metadata

        Returns:
            Dict with scores for each dimension + composite
        """
        scores = {}

        # 1. Shape Similarity (curve features)
        scores['shape'] = self._compute_shape_similarity(target_metadata, candidate_metadata)

        # 2. Pattern Match (categorical)
        scores['pattern'] = self._compute_pattern_match(target_metadata, candidate_metadata)

        # 3. Temporal Similarity (year proximity)
        scores['temporal'] = self._compute_temporal_similarity(target_metadata, candidate_metadata)

        # 4. Farm Match (same management)
        scores['farm'] = self._compute_farm_match(target_metadata, candidate_metadata)

        # 5. Volatility Similarity (curve smoothness)
        scores['volatility'] = self._compute_volatility_similarity(target_metadata, candidate_metadata)

        # Composite score (weighted combination)
        scores['composite'] = sum(scores[k] * self.weights[k] for k in self.weights.keys())

        return scores

    def _compute_shape_similarity(self, target: Dict, candidate: Dict) -> float:
        """
        Similarity based on curve shape features
        Higher = more similar shapes
        """
        shape_features = ['mean_yield', 'trend_slope', 'peak_week', 'range', 'peak_value']

        diffs = []
        for feat in shape_features:
            if feat in target and feat in candidate:
                t_val = target[feat]
                c_val = candidate[feat]

                if t_val is not None and c_val is not None:
                    # Relative difference
                    if abs(t_val) > 1e-8:
                        diff = abs(t_val - c_val) / abs(t_val)
                    else:
                        diff = abs(c_val)
                    diffs.append(diff)

        if not diffs:
            return 0.0

        # Convert average difference to similarity (0-1)
        avg_diff = np.mean(diffs)
        similarity = 1.0 / (1.0 + avg_diff)

        return similarity

    def _compute_pattern_match(self, target: Dict, candidate: Dict) -> float:
        """
        Similarity based on growth pattern match
        1.0 = exact match, 0.0 = different
        """
        t_pattern = target.get('growth_pattern')
        c_pattern = candidate.get('growth_pattern')

        if t_pattern is None or c_pattern is None:
            return 0.5  # Unknown

        return 1.0 if t_pattern == c_pattern else 0.0

    def _compute_temporal_similarity(self, target: Dict, candidate: Dict) -> float:
        """
        Similarity based on year proximity
        Recent years more similar
        """
        t_year = target.get('year')
        c_year = candidate.get('year')

        if t_year is None or c_year is None:
            return 0.5  # Unknown

        year_diff = abs(int(t_year) - int(c_year))

        # Exponential decay: adjacent years very similar, distant years not
        similarity = np.exp(-year_diff / 2.0)

        return similarity

    def _compute_farm_match(self, target: Dict, candidate: Dict) -> float:
        """
        Similarity based on farm match
        Same farm = 1.0, different = 0.0
        """
        t_farm = target.get('farm')
        c_farm = candidate.get('farm')

        if t_farm is None or c_farm is None:
            return 0.5  # Unknown

        return 1.0 if t_farm == c_farm else 0.0

    def _compute_volatility_similarity(self, target: Dict, candidate: Dict) -> float:
        """
        Similarity based on volatility (curve smoothness)
        """
        t_vol = target.get('volatility')
        c_vol = candidate.get('volatility')

        if t_vol is None or c_vol is None:
            return 0.5  # Unknown

        # Relative difference in volatility
        vol_diff = abs(t_vol - c_vol) / (abs(t_vol) + 1e-8)
        similarity = 1.0 / (1.0 + vol_diff)

        return similarity

    def rank_candidates(self,
                       target_metadata: Dict,
                       candidates: List[Tuple[int, Dict, float]]) -> List[Tuple[int, Dict, Dict]]:
        """
        Rank all candidates by composite score

        Args:
            target_metadata: Target plot metadata
            candidates: List of (plot_id, metadata, dtw_score) tuples

        Returns:
            List of (plot_id, metadata, feature_scores) sorted by composite score
        """
        results = []

        for plot_id, metadata, dtw_score in candidates:
            # Compute feature scores
            scores = self.compute_similarity(target_metadata, metadata)

            # Optionally blend with DTW score
            scores['dtw'] = dtw_score
            scores['blended'] = 0.5 * scores['composite'] + 0.5 * dtw_score

            results.append((plot_id, metadata, scores))

        # Sort by composite score (descending)
        results.sort(key=lambda x: x[2]['composite'], reverse=True)

        return results

    def update_weights(self, new_weights: Dict[str, float]):
        """
        Update feature weights dynamically

        Args:
            new_weights: Dict with new weight values
        """
        for key, value in new_weights.items():
            if key in self.weights:
                self.weights[key] = value

        # Renormalize
        total = sum(self.weights.values())
        self.weights = {k: v/total for k, v in self.weights.items()}

    def get_weights(self) -> Dict[str, float]:
        """Return current weights"""
        return self.weights.copy()
