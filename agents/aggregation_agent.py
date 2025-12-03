"""
Aggregation Agent - Compute aggregate statistics across multiple plots

Deterministic agent that computes aggregate profiles.
No LLM needed - pure statistics.
"""

import numpy as np
from typing import List, Dict
from collections import Counter


class AggregationAgent:
    """
    Computes aggregate statistics across multiple plots
    Useful for understanding overall characteristics of test set
    """

    def __init__(self):
        """Initialize aggregation agent"""
        self.numeric_features = [
            'mean_yield', 'std_yield', 'cv_yield', 'trend_slope',
            'peak_week', 'peak_value', 'valley_value', 'range',
            'volatility', 'early_mean', 'mid_mean', 'late_mean'
        ]

        self.categorical_features = [
            'growth_pattern', 'farm', 'crop', 'soil', 'location', 'year'
        ]

    def compute_aggregate_profile(self, metadatas: List[Dict]) -> Dict:
        """
        Compute aggregate statistics across multiple plots

        Args:
            metadatas: List of metadata dicts

        Returns:
            Dict with aggregate statistics
        """
        if not metadatas:
            return {}

        aggregates = {
            'num_plots': len(metadatas)
        }

        # Numeric features: mean, std, min, max
        for feature in self.numeric_features:
            values = [m.get(feature) for m in metadatas if m.get(feature) is not None]

            if values:
                aggregates[f'{feature}_mean'] = float(np.mean(values))
                aggregates[f'{feature}_std'] = float(np.std(values))
                aggregates[f'{feature}_min'] = float(np.min(values))
                aggregates[f'{feature}_max'] = float(np.max(values))
                aggregates[f'{feature}_median'] = float(np.median(values))
            else:
                aggregates[f'{feature}_mean'] = None
                aggregates[f'{feature}_std'] = None
                aggregates[f'{feature}_min'] = None
                aggregates[f'{feature}_max'] = None
                aggregates[f'{feature}_median'] = None

        # Categorical features: mode and distribution
        for feature in self.categorical_features:
            values = [m.get(feature) for m in metadatas if m.get(feature) is not None]

            if values:
                counter = Counter(values)
                aggregates[f'{feature}_mode'] = counter.most_common(1)[0][0]
                aggregates[f'{feature}_distribution'] = dict(counter)
                aggregates[f'{feature}_diversity'] = len(counter) / len(values)  # 0-1
            else:
                aggregates[f'{feature}_mode'] = None
                aggregates[f'{feature}_distribution'] = {}
                aggregates[f'{feature}_diversity'] = 0.0

        return aggregates

    def compute_representative_metadata(self, metadatas: List[Dict]) -> Dict:
        """
        Compute a single "representative" metadata from aggregates

        Args:
            metadatas: List of metadata dicts

        Returns:
            Single metadata dict representing the aggregate
        """
        aggregates = self.compute_aggregate_profile(metadatas)

        representative = {}

        # Use mean/median for numeric features
        for feature in self.numeric_features:
            mean_key = f'{feature}_mean'
            if mean_key in aggregates and aggregates[mean_key] is not None:
                representative[feature] = aggregates[mean_key]

        # Use mode for categorical features
        for feature in self.categorical_features:
            mode_key = f'{feature}_mode'
            if mode_key in aggregates and aggregates[mode_key] is not None:
                representative[feature] = aggregates[mode_key]

        return representative

    def compare_to_aggregate(self,
                            candidate_metadata: Dict,
                            aggregate_profile: Dict) -> float:
        """
        Compare a candidate plot to an aggregate profile

        Args:
            candidate_metadata: Single plot metadata
            aggregate_profile: Aggregate profile from compute_aggregate_profile()

        Returns:
            Similarity score (0-1) to aggregate
        """
        similarities = []

        # Compare numeric features
        for feature in self.numeric_features:
            c_val = candidate_metadata.get(feature)
            a_mean = aggregate_profile.get(f'{feature}_mean')
            a_std = aggregate_profile.get(f'{feature}_std')

            if c_val is not None and a_mean is not None and a_std is not None:
                # Z-score similarity
                if a_std > 1e-8:
                    z_score = abs(c_val - a_mean) / a_std
                    # Convert to similarity (0-1)
                    similarity = np.exp(-z_score / 2.0)
                else:
                    # No variation - exact match or not
                    similarity = 1.0 if abs(c_val - a_mean) < 1e-8 else 0.5

                similarities.append(similarity)

        # Compare categorical features
        for feature in self.categorical_features:
            c_val = candidate_metadata.get(feature)
            a_mode = aggregate_profile.get(f'{feature}_mode')

            if c_val is not None and a_mode is not None:
                similarity = 1.0 if c_val == a_mode else 0.0
                similarities.append(similarity)

        if not similarities:
            return 0.5  # Unknown

        return float(np.mean(similarities))

    def identify_outlier_plots(self,
                               metadatas: List[Dict],
                               threshold: float = 2.0) -> List[Dict]:
        """
        Identify plots that are outliers relative to the aggregate

        Args:
            metadatas: List of metadata dicts
            threshold: Z-score threshold for outlier (default 2.0 = 2 std devs)

        Returns:
            List of outlier metadata dicts with outlier scores
        """
        aggregates = self.compute_aggregate_profile(metadatas)
        outliers = []

        for metadata in metadatas:
            outlier_scores = []

            # Check each numeric feature
            for feature in self.numeric_features:
                val = metadata.get(feature)
                mean = aggregates.get(f'{feature}_mean')
                std = aggregates.get(f'{feature}_std')

                if val is not None and mean is not None and std is not None and std > 1e-8:
                    z_score = abs(val - mean) / std
                    if z_score > threshold:
                        outlier_scores.append({
                            'feature': feature,
                            'value': val,
                            'mean': mean,
                            'z_score': z_score
                        })

            if outlier_scores:
                outliers.append({
                    'metadata': metadata,
                    'outlier_features': outlier_scores,
                    'max_z_score': max(s['z_score'] for s in outlier_scores)
                })

        # Sort by severity (max z-score)
        outliers.sort(key=lambda x: x['max_z_score'], reverse=True)

        return outliers
