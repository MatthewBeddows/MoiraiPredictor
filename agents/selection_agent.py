"""
Selection Agent - Combine multiple strategies for plot selection

Deterministic agent that implements various selection strategies.
Combines scores from multiple sources (DTW, features, diversity, etc.)
"""

import numpy as np
from typing import List, Dict, Tuple


class SelectionAgent:
    """
    Implements various selection strategies
    All deterministic - no LLM needed
    """

    def __init__(self):
        """Initialize selection agent"""
        pass

    def borda_count(self,
                   rankings: List[List[int]],
                   weights: List[float] = None) -> List[int]:
        """
        Combine multiple rankings using Borda count method

        Args:
            rankings: List of rankings (each is list of plot IDs in order)
            weights: Optional weights for each ranking (default: equal)

        Returns:
            Combined ranking (list of plot IDs)
        """
        if not rankings:
            return []

        if weights is None:
            weights = [1.0] * len(rankings)

        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

        # Collect all plot IDs
        all_plots = set()
        for ranking in rankings:
            all_plots.update(ranking)

        # Compute Borda scores
        borda_scores = {}

        for plot_id in all_plots:
            score = 0.0

            for ranking, weight in zip(rankings, weights):
                if plot_id in ranking:
                    # Points = (num_plots - rank)
                    rank = ranking.index(plot_id) + 1
                    points = len(ranking) - rank + 1
                    score += points * weight

            borda_scores[plot_id] = score

        # Sort by Borda score (descending)
        sorted_plots = sorted(borda_scores.keys(),
                            key=lambda x: borda_scores[x],
                            reverse=True)

        return sorted_plots

    def diversity_weighted_selection(self,
                                    candidates: List[Tuple[int, Dict, float]],
                                    top_k: int,
                                    diversity_weight: float = 0.15) -> List[int]:
        """
        Select plots with diversity bonus

        Args:
            candidates: List of (plot_id, metadata, score) sorted by score
            top_k: Number to select
            diversity_weight: Weight for diversity bonus

        Returns:
            Selected plot IDs
        """
        selected = []
        farms_used = set()
        years_used = set()
        patterns_used = set()

        # Make adjustable scores
        adjusted_candidates = []

        for plot_id, metadata, base_score in candidates:
            adjusted_candidates.append({
                'plot_id': plot_id,
                'metadata': metadata,
                'base_score': base_score,
                'adjusted_score': base_score
            })

        # Greedy selection with diversity boost
        while len(selected) < top_k and adjusted_candidates:
            # Recalculate adjusted scores based on current selection
            for cand in adjusted_candidates:
                if cand['plot_id'] in selected:
                    cand['adjusted_score'] = -999  # Already selected
                    continue

                diversity_bonus = 0.0
                meta = cand['metadata']

                # Bonus for new farm
                if meta.get('farm') and meta['farm'] not in farms_used:
                    diversity_bonus += diversity_weight

                # Bonus for new year
                if meta.get('year') and meta['year'] not in years_used:
                    diversity_bonus += diversity_weight * 0.8

                # Bonus for new pattern
                if meta.get('growth_pattern') and meta['growth_pattern'] not in patterns_used:
                    diversity_bonus += diversity_weight * 0.5

                cand['adjusted_score'] = cand['base_score'] + diversity_bonus

            # Select best remaining
            best = max(adjusted_candidates, key=lambda x: x['adjusted_score'])

            if best['adjusted_score'] <= -999:
                break  # All selected

            selected.append(best['plot_id'])

            # Update diversity tracking
            meta = best['metadata']
            if meta.get('farm'):
                farms_used.add(meta['farm'])
            if meta.get('year'):
                years_used.add(meta['year'])
            if meta.get('growth_pattern'):
                patterns_used.add(meta['growth_pattern'])

        return selected

    def redundancy_filtered_selection(self,
                                     candidates: List[Tuple[int, Dict, float]],
                                     top_k: int,
                                     redundancy_threshold: float = 0.95) -> List[int]:
        """
        Select plots while avoiding redundant similar plots

        Args:
            candidates: List of (plot_id, metadata, score)
            top_k: Number to select
            redundancy_threshold: Similarity threshold for redundancy (0-1)

        Returns:
            Selected plot IDs
        """
        selected = []
        selected_metadata = []

        for plot_id, metadata, score in candidates:
            if len(selected) >= top_k:
                break

            # Check if too similar to already selected
            is_redundant = False

            for sel_meta in selected_metadata:
                similarity = self._compute_metadata_similarity(metadata, sel_meta)
                if similarity > redundancy_threshold:
                    is_redundant = True
                    break

            if not is_redundant:
                selected.append(plot_id)
                selected_metadata.append(metadata)

        # If we don't have enough (too strict), fill with best remaining
        if len(selected) < top_k:
            remaining = [pid for pid, _, _ in candidates if pid not in selected]
            selected.extend(remaining[:top_k - len(selected)])

        return selected[:top_k]

    def _compute_metadata_similarity(self, meta1: Dict, meta2: Dict) -> float:
        """
        Compute similarity between two metadata dicts

        Args:
            meta1, meta2: Metadata dicts

        Returns:
            Similarity score (0-1)
        """
        similarities = []

        # Categorical matches
        categorical_features = ['farm', 'year', 'growth_pattern', 'crop']
        for feat in categorical_features:
            if feat in meta1 and feat in meta2:
                sim = 1.0 if meta1[feat] == meta2[feat] else 0.0
                similarities.append(sim)

        # Numeric features
        numeric_features = ['mean_yield', 'peak_week', 'volatility', 'trend_slope']
        for feat in numeric_features:
            if feat in meta1 and feat in meta2:
                v1, v2 = meta1[feat], meta2[feat]
                if v1 is not None and v2 is not None:
                    # Relative difference
                    if abs(v1) > 1e-8:
                        diff = abs(v1 - v2) / abs(v1)
                    else:
                        diff = abs(v2)
                    sim = 1.0 / (1.0 + diff)
                    similarities.append(sim)

        if not similarities:
            return 0.5

        return float(np.mean(similarities))

    def weighted_ensemble_selection(self,
                                   score_sources: List[Dict],
                                   weights: List[float],
                                   top_k: int) -> List[int]:
        """
        Combine multiple score sources with weights

        Args:
            score_sources: List of dicts mapping plot_id -> score
            weights: Weight for each score source
            top_k: Number to select

        Returns:
            Selected plot IDs
        """
        if not score_sources:
            return []

        # Normalize weights
        total = sum(weights)
        weights = [w / total for w in weights]

        # Get all plots
        all_plots = set()
        for source in score_sources:
            all_plots.update(source.keys())

        # Compute weighted combined scores
        combined_scores = {}

        for plot_id in all_plots:
            score = 0.0

            for source, weight in zip(score_sources, weights):
                if plot_id in source:
                    # Normalize scores to 0-1 within each source
                    source_values = list(source.values())
                    min_val = min(source_values)
                    max_val = max(source_values)

                    if max_val > min_val:
                        normalized = (source[plot_id] - min_val) / (max_val - min_val)
                    else:
                        normalized = 1.0

                    score += normalized * weight
                else:
                    # Missing from this source = 0 score
                    score += 0.0

            combined_scores[plot_id] = score

        # Sort and select top K
        sorted_plots = sorted(combined_scores.keys(),
                            key=lambda x: combined_scores[x],
                            reverse=True)

        return sorted_plots[:top_k]

    def stratified_selection(self,
                           candidates: List[Tuple[int, Dict, float]],
                           top_k: int,
                           stratify_by: str = 'farm') -> List[int]:
        """
        Select plots ensuring representation across strata

        Args:
            candidates: List of (plot_id, metadata, score)
            top_k: Number to select
            stratify_by: Feature to stratify by ('farm', 'year', 'pattern')

        Returns:
            Selected plot IDs
        """
        # Group by strata
        strata = {}

        for plot_id, metadata, score in candidates:
            stratum = metadata.get(stratify_by, 'unknown')

            if stratum not in strata:
                strata[stratum] = []

            strata[stratum].append((plot_id, score))

        # Sort within each stratum
        for stratum in strata:
            strata[stratum].sort(key=lambda x: x[1], reverse=True)

        # Allocate slots proportionally to stratum size
        total_candidates = sum(len(s) for s in strata.values())
        selected = []

        # Round-robin selection from strata
        stratum_lists = list(strata.values())
        stratum_indices = [0] * len(stratum_lists)

        while len(selected) < top_k:
            for i, stratum_list in enumerate(stratum_lists):
                if len(selected) >= top_k:
                    break

                if stratum_indices[i] < len(stratum_list):
                    plot_id, score = stratum_list[stratum_indices[i]]
                    if plot_id not in selected:
                        selected.append(plot_id)
                    stratum_indices[i] += 1

            # If no progress, break
            if all(idx >= len(lst) for idx, lst in zip(stratum_indices, stratum_lists)):
                break

        return selected[:top_k]
