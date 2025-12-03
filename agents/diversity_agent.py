"""
Diversity Agent - Ensures training set diversity

Deterministic agent that analyzes and enforces diversity in selected plots.
"""

import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict


class DiversityAgent:
    """
    Analyzes and ensures diversity in training plot selection
    Pure mathematical approach - no LLM needed
    """

    def __init__(self, min_farms: int = 2, min_years: int = 2, min_patterns: int = 2):
        """
        Initialize diversity requirements

        Args:
            min_farms: Minimum number of unique farms desired
            min_years: Minimum number of unique years desired
            min_patterns: Minimum number of unique growth patterns desired
        """
        self.min_farms = min_farms
        self.min_years = min_years
        self.min_patterns = min_patterns

    def analyze_diversity(self, selected_plot_ids: List[int],
                         metadata_lookup: Dict[int, Dict]) -> Dict:
        """
        Analyze diversity of selected plots

        Args:
            selected_plot_ids: List of selected plot IDs
            metadata_lookup: Dict mapping plot_id -> metadata

        Returns:
            Diversity report dict
        """
        farms = set()
        years = set()
        growth_patterns = defaultdict(int)
        locations = set()

        for pid in selected_plot_ids:
            if pid not in metadata_lookup:
                continue

            meta = metadata_lookup[pid]

            if meta.get('farm'):
                farms.add(meta['farm'])
            if meta.get('year'):
                years.add(meta['year'])
            if meta.get('growth_pattern'):
                growth_patterns[meta['growth_pattern']] += 1
            if meta.get('location'):
                locations.add(meta['location'])

        # Calculate diversity metrics
        num_plots = len(selected_plot_ids)
        farm_diversity = len(farms) / num_plots if num_plots > 0 else 0
        year_diversity = len(years) / num_plots if num_plots > 0 else 0
        pattern_diversity = len(growth_patterns) / num_plots if num_plots > 0 else 0

        # Overall diversity score (0-1)
        diversity_score = np.mean([farm_diversity, year_diversity, pattern_diversity])

        # Check if requirements met
        meets_requirements = (
            len(farms) >= self.min_farms and
            len(years) >= self.min_years and
            len(growth_patterns) >= self.min_patterns
        )

        # Generate warnings
        warnings = []
        if len(farms) < self.min_farms:
            warnings.append(f"Only {len(farms)} farm(s) - need {self.min_farms}+ for robustness")
        if len(years) < self.min_years:
            warnings.append(f"Only {len(years)} year(s) - need {self.min_years}+ for temporal diversity")
        if len(growth_patterns) < self.min_patterns:
            warnings.append(f"Only {len(growth_patterns)} growth pattern(s) - need variety")

        # Check for dominance
        if growth_patterns:
            dominant_pattern = max(growth_patterns.items(), key=lambda x: x[1])
            if dominant_pattern[1] > num_plots * 0.7:
                warnings.append(f"70%+ plots have '{dominant_pattern[0]}' pattern - low pattern diversity")

        return {
            'num_plots': num_plots,
            'num_farms': len(farms),
            'num_years': len(years),
            'num_patterns': len(growth_patterns),
            'num_locations': len(locations),
            'farms': sorted(list(farms)),
            'years': sorted(list(years)),
            'patterns': dict(growth_patterns),
            'locations': list(locations),
            'diversity_score': diversity_score,
            'meets_requirements': meets_requirements,
            'warnings': warnings
        }

    def boost_for_diversity(self,
                           candidates: List[Tuple[int, Dict, float]],
                           current_selection: List[int],
                           boost_weight: float = 0.15) -> List[Tuple[int, float]]:
        """
        Apply diversity boost to candidate scores

        Args:
            candidates: List of (plot_id, metadata, score) tuples
            current_selection: Already selected plot IDs
            boost_weight: Weight for diversity bonus

        Returns:
            List of (plot_id, adjusted_score) sorted by adjusted score
        """
        # Analyze current selection
        current_farms = set()
        current_years = set()
        current_patterns = set()

        metadata_lookup = {pid: meta for pid, meta, _ in candidates}

        for pid in current_selection:
            if pid in metadata_lookup:
                meta = metadata_lookup[pid]
                if meta.get('farm'):
                    current_farms.add(meta['farm'])
                if meta.get('year'):
                    current_years.add(meta['year'])
                if meta.get('growth_pattern'):
                    current_patterns.add(meta['growth_pattern'])

        # Score each candidate with diversity bonus
        adjusted_scores = []

        for plot_id, metadata, base_score in candidates:
            if plot_id in current_selection:
                continue  # Already selected

            diversity_bonus = 0.0

            # Bonus for new farm
            if metadata.get('farm') and metadata['farm'] not in current_farms:
                diversity_bonus += boost_weight

            # Bonus for new year
            if metadata.get('year') and metadata['year'] not in current_years:
                diversity_bonus += boost_weight * 0.8

            # Bonus for new pattern
            if metadata.get('growth_pattern') and metadata['growth_pattern'] not in current_patterns:
                diversity_bonus += boost_weight * 0.5

            adjusted_score = base_score + diversity_bonus
            adjusted_scores.append((plot_id, adjusted_score))

        # Sort by adjusted score
        adjusted_scores.sort(key=lambda x: x[1], reverse=True)

        return adjusted_scores

    def enforce_minimum_diversity(self,
                                  ranked_candidates: List[Tuple[int, Dict, float]],
                                  top_k: int,
                                  force_diversity: bool = True) -> List[int]:
        """
        Select top K plots while enforcing minimum diversity

        Args:
            ranked_candidates: Candidates sorted by score (plot_id, metadata, score)
            top_k: Number of plots to select
            force_diversity: If True, enforce diversity even if scores are lower

        Returns:
            List of selected plot IDs
        """
        if not force_diversity:
            # Just take top K
            return [pid for pid, _, _ in ranked_candidates[:top_k]]

        selected = []
        farms_used = set()
        years_used = set()
        patterns_used = set()

        # First pass: Select while maintaining diversity
        for plot_id, metadata, score in ranked_candidates:
            if len(selected) >= top_k:
                break

            farm = metadata.get('farm')
            year = metadata.get('year')
            pattern = metadata.get('growth_pattern')

            # Allow selection if adds diversity OR if we're below minimum requirements
            add_plot = False

            # Always add first few plots
            if len(selected) < 3:
                add_plot = True
            # Enforce diversity constraints
            elif len(selected) < top_k:
                # Check if this adds diversity
                adds_diversity = (
                    (farm and farm not in farms_used) or
                    (year and year not in years_used) or
                    (pattern and pattern not in patterns_used)
                )

                # Check if we meet minimum diversity already
                meets_minimums = (
                    len(farms_used) >= self.min_farms and
                    len(years_used) >= self.min_years and
                    len(patterns_used) >= self.min_patterns
                )

                # Add if adds diversity OR if we already meet minimums
                add_plot = adds_diversity or meets_minimums

            if add_plot:
                selected.append(plot_id)
                if farm:
                    farms_used.add(farm)
                if year:
                    years_used.add(year)
                if pattern:
                    patterns_used.add(pattern)

        # If we didn't get enough plots, fill remainder with best remaining
        if len(selected) < top_k:
            remaining = [pid for pid, _, _ in ranked_candidates if pid not in selected]
            selected.extend(remaining[:top_k - len(selected)])

        return selected[:top_k]

    def compute_diversity_penalty(self, selected_plots: List[int],
                                  metadata_lookup: Dict[int, Dict]) -> float:
        """
        Compute penalty score for lack of diversity (higher = worse diversity)

        Args:
            selected_plots: List of plot IDs
            metadata_lookup: Dict mapping plot_id -> metadata

        Returns:
            Penalty score (0 = perfect diversity, 1 = no diversity)
        """
        report = self.analyze_diversity(selected_plots, metadata_lookup)

        # Invert diversity score to get penalty
        penalty = 1.0 - report['diversity_score']

        # Add extra penalty for not meeting requirements
        if not report['meets_requirements']:
            penalty += 0.2

        return min(1.0, penalty)
