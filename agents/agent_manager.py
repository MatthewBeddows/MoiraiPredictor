"""
Agent Manager - Decides which agents to use based on data characteristics

This is the intelligence layer that analyzes the data and decides
which combination of agents will work best.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from .diversity_agent import DiversityAgent
from .feature_scoring_agent import FeatureScoringAgent
from .aggregation_agent import AggregationAgent
from .selection_agent import SelectionAgent


class AgentManager:
    """
    Coordinates multiple agents and decides which strategies to use
    Based on data characteristics
    """

    def __init__(self, llm_agent=None, verbose: bool = True):
        """
        Initialize agent manager

        Args:
            llm_agent: Optional LLM agent for AI-powered decisions
            verbose: Print decisions and reasoning
        """
        self.llm_agent = llm_agent
        self.verbose = verbose

        # Initialize deterministic agents
        self.diversity_agent = DiversityAgent()
        self.feature_agent = FeatureScoringAgent()
        self.aggregation_agent = AggregationAgent()
        self.selection_agent = SelectionAgent()

    def analyze_data_characteristics(self,
                                     training_plots: List[Tuple[int, Dict]],
                                     test_plots_metadata: List[Dict]) -> Dict:
        """
        Analyze data to understand characteristics

        Args:
            training_plots: List of (plot_id, metadata) for training
            test_plots_metadata: List of metadata for test plots

        Returns:
            Dict with data characteristics
        """
        train_metas = [meta for _, meta in training_plots]

        # Compute aggregates
        train_profile = self.aggregation_agent.compute_aggregate_profile(train_metas)
        test_profile = self.aggregation_agent.compute_aggregate_profile(test_plots_metadata)

        # Analyze diversity
        train_diversity = {
            'num_farms': len(train_profile.get('farm_distribution', {})),
            'num_years': len(train_profile.get('year_distribution', {})),
            'num_patterns': len(train_profile.get('growth_pattern_distribution', {})),
            'farm_diversity': train_profile.get('farm_diversity', 0),
            'year_diversity': train_profile.get('year_diversity', 0),
            'pattern_diversity': train_profile.get('growth_pattern_diversity', 0)
        }

        # Check data quality
        characteristics = {
            'num_training_plots': len(training_plots),
            'num_test_plots': len(test_plots_metadata),
            'train_profile': train_profile,
            'test_profile': test_profile,
            'train_diversity': train_diversity,
            'data_size': 'small' if len(training_plots) < 50 else ('medium' if len(training_plots) < 150 else 'large'),
            'diversity_level': 'low' if train_diversity['farm_diversity'] < 0.3 else ('medium' if train_diversity['farm_diversity'] < 0.6 else 'high')
        }

        return characteristics

    def decide_strategy(self,
                       data_characteristics: Dict,
                       top_k: int,
                       use_llm: bool = False) -> Dict:
        """
        Decide which agents and strategies to use

        Args:
            data_characteristics: Output from analyze_data_characteristics
            top_k: Number of plots to select
            use_llm: Whether LLM is available

        Returns:
            Strategy configuration dict
        """
        strategy = {
            'use_feature_scoring': True,  # Always useful
            'use_diversity_boost': False,
            'use_redundancy_filter': False,
            'use_stratified_selection': False,
            'use_aggregate_profile': False,
            'use_llm_ranking': False,
            'diversity_weight': 0.15,
            'selection_method': 'score_based'  # 'score_based', 'diversity_weighted', 'stratified', 'redundancy_filtered'
        }

        num_training = data_characteristics['num_training_plots']
        num_test = data_characteristics['num_test_plots']
        diversity_level = data_characteristics['diversity_level']

        # Decision logic based on data

        # 1. Low diversity → Need diversity boost
        if diversity_level == 'low':
            strategy['use_diversity_boost'] = True
            strategy['diversity_weight'] = 0.25  # Higher weight
            strategy['selection_method'] = 'diversity_weighted'

            if self.verbose:
                print(f"📊 Decision: Low diversity detected → Using diversity-weighted selection")

        # 2. Many test plots → Use aggregate profile
        if num_test >= 50:
            strategy['use_aggregate_profile'] = True

            if self.verbose:
                print(f"📊 Decision: {num_test} test plots → Using aggregate target profile")

        # 3. Large training set → Filter redundancy
        if num_training > 100 and top_k < 30:
            strategy['use_redundancy_filter'] = True
            strategy['selection_method'] = 'redundancy_filtered'

            if self.verbose:
                print(f"📊 Decision: {num_training} training plots, selecting {top_k} → Filter redundancy")

        # 4. Medium data + good diversity → Stratified selection
        if num_training >= 50 and diversity_level in ['medium', 'high'] and top_k >= 15:
            strategy['use_stratified_selection'] = True
            strategy['selection_method'] = 'stratified'

            if self.verbose:
                print(f"📊 Decision: Good diversity + K={top_k} → Stratified selection by farm")

        # 5. LLM available + enough candidates → Use LLM for refinement
        if use_llm and self.llm_agent and num_training >= 30 and top_k >= 15:
            strategy['use_llm_ranking'] = True

            if self.verbose:
                print(f"📊 Decision: LLM available + K≥15 → Enable LLM ranking")

        # 6. Small K → Just use top scores (no fancy strategies)
        if top_k < 10:
            strategy['use_diversity_boost'] = False
            strategy['use_redundancy_filter'] = False
            strategy['use_stratified_selection'] = False
            strategy['selection_method'] = 'score_based'

            if self.verbose:
                print(f"📊 Decision: K={top_k} (small) → Simple score-based selection")

        return strategy

    def select_training_plots(self,
                             target_metadata: Dict,
                             candidates: List[Tuple[int, Dict, float]],
                             top_k: int,
                             strategy: Dict) -> List[int]:
        """
        Select training plots using decided strategy

        Args:
            target_metadata: Target plot metadata (or aggregate)
            candidates: List of (plot_id, metadata, dtw_score)
            top_k: Number to select
            strategy: Strategy config from decide_strategy()

        Returns:
            List of selected plot IDs
        """
        # Step 1: Feature scoring
        if strategy['use_feature_scoring']:
            if self.verbose:
                print(f"\n🔄 Step 1: Computing multi-dimensional feature scores...")

            scored_candidates = self.feature_agent.rank_candidates(
                target_metadata,
                candidates
            )

            # Convert to (plot_id, metadata, composite_score) format
            candidates = [
                (pid, meta, scores['composite'])
                for pid, meta, scores in scored_candidates
            ]

        # Step 2: Apply selection method
        if self.verbose:
            print(f"🔄 Step 2: Applying {strategy['selection_method']} selection...")

        if strategy['selection_method'] == 'diversity_weighted':
            selected = self.selection_agent.diversity_weighted_selection(
                candidates,
                top_k,
                diversity_weight=strategy['diversity_weight']
            )

        elif strategy['selection_method'] == 'redundancy_filtered':
            selected = self.selection_agent.redundancy_filtered_selection(
                candidates,
                top_k,
                redundancy_threshold=0.92
            )

        elif strategy['selection_method'] == 'stratified':
            selected = self.selection_agent.stratified_selection(
                candidates,
                top_k,
                stratify_by='farm'
            )

        else:  # score_based (default)
            selected = [pid for pid, _, _ in candidates[:top_k]]

        # Step 3: LLM ranking (if enabled)
        if strategy['use_llm_ranking'] and self.llm_agent:
            if self.verbose:
                print(f"🔄 Step 3: LLM ranking refinement...")

            # Use LLM to rerank the selected plots (not add new ones)
            # This is less intrusive - LLM just reorders, doesn't change selection
            # TODO: Implement LLM reranking here
            pass

        # Step 4: Diversity check
        if self.verbose:
            print(f"\n🔄 Step 4: Diversity check...")

        metadata_lookup = {pid: meta for pid, meta, _ in candidates}
        diversity_report = self.diversity_agent.analyze_diversity(selected, metadata_lookup)

        if self.verbose:
            print(f"✓ Selected {len(selected)} plots:")
            print(f"  - {diversity_report['num_farms']} farms: {diversity_report['farms']}")
            print(f"  - {diversity_report['num_years']} years: {diversity_report['years']}")
            print(f"  - {diversity_report['num_patterns']} growth patterns")
            print(f"  - Diversity score: {diversity_report['diversity_score']:.2f}/1.0")

            if diversity_report['warnings']:
                print(f"\n⚠️  Diversity warnings:")
                for warning in diversity_report['warnings']:
                    print(f"  - {warning}")

        return selected

    def execute_full_pipeline(self,
                             training_plots: List[Tuple[int, Dict]],
                             test_plots_metadata: List[Dict],
                             dtw_scores: Dict[int, float],
                             top_k: int,
                             use_llm: bool = False) -> Tuple[List[int], Dict]:
        """
        Execute complete selection pipeline with automatic strategy decision

        Args:
            training_plots: List of (plot_id, metadata)
            test_plots_metadata: List of test plot metadata
            dtw_scores: Dict mapping plot_id -> dtw_similarity_score
            top_k: Number of plots to select
            use_llm: Whether to use LLM if available

        Returns:
            Tuple of (selected_plot_ids, strategy_used)
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print("AGENT MANAGER: AUTOMATIC STRATEGY SELECTION")
            print(f"{'='*60}")

        # Step 1: Analyze data
        if self.verbose:
            print(f"\n📊 Analyzing data characteristics...")

        characteristics = self.analyze_data_characteristics(training_plots, test_plots_metadata)

        if self.verbose:
            print(f"✓ Training plots: {characteristics['num_training_plots']}")
            print(f"✓ Test plots: {characteristics['num_test_plots']}")
            print(f"✓ Data size: {characteristics['data_size']}")
            print(f"✓ Diversity level: {characteristics['diversity_level']}")

        # Step 2: Decide strategy
        if self.verbose:
            print(f"\n🤔 Deciding optimal strategy...")

        strategy = self.decide_strategy(characteristics, top_k, use_llm)

        # Step 3: Prepare target metadata
        if strategy['use_aggregate_profile']:
            target_metadata = self.aggregation_agent.compute_representative_metadata(test_plots_metadata)
            if self.verbose:
                print(f"✓ Using aggregate profile from {len(test_plots_metadata)} test plots")
        else:
            target_metadata = test_plots_metadata[0] if test_plots_metadata else {}
            if self.verbose:
                print(f"✓ Using single representative test plot")

        # Step 4: Prepare candidates
        candidates = [
            (plot_id, metadata, dtw_scores.get(plot_id, 0.0))
            for plot_id, metadata in training_plots
        ]

        # Step 5: Select plots
        selected = self.select_training_plots(target_metadata, candidates, top_k, strategy)

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"✅ SELECTION COMPLETE: {len(selected)} plots selected")
            print(f"{'='*60}")

        return selected, strategy
