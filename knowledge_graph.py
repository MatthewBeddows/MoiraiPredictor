# knowledge_graph.py
import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional
import pickle
from scipy.spatial.distance import euclidean
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import cosine_similarity
from dtaidistance import dtw

class AgriculturalKnowledgeGraph:
    """Simple knowledge graph for agricultural time series data"""
    
    def __init__(self):
        self.G = nx.DiGraph()
        self.plot_data = {}  # Store actual time series data
        
    def _extract_curve_features(self, time_series: pd.DataFrame) -> Dict:
        """
        Extract mathematical features that describe the shape of a yield curve

        Args:
            time_series: DataFrame with 'target' column

        Returns:
            Dict of curve shape features
        """
        values = time_series['target'].dropna().values

        if len(values) < 3:
            return {}

        # Statistical features
        mean_val = float(np.mean(values))
        std_val = float(np.std(values))
        cv = std_val / (mean_val + 1e-8)  # Coefficient of variation

        # Trend features
        x = np.arange(len(values))
        slope, _ = np.polyfit(x, values, 1)  # Linear trend

        # Shape features
        peak_idx = int(np.argmax(values))
        peak_week = peak_idx / len(values)  # Normalized position of peak (0-1)
        peak_value = float(np.max(values))
        valley_value = float(np.min(values))
        range_val = peak_value - valley_value

        # Volatility/smoothness
        diffs = np.diff(values)
        volatility = float(np.std(diffs))

        # Growth pattern
        early_mean = float(np.mean(values[:len(values)//3]))  # First third
        mid_mean = float(np.mean(values[len(values)//3:2*len(values)//3]))  # Middle third
        late_mean = float(np.mean(values[2*len(values)//3:]))  # Last third

        growth_pattern = "flat"
        if late_mean > early_mean * 1.1:
            growth_pattern = "increasing"
        elif late_mean < early_mean * 0.9:
            growth_pattern = "decreasing"
        elif mid_mean > early_mean * 1.1 and mid_mean > late_mean * 1.1:
            growth_pattern = "peak_middle"

        return {
            'mean_yield': mean_val,
            'std_yield': std_val,
            'cv_yield': cv,
            'trend_slope': float(slope),
            'peak_week': peak_week,
            'peak_value': peak_value,
            'valley_value': valley_value,
            'range': range_val,
            'volatility': volatility,
            'early_avg': early_mean,
            'mid_avg': mid_mean,
            'late_avg': late_mean,
            'growth_pattern': growth_pattern
        }

    def add_plot(self, plot_id: int, metadata: Dict, time_series: pd.DataFrame, quality_score: float = None):
        """
        Add a plot to the knowledge graph

        Args:
            plot_id: Unique plot identifier
            metadata: Dict with plot metadata (farm, soil, crop, etc.)
            time_series: DataFrame with date index and target column
            quality_score: Optional data quality score (0-100)
        """
        # Extract curve shape features
        curve_features = self._extract_curve_features(time_series)

        # Add node with metadata AND curve features
        self.G.add_node(
            plot_id,
            farm=metadata.get('farm'),
            soil=metadata.get('soil'),
            crop=metadata.get('crop'),
            location=metadata.get('location'),
            year=metadata.get('year'),
            num_weeks=len(time_series),
            quality_score=quality_score,  # Add quality score
            **curve_features  # Add all curve features to node
        )

        # Store time series data separately (more efficient)
        self.plot_data[plot_id] = time_series.copy()

        quality_msg = f" (quality: {quality_score:.1f})" if quality_score is not None else ""
        print(f"Added plot {plot_id}: {metadata.get('crop')} at {metadata.get('farm')}{quality_msg}")
    
    def add_relationship(self, plot1: int, plot2: int, relationship_type: str, weight: float = 1.0):
        """Add a relationship edge between two plots"""
        self.G.add_edge(
            plot1, plot2,
            type=relationship_type,
            weight=weight
        )
    
    def get_plot_metadata(self, plot_id: int) -> Dict:
        """Get metadata for a specific plot"""
        if plot_id not in self.G.nodes:
            raise ValueError(f"Plot {plot_id} not in knowledge graph")
        return dict(self.G.nodes[plot_id])
    
    def get_plot_data(self, plot_id: int) -> pd.DataFrame:
        """Get time series data for a specific plot"""
        if plot_id not in self.plot_data:
            raise ValueError(f"No data for plot {plot_id}")
        return self.plot_data[plot_id].copy()
    
    def get_all_plots(self) -> List[int]:
        """Get list of all plot IDs in the graph"""
        return list(self.G.nodes())
    
    def compute_yield_curve_similarity(self,
                                      plot1_id: int,
                                      plot2_id: int,
                                      method: str = 'dtw') -> float:
        """
        Compute similarity between two yield curves

        Args:
            plot1_id: First plot ID
            plot2_id: Second plot ID
            method: Similarity method - 'dtw', 'pearson', 'spearman', 'cosine', 'euclidean'

        Returns:
            Similarity score (higher = more similar, except for euclidean/dtw which are distances)
        """
        # Get target column data
        data1 = self.plot_data[plot1_id]['target'].dropna().values
        data2 = self.plot_data[plot2_id]['target'].dropna().values

        if len(data1) == 0 or len(data2) == 0:
            return 0.0

        # Normalize to comparable scale (0-1)
        data1_norm = (data1 - data1.min()) / (data1.max() - data1.min() + 1e-8)
        data2_norm = (data2 - data2.min()) / (data2.max() - data2.min() + 1e-8)

        if method == 'dtw':
            # Dynamic Time Warping - handles different lengths and temporal shifts
            distance = dtw.distance(data1_norm, data2_norm)
            # Convert distance to similarity (0-1 scale, 1 = identical)
            similarity = 1.0 / (1.0 + distance)
            return similarity

        elif method == 'pearson':
            # Pearson correlation - linear relationship
            min_len = min(len(data1_norm), len(data2_norm))
            if min_len < 2:
                return 0.0
            corr, _ = pearsonr(data1_norm[:min_len], data2_norm[:min_len])
            return corr if not np.isnan(corr) else 0.0

        elif method == 'spearman':
            # Spearman correlation - monotonic relationship
            min_len = min(len(data1_norm), len(data2_norm))
            if min_len < 2:
                return 0.0
            corr, _ = spearmanr(data1_norm[:min_len], data2_norm[:min_len])
            return corr if not np.isnan(corr) else 0.0

        elif method == 'cosine':
            # Cosine similarity - direction similarity
            min_len = min(len(data1_norm), len(data2_norm))
            sim = cosine_similarity(
                data1_norm[:min_len].reshape(1, -1),
                data2_norm[:min_len].reshape(1, -1)
            )[0, 0]
            return sim if not np.isnan(sim) else 0.0

        elif method == 'euclidean':
            # Euclidean distance
            min_len = min(len(data1_norm), len(data2_norm))
            dist = euclidean(data1_norm[:min_len], data2_norm[:min_len])
            # Convert to similarity
            similarity = 1.0 / (1.0 + dist)
            return similarity

        else:
            raise ValueError(f"Unknown similarity method: {method}")

    def find_similar_plots(self,
                          target_plot_id: int,
                          method: str = 'dtw',
                          top_k: Optional[int] = None,
                          threshold: Optional[float] = None) -> List[Tuple[int, float]]:
        """
        Find plots with similar yield curves to the target

        Args:
            target_plot_id: Target plot to compare against
            method: Similarity method to use
            top_k: Return top K most similar plots (if None, use threshold)
            threshold: Minimum similarity threshold (if None, use top_k)

        Returns:
            List of (plot_id, similarity_score) tuples, sorted by similarity (descending)
        """
        similarities = []

        for plot_id in self.G.nodes():
            if plot_id == target_plot_id:
                continue

            try:
                sim = self.compute_yield_curve_similarity(target_plot_id, plot_id, method)
                similarities.append((plot_id, sim))
            except Exception as e:
                print(f"Warning: Could not compute similarity for plot {plot_id}: {e}")
                continue

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Apply filtering
        if top_k is not None:
            return similarities[:top_k]
        elif threshold is not None:
            return [(pid, sim) for pid, sim in similarities if sim >= threshold]
        else:
            return similarities

    def rank_all_plots(self,
                      target_plot_id: int,
                      similarity_method: str = 'dtw',
                      use_llm_agent: bool = False,
                      llm_agent = None,
                      output_file: str = None) -> Dict:
        """
        Rank ALL plots by both similarity score and LLM relevance

        Args:
            target_plot_id: The target plot to compare against
            similarity_method: Method to compute similarity
            use_llm_agent: Whether to also get LLM rankings
            llm_agent: LLMAgent instance (required if use_llm_agent=True)
            output_file: Path to save rankings JSON file

        Returns:
            Dict with similarity rankings and LLM rankings
        """
        print(f"\n{'='*60}")
        print(f"RANKING ALL PLOTS FOR TARGET PLOT {target_plot_id}")
        print(f"{'='*60}")

        # Get similarity scores for ALL plots
        all_plots = [pid for pid in self.G.nodes() if pid != target_plot_id]

        print(f"Computing {similarity_method} similarity for {len(all_plots)} plots...")
        similarity_rankings = []

        for plot_id in all_plots:
            try:
                sim = self.compute_yield_curve_similarity(target_plot_id, plot_id, similarity_method)
                metadata = self.get_plot_metadata(plot_id)
                similarity_rankings.append({
                    'plot_id': plot_id,
                    'similarity_score': float(sim),
                    'metadata': dict(metadata)
                })
            except Exception as e:
                print(f"Warning: Could not compute similarity for plot {plot_id}: {e}")
                continue

        # Sort by similarity (descending)
        similarity_rankings.sort(key=lambda x: x['similarity_score'], reverse=True)

        print(f"✓ Similarity ranking complete")
        print(f"  Top 5 by {similarity_method}: {[r['plot_id'] for r in similarity_rankings[:5]]}")
        print(f"  Similarity range: {similarity_rankings[-1]['similarity_score']:.3f} to {similarity_rankings[0]['similarity_score']:.3f}")

        # Get LLM rankings if requested
        llm_rankings = None
        if use_llm_agent and llm_agent is not None:
            print(f"\n🤖 Getting LLM relevance rankings for ALL plots...")

            target_metadata = self.get_plot_metadata(target_plot_id)

            # Prepare all candidates with metadata for LLM
            candidates_with_metadata = [
                (r['plot_id'], r['metadata'], r['similarity_score'])
                for r in similarity_rankings
            ]

            # Get LLM's ranked list of ALL plots
            llm_rankings = llm_agent.rank_all_plots_by_relevance(
                target_metadata,
                candidates_with_metadata
            )

            print(f"✓ LLM ranking complete")
            if llm_rankings:
                print(f"  Top 5 by LLM: {llm_rankings[:5]}")

        # Create comparison data structure
        result = {
            'target_plot_id': target_plot_id,
            'target_metadata': dict(self.get_plot_metadata(target_plot_id)),
            'similarity_method': similarity_method,
            'total_plots_ranked': len(similarity_rankings),
            'similarity_rankings': similarity_rankings,
            'llm_rankings': llm_rankings
        }

        # Save to CSV file if specified
        if output_file:
            import csv
            csv_file = output_file.replace('.json', '.csv')

            # Create LLM rank mapping and Borda scores
            llm_rank_map = {}
            borda_scores = {}
            borda_rank_map = {}

            if llm_rankings:
                # LLM ranks
                for rank, plot_id in enumerate(llm_rankings, 1):
                    llm_rank_map[plot_id] = rank

                # Calculate Borda scores (sum of inverted ranks from both methods)
                total_plots = len(similarity_rankings)
                for rank, r in enumerate(similarity_rankings, 1):
                    plot_id = r['plot_id']
                    borda_scores[plot_id] = total_plots - rank

                for rank, plot_id in enumerate(llm_rankings, 1):
                    borda_scores[plot_id] = borda_scores.get(plot_id, 0) + (total_plots - rank)

                # Create Borda rank mapping
                sorted_by_borda = sorted(borda_scores.keys(), key=lambda x: borda_scores[x], reverse=True)
                for rank, plot_id in enumerate(sorted_by_borda, 1):
                    borda_rank_map[plot_id] = rank

            with open(csv_file, 'w', newline='') as f:
                # Define all columns including curve features and Borda
                fieldnames = [
                    'similarity_rank', 'llm_rank', 'borda_rank', 'borda_score', 'plot_id', 'similarity_score',
                    'farm', 'year', 'crop', 'soil', 'location',
                    'mean_yield', 'std_yield', 'cv_yield', 'trend_slope',
                    'peak_week', 'peak_value', 'valley_value', 'range',
                    'volatility', 'early_avg', 'mid_avg', 'late_avg', 'growth_pattern'
                ]

                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

                # Write all plots with their rankings
                for sim_rank, r in enumerate(similarity_rankings, 1):
                    plot_id = r['plot_id']
                    meta = r['metadata']
                    llm_rank = llm_rank_map.get(plot_id, '')
                    borda_rank = borda_rank_map.get(plot_id, '')
                    borda_score = borda_scores.get(plot_id, '')

                    row = {
                        'similarity_rank': sim_rank,
                        'llm_rank': llm_rank,
                        'borda_rank': borda_rank,
                        'borda_score': borda_score,
                        'plot_id': plot_id,
                        'similarity_score': f"{r['similarity_score']:.4f}",
                        'farm': meta.get('farm', ''),
                        'year': meta.get('year', ''),
                        'crop': meta.get('crop', ''),
                        'soil': meta.get('soil', ''),
                        'location': meta.get('location', ''),
                        'mean_yield': f"{meta.get('mean_yield', ''):.2f}" if meta.get('mean_yield') else '',
                        'std_yield': f"{meta.get('std_yield', ''):.2f}" if meta.get('std_yield') else '',
                        'cv_yield': f"{meta.get('cv_yield', ''):.4f}" if meta.get('cv_yield') else '',
                        'trend_slope': f"{meta.get('trend_slope', ''):.4f}" if meta.get('trend_slope') else '',
                        'peak_week': f"{meta.get('peak_week', ''):.4f}" if meta.get('peak_week') else '',
                        'peak_value': f"{meta.get('peak_value', ''):.2f}" if meta.get('peak_value') else '',
                        'valley_value': f"{meta.get('valley_value', ''):.2f}" if meta.get('valley_value') else '',
                        'range': f"{meta.get('range', ''):.2f}" if meta.get('range') else '',
                        'volatility': f"{meta.get('volatility', ''):.2f}" if meta.get('volatility') else '',
                        'early_avg': f"{meta.get('early_avg', ''):.2f}" if meta.get('early_avg') else '',
                        'mid_avg': f"{meta.get('mid_avg', ''):.2f}" if meta.get('mid_avg') else '',
                        'late_avg': f"{meta.get('late_avg', ''):.2f}" if meta.get('late_avg') else '',
                        'growth_pattern': meta.get('growth_pattern', '')
                    }
                    writer.writerow(row)

            print(f"\n✓ Rankings saved to CSV: {csv_file}")

        return result

    def retrieve_training_data(self,
                              target_plot_id: int = None,
                              filters: Dict = None,
                              similarity_method: Optional[str] = None,
                              top_k_similar: Optional[int] = None,
                              similarity_threshold: Optional[float] = None,
                              use_llm_agent: bool = False,
                              llm_agent = None) -> List[Tuple[int, pd.DataFrame]]:
        """
        Retrieve training data from knowledge graph with optional similarity filtering

        Args:
            target_plot_id: If provided, exclude this plot (for testing)
            filters: Dict of filters to apply (e.g., {'crop': 'wheat', 'year': 2020})
            similarity_method: If provided, filter by similarity to target plot
            top_k_similar: Return top K most similar plots
            similarity_threshold: Minimum similarity threshold
            use_llm_agent: If True, use LLM agent to refine selection
            llm_agent: LLMAgent instance (required if use_llm_agent=True)

        Returns:
            List of (plot_id, time_series) tuples
        """
        results = []

        # If similarity-based retrieval is requested
        if target_plot_id and similarity_method:
            similar_plots = self.find_similar_plots(
                target_plot_id,
                method=similarity_method,
                top_k=top_k_similar,
                threshold=similarity_threshold
            )

            print(f"\nSimilarity-based retrieval using {similarity_method}:")
            print(f"Found {len(similar_plots)} similar plots")
            if len(similar_plots) > 0:
                print(f"Similarity range: {similar_plots[-1][1]:.3f} to {similar_plots[0][1]:.3f}")

            # Use LLM agent to refine selection if requested
            if use_llm_agent and llm_agent is not None:
                print("\n🤖 Using LLM agent to refine plot selection...")

                # Prepare metadata for LLM
                target_metadata = self.get_plot_metadata(target_plot_id)
                candidates_with_metadata = [
                    (plot_id, self.get_plot_metadata(plot_id), similarity)
                    for plot_id, similarity in similar_plots
                ]

                # Get LLM's selection
                selected_ids = llm_agent.analyze_plot_metadata(
                    target_metadata,
                    candidates_with_metadata,
                    top_k=top_k_similar if top_k_similar else len(similar_plots)
                )

                # Filter to only selected plots
                similar_plots = [(pid, sim) for pid, sim in similar_plots if pid in selected_ids]
                print(f"✓ LLM refined selection to {len(similar_plots)} plots")

            for plot_id, similarity in similar_plots:
                # Apply additional filters if provided
                if filters:
                    metadata = self.G.nodes[plot_id]
                    skip = False
                    for key, value in filters.items():
                        if metadata.get(key) != value:
                            skip = True
                            break
                    if skip:
                        continue

                data = self.plot_data[plot_id]
                results.append((plot_id, data))

            return results

        # Otherwise, use original logic (retrieve all except target)
        for plot_id in self.G.nodes():
            # Skip target plot if specified
            if target_plot_id and plot_id == target_plot_id:
                continue

            # Apply filters if provided
            if filters:
                metadata = self.G.nodes[plot_id]
                skip = False
                for key, value in filters.items():
                    if metadata.get(key) != value:
                        skip = True
                        break
                if skip:
                    continue

            # Get the time series data
            data = self.plot_data[plot_id]
            results.append((plot_id, data))

        return results
    
    def save(self, filepath: str):
        """Save knowledge graph to disk"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'graph': self.G,
                'plot_data': self.plot_data
            }, f)
        print(f"Knowledge graph saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str):
        """Load knowledge graph from disk"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        kg = cls()
        kg.G = data['graph']
        kg.plot_data = data['plot_data']
        print(f"Knowledge graph loaded from {filepath}")
        return kg
    
    def summary(self):
        """Print summary statistics of the knowledge graph"""
        print("\n" + "="*60)
        print("KNOWLEDGE GRAPH SUMMARY")
        print("="*60)
        print(f"Total plots: {len(self.G.nodes())}")
        print(f"Total edges: {len(self.G.edges())}")
        
        # Count by crop type
        crops = {}
        for plot_id in self.G.nodes():
            crop = self.G.nodes[plot_id].get('crop', 'unknown')
            crops[crop] = crops.get(crop, 0) + 1
        
        print("\nPlots by crop:")
        for crop, count in sorted(crops.items()):
            print(f"  {crop}: {count}")
        
        # Count by year
        years = {}
        for plot_id in self.G.nodes():
            year = self.G.nodes[plot_id].get('year', 'unknown')
            years[year] = years.get(year, 0) + 1
        
        print("\nPlots by year:")
        for year, count in sorted(years.items()):
            print(f"  {year}: {count}")
        
        print("="*60 + "\n")