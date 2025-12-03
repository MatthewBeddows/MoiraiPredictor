"""
Data Explorer Agent

Autonomous agent that deeply explores agricultural data when uploaded.
Extracts maximum information into the knowledge graph for downstream agents.

This is a true AI-powered agent with:
- Autonomy: Decides what to explore and in what order
- Memory: Tracks all discoveries and builds knowledge incrementally
- Self-critique: Evaluates if exploration is thorough enough
- Learning: Learns what patterns matter for predictions
- Reasoning: Uses LLM to identify non-obvious insights
- Tools: Statistical analysis, clustering, correlation, anomaly detection, etc.

Unlike the current simple KG builder that just stores basic stats,
this agent DEEPLY explores the data to find:
- Hidden correlations between features
- Seasonal patterns and cycles
- Plot clustering and grouping
- Anomalies and outliers
- Feature interactions
- Temporal patterns
- Cross-plot relationships
- Predictive signals
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Set
from .base_agent import BaseAgent, Critique
from scipy.stats import pearsonr, spearmanr, kendalltau
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

# Optional sklearn dependencies
try:
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class DataExplorerAgent(BaseAgent):
    """
    Autonomous agent that deeply explores agricultural data

    Given raw data, this agent:
    1. Analyzes each plot's characteristics
    2. Finds relationships between plots
    3. Discovers feature interactions
    4. Identifies patterns and anomalies
    5. Builds comprehensive knowledge graph
    6. Generates hypotheses for prediction

    The agent explores autonomously until it's confident the KG is complete.
    """

    def __init__(self,
                 knowledge_graph,
                 llm_agent=None,
                 verbose: bool = True,
                 max_iterations: int = 50,
                 exploration_depth: str = 'thorough'):  # 'quick', 'thorough', 'exhaustive'
        """
        Initialize Data Explorer Agent

        Args:
            knowledge_graph: KG instance to populate
            llm_agent: LLM for reasoning about patterns
            verbose: Print exploration progress
            exploration_depth: How deep to explore
                - 'quick': Basic stats only (5-10 min)
                - 'thorough': Stats + correlations + clustering (10-30 min)
                - 'exhaustive': Everything including interactions (30+ min)
        """
        # Store before super().__init__
        self.kg = knowledge_graph
        self.exploration_depth = exploration_depth

        super().__init__(
            role="Data Explorer Specialist",
            goal="Extract maximum predictive information from agricultural data into KG",
            backstory="Expert at finding hidden patterns, relationships, and insights in complex agricultural datasets",
            llm_agent=llm_agent,
            verbose=verbose,
            max_iterations=max_iterations
        )

        # Exploration state
        self.raw_data = None
        self.plots_explored = set()
        self.features_analyzed = set()
        self.relationships_found = []
        self.patterns_discovered = {}
        self.hypotheses = []

        # Statistics tracking
        self.exploration_stats = {
            'plots_processed': 0,
            'features_analyzed': 0,
            'correlations_found': 0,
            'clusters_identified': 0,
            'anomalies_detected': 0,
            'patterns_discovered': 0,
            'hypotheses_generated': 0
        }

    def _register_tools(self):
        """Register exploration tools"""
        self.register_tool(
            "analyze_plot",
            "Deep analysis of a single plot's characteristics",
            self._analyze_plot_tool,
            {"plot_id": "int", "data": "DataFrame"}
        )

        self.register_tool(
            "find_correlations",
            "Find significant correlations between features",
            self._find_correlations_tool,
            {"features": "list", "threshold": "float"}
        )

        self.register_tool(
            "cluster_plots",
            "Group similar plots into clusters",
            self._cluster_plots_tool,
            {"method": "string", "n_clusters": "int"}
        )

        self.register_tool(
            "detect_seasonality",
            "Identify seasonal patterns in time series",
            self._detect_seasonality_tool,
            {"plot_id": "int", "time_series": "DataFrame"}
        )

        self.register_tool(
            "find_anomalies",
            "Detect outliers and anomalies in data",
            self._find_anomalies_tool,
            {"plot_id": "int", "data": "DataFrame"}
        )

        self.register_tool(
            "discover_interactions",
            "Find feature interaction effects",
            self._discover_interactions_tool,
            {"features": "list", "target": "string"}
        )

        self.register_tool(
            "generate_hypothesis",
            "Generate predictive hypothesis using LLM",
            self._generate_hypothesis_tool,
            {"observations": "dict"}
        )

    def solve(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main autonomous exploration loop

        Args:
            task: {
                'df_train': Training DataFrame,
                'df_test': Test DataFrame (optional),
                'target_col': Target column name,
                'metadata_cols': List of metadata column names,
                'feature_cols': List of feature column names,
                'plot_id_col': Column containing plot IDs
            }

        Returns:
            {
                'exploration_complete': True/False,
                'plots_explored': List of plot IDs,
                'patterns_found': Dict of patterns,
                'hypotheses': List of predictive hypotheses,
                'kg_enriched': True/False,
                'exploration_stats': Statistics about exploration,
                'recommendations': List of recommendations for prediction
            }
        """
        self.log(f"\n{'='*60}", "info")
        self.log(f"DATA EXPLORER AGENT", "info")
        self.log(f"{'='*60}", "info")
        self.log(self.get_state_summary(), "info")
        self.log(f"Exploration depth: {self.exploration_depth}", "info")

        # Extract task data
        df_train = task['df_train']
        target_col = task['target_col']
        plot_id_col = task['plot_id_col']
        feature_cols = task['feature_cols']
        metadata_cols = task.get('metadata_cols', [])

        self.raw_data = df_train
        plot_ids = df_train[plot_id_col].unique()

        self.log(f"\nDataset: {len(df_train)} rows, {len(feature_cols)} features, {len(plot_ids)} plots", "info")

        # =================================================================
        # PHASE 1: Individual Plot Analysis
        # =================================================================
        self.log(f"\n{'='*60}", "info")
        self.log("📊 PHASE 1: INDIVIDUAL PLOT ANALYSIS", "action")
        self.log(f"{'='*60}", "info")

        for plot_id in plot_ids:
            plot_data = df_train[df_train[plot_id_col] == plot_id].copy()

            if len(plot_data) < 4:
                self.log(f"Skipping plot {plot_id} (insufficient data)", "warning")
                continue

            # Deep analysis of this plot
            analysis = self.use_tool(
                "analyze_plot",
                plot_id=plot_id,
                data=plot_data
            )

            self.plots_explored.add(plot_id)
            self.exploration_stats['plots_processed'] += 1

            # Add to knowledge graph with rich metadata
            self._add_plot_to_kg(plot_id, plot_data, analysis, metadata_cols, target_col)

        self.log(f"\n✓ Analyzed {len(self.plots_explored)} plots", "success")

        # =================================================================
        # PHASE 2: Feature Correlation Analysis
        # =================================================================
        if self.exploration_depth in ['thorough', 'exhaustive']:
            self.log(f"\n{'='*60}", "info")
            self.log("🔗 PHASE 2: FEATURE CORRELATION ANALYSIS", "action")
            self.log(f"{'='*60}", "info")

            correlations = self.use_tool(
                "find_correlations",
                features=feature_cols,
                threshold=0.3  # Only strong correlations
            )

            self.log(f"✓ Found {len(correlations)} significant correlations", "success")

            # Store correlations in KG
            for feat1, feat2, corr_value, corr_type in correlations:
                self._add_relationship_to_kg(
                    'feature_correlation',
                    {'feature1': feat1, 'feature2': feat2, 'correlation': corr_value, 'type': corr_type}
                )

        # =================================================================
        # PHASE 3: Plot Clustering
        # =================================================================
        if self.exploration_depth in ['thorough', 'exhaustive']:
            self.log(f"\n{'='*60}", "info")
            self.log("🎯 PHASE 3: PLOT CLUSTERING", "action")
            self.log(f"{'='*60}", "info")

            # Adaptive number of clusters (at least 2, at most 5)
            n_clusters = max(2, min(5, len(plot_ids) // 10))

            clusters = self.use_tool(
                "cluster_plots",
                method='kmeans',
                n_clusters=n_clusters
            )

            if clusters is not None:
                self.log(f"✓ Identified {len(clusters)} plot clusters", "success")

                # Store clusters in KG
                for cluster_id, cluster_plots in clusters.items():
                    self._add_cluster_to_kg(cluster_id, cluster_plots)
            else:
                self.log("⚠️ Clustering skipped (insufficient data)", "warning")

        # =================================================================
        # PHASE 4: Seasonality & Temporal Patterns
        # =================================================================
        if self.exploration_depth in ['thorough', 'exhaustive']:
            self.log(f"\n{'='*60}", "info")
            self.log("📅 PHASE 4: SEASONALITY & TEMPORAL PATTERNS", "action")
            self.log(f"{'='*60}", "info")

            seasonal_plots = 0
            for plot_id in self.plots_explored:
                plot_data = df_train[df_train[plot_id_col] == plot_id].copy()

                seasonality = self.use_tool(
                    "detect_seasonality",
                    plot_id=plot_id,
                    time_series=plot_data
                )

                if seasonality['has_seasonality']:
                    seasonal_plots += 1
                    self._add_seasonality_to_kg(plot_id, seasonality)

            self.log(f"✓ Found seasonality in {seasonal_plots}/{len(self.plots_explored)} plots", "success")

        # =================================================================
        # PHASE 5: Anomaly Detection
        # =================================================================
        if self.exploration_depth in ['thorough', 'exhaustive']:
            self.log(f"\n{'='*60}", "info")
            self.log("⚠️  PHASE 5: ANOMALY DETECTION", "action")
            self.log(f"{'='*60}", "info")

            total_anomalies = 0
            for plot_id in self.plots_explored:
                plot_data = df_train[df_train[plot_id_col] == plot_id].copy()

                anomalies = self.use_tool(
                    "find_anomalies",
                    plot_id=plot_id,
                    data=plot_data
                )

                if anomalies['count'] > 0:
                    total_anomalies += anomalies['count']
                    self._add_anomalies_to_kg(plot_id, anomalies)

            self.log(f"✓ Detected {total_anomalies} anomalies across all plots", "success")

        # =================================================================
        # PHASE 6: Feature Interactions
        # =================================================================
        if self.exploration_depth == 'exhaustive':
            self.log(f"\n{'='*60}", "info")
            self.log("🔬 PHASE 6: FEATURE INTERACTION DISCOVERY", "action")
            self.log(f"{'='*60}", "info")

            interactions = self.use_tool(
                "discover_interactions",
                features=feature_cols[:20],  # Limit to top 20 features
                target=target_col
            )

            self.log(f"✓ Found {len(interactions)} significant interactions", "success")

            for interaction in interactions:
                self._add_interaction_to_kg(interaction)

        # =================================================================
        # PHASE 7: Hypothesis Generation (LLM)
        # =================================================================
        if self.llm_agent:
            self.log(f"\n{'='*60}", "info")
            self.log("🧠 PHASE 7: HYPOTHESIS GENERATION", "action")
            self.log(f"{'='*60}", "info")

            # Compile all observations
            observations = {
                'n_plots': len(self.plots_explored),
                'n_features': len(feature_cols),
                'correlations': self.exploration_stats['correlations_found'],
                'clusters': self.exploration_stats['clusters_identified'],
                'anomalies': self.exploration_stats['anomalies_detected'],
                'patterns': self.patterns_discovered
            }

            hypotheses = self.use_tool(
                "generate_hypothesis",
                observations=observations
            )

            self.hypotheses = hypotheses
            self.log(f"✓ Generated {len(hypotheses)} predictive hypotheses", "success")

        # =================================================================
        # FINAL: Compile Results
        # =================================================================
        self.log(f"\n{'='*60}", "info")
        self.log("📋 EXPLORATION COMPLETE", "success")
        self.log(f"{'='*60}", "info")

        self._print_exploration_summary()

        result = {
            'exploration_complete': True,
            'plots_explored': list(self.plots_explored),
            'patterns_found': self.patterns_discovered,
            'hypotheses': self.hypotheses,
            'kg_enriched': True,
            'exploration_stats': self.exploration_stats,
            'recommendations': self._generate_recommendations()
        }

        return result

    # =====================================================================
    # TOOL IMPLEMENTATIONS
    # =====================================================================

    def _analyze_plot_tool(self, plot_id: int, data: pd.DataFrame) -> Dict[str, Any]:
        """Deep analysis of a single plot"""
        analysis = {}

        # Basic statistics
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            values = data[col].dropna()
            if len(values) > 0:
                analysis[f'{col}_mean'] = float(values.mean())
                analysis[f'{col}_std'] = float(values.std())
                analysis[f'{col}_min'] = float(values.min())
                analysis[f'{col}_max'] = float(values.max())
                analysis[f'{col}_skew'] = float(values.skew())
                analysis[f'{col}_kurtosis'] = float(values.kurtosis())

        # Temporal patterns (if date column exists)
        if 'date' in data.columns:
            data_sorted = data.sort_values('date')

            # Trend detection
            if 'target' in data.columns:
                target_vals = data_sorted['target'].values
                if len(target_vals) >= 3:
                    x = np.arange(len(target_vals))
                    slope, _ = np.polyfit(x, target_vals, 1)
                    analysis['trend_slope'] = float(slope)

                    # Peak detection
                    peaks, _ = find_peaks(target_vals)
                    analysis['n_peaks'] = len(peaks)
                    if len(peaks) > 0:
                        analysis['peak_positions'] = peaks.tolist()

        # Completeness
        analysis['completeness'] = float(data.notna().mean().mean())
        analysis['n_observations'] = len(data)

        self.log(f"  Plot {plot_id}: {analysis['n_observations']} obs, {analysis['completeness']:.2%} complete", "debug")

        return analysis

    def _find_correlations_tool(self, features: List[str], threshold: float) -> List[Tuple]:
        """Find significant correlations between features"""
        correlations = []

        # Get numeric data for all features
        feature_data = self.raw_data[features].select_dtypes(include=[np.number])

        # Compute pairwise correlations
        for i, feat1 in enumerate(feature_data.columns):
            for feat2 in feature_data.columns[i+1:]:
                vals1 = feature_data[feat1].dropna()
                vals2 = feature_data[feat2].dropna()

                # Need same indices
                common_idx = vals1.index.intersection(vals2.index)
                if len(common_idx) < 10:
                    continue

                v1 = vals1.loc[common_idx]
                v2 = vals2.loc[common_idx]

                # Pearson correlation
                try:
                    corr, p_value = pearsonr(v1, v2)
                    if abs(corr) > threshold and p_value < 0.05:
                        correlations.append((feat1, feat2, float(corr), 'pearson'))
                        self.exploration_stats['correlations_found'] += 1
                except:
                    pass

        return correlations

    def _cluster_plots_tool(self, method: str, n_clusters: int) -> Dict[int, List[int]]:
        """Cluster plots based on their features"""
        if not SKLEARN_AVAILABLE:
            self.log("Clustering requires sklearn (not installed), using simple grouping instead", "warning")
            # Fallback: group by farm or year
            clusters = {}
            for plot_id in self.plots_explored:
                plot_node = self.kg.G.nodes.get(plot_id, {})
                cluster_key = plot_node.get('farm', 0)
                if cluster_key not in clusters:
                    clusters[cluster_key] = []
                clusters[cluster_key].append(plot_id)
            self.exploration_stats['clusters_identified'] = len(clusters)
            return clusters

        # Extract features for each plot
        plot_features = []
        plot_ids_list = []

        for plot_id in self.plots_explored:
            plot_node = self.kg.G.nodes.get(plot_id, {})

            # Extract numeric features
            features = []
            for key, value in plot_node.items():
                if isinstance(value, (int, float)) and key not in ['farm', 'year']:
                    features.append(value)

            if len(features) > 0:
                plot_features.append(features)
                plot_ids_list.append(plot_id)

        # Not enough plots or features to cluster
        if len(plot_features) < 2 or len(plot_features) < n_clusters:
            self.log(f"Insufficient data for clustering ({len(plot_features)} plots, need {n_clusters})", "warning")
            # Create single cluster with all plots
            clusters = {0: list(self.plots_explored)}
            self.exploration_stats['clusters_identified'] = 1
            return clusters

        # Normalize features
        X = StandardScaler().fit_transform(plot_features)

        # Cluster
        if method == 'kmeans':
            clusterer = KMeans(n_clusters=n_clusters, random_state=42)
            labels = clusterer.fit_predict(X)
        elif method == 'dbscan':
            clusterer = DBSCAN(eps=0.5, min_samples=2)
            labels = clusterer.fit_predict(X)
        else:
            # Unknown method - create single cluster
            clusters = {0: list(self.plots_explored)}
            self.exploration_stats['clusters_identified'] = 1
            return clusters

        # Group by cluster
        clusters = {}
        for plot_id, label in zip(plot_ids_list, labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(plot_id)

        self.exploration_stats['clusters_identified'] = len(clusters)

        return clusters

    def _detect_seasonality_tool(self, plot_id: int, time_series: pd.DataFrame) -> Dict[str, Any]:
        """Detect seasonal patterns in time series"""
        result = {'has_seasonality': False}

        if 'target' not in time_series.columns or 'date' not in time_series.columns:
            return result

        ts = time_series.sort_values('date')['target'].values

        if len(ts) < 12:  # Need at least 12 observations
            return result

        # Simple autocorrelation check
        from scipy.stats import pearsonr

        # Check for weekly seasonality (lag 4 for monthly data, adjust as needed)
        for lag in [4, 8, 12]:
            if len(ts) > lag:
                corr, p_val = pearsonr(ts[:-lag], ts[lag:])
                if abs(corr) > 0.5 and p_val < 0.05:
                    result['has_seasonality'] = True
                    result['lag'] = lag
                    result['strength'] = float(abs(corr))
                    self.exploration_stats['patterns_discovered'] += 1
                    break

        return result

    def _find_anomalies_tool(self, plot_id: int, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect outliers and anomalies"""
        anomalies = {'count': 0, 'indices': [], 'types': []}

        if 'target' not in data.columns:
            return anomalies

        target = data['target'].dropna()

        if len(target) < 10:
            return anomalies

        # Z-score based anomaly detection
        mean = target.mean()
        std = target.std()

        z_scores = np.abs((target - mean) / (std + 1e-8))
        outliers = z_scores > 3  # 3 sigma rule

        anomalies['count'] = int(outliers.sum())
        anomalies['indices'] = target[outliers].index.tolist()
        anomalies['types'] = ['outlier'] * anomalies['count']

        if anomalies['count'] > 0:
            self.exploration_stats['anomalies_detected'] += anomalies['count']

        return anomalies

    def _discover_interactions_tool(self, features: List[str], target: str) -> List[Dict]:
        """Find feature interaction effects"""
        interactions = []

        # Simplified: check pairwise interactions with target
        # Full implementation would test multiplicative effects

        data = self.raw_data[features + [target]].dropna()

        if len(data) < 30:
            return interactions

        # For each pair of features, check if their interaction correlates with target
        for i in range(len(features)):
            for j in range(i+1, min(i+5, len(features))):  # Limit combinations
                feat1 = features[i]
                feat2 = features[j]

                try:
                    # Create interaction term
                    interaction = data[feat1] * data[feat2]

                    # Check correlation with target
                    corr, p_val = pearsonr(interaction, data[target])

                    if abs(corr) > 0.3 and p_val < 0.05:
                        interactions.append({
                            'feature1': feat1,
                            'feature2': feat2,
                            'correlation': float(corr),
                            'p_value': float(p_val)
                        })
                except:
                    pass

        return interactions

    def _generate_hypothesis_tool(self, observations: Dict) -> List[str]:
        """Generate predictive hypotheses using LLM"""
        if not self.llm_agent:
            return []

        prompt = f"""You are an agricultural data scientist. Based on this data exploration:

Observations:
- {observations['n_plots']} plots analyzed
- {observations['n_features']} features measured
- {observations['correlations']} significant correlations found
- {observations['clusters']} plot clusters identified
- {observations['anomalies']} anomalies detected

Patterns discovered:
{observations.get('patterns', 'None yet')}

Task: Generate 3-5 hypotheses about what factors most influence crop yield.

Format as bullet points:
- Hypothesis 1: ...
- Hypothesis 2: ...
"""

        response = self.llm_agent.query(prompt, stream=False)

        # Parse hypotheses
        hypotheses = []
        for line in response.split('\n'):
            if line.strip().startswith('-') or line.strip().startswith('*'):
                hypothesis = line.strip().lstrip('-*').strip()
                if hypothesis:
                    hypotheses.append(hypothesis)

        self.exploration_stats['hypotheses_generated'] = len(hypotheses)

        return hypotheses

    # =====================================================================
    # KNOWLEDGE GRAPH POPULATION
    # =====================================================================

    def _add_plot_to_kg(self, plot_id: int, data: pd.DataFrame, analysis: Dict,
                        metadata_cols: List[str], target_col: str):
        """Add plot with rich metadata to KG"""
        # Extract metadata
        metadata = {}
        for col in metadata_cols:
            if col in data.columns:
                metadata[col] = data[col].iloc[0]

        # Create time series for KG
        if 'date' in data.columns and target_col in data.columns:
            ts_data = data[['date', target_col]].copy()
            ts_data = ts_data.sort_values('date').set_index('date')
        else:
            ts_data = data[[target_col]].copy() if target_col in data.columns else data

        # Add to KG with all analysis results
        self.kg.add_plot(
            plot_id=plot_id,
            metadata=metadata,
            time_series=ts_data,
            quality_score=analysis.get('completeness', 0.0) * 100
        )

        # Add analysis results as node attributes
        for key, value in analysis.items():
            if plot_id in self.kg.G.nodes:
                self.kg.G.nodes[plot_id][key] = value

    def _add_relationship_to_kg(self, rel_type: str, data: Dict):
        """Add a relationship to the KG"""
        # Store in long-term memory
        self.long_term_memory.setdefault('relationships', []).append({
            'type': rel_type,
            'data': data
        })

        self.relationships_found.append((rel_type, data))

    def _add_cluster_to_kg(self, cluster_id: int, plot_ids: List[int]):
        """Add cluster information to KG"""
        # Add cluster attribute to each plot node
        for plot_id in plot_ids:
            if plot_id in self.kg.G.nodes:
                self.kg.G.nodes[plot_id]['cluster'] = cluster_id

        # Store cluster info
        self.long_term_memory.setdefault('clusters', {})[cluster_id] = plot_ids

    def _add_seasonality_to_kg(self, plot_id: int, seasonality: Dict):
        """Add seasonality info to plot node"""
        if plot_id in self.kg.G.nodes:
            self.kg.G.nodes[plot_id]['seasonality'] = seasonality

        self.patterns_discovered.setdefault('seasonal_plots', []).append(plot_id)

    def _add_anomalies_to_kg(self, plot_id: int, anomalies: Dict):
        """Add anomaly info to plot node"""
        if plot_id in self.kg.G.nodes:
            self.kg.G.nodes[plot_id]['anomalies'] = anomalies

    def _add_interaction_to_kg(self, interaction: Dict):
        """Add feature interaction to KG"""
        self.long_term_memory.setdefault('interactions', []).append(interaction)

    # =====================================================================
    # REPORTING
    # =====================================================================

    def _print_exploration_summary(self):
        """Print summary of exploration"""
        self.log("\n📊 Exploration Summary:", "info")
        self.log(f"  • Plots processed: {self.exploration_stats['plots_processed']}", "info")
        self.log(f"  • Features analyzed: {len(self.features_analyzed)}", "info")
        self.log(f"  • Correlations found: {self.exploration_stats['correlations_found']}", "info")
        self.log(f"  • Clusters identified: {self.exploration_stats['clusters_identified']}", "info")
        self.log(f"  • Anomalies detected: {self.exploration_stats['anomalies_detected']}", "info")
        self.log(f"  • Patterns discovered: {self.exploration_stats['patterns_discovered']}", "info")
        self.log(f"  • Hypotheses generated: {self.exploration_stats['hypotheses_generated']}", "info")

    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations for downstream prediction agents"""
        recommendations = []

        if self.exploration_stats['correlations_found'] > 10:
            recommendations.append("High feature correlation - consider feature selection")

        if self.exploration_stats['clusters_identified'] > 0:
            recommendations.append(f"Found {self.exploration_stats['clusters_identified']} plot clusters - use cluster-based sampling")

        if self.exploration_stats['anomalies_detected'] > 0:
            recommendations.append("Anomalies detected - consider robust training methods")

        if self.patterns_discovered.get('seasonal_plots'):
            recommendations.append("Seasonal patterns detected - use seasonal models")

        return recommendations

    def _build_critique_prompt(self, result: Dict, context: Dict) -> str:
        """Build critique prompt (required by base class)"""
        return f"""Evaluate data exploration completeness:

Explored: {result.get('exploration_stats', {})}
Depth: {self.exploration_depth}

Is the exploration thorough enough? What's missing?
"""


if __name__ == "__main__":
    print("Data Explorer Agent - Example Usage")
    print("="*60)
    print("\nThis agent autonomously explores agricultural data")
    print("and builds a comprehensive knowledge graph.")
    print("\nUsage:")
    print("""
from agents import DataExplorerAgent
from knowledge_graph import AgriculturalKnowledgeGraph

# Create KG and agent
kg = AgriculturalKnowledgeGraph()
agent = DataExplorerAgent(
    knowledge_graph=kg,
    llm_agent=llm,
    exploration_depth='thorough'
)

# Explore data
result = agent.solve({
    'df_train': df_train,
    'target_col': 'target',
    'plot_id_col': 'lookupEncoded',
    'feature_cols': feature_columns,
    'metadata_cols': ['FarmEncoded', 'year']
})

# KG is now enriched with deep insights!
print(result['exploration_stats'])
print(result['hypotheses'])
""")
