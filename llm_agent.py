"""
LLM Agent for intelligent plot selection and analysis
Can reason about which plots are most relevant based on metadata
Enhanced with domain knowledge, feature-based scoring, and chain-of-thought reasoning
"""
import requests
import json
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
from collections import defaultdict


class LLMAgent:
    """Agent that uses LLM to make intelligent decisions about data selection"""

    def __init__(self, remote_url: str, model: str = "llama3.2:latest"):
        """
        Initialize LLM agent

        Args:
            remote_url: URL of the Ollama/LLM server (e.g., ngrok URL)
            model: Model name to use
        """
        self.remote_url = remote_url
        self.model = model
        self.conversation_history = []

        # Test connection
        try:
            response = requests.get(f"{self.remote_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                available = [m["name"] for m in models]
                print(f"✓ Connected to LLM server")
                print(f"Available models: {available}")
                if model not in available:
                    print(f"⚠ Warning: {model} not available, using {available[0] if available else 'none'}")
                    if available:
                        self.model = available[0]
            else:
                print(f"⚠ Warning: Could not fetch models (status {response.status_code})")
        except Exception as e:
            print(f"⚠ Warning: Could not connect to LLM server: {e}")

    def query(self, prompt: str, stream: bool = False) -> str:
        """
        Send a query to the LLM

        Args:
            prompt: The question or instruction
            stream: Whether to stream the response

        Returns:
            LLM's response as a string
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": stream
        }

        try:
            response = requests.post(
                f"{self.remote_url}/api/generate",
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload),
                stream=stream,
                timeout=60
            )

            if response.status_code == 200:
                if stream:
                    full_response = ""
                    for chunk in response.iter_content(chunk_size=1024, decode_unicode=True):
                        if chunk:
                            try:
                                data = json.loads(chunk)
                                chunk_text = data.get("response", "")
                                full_response += chunk_text
                                print(chunk_text, end='', flush=True)
                            except json.JSONDecodeError:
                                continue
                    print()  # Newline after streaming
                    return full_response
                else:
                    # Non-streaming response
                    full_response = ""
                    for line in response.iter_lines(decode_unicode=True):
                        if line:
                            try:
                                data = json.loads(line)
                                full_response += data.get("response", "")
                            except json.JSONDecodeError:
                                continue
                    return full_response
            else:
                return f"Error: HTTP {response.status_code}"

        except Exception as e:
            return f"Error: {e}"

    def compute_aggregate_target_profile(self, target_metadatas: List[Dict]) -> Dict:
        """
        Compute aggregate statistics across multiple target plots (e.g., all 2023 plots)
        This gives a better representation of what we're trying to predict

        Args:
            target_metadatas: List of metadata dicts for all target plots

        Returns:
            Dict with aggregate statistics
        """
        if not target_metadatas:
            return {}

        numeric_features = [
            'mean_yield', 'std_yield', 'cv_yield', 'trend_slope',
            'peak_week', 'peak_value', 'valley_value', 'range',
            'volatility', 'early_mean', 'mid_mean', 'late_mean'
        ]

        aggregates = {}

        # Compute mean and std for each numeric feature
        for feature in numeric_features:
            values = [m.get(feature) for m in target_metadatas if m.get(feature) is not None]
            if values:
                aggregates[f'{feature}_mean'] = np.mean(values)
                aggregates[f'{feature}_std'] = np.std(values)
                aggregates[f'{feature}_min'] = np.min(values)
                aggregates[f'{feature}_max'] = np.max(values)

        # Categorical features - get distributions
        categorical_features = ['growth_pattern', 'farm', 'crop', 'soil', 'location']
        for feature in categorical_features:
            values = [m.get(feature) for m in target_metadatas if m.get(feature) is not None]
            if values:
                # Get mode (most common)
                from collections import Counter
                counter = Counter(values)
                aggregates[f'{feature}_mode'] = counter.most_common(1)[0][0]
                aggregates[f'{feature}_distribution'] = dict(counter)

        aggregates['num_plots'] = len(target_metadatas)

        return aggregates

    def compute_feature_similarity_score(self, target_metadata: Dict, candidate_metadata: Dict) -> Dict[str, float]:
        """
        Compute similarity scores for different types of features
        Returns a breakdown of similarity by feature category

        Args:
            target_metadata: Target plot metadata
            candidate_metadata: Candidate plot metadata

        Returns:
            Dict with similarity scores for different feature groups
        """
        scores = {}

        # Curve shape similarity (most important for fine-tuning)
        shape_features = ['mean_yield', 'trend_slope', 'peak_week', 'volatility', 'range']
        shape_diffs = []
        for feat in shape_features:
            if feat in target_metadata and feat in candidate_metadata:
                t_val = target_metadata[feat]
                c_val = candidate_metadata[feat]
                if t_val is not None and c_val is not None:
                    # Normalize by target value to get relative difference
                    diff = abs(t_val - c_val) / (abs(t_val) + 1e-8)
                    shape_diffs.append(diff)

        scores['shape_similarity'] = 1.0 / (1.0 + np.mean(shape_diffs)) if shape_diffs else 0.0

        # Growth pattern match (categorical)
        if target_metadata.get('growth_pattern') == candidate_metadata.get('growth_pattern'):
            scores['pattern_match'] = 1.0
        else:
            scores['pattern_match'] = 0.0

        # Temporal relevance (same year is very important)
        t_year = target_metadata.get('year')
        c_year = candidate_metadata.get('year')
        if t_year and c_year:
            year_diff = abs(int(t_year) - int(c_year))
            scores['temporal_similarity'] = 1.0 / (1.0 + year_diff)
        else:
            scores['temporal_similarity'] = 0.0

        # Farm match (shared management practices)
        if target_metadata.get('farm') == candidate_metadata.get('farm'):
            scores['farm_match'] = 1.0
        else:
            scores['farm_match'] = 0.0

        # Volatility match (smoothness of curve)
        t_vol = target_metadata.get('volatility')
        c_vol = candidate_metadata.get('volatility')
        if t_vol is not None and c_vol is not None:
            vol_diff = abs(t_vol - c_vol) / (t_vol + 1e-8)
            scores['volatility_similarity'] = 1.0 / (1.0 + vol_diff)
        else:
            scores['volatility_similarity'] = 0.0

        # Overall composite score (weighted combination)
        weights = {
            'shape_similarity': 0.35,      # Most important
            'pattern_match': 0.20,
            'temporal_similarity': 0.20,
            'farm_match': 0.15,
            'volatility_similarity': 0.10
        }

        scores['composite_score'] = sum(scores.get(k, 0) * w for k, w in weights.items())

        return scores

    def analyze_plot_metadata(self,
                             target_metadata: Dict,
                             candidate_plots: List[Tuple[int, Dict, float]],
                             top_k: int = 10) -> List[int]:
        """
        Use LLM to intelligently select most relevant plots based on metadata

        Args:
            target_metadata: Metadata of the target plot
            candidate_plots: List of (plot_id, metadata, similarity_score) tuples
            top_k: Number of plots to select

        Returns:
            List of selected plot IDs
        """
        # Format metadata for LLM
        target_info = self._format_metadata(target_metadata)

        candidates_info = []
        for plot_id, metadata, sim_score in candidate_plots:
            candidates_info.append({
                'plot_id': plot_id,
                'metadata': self._format_metadata(metadata),
                'similarity_score': f"{sim_score:.3f}"
            })

        # Create prompt
        prompt = f"""You are an agricultural data scientist helping select the most relevant training data for yield prediction.

TARGET PLOT TO PREDICT:
{target_info}

CANDIDATE PLOTS (with DTW similarity scores):
"""
        for i, cand in enumerate(candidates_info[:20], 1):  # Limit to top 20 for context
            prompt += f"\n{i}. Plot {cand['plot_id']} (similarity: {cand['similarity_score']})\n"
            prompt += f"   {cand['metadata']}\n"

        prompt += f"""
TASK:
Select the {top_k} most relevant plots for training a yield prediction model for the target plot.

CONSIDER:
1. DTW similarity scores (higher = more similar yield patterns)
2. Same farm (plots from same farm may share management practices)
3. Same year (temporal relevance, weather patterns)
4. Location proximity
5. Any other relevant factors

RESPOND WITH:
Just the plot IDs separated by commas (e.g., "5,12,18,3,7,9,15,22,1,11")

Your selection of {top_k} plot IDs:"""

        # Query LLM
        print("\n" + "="*60)
        print("CONSULTING LLM AGENT FOR PLOT SELECTION")
        print("="*60)

        response = self.query(prompt, stream=False)

        # Parse response to extract plot IDs
        try:
            # Clean response and extract numbers
            cleaned = response.strip().replace(" ", "")
            plot_ids = [int(x) for x in cleaned.split(",") if x.isdigit()]

            # Validate all IDs are in candidate list
            valid_ids = [pid for pid, _, _ in candidate_plots]
            selected_ids = [pid for pid in plot_ids if pid in valid_ids]

            # If we got fewer than top_k, fill with highest similarity
            if len(selected_ids) < top_k:
                remaining = [pid for pid, _, _ in candidate_plots if pid not in selected_ids]
                selected_ids.extend(remaining[:top_k - len(selected_ids)])

            # Truncate to top_k
            selected_ids = selected_ids[:top_k]

            print(f"\n✓ LLM selected {len(selected_ids)} plots: {selected_ids}")
            return selected_ids

        except Exception as e:
            print(f"⚠ Error parsing LLM response: {e}")
            print(f"Raw response: {response}")
            # Fallback: return top k by similarity
            return [pid for pid, _, _ in candidate_plots[:top_k]]

    def rank_all_plots_by_relevance(self,
                                    target_metadata: Dict,
                                    candidate_plots: List[Tuple[int, Dict, float]],
                                    batch_size: int = 25,
                                    use_aggregate_profile: bool = True,
                                    target_metadatas: Optional[List[Dict]] = None) -> List[int]:
        """
        Use LLM to rank ALL plots by relevance with enhanced reasoning
        Now supports aggregate target profiles for better multi-plot prediction

        Args:
            target_metadata: Metadata of a single target plot (or representative)
            candidate_plots: List of (plot_id, metadata, similarity_score) tuples for ALL plots
            batch_size: Process this many plots at a time (to avoid context limits)
            use_aggregate_profile: If True and target_metadatas provided, use aggregate statistics
            target_metadatas: List of all target plot metadatas for computing aggregate profile

        Returns:
            List of plot IDs ranked by LLM relevance (most relevant first)
        """
        print(f"\n{'='*60}")
        print(f"LLM RANKING ALL {len(candidate_plots)} PLOTS WITH ENHANCED REASONING")
        print(f"{'='*60}")

        # Compute aggregate target profile if available
        aggregate_profile = None
        if use_aggregate_profile and target_metadatas:
            aggregate_profile = self.compute_aggregate_target_profile(target_metadatas)
            print(f"✓ Using aggregate profile from {len(target_metadatas)} target plots")

        # Pre-filter candidates using feature-based scoring
        print(f"\n📊 Computing feature-based similarity scores...")
        candidates_with_scores = []
        for plot_id, metadata, dtw_sim in candidate_plots:
            feature_scores = self.compute_feature_similarity_score(target_metadata, metadata)
            candidates_with_scores.append((
                plot_id,
                metadata,
                dtw_sim,
                feature_scores
            ))

        # Sort by composite score to prioritize best candidates
        candidates_with_scores.sort(key=lambda x: x[3]['composite_score'], reverse=True)

        print(f"✓ Top candidate by composite score: Plot {candidates_with_scores[0][0]} (score: {candidates_with_scores[0][3]['composite_score']:.3f})")
        print(f"✓ Lowest candidate by composite score: Plot {candidates_with_scores[-1][0]} (score: {candidates_with_scores[-1][3]['composite_score']:.3f})")

        all_ranked_ids = []

        # Process in batches to avoid context limits
        num_batches = (len(candidates_with_scores) + batch_size - 1) // batch_size

        for batch_num in range(0, len(candidates_with_scores), batch_size):
            batch = candidates_with_scores[batch_num:batch_num + batch_size]
            batch_idx = batch_num // batch_size + 1

            # Build enhanced prompt with domain knowledge
            prompt = f"""You are an expert agricultural data scientist specializing in time series forecasting and machine learning.

CONTEXT: Fine-Tuning for Yield Prediction
We are fine-tuning a foundation time series model (Moirai) on historical agricultural data to predict future yields.
The model learns patterns through gradient descent - it adjusts its weights based on the training data's yield curves.

"""

            if aggregate_profile:
                prompt += f"""TARGET: Aggregate Profile of {aggregate_profile['num_plots']} 2023 Test Plots
We're predicting MULTIPLE plots in 2023. Here's their aggregate characteristics:

Statistical Profile:
- Mean Yield: {aggregate_profile.get('mean_yield_mean', 'N/A'):.2f} ± {aggregate_profile.get('mean_yield_std', 0):.2f}
- Trend Slope: {aggregate_profile.get('trend_slope_mean', 'N/A'):.4f} (overall trend direction)
- Peak Timing: Week {aggregate_profile.get('peak_week_mean', 'N/A'):.1f} of season ({aggregate_profile.get('peak_week_mean', 0)*100:.0f}%)
- Volatility: {aggregate_profile.get('volatility_mean', 'N/A'):.3f} ± {aggregate_profile.get('volatility_std', 0):.3f}

Growth Patterns:
{aggregate_profile.get('growth_pattern_distribution', {})}

Farm Distribution:
{aggregate_profile.get('farm_distribution', {})}

"""
            else:
                target_info = self._format_metadata_detailed(target_metadata)
                prompt += f"""TARGET PLOT TO PREDICT:
{target_info}

"""

            prompt += f"""CANDIDATE TRAINING PLOTS (Batch {batch_idx}/{num_batches}):
"""

            for i, (plot_id, metadata, dtw_sim, feat_scores) in enumerate(batch, 1):
                prompt += f"\n{i}. Plot {plot_id}\n"
                prompt += f"   DTW Similarity: {dtw_sim:.3f}\n"
                prompt += f"   Composite Feature Score: {feat_scores['composite_score']:.3f}\n"

                # Show key curve features
                prompt += f"   Curve: mean={metadata.get('mean_yield', 'N/A'):.2f}, "
                prompt += f"trend={metadata.get('trend_slope', 'N/A'):.4f}, "
                prompt += f"peak@{metadata.get('peak_week', 'N/A'):.2f}, "
                prompt += f"pattern={metadata.get('growth_pattern', 'N/A')}, "
                prompt += f"volatility={metadata.get('volatility', 'N/A'):.3f}\n"

                # Show metadata
                prompt += f"   Farm: {metadata.get('farm', 'N/A')}, "
                prompt += f"Year: {metadata.get('year', 'N/A')}, "
                prompt += f"Crop: {metadata.get('crop', 'N/A')}\n"

            prompt += f"""

TASK: Rank these {len(batch)} plots by their value as TRAINING DATA for fine-tuning.

AGRICULTURAL DOMAIN KNOWLEDGE:
1. **Curve Shape Matters Most**: The model learns from yield curve patterns. Plots with similar curve shapes (peak timing, trend, volatility) teach the model patterns relevant to the target.

2. **Growth Pattern Alignment**: If target plots show "peak_middle" pattern, training on similar patterns helps the model recognize that seasonal behavior.

3. **Temporal Context**: Recent years (2020-2022) may be more relevant than older data due to changing climate/practices, but diversity across years can help generalization.

4. **Farm-Specific Practices**: Same farm = similar soil management, irrigation, fertilization. This consistency is valuable.

5. **Volatility Matching**: If target plots have smooth curves (low volatility), training on similarly smooth curves helps. Mixing volatile and smooth can confuse the model.

6. **Diversity vs. Similarity Trade-off**:
   - Too similar: Model may overfit to specific pattern
   - Too diverse: Model learns contradictory patterns
   - IDEAL: Core group of very similar plots + few diverse plots for robustness

REASONING PROCESS (think step-by-step):

Step 1: Identify plots with HIGHEST curve shape similarity
- Look at composite_score (weights shape, pattern, temporal, farm factors)
- DTW similarity confirms overall curve shape match
- Higher scores = more relevant for fine-tuning

Step 2: Check for beneficial diversity
- Do we have multiple farms? (different management practices)
- Do we have multiple years? (different weather conditions)
- Different growth patterns might help OR hurt - use judgment

Step 3: Prioritize temporal relevance
- 2022 data >> 2021 >> 2020 >> older
- Recent data reflects current agricultural conditions

Step 4: Consider practical constraints
- Extremely low DTW similarity (<0.1) = probably not useful
- Missing key features = less reliable

Now rank ALL {len(batch)} plots from MOST to LEAST valuable for training.

RESPOND WITH:
ONLY a comma-separated list of plot IDs, ranked from best to worst.
Example: "42,15,8,31,19,..."

Your ranked list:"""

            # Query LLM
            print(f"\n  🤖 Processing batch {batch_idx}/{num_batches} ({len(batch)} plots)...")
            response = self.query(prompt, stream=False)

            # Parse response to extract plot IDs
            try:
                cleaned = response.strip().replace(" ", "")
                batch_ranked_ids = [int(x) for x in cleaned.split(",") if x.isdigit()]

                # Validate all IDs are in batch (batch contains 4 elements: pid, metadata, dtw_sim, feat_scores)
                valid_ids = [pid for pid, _, _, _ in batch]
                batch_ranked_ids = [pid for pid in batch_ranked_ids if pid in valid_ids]

                # Add any missing IDs from batch (LLM might have missed some)
                missing = [pid for pid in valid_ids if pid not in batch_ranked_ids]
                batch_ranked_ids.extend(missing)

                all_ranked_ids.extend(batch_ranked_ids)
                print(f"    ✓ Ranked {len(batch_ranked_ids)} plots")

            except Exception as e:
                print(f"    ⚠ Error parsing LLM response for batch: {e}")
                # Fallback: use original order
                all_ranked_ids.extend([pid for pid, _, _, _ in batch])

        print(f"\n✓ LLM ranked all {len(all_ranked_ids)} plots")
        return all_ranked_ids

    def explain_selection(self,
                         target_metadata: Dict,
                         selected_plots: List[Tuple[int, Dict, float]]) -> str:
        """
        Ask LLM to explain why these plots were selected

        Args:
            target_metadata: Target plot metadata
            selected_plots: List of (plot_id, metadata, similarity_score) tuples

        Returns:
            Explanation text
        """
        target_info = self._format_metadata(target_metadata)

        selected_info = []
        for plot_id, metadata, sim_score in selected_plots:
            selected_info.append(
                f"Plot {plot_id} (similarity: {sim_score:.3f}) - {self._format_metadata(metadata)}"
            )

        prompt = f"""Target Plot:
{target_info}

Selected Training Plots:
{chr(10).join(selected_info)}

Explain in 2-3 sentences why these plots are good training data for predicting the target plot's yield."""

        return self.query(prompt, stream=False)

    def compare_similarity_methods(self,
                                  dtw_plots: List[int],
                                  pearson_plots: List[int],
                                  metadata_dict: Dict[int, Dict]) -> Dict[str, any]:
        """
        Ask LLM to analyze differences between similarity methods

        Args:
            dtw_plots: Plot IDs selected by DTW
            pearson_plots: Plot IDs selected by Pearson
            metadata_dict: Dictionary mapping plot_id to metadata

        Returns:
            Analysis results
        """
        dtw_only = set(dtw_plots) - set(pearson_plots)
        pearson_only = set(pearson_plots) - set(dtw_plots)
        both = set(dtw_plots) & set(pearson_plots)

        prompt = f"""Two similarity methods selected different plots:

DTW selected: {dtw_plots}
Pearson selected: {pearson_plots}

Plots in BOTH: {list(both)}
Only in DTW: {list(dtw_only)}
Only in Pearson: {list(pearson_only)}

Explain in 2-3 sentences:
1. Why might DTW and Pearson disagree?
2. Which method might be better for agricultural yield prediction?"""

        explanation = self.query(prompt, stream=False)

        return {
            'both': list(both),
            'dtw_only': list(dtw_only),
            'pearson_only': list(pearson_only),
            'explanation': explanation
        }

    def suggest_additional_features(self,
                                   available_columns: List[str],
                                   target: str = "target") -> List[str]:
        """
        Ask LLM which features might be most useful for prediction

        Args:
            available_columns: List of available feature columns
            target: Target variable name

        Returns:
            List of suggested feature names
        """
        prompt = f"""Available features in agricultural yield dataset:
{', '.join(available_columns)}

Target variable: {target}

Which 5-10 features are most likely to be important for predicting agricultural yield?
Respond with just the feature names separated by commas."""

        response = self.query(prompt, stream=False)

        try:
            suggested = [f.strip() for f in response.split(",")]
            # Validate against available columns
            valid = [f for f in suggested if f in available_columns]
            return valid
        except:
            return []

    def analyze_dataset_statistics(self, plot_data: pd.DataFrame) -> Dict:
        """
        Use LLM to analyze dataset statistics and extract insights

        Args:
            plot_data: DataFrame with time series data

        Returns:
            Dictionary with analysis results
        """
        # Calculate statistics
        stats = {}
        for col in plot_data.select_dtypes(include=[np.number]).columns:
            stats[col] = {
                'mean': float(plot_data[col].mean()),
                'std': float(plot_data[col].std()),
                'min': float(plot_data[col].min()),
                'max': float(plot_data[col].max()),
                'missing': int(plot_data[col].isna().sum())
            }

        # Build prompt
        prompt = f"""Analyze this agricultural dataset statistics and identify key patterns:

Dataset shape: {plot_data.shape[0]} weeks x {plot_data.shape[1]} features

Feature Statistics:
"""
        for col, stat in list(stats.items())[:10]:  # First 10 features
            prompt += f"\n{col}:"
            prompt += f"\n  Mean: {stat['mean']:.2f}, Std: {stat['std']:.2f}"
            prompt += f"\n  Range: [{stat['min']:.2f}, {stat['max']:.2f}]"
            prompt += f"\n  Missing: {stat['missing']} values"

        prompt += """

Provide a 3-4 sentence analysis:
1. Which features show high variability?
2. Are there any concerning patterns (e.g., many missing values)?
3. Which features might be most predictive for yield?"""

        analysis = self.query(prompt, stream=False)

        return {
            'statistics': stats,
            'llm_analysis': analysis
        }

    def extract_temporal_patterns(self, target_series: pd.Series) -> Dict:
        """
        Ask LLM to identify temporal patterns in yield data

        Args:
            target_series: Time series of target values

        Returns:
            Dictionary with pattern analysis
        """
        # Calculate basic temporal features
        values = target_series.values

        prompt = f"""Analyze this weekly yield time series pattern:

Total weeks: {len(values)}
Mean yield: {np.mean(values):.2f}
Std deviation: {np.std(values):.2f}
Min yield: {np.min(values):.2f}
Max yield: {np.max(values):.2f}

First 10 weeks: {values[:10].tolist()}
Last 10 weeks: {values[-10:].tolist()}

Week-to-week changes (first 10): {np.diff(values[:11]).tolist()}

Questions:
1. Is there a clear trend (increasing/decreasing/stable)?
2. Is there seasonality or cyclical behavior?
3. Are there any anomalies or outliers?
4. What does this tell us about the growing season?

Provide 2-3 sentence analysis."""

        analysis = self.query(prompt, stream=False)

        return {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'trend': analysis
        }

    def suggest_feature_engineering(self,
                                   available_features: List[str],
                                   target_description: str = "agricultural yield") -> List[str]:
        """
        Ask LLM what new features to create from existing ones

        Args:
            available_features: List of current feature names
            target_description: What we're trying to predict

        Returns:
            List of feature engineering suggestions
        """
        prompt = f"""Given these features in an agricultural dataset:

Available features: {', '.join(available_features[:20])}

Target: {target_description}

Suggest 5-10 engineered features that could improve prediction:
- Ratios between features
- Interactions
- Aggregations
- Domain-specific calculations

Format: One suggestion per line, starting with a dash."""

        response = self.query(prompt, stream=False)

        # Parse suggestions
        suggestions = []
        for line in response.split('\n'):
            line = line.strip()
            if line.startswith('-') or line.startswith('•'):
                suggestions.append(line.lstrip('-•').strip())

        return suggestions

    def compare_plots_deeply(self,
                            plot1_id: int,
                            plot1_data: pd.DataFrame,
                            plot1_meta: Dict,
                            plot2_id: int,
                            plot2_data: pd.DataFrame,
                            plot2_meta: Dict) -> str:
        """
        Deep comparison between two plots

        Args:
            plot1_id, plot2_id: Plot identifiers
            plot1_data, plot2_data: Time series data
            plot1_meta, plot2_meta: Metadata

        Returns:
            Detailed comparison text
        """
        prompt = f"""Compare these two agricultural plots in detail:

PLOT {plot1_id}:
Metadata: {self._format_metadata(plot1_meta)}
Yield: mean={plot1_data['target'].mean():.2f}, std={plot1_data['target'].std():.2f}
Pattern: {plot1_data['target'].values[:5].tolist()} ... {plot1_data['target'].values[-5:].tolist()}

PLOT {plot2_id}:
Metadata: {self._format_metadata(plot2_meta)}
Yield: mean={plot2_data['target'].mean():.2f}, std={plot2_data['target'].std():.2f}
Pattern: {plot2_data['target'].values[:5].tolist()} ... {plot2_data['target'].values[-5:].tolist()}

Provide detailed comparison:
1. Similarities
2. Key differences
3. Would Plot {plot2_id} be good training data for predicting Plot {plot1_id}? Why?

Answer in 3-4 sentences."""

        return self.query(prompt, stream=False)

    def identify_outlier_plots(self,
                              all_plots: List[Tuple[int, Dict, pd.DataFrame]],
                              top_n: int = 5) -> List[int]:
        """
        Identify which plots are outliers/anomalous

        Args:
            all_plots: List of (plot_id, metadata, data) tuples
            top_n: Number of outliers to identify

        Returns:
            List of outlier plot IDs
        """
        # Calculate statistics for each plot
        plot_stats = []
        for plot_id, meta, data in all_plots[:20]:  # Limit for context
            plot_stats.append({
                'id': plot_id,
                'farm': meta.get('farm', 'unknown'),
                'year': meta.get('year', 'unknown'),
                'mean_yield': float(data['target'].mean()),
                'std_yield': float(data['target'].std()),
                'weeks': len(data)
            })

        prompt = f"""Analyze these agricultural plots and identify {top_n} that are outliers/anomalous:

"""
        for ps in plot_stats:
            prompt += f"Plot {ps['id']}: farm={ps['farm']}, year={ps['year']}, "
            prompt += f"mean_yield={ps['mean_yield']:.2f}, std={ps['std_yield']:.2f}, weeks={ps['weeks']}\n"

        prompt += f"""
Which {top_n} plots are most different from the others? Consider:
- Unusual yield levels
- High variability
- Different farm/year patterns
- Data quality issues

Respond with just the plot IDs separated by commas (e.g., "5,12,18,3,7")"""

        response = self.query(prompt, stream=False)

        try:
            outlier_ids = [int(x.strip()) for x in response.split(',') if x.strip().isdigit()]
            return outlier_ids[:top_n]
        except:
            return []

    def _format_metadata(self, metadata: Dict) -> str:
        """Format metadata dict into readable string (simple version)"""
        parts = []
        for key, value in metadata.items():
            if value is not None and value != 'unknown':
                parts.append(f"{key}={value}")
        return ", ".join(parts) if parts else "No metadata"

    def _format_metadata_detailed(self, metadata: Dict) -> str:
        """Format metadata dict into detailed, structured string"""
        output = []

        # Curve features
        curve_features = ['mean_yield', 'std_yield', 'cv_yield', 'trend_slope',
                         'peak_week', 'peak_value', 'valley_value', 'range',
                         'volatility', 'growth_pattern']
        curve_parts = []
        for key in curve_features:
            if key in metadata and metadata[key] is not None:
                curve_parts.append(f"{key}={metadata[key]}")
        if curve_parts:
            output.append("Curve: " + ", ".join(curve_parts))

        # Context features
        context_features = ['farm', 'year', 'crop', 'soil', 'location']
        context_parts = []
        for key in context_features:
            if key in metadata and metadata[key] is not None:
                context_parts.append(f"{key}={metadata[key]}")
        if context_parts:
            output.append("Context: " + ", ".join(context_parts))

        return "\n".join(output) if output else "No metadata"

    def ensure_diversity(self,
                        selected_ids: List[int],
                        all_candidates: List[Tuple[int, Dict, float]],
                        min_farms: int = 2,
                        min_years: int = 2) -> Tuple[List[int], Dict]:
        """
        Check diversity in selected plots and optionally add diverse candidates

        Args:
            selected_ids: Currently selected plot IDs
            all_candidates: All candidate plots (plot_id, metadata, score)
            min_farms: Minimum number of different farms desired
            min_years: Minimum number of different years desired

        Returns:
            Tuple of (potentially modified list of IDs, diversity report)
        """
        # Build metadata lookup
        metadata_lookup = {pid: meta for pid, meta, _ in all_candidates}

        # Analyze current selection
        farms = set()
        years = set()
        growth_patterns = defaultdict(int)

        for pid in selected_ids:
            if pid in metadata_lookup:
                meta = metadata_lookup[pid]
                if meta.get('farm'):
                    farms.add(meta['farm'])
                if meta.get('year'):
                    years.add(meta['year'])
                if meta.get('growth_pattern'):
                    growth_patterns[meta['growth_pattern']] += 1

        diversity_report = {
            'num_farms': len(farms),
            'num_years': len(years),
            'farms': list(farms),
            'years': list(years),
            'growth_patterns': dict(growth_patterns),
            'needs_improvement': False,
            'suggestions': []
        }

        # Check if diversity is lacking
        if len(farms) < min_farms:
            diversity_report['needs_improvement'] = True
            diversity_report['suggestions'].append(
                f"Only {len(farms)} farm(s) represented. Consider adding plots from different farms for robustness."
            )

        if len(years) < min_years:
            diversity_report['needs_improvement'] = True
            diversity_report['suggestions'].append(
                f"Only {len(years)} year(s) represented. Consider adding plots from different years to capture varying conditions."
            )

        # Check pattern diversity
        dominant_pattern = max(growth_patterns.items(), key=lambda x: x[1])[0] if growth_patterns else None
        if dominant_pattern and growth_patterns[dominant_pattern] > len(selected_ids) * 0.8:
            diversity_report['suggestions'].append(
                f"80%+ plots have '{dominant_pattern}' pattern. Some pattern diversity might help generalization."
            )

        return selected_ids, diversity_report


# Convenience function for quick queries
def quick_query(prompt: str,
                remote_url: str = "https://7167172d87c6.ngrok-free.app",
                model: str = "llama3.2:latest") -> str:
    """Quick one-off query to LLM"""
    agent = LLMAgent(remote_url, model)
    return agent.query(prompt, stream=True)


if __name__ == "__main__":
    # Test the agent
    agent = LLMAgent("https://7167172d87c6.ngrok-free.app")

    # Example: Analyze plot selection
    target = {
        'farm': 11,
        'year': 2020,
        'location': 'Aberdeen',
        'crop': 'strawberries'
    }

    candidates = [
        (5, {'farm': 11, 'year': 2020, 'location': 'Aberdeen'}, 0.847),
        (12, {'farm': 11, 'year': 2019, 'location': 'Aberdeen'}, 0.823),
        (18, {'farm': 15, 'year': 2020, 'location': 'Aberdeen'}, 0.801),
        (3, {'farm': 11, 'year': 2020, 'location': 'Aberdeen'}, 0.789),
    ]

    selected = agent.analyze_plot_metadata(target, candidates, top_k=3)
    print(f"\nSelected plots: {selected}")

    # Get explanation
    selected_with_meta = [(pid, meta, score) for pid, meta, score in candidates if pid in selected]
    explanation = agent.explain_selection(target, selected_with_meta)
    print(f"\nExplanation:\n{explanation}")
