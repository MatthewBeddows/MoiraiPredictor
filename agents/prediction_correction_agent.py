"""
Prediction Correction Agent

Autonomous agent that examines predictions and corrects anomalies based on
historical patterns from the SAME plot.

This is a true AI-powered agent with:
- Autonomy: Decides which predictions look suspicious
- Memory: Learns typical patterns for each plot
- Self-critique: Evaluates if corrections make sense
- Learning: Improves correction strategy over time
- Reasoning: Uses LLM to understand growth curves and spot anomalies
- Tools: Statistical analysis, smoothing, outlier detection

CRITICAL: This agent NEVER sees actual target values - only historical patterns
from the same plot to maintain prediction integrity.

Example anomalies it catches:
- Sudden spikes in an otherwise smooth growth curve
- Negative predictions for yield
- Predictions far outside historical range
- Predictions that violate known agricultural constraints
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from .base_agent import BaseAgent, Critique
from scipy.signal import savgol_filter
from scipy.stats import zscore
import warnings
warnings.filterwarnings('ignore')


class PredictionCorrectionAgent(BaseAgent):
    """
    Autonomous agent that corrects anomalous predictions

    Given raw predictions and historical data from the SAME plot,
    this agent:
    1. Analyzes historical patterns (growth curve, range, seasonality)
    2. Identifies suspicious predictions (spikes, out-of-range, etc.)
    3. Uses LLM to reason about what's realistic
    4. Corrects predictions to match expected patterns
    5. Never sees actual target values (no data leakage)

    The agent is conservative - only corrects obvious anomalies.
    """

    def __init__(self,
                 llm_agent=None,
                 verbose: bool = True,
                 max_iterations: int = 10,
                 correction_threshold: float = 3.0):  # Z-score threshold
        """
        Initialize Prediction Correction Agent

        Args:
            llm_agent: LLM for reasoning about patterns
            verbose: Print correction decisions
            correction_threshold: Z-score threshold for anomaly detection
            max_iterations: Max correction iterations
        """
        # Store before super().__init__
        self.correction_threshold = correction_threshold

        super().__init__(
            role="Prediction Correction Specialist",
            goal="Identify and fix anomalous predictions using historical patterns",
            backstory="Expert at spotting unrealistic predictions and correcting them based on agricultural knowledge",
            llm_agent=llm_agent,
            verbose=verbose,
            max_iterations=max_iterations
        )

        # Agent state
        self.plot_patterns = {}  # Store learned patterns per plot
        self.corrections_made = []
        self.correction_stats = {
            'predictions_checked': 0,
            'anomalies_detected': 0,
            'corrections_applied': 0,
            'avg_correction_size': 0.0
        }

    def _register_tools(self):
        """Register correction tools"""
        self.register_tool(
            "analyze_historical_pattern",
            "Extract pattern from plot's historical data",
            self._analyze_pattern_tool,
            {"plot_id": "int", "historical_data": "DataFrame"}
        )

        self.register_tool(
            "detect_anomalies",
            "Find suspicious predictions",
            self._detect_anomalies_tool,
            {"predictions": "array", "pattern": "dict"}
        )

        self.register_tool(
            "smooth_prediction",
            "Apply smoothing to anomalous prediction",
            self._smooth_prediction_tool,
            {"predictions": "array", "anomaly_indices": "list"}
        )

        self.register_tool(
            "reason_about_correction",
            "Use LLM to validate correction",
            self._reason_correction_tool,
            {"prediction": "float", "corrected": "float", "context": "dict"}
        )

    def solve(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main autonomous loop - correct predictions

        Args:
            task: {
                'predictions': Array of predictions to check,
                'plot_id': Plot ID being predicted,
                'historical_data': Historical data from THIS plot (no targets!),
                'prediction_dates': Optional dates for the predictions (for seasonal analysis),
                'feature_context': Optional features for the prediction period
            }

        Returns:
            {
                'corrected_predictions': Fixed predictions,
                'corrections_made': List of corrections with reasoning,
                'anomalies_found': Indices of anomalous predictions,
                'correction_stats': Statistics about corrections,
                'confidence': Confidence in corrections (0-1)
            }
        """
        self.log(f"\n{'='*60}", "info")
        self.log(f"PREDICTION CORRECTION AGENT", "info")
        self.log(f"{'='*60}", "info")
        self.log(self.get_state_summary(), "info")

        # Extract task data
        predictions = np.array(task['predictions'])
        plot_id = task['plot_id']
        historical_data = task['historical_data']
        prediction_dates = task.get('prediction_dates', None)
        feature_context = task.get('feature_context', {})

        self.log(f"\nChecking {len(predictions)} predictions for plot {plot_id}", "info")

        self.correction_stats['predictions_checked'] = len(predictions)

        # =================================================================
        # PHASE 1: Analyze Historical Pattern
        # =================================================================
        self.log(f"\n{'='*60}", "info")
        self.log("📊 PHASE 1: ANALYZING HISTORICAL PATTERN", "action")
        self.log(f"{'='*60}", "info")

        pattern = self.use_tool(
            "analyze_historical_pattern",
            plot_id=plot_id,
            historical_data=historical_data,
            prediction_dates=prediction_dates
        )

        # Check if pattern analysis failed
        if 'error' in pattern:
            self.log(f"⚠️ Pattern analysis failed: {pattern['error']}", "warning")
            # Return no corrections if we can't analyze the pattern
            return {
                'corrected_predictions': predictions,
                'corrections_made': [],
                'anomalies_found': 0,
                'correction_stats': {
                    'corrections_applied': 0,
                    'avg_correction_size': 0.0,
                    'max_correction_size': 0.0
                }
            }

        self.log(f"Historical range: [{pattern['min_value']:.2f}, {pattern['max_value']:.2f}]", "info")
        self.log(f"Typical growth: {pattern['growth_pattern']}", "info")
        self.log(f"Volatility: {pattern['volatility']:.3f}", "info")

        # =================================================================
        # PHASE 2: Detect Anomalies
        # =================================================================
        self.log(f"\n{'='*60}", "info")
        self.log("🔍 PHASE 2: DETECTING ANOMALIES", "action")
        self.log(f"{'='*60}", "info")

        # Convert prediction_dates to list of date strings if available
        date_list = None
        if prediction_dates is not None:
            date_list = [d.strftime('%Y-%m-%d') for d in prediction_dates]

        anomalies = self.use_tool(
            "detect_anomalies",
            predictions=predictions,
            pattern=pattern,
            prediction_dates=date_list
        )

        if len(anomalies) == 0:
            self.log("✓ No anomalies detected - predictions look good!", "success")
            return self._build_result(predictions, [], pattern)

        self.log(f"⚠️  Found {len(anomalies)} anomalous predictions", "warning")
        self.correction_stats['anomalies_detected'] = len(anomalies)

        for idx, reason in anomalies:
            self.log(f"  • Index {idx}: {predictions[idx]:.2f} - {reason}", "debug")

        # =================================================================
        # PHASE 3: Correct Anomalies
        # =================================================================
        self.log(f"\n{'='*60}", "info")
        self.log("🔧 PHASE 3: CORRECTING ANOMALIES", "action")
        self.log(f"{'='*60}", "info")

        corrected = predictions.copy()
        anomaly_indices = [idx for idx, _ in anomalies]

        corrected = self.use_tool(
            "smooth_prediction",
            predictions=corrected,
            anomaly_indices=anomaly_indices
        )

        # =================================================================
        # PHASE 4: Validate Corrections (LLM)
        # =================================================================
        if self.llm_agent:
            self.log(f"\n{'='*60}", "info")
            self.log("🧠 PHASE 4: VALIDATING CORRECTIONS", "action")
            self.log(f"{'='*60}", "info")

            for idx, reason in anomalies[:3]:  # Validate first 3
                validation = self.use_tool(
                    "reason_about_correction",
                    prediction=float(predictions[idx]),
                    corrected=float(corrected[idx]),
                    context={'reason': reason, 'pattern': pattern}
                )

                if validation['approved']:
                    self.log(f"✓ Correction {idx}: {predictions[idx]:.2f} → {corrected[idx]:.2f}", "success")
                else:
                    self.log(f"⚠️  Correction {idx} questionable, reverting", "warning")
                    corrected[idx] = predictions[idx]

        # =================================================================
        # PHASE 5: Final Statistics
        # =================================================================
        corrections_applied = np.sum(corrected != predictions)
        avg_correction = np.mean(np.abs(corrected - predictions)[corrected != predictions]) if corrections_applied > 0 else 0

        self.correction_stats['corrections_applied'] = int(corrections_applied)
        self.correction_stats['avg_correction_size'] = float(avg_correction)

        self.log(f"\n{'='*60}", "info")
        self.log("📋 CORRECTION COMPLETE", "success")
        self.log(f"{'='*60}", "info")
        self.log(f"Corrections applied: {corrections_applied}/{len(predictions)}", "info")
        self.log(f"Average correction size: {avg_correction:.3f}", "info")

        # Build detailed correction list
        correction_list = []
        for idx in range(len(predictions)):
            if predictions[idx] != corrected[idx]:
                correction_list.append({
                    'index': idx,
                    'original': float(predictions[idx]),
                    'corrected': float(corrected[idx]),
                    'change': float(corrected[idx] - predictions[idx]),
                    'reason': next((r for i, r in anomalies if i == idx), 'Unknown')
                })

        return self._build_result(corrected, correction_list, pattern)

    # =====================================================================
    # TOOL IMPLEMENTATIONS
    # =====================================================================

    def _analyze_pattern_tool(self, plot_id: int, historical_data: pd.DataFrame,
                             prediction_dates: Optional[pd.DatetimeIndex] = None) -> Dict[str, Any]:
        """Analyze historical pattern from plot's past data, including seasonal patterns"""
        # Extract target column (without looking at future actuals!)
        target_col = 'target'
        if target_col not in historical_data.columns:
            # Try to find any numeric column that looks like a target
            numeric_cols = historical_data.select_dtypes(include=[np.number]).columns
            target_col = numeric_cols[0] if len(numeric_cols) > 0 else None

        if target_col is None:
            return {'error': 'No target column found'}

        values = historical_data[target_col].dropna().values

        if len(values) < 3:
            return {'error': 'Insufficient historical data'}

        # Extract pattern characteristics
        pattern = {
            'plot_id': plot_id,
            'n_observations': len(values),
            'min_value': float(np.min(values)),
            'max_value': float(np.max(values)),
            'mean_value': float(np.mean(values)),
            'std_value': float(np.std(values)),
            'volatility': float(np.std(np.diff(values))),  # How much it jumps around
            'typical_range': float(np.max(values) - np.min(values)),
        }

        # Detect growth pattern
        if len(values) >= 5:
            trend = np.polyfit(np.arange(len(values)), values, 1)[0]
            pattern['trend_slope'] = float(trend)

            if trend > 0.1:
                pattern['growth_pattern'] = 'increasing'
            elif trend < -0.1:
                pattern['growth_pattern'] = 'decreasing'
            else:
                pattern['growth_pattern'] = 'stable'
        else:
            pattern['growth_pattern'] = 'unknown'

        # =====================================================================
        # CURVE SHAPE ANALYSIS: Model typical smooth curve shape
        # =====================================================================
        # Most agricultural crops follow a smooth bell curve: 0,0,1,2,3,5,7,8,5,3,1,0,0
        # We want to detect predictions with spikes that deviate from this smooth shape
        curve_shape = self._analyze_curve_shape(values)
        pattern.update(curve_shape)

        # =====================================================================
        # SEASONAL ANALYSIS: Compare to previous years at same dates
        # =====================================================================
        if prediction_dates is not None and historical_data.index.dtype.name.startswith('datetime'):
            seasonal_patterns = self._analyze_seasonal_patterns(
                historical_data, target_col, prediction_dates
            )
            pattern.update(seasonal_patterns)

        # Store for future use
        self.plot_patterns[plot_id] = pattern

        return pattern

    def _analyze_curve_shape(self, values: np.ndarray) -> Dict[str, Any]:
        """
        Analyze the typical smooth curve shape from historical data

        Agricultural yields typically follow smooth bell curves like:
        0,0,1,2,3,5,7,8,5,3,1,0,0

        This detects the typical smoothness and helps identify spiky predictions
        """
        curve_info = {
            'has_curve_model': False,
            'expected_smoothness': 0.0,
            'typical_acceleration': 0.0
        }

        if len(values) < 5:
            return curve_info

        try:
            # Calculate smoothness metrics from historical data
            # Smoothness = how gradually values change (low 2nd derivative)
            first_derivative = np.diff(values)  # Rate of change
            second_derivative = np.diff(first_derivative)  # Acceleration/curvature

            # Typical smoothness: std of acceleration (lower = smoother)
            typical_smoothness = float(np.std(second_derivative))

            # Typical rate of change
            typical_change_rate = float(np.std(first_derivative))

            # Fit a smooth polynomial to see expected shape
            # Use degree 3 or 4 for bell curve
            x = np.arange(len(values))
            degree = min(4, len(values) - 1)
            poly_coeffs = np.polyfit(x, values, degree)
            poly_fit = np.polyval(poly_coeffs, x)

            # How much do actual values deviate from smooth curve?
            fit_residuals = values - poly_fit
            typical_deviation = float(np.std(fit_residuals))

            # Maximum acceptable spike size (relative to smooth curve)
            # If historical data has spikes, we'll allow similar spikes
            # If historical is smooth, we'll be strict about spikes
            max_spike_ratio = float(np.max(np.abs(fit_residuals)) / (np.mean(values) + 1e-8))

            curve_info.update({
                'has_curve_model': True,
                'expected_smoothness': typical_smoothness,
                'typical_change_rate': typical_change_rate,
                'typical_deviation_from_smooth': typical_deviation,
                'max_historical_spike_ratio': max_spike_ratio,
                'polynomial_degree': degree,
                'polynomial_coeffs': poly_coeffs.tolist()
            })

        except Exception as e:
            curve_info['error'] = str(e)

        return curve_info

    def _analyze_seasonal_patterns(self, historical_data: pd.DataFrame,
                                   target_col: str,
                                   prediction_dates: pd.DatetimeIndex) -> Dict[str, Any]:
        """
        Analyze seasonal patterns from previous years at the same dates

        For each prediction date, look at historical data from:
        - Same week in previous years
        - Same month in previous years

        Returns expected ranges and patterns for those specific dates
        """
        seasonal_info = {
            'has_seasonal_data': False,
            'seasonal_patterns': {}
        }

        try:
            # Group historical data by week-of-year or month
            hist_df = historical_data.copy()
            hist_df['week_of_year'] = hist_df.index.isocalendar().week
            hist_df['month'] = hist_df.index.month
            hist_df['year'] = hist_df.index.year

            # For each prediction date, find historical values at same time of year
            for pred_date in prediction_dates:
                pred_week = pred_date.isocalendar().week
                pred_month = pred_date.month

                # Find historical data from same week in previous years
                same_week_data = hist_df[
                    (hist_df['week_of_year'] == pred_week) &
                    (hist_df['year'] < pred_date.year)
                ][target_col].values

                # Find historical data from same month in previous years
                same_month_data = hist_df[
                    (hist_df['month'] == pred_month) &
                    (hist_df['year'] < pred_date.year)
                ][target_col].values

                # Store seasonal expectations if we have data
                if len(same_week_data) > 0 or len(same_month_data) > 0:
                    seasonal_info['has_seasonal_data'] = True

                    date_key = pred_date.strftime('%Y-%m-%d')
                    seasonal_info['seasonal_patterns'][date_key] = {
                        'week_of_year': int(pred_week),
                        'month': int(pred_month),
                    }

                    # Weekly seasonal pattern (more precise)
                    if len(same_week_data) >= 2:
                        seasonal_info['seasonal_patterns'][date_key]['weekly'] = {
                            'n_years': len(same_week_data),
                            'mean': float(np.mean(same_week_data)),
                            'std': float(np.std(same_week_data)),
                            'min': float(np.min(same_week_data)),
                            'max': float(np.max(same_week_data)),
                            'range': float(np.max(same_week_data) - np.min(same_week_data))
                        }

                    # Monthly seasonal pattern (fallback if weekly not available)
                    if len(same_month_data) >= 2:
                        seasonal_info['seasonal_patterns'][date_key]['monthly'] = {
                            'n_years': len(same_month_data),
                            'mean': float(np.mean(same_month_data)),
                            'std': float(np.std(same_month_data)),
                            'min': float(np.min(same_month_data)),
                            'max': float(np.max(same_month_data)),
                            'range': float(np.max(same_month_data) - np.min(same_month_data))
                        }

        except Exception as e:
            # If seasonal analysis fails, just skip it
            seasonal_info['error'] = str(e)

        return seasonal_info

    def _detect_anomalies_tool(self, predictions: np.ndarray, pattern: Dict,
                              prediction_dates: Optional[List] = None) -> List[Tuple[int, str]]:
        """Detect anomalous predictions using historical and seasonal patterns"""
        anomalies = []

        for idx, pred in enumerate(predictions):
            reasons = []

            # Check 1: Negative yield (impossible)
            if pred < 0:
                reasons.append("Negative prediction (impossible)")

            # Check 2: Way outside historical range (more aggressive)
            buffer = pattern['std_value'] * self.correction_threshold  # Use configurable threshold
            if pred > pattern['max_value'] + buffer:
                reasons.append(f"Exceeds historical max by {pred - pattern['max_value']:.2f}")
            elif pred < pattern['min_value'] - buffer and pattern['min_value'] > 0:
                reasons.append(f"Below historical min by {pattern['min_value'] - pred:.2f}")

            # Check 3: Sudden spike in sequence (more aggressive)
            if idx > 0 and len(predictions) > 1:
                prev_pred = predictions[idx - 1]
                change = abs(pred - prev_pred)
                # Tighter threshold: use correction_threshold instead of 5x
                volatility_threshold = max(pattern['volatility'] * self.correction_threshold, pattern['std_value'])
                if change > volatility_threshold:
                    reasons.append(f"Sudden spike: {change:.2f} (threshold: {volatility_threshold:.2f})")

            # Check 4: Unrealistic for growth pattern (more aggressive)
            if pattern.get('growth_pattern') == 'stable':
                if abs(pred - pattern['mean_value']) > pattern['std_value'] * self.correction_threshold:
                    reasons.append(f"Too far from stable mean")

            # Check 5: SEASONAL WEIRDNESS - Compare to same date in previous years
            # IMPORTANT: Skip this check for sparse/seasonal crops where most weeks are zero
            if pattern.get('has_seasonal_data') and prediction_dates and idx < len(prediction_dates):
                seasonal_patterns = pattern.get('seasonal_patterns', {})

                # Get seasonal pattern for this specific date
                if prediction_dates[idx] in seasonal_patterns:
                    date_pattern = seasonal_patterns[prediction_dates[idx]]

                    # Use weekly pattern if available (more precise)
                    if 'weekly' in date_pattern:
                        weekly = date_pattern['weekly']
                        seasonal_mean = weekly['mean']
                        seasonal_std = weekly['std']

                        # CRITICAL FIX: Skip seasonal check if historical data for this week is sparse/zero
                        # For crops like strawberries that only produce a few weeks per year,
                        # comparing to zero-weeks is meaningless
                        if seasonal_mean > 0 or weekly['max'] > 0:
                            seasonal_buffer = max(seasonal_std * self.correction_threshold, seasonal_mean * 0.3)

                            # More aggressive: flag if 2-3x historical max (not 5x)
                            multiplier = 2.0 + self.correction_threshold  # 2.0=4x, 3.0=5x
                            if pred > weekly['max'] * multiplier:
                                reasons.append(
                                    f"Week {date_pattern['week_of_year']}: {pred:.1f} much higher than typical "
                                    f"{seasonal_mean:.1f}±{seasonal_std:.1f} (from {weekly['n_years']} prev years)"
                                )
                            elif seasonal_mean > 0 and pred < weekly['min'] * (0.5 - self.correction_threshold * 0.1):
                                reasons.append(
                                    f"Week {date_pattern['week_of_year']}: {pred:.1f} much lower than typical "
                                    f"{seasonal_mean:.1f}±{seasonal_std:.1f} (from {weekly['n_years']} prev years)"
                                )

                    # Fallback to monthly pattern
                    elif 'monthly' in date_pattern:
                        monthly = date_pattern['monthly']
                        seasonal_mean = monthly['mean']
                        seasonal_std = monthly['std']

                        # Same sparse data check for monthly
                        if seasonal_mean > 0 or monthly['max'] > 0:
                            # More aggressive threshold
                            multiplier = 2.0 + self.correction_threshold
                            if pred > monthly['max'] * multiplier:
                                reasons.append(
                                    f"Month {date_pattern['month']}: {pred:.1f} outside typical range "
                                    f"{seasonal_mean:.1f}±{seasonal_std:.1f} (from {monthly['n_years']} prev years)"
                                )

            # Check 6: CURVE SHAPE VIOLATION - Spikes that don't fit smooth curve
            # IMPORTANT: Only apply this for crops with dense historical data
            # Skip for sparse crops where curve fitting is unreliable
            has_dense_history = pattern.get('n_observations', 0) >= 10 and pattern.get('mean_value', 0) > 1

            if pattern.get('has_curve_model') and has_dense_history and len(predictions) >= 3:
                # Check if this prediction creates an unusual spike in the curve
                # Compare to what a smooth polynomial would predict
                if idx > 0 and idx < len(predictions) - 1:
                    # Fit polynomial through neighboring predictions
                    neighbor_indices = []
                    neighbor_values = []

                    # Include context around this point (but not the point itself)
                    for j in range(max(0, idx-2), min(len(predictions), idx+3)):
                        if j != idx:
                            neighbor_indices.append(j)
                            neighbor_values.append(predictions[j])

                    if len(neighbor_values) >= 3:
                        # Fit smooth curve through neighbors
                        neighbor_poly = np.polyfit(neighbor_indices, neighbor_values, min(2, len(neighbor_values)-1))
                        expected_value = np.polyval(neighbor_poly, idx)

                        # How much does this prediction deviate from smooth expectation?
                        deviation = abs(pred - expected_value)

                        # Compare to historical curve smoothness
                        typical_deviation = pattern.get('typical_deviation_from_smooth', pattern['std_value'])

                        # More aggressive: use correction_threshold (2.0 = 6x, 3.0 = 9x)
                        spike_threshold = typical_deviation * (3 * self.correction_threshold)

                        relative_change_threshold = 1.5 if self.correction_threshold < 2.5 else 2.0

                        if deviation > spike_threshold and deviation > expected_value * relative_change_threshold:
                            # Flag if deviation is both:
                            # 1. Much larger than historical smoothness
                            # 2. Large relative to expected value
                            reasons.append(
                                f"Spike breaks curve shape: {pred:.1f} vs expected {expected_value:.1f} "
                                f"(deviation {deviation:.1f} >> typical {typical_deviation:.1f})"
                            )

            if reasons:
                anomalies.append((idx, "; ".join(reasons)))

        return anomalies

    def _smooth_prediction_tool(self, predictions: np.ndarray, anomaly_indices: List[int]) -> np.ndarray:
        """Smooth anomalous predictions"""
        corrected = predictions.copy()

        for idx in anomaly_indices:
            # Strategy: Replace with interpolation from neighbors
            if idx == 0:
                # First prediction - use next value
                if len(predictions) > 1:
                    corrected[idx] = predictions[1]
            elif idx == len(predictions) - 1:
                # Last prediction - use previous value
                corrected[idx] = predictions[idx - 1]
            else:
                # Middle - interpolate from neighbors
                corrected[idx] = (predictions[idx - 1] + predictions[idx + 1]) / 2

            # Ensure it stays in reasonable bounds
            if self.plot_patterns:
                pattern = list(self.plot_patterns.values())[0]
                corrected[idx] = np.clip(
                    corrected[idx],
                    max(0, pattern['min_value'] - pattern['std_value']),
                    pattern['max_value'] + pattern['std_value'] * 2
                )

        return corrected

    def _reason_correction_tool(self, prediction: float, corrected: float, context: Dict) -> Dict[str, Any]:
        """Use LLM to validate correction"""
        if not self.llm_agent:
            return {'approved': True, 'reasoning': 'No LLM available'}

        pattern = context.get('pattern', {})
        reason = context.get('reason', 'Unknown')

        prompt = f"""You are an agricultural yield prediction expert. Evaluate this correction:

Original Prediction: {prediction:.2f}
Corrected Prediction: {corrected:.2f}
Correction Size: {abs(corrected - prediction):.2f}

Historical Context:
- Historical range: [{pattern.get('min_value', 0):.2f}, {pattern.get('max_value', 100):.2f}]
- Typical mean: {pattern.get('mean_value', 50):.2f}
- Growth pattern: {pattern.get('growth_pattern', 'unknown')}

Reason for flagging: {reason}

Question: Is this correction reasonable and conservative?

Consider:
1. Does the correction make agricultural sense?
2. Is it a minimal change (conservative)?
3. Does it respect historical patterns?

Answer with APPROVED or REJECTED and brief reasoning (1 sentence).
"""

        response = self.llm_agent.query(prompt, stream=False)

        approved = 'APPROVED' in response.upper() or 'YES' in response.upper()

        return {
            'approved': approved,
            'reasoning': response
        }

    def _build_result(self, corrected: np.ndarray, corrections: List[Dict], pattern: Dict) -> Dict[str, Any]:
        """Build final result"""
        confidence = 1.0
        if len(corrections) > 0:
            # Lower confidence if many corrections
            confidence = max(0.5, 1.0 - (len(corrections) / len(corrected)) * 0.5)

        return {
            'corrected_predictions': corrected,
            'corrections_made': corrections,
            'anomalies_found': len(corrections),
            'correction_stats': self.correction_stats,
            'pattern_used': pattern,
            'confidence': confidence
        }

    def _build_critique_prompt(self, result: Dict, context: Dict) -> str:
        """Build critique prompt (required by base class)"""
        return f"""Evaluate prediction correction results:

Corrections made: {result['anomalies_found']}
Confidence: {result['confidence']:.2f}

Were corrections conservative and appropriate?
"""


if __name__ == "__main__":
    print("Prediction Correction Agent - Example Usage")
    print("="*60)

    # Create mock historical data
    historical = pd.DataFrame({
        'target': [10, 15, 20, 25, 30, 35, 40, 45]  # Smooth increasing trend
    })

    # Create predictions with anomalies
    predictions = np.array([50, 55, 200, 65, 70, -5, 80])  # Spike at idx 2, negative at idx 5

    print(f"\nOriginal predictions: {predictions}")
    print(f"Anomalies: index 2 (200 - spike), index 5 (-5 - negative)")

    # Create agent
    agent = PredictionCorrectionAgent(
        llm_agent=None,
        verbose=True,
        correction_threshold=3.0
    )

    # Run correction
    result = agent.solve({
        'predictions': predictions,
        'plot_id': 1,
        'historical_data': historical
    })

    print(f"\n{'='*60}")
    print("RESULTS:")
    print(f"{'='*60}")
    print(f"Corrected predictions: {result['corrected_predictions']}")
    print(f"Anomalies found: {result['anomalies_found']}")
    print(f"Corrections applied: {result['correction_stats']['corrections_applied']}")
    print(f"Confidence: {result['confidence']:.2f}")

    print(f"\nDetailed corrections:")
    for corr in result['corrections_made']:
        print(f"  • Index {corr['index']}: {corr['original']:.2f} → {corr['corrected']:.2f}")
        print(f"    Reason: {corr['reason']}")

    print("\n✓ Test complete!")
