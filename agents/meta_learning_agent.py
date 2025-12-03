"""
Meta-Learning Agent

Autonomous agent that learns optimal hyperparameters from past experiments.

Instead of manually tuning:
- TOP_SIMILAR (how many training plots?)
- diversity_weight (how much to weight diversity?)
- min_quality_score (quality threshold?)
- finetuning_epochs, learning_rate, etc.

This agent runs experiments, tracks results, and learns what works best.

This is a true AI-powered agent with:
- Memory: Tracks all experiments and their MAE results
- Autonomy: Runs experiments until finds optimal hyperparameters
- Self-critique: Evaluates if hyperparameters are improving
- Learning: Builds a model of hyperparameter → MAE relationship
- Planning: Uses Bayesian optimization to intelligently search space
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from .base_agent import BaseAgent, Critique
import json
from datetime import datetime


class MetaLearningAgent(BaseAgent):
    """
    Learns optimal hyperparameters through systematic experimentation

    How it works:
    1. Define hyperparameter search space
    2. Run experiment with initial hyperparameters
    3. Evaluate MAE
    4. Use LLM + Bayesian optimization to suggest next hyperparameters
    5. Repeat until convergence
    6. Return best hyperparameters found
    """

    def __init__(self,
                 experiment_runner,  # Function(hyperparams) -> mae
                 llm_agent=None,
                 verbose: bool = True,
                 max_iterations: int = 20,
                 optimization_method: str = 'bayesian'):  # 'bayesian', 'grid', 'random', 'llm'
        """
        Initialize Meta-Learning Agent

        Args:
            experiment_runner: Function(hyperparams: Dict) -> mae: float
            llm_agent: LLM for reasoning about hyperparameter choices
            verbose: Print agent reasoning
            max_iterations: Max experiments to run
            optimization_method: 'bayesian', 'grid', 'random', or 'llm'
        """
        # Store before super().__init__
        self.experiment_runner = experiment_runner
        self.optimization_method = optimization_method

        super().__init__(
            role="Meta-Learning Specialist",
            goal="Find hyperparameters that minimize MAE",
            backstory="Expert at learning from experiments and optimizing model performance",
            llm_agent=llm_agent,
            verbose=verbose,
            max_iterations=max_iterations
        )

        # Agent state
        self.hyperparameter_space = {}
        self.experiment_history = []  # List of (hyperparams, mae, timestamp)
        self.best_hyperparams = None
        self.best_mae = float('inf')

    def _register_tools(self):
        """Register tools for this agent"""
        self.register_tool(
            "run_experiment",
            "Run experiment with given hyperparameters and return MAE",
            self.experiment_runner,
            {"hyperparams": "dict"}
        )

        self.register_tool(
            "suggest_next_hyperparams",
            "Suggest next hyperparameters to try based on history",
            self._suggest_next_hyperparams,
            {"method": "string"}
        )

    def solve(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main autonomous loop - find optimal hyperparameters

        Args:
            task: {
                'hyperparameter_space': {
                    'top_similar': {'type': 'int', 'min': 5, 'max': 50},
                    'diversity_weight': {'type': 'float', 'min': 0.0, 'max': 0.5},
                    'quality_threshold': {'type': 'int', 'min': 30, 'max': 80},
                    'finetuning_epochs': {'type': 'int', 'min': 3, 'max': 10},
                    ...
                },
                'initial_hyperparams': Optional starting point,
                'baseline_mae': Optional baseline to compare against
            }

        Returns:
            {
                'best_hyperparams': Dict with optimal hyperparameters,
                'best_mae': Best MAE achieved,
                'improvement_from_baseline': Improvement over baseline,
                'experiment_history': All experiments run,
                'learned_insights': LLM insights about what works,
                'convergence_iteration': When best result was found
            }
        """
        self.log(f"\n{'='*60}", "info")
        self.log(f"META-LEARNING AGENT", "info")
        self.log(f"{'='*60}", "info")
        self.log(self.get_state_summary(), "info")

        # Extract task data
        self.hyperparameter_space = task['hyperparameter_space']
        initial_hyperparams = task.get('initial_hyperparams', None)
        baseline_mae = task.get('baseline_mae', None)

        self.log(f"Hyperparameter space: {len(self.hyperparameter_space)} parameters", "info")
        for param, config in self.hyperparameter_space.items():
            self.log(f"  {param}: {config}", "debug")

        # Phase 1: Baseline experiment (if initial hyperparams provided)
        if initial_hyperparams:
            self.log("\n📊 Phase 1: Baseline experiment...", "action")
            mae = self.use_tool("run_experiment", hyperparams=initial_hyperparams)
            self._record_experiment(initial_hyperparams, mae, is_baseline=True)
            baseline_mae = mae
            self.log(f"Baseline MAE: {baseline_mae:.4f}", "success")
        elif baseline_mae:
            self.log(f"Using provided baseline: {baseline_mae:.4f}", "info")
        else:
            self.log("No baseline provided, will establish one", "info")

        # Phase 2: Iterative optimization
        self.log("\n🔍 Phase 2: Hyperparameter optimization...", "action")
        self.log(f"Method: {self.optimization_method}", "info")

        for iteration in range(1, self.max_iterations + 1):
            if self.check_convergence():
                break

            self.iteration_count = iteration

            self.log(f"\n--- Iteration {iteration}/{self.max_iterations} ---", "thinking")

            # Generate next hyperparameters to try
            next_hyperparams = self.use_tool(
                "suggest_next_hyperparams",
                method=self.optimization_method
            )

            self.log(f"Testing hyperparameters:", "action")
            for param, value in next_hyperparams.items():
                self.log(f"  {param}: {value}", "debug")

            # Run experiment
            mae = self.use_tool("run_experiment", hyperparams=next_hyperparams)

            # Record result
            self._record_experiment(next_hyperparams, mae)

            # Check if this is best so far
            improvement = self.best_mae - mae
            if mae < self.best_mae:
                self.log(f"✓ New best! MAE: {mae:.4f} (improved by {improvement:.4f})", "success")
                self.best_mae = mae
                self.best_hyperparams = next_hyperparams.copy()
                self.learn('best_iteration', iteration)
            else:
                self.log(f"✗ MAE: {mae:.4f} (not better than {self.best_mae:.4f})", "info")

            # Self-critique
            critique = self._critique_hyperparams(next_hyperparams, mae, improvement)
            self.remember(
                action="test_hyperparameters",
                params=next_hyperparams,
                result={'mae': mae, 'improvement': improvement},
                critique=critique
            )

        # Phase 3: Final analysis
        self.log("\n📋 Phase 3: Generating insights...", "action")

        insights = self._generate_insights(baseline_mae)

        result = {
            'best_hyperparams': self.best_hyperparams,
            'best_mae': self.best_mae,
            'improvement_from_baseline': (baseline_mae - self.best_mae) if baseline_mae else None,
            'improvement_pct': ((baseline_mae - self.best_mae) / baseline_mae * 100) if baseline_mae else None,
            'experiment_count': len(self.experiment_history),
            'convergence_iteration': self.retrieve_learning('best_iteration', self.iteration_count),
            'experiment_history': self.experiment_history,
            'learned_insights': insights,
            'hyperparameter_importance': self._compute_importance()
        }

        self.log("\n" + "="*60, "info")
        self.log("META-LEARNING COMPLETE", "success")
        self.log("="*60, "info")
        self.log(f"Best MAE: {self.best_mae:.4f}", "success")
        if baseline_mae:
            self.log(f"Improvement: {result['improvement_from_baseline']:.4f} ({result['improvement_pct']:.2f}%)", "success")
        self.log(f"Experiments run: {result['experiment_count']}", "success")
        self.log(f"Best found at iteration: {result['convergence_iteration']}", "success")

        return result

    def _suggest_next_hyperparams(self, method: str) -> Dict[str, Any]:
        """
        Suggest next hyperparameters to try

        Args:
            method: 'bayesian', 'grid', 'random', or 'llm'

        Returns:
            Dict of hyperparameter values
        """
        if len(self.experiment_history) == 0:
            # First experiment: use defaults or random
            return self._random_hyperparams()

        if method == 'random':
            return self._random_hyperparams()

        elif method == 'grid':
            return self._grid_search_next()

        elif method == 'bayesian':
            return self._bayesian_suggestion()

        elif method == 'llm':
            return self._llm_suggestion()

        else:
            raise ValueError(f"Unknown optimization method: {method}")

    def _random_hyperparams(self) -> Dict[str, Any]:
        """Generate random hyperparameters from search space"""
        hyperparams = {}

        for param, config in self.hyperparameter_space.items():
            param_type = config['type']

            if param_type == 'int':
                value = np.random.randint(config['min'], config['max'] + 1)
            elif param_type == 'float':
                value = np.random.uniform(config['min'], config['max'])
            elif param_type == 'categorical':
                value = np.random.choice(config['choices'])
            else:
                raise ValueError(f"Unknown parameter type: {param_type}")

            hyperparams[param] = value

        return hyperparams

    def _grid_search_next(self) -> Dict[str, Any]:
        """
        Grid search: systematically try all combinations

        For now, simplified version that samples grid points
        """
        # Create grid points for each parameter
        grid_points = {}

        for param, config in self.hyperparameter_space.items():
            param_type = config['type']

            if param_type == 'int':
                # Sample 5 points evenly spaced
                points = np.linspace(config['min'], config['max'], 5, dtype=int)
                grid_points[param] = list(set(points))  # Remove duplicates

            elif param_type == 'float':
                # Sample 5 points evenly spaced
                points = np.linspace(config['min'], config['max'], 5)
                grid_points[param] = list(points)

            elif param_type == 'categorical':
                grid_points[param] = config['choices']

        # Find untried combination
        # Simplified: just sample one point per parameter that hasn't been tried much
        hyperparams = {}
        for param, points in grid_points.items():
            # Count how many times each value has been tried
            tried_counts = {point: 0 for point in points}
            for exp_hyp, _, _ in self.experiment_history:
                if param in exp_hyp and exp_hyp[param] in tried_counts:
                    tried_counts[exp_hyp[param]] += 1

            # Pick least-tried value
            least_tried = min(tried_counts.items(), key=lambda x: x[1])
            hyperparams[param] = least_tried[0]

        return hyperparams

    def _bayesian_suggestion(self) -> Dict[str, Any]:
        """
        Bayesian optimization: model MAE as function of hyperparams

        Simplified implementation using exploitation + exploration
        """
        if len(self.experiment_history) < 3:
            # Need more data, use random
            return self._random_hyperparams()

        # Extract data
        X = []  # Hyperparameter vectors
        y = []  # MAE values

        for hyperparams, mae, _ in self.experiment_history:
            x_vec = self._hyperparams_to_vector(hyperparams)
            X.append(x_vec)
            y.append(mae)

        X = np.array(X)
        y = np.array(y)

        # Simple approach: find region with lowest MAE and explore nearby
        best_idx = np.argmin(y)
        best_x = X[best_idx]

        # Add Gaussian noise to explore nearby
        noise_scale = 0.2  # 20% of range
        new_x = best_x + np.random.randn(len(best_x)) * noise_scale

        # Clip to valid ranges
        new_x = np.clip(new_x, 0, 1)  # Normalized space

        # Convert back to hyperparameters
        return self._vector_to_hyperparams(new_x)

    def _llm_suggestion(self) -> Dict[str, Any]:
        """
        Use LLM to suggest next hyperparameters based on experiment history
        """
        if not self.llm_agent:
            # Fallback to Bayesian
            return self._bayesian_suggestion()

        # Build prompt with experiment history
        history_summary = self._format_experiment_history(top_n=10)

        prompt = f"""You are a hyperparameter optimization expert for agricultural forecasting.

Experiment History (sorted by MAE, best first):
{history_summary}

Hyperparameter Space:
{json.dumps(self.hyperparameter_space, indent=2)}

Current Best MAE: {self.best_mae:.4f}

Task: Suggest the NEXT set of hyperparameters to try.

Consider:
1. What patterns do you see in successful experiments?
2. Which hyperparameters seem most important?
3. Should we exploit (refine best settings) or explore (try new regions)?
4. Are there any interactions between hyperparameters?

Provide your reasoning and then suggest specific values.

Format your response as:
Reasoning: [your analysis]
Suggestion:
- param1: value1
- param2: value2
...
"""

        response = self.llm_agent.query(prompt, stream=False)

        # Parse LLM response
        suggested = self._parse_llm_suggestion(response)

        if suggested:
            return suggested
        else:
            # LLM parsing failed, fallback
            self.log("LLM suggestion parsing failed, using Bayesian", "warning")
            return self._bayesian_suggestion()

    def _hyperparams_to_vector(self, hyperparams: Dict) -> np.ndarray:
        """Convert hyperparameters to normalized vector [0, 1]"""
        vector = []

        for param in sorted(self.hyperparameter_space.keys()):
            config = self.hyperparameter_space[param]
            value = hyperparams.get(param, config.get('default', 0))

            if config['type'] == 'int' or config['type'] == 'float':
                # Normalize to [0, 1]
                normalized = (value - config['min']) / (config['max'] - config['min'])
                vector.append(normalized)

            elif config['type'] == 'categorical':
                # One-hot encoding (simplified: just index)
                choices = config['choices']
                idx = choices.index(value) if value in choices else 0
                normalized = idx / len(choices)
                vector.append(normalized)

        return np.array(vector)

    def _vector_to_hyperparams(self, vector: np.ndarray) -> Dict[str, Any]:
        """Convert normalized vector back to hyperparameters"""
        hyperparams = {}

        for i, param in enumerate(sorted(self.hyperparameter_space.keys())):
            config = self.hyperparameter_space[param]
            normalized = vector[i]

            if config['type'] == 'int':
                value = int(config['min'] + normalized * (config['max'] - config['min']))
                value = max(config['min'], min(config['max'], value))

            elif config['type'] == 'float':
                value = config['min'] + normalized * (config['max'] - config['min'])
                value = max(config['min'], min(config['max'], value))

            elif config['type'] == 'categorical':
                choices = config['choices']
                idx = int(normalized * len(choices))
                idx = max(0, min(len(choices) - 1, idx))
                value = choices[idx]

            hyperparams[param] = value

        return hyperparams

    def _record_experiment(self, hyperparams: Dict, mae: float, is_baseline: bool = False):
        """Record experiment result"""
        self.experiment_history.append((
            hyperparams.copy(),
            mae,
            datetime.now().isoformat()
        ))

        # Update best if this is better
        if mae < self.best_mae:
            self.best_mae = mae
            self.best_hyperparams = hyperparams.copy()

    def _critique_hyperparams(self, hyperparams: Dict, mae: float, improvement: float) -> Critique:
        """Evaluate hyperparameter choice"""

        is_improvement = mae < self.best_mae

        if self.llm_agent and len(self.experiment_history) >= 3:
            # Use LLM for nuanced critique
            prompt = f"""Evaluate this hyperparameter experiment:

Hyperparameters tested:
{json.dumps(hyperparams, indent=2)}

Result: MAE = {mae:.4f}
Best MAE so far: {self.best_mae:.4f}
Improvement: {improvement:.4f}

Recent experiments (for context):
{self._format_experiment_history(top_n=5)}

Question: Is this a good result? What should we try next?

Provide brief analysis.
"""
            reasoning = self.llm_agent.query(prompt, stream=False)
        else:
            reasoning = f"MAE {mae:.4f} {'improves' if is_improvement else 'does not improve'} on best {self.best_mae:.4f}"

        return Critique(
            is_success=is_improvement,
            score=(self.best_mae / mae) if mae > 0 else 1.0,
            reasoning=reasoning,
            suggestions=[],
            next_action="exploit" if is_improvement else "explore"
        )

    def _format_experiment_history(self, top_n: int = 10) -> str:
        """Format experiment history for display"""
        if not self.experiment_history:
            return "No experiments yet"

        # Sort by MAE (best first)
        sorted_history = sorted(self.experiment_history, key=lambda x: x[1])

        lines = []
        for i, (hyperparams, mae, timestamp) in enumerate(sorted_history[:top_n], 1):
            params_str = ", ".join([f"{k}={v}" for k, v in hyperparams.items()])
            lines.append(f"{i}. MAE={mae:.4f} | {params_str}")

        return "\n".join(lines)

    def _parse_llm_suggestion(self, response: str) -> Optional[Dict]:
        """Parse LLM's hyperparameter suggestion"""
        # Look for lines like "- param: value"
        hyperparams = {}

        for line in response.split('\n'):
            line = line.strip()
            if line.startswith('-') or line.startswith('•'):
                # Extract param: value
                parts = line.lstrip('-•').strip().split(':', 1)
                if len(parts) == 2:
                    param = parts[0].strip()
                    value_str = parts[1].strip()

                    if param in self.hyperparameter_space:
                        config = self.hyperparameter_space[param]

                        try:
                            if config['type'] == 'int':
                                value = int(float(value_str))
                            elif config['type'] == 'float':
                                value = float(value_str)
                            elif config['type'] == 'categorical':
                                value = value_str
                            else:
                                continue

                            hyperparams[param] = value
                        except (ValueError, TypeError):
                            continue

        # Check if we got all parameters
        if len(hyperparams) == len(self.hyperparameter_space):
            return hyperparams
        else:
            return None

    def _generate_insights(self, baseline_mae: Optional[float]) -> str:
        """Generate insights about what was learned"""
        if not self.llm_agent or len(self.experiment_history) < 5:
            return f"Found best hyperparameters with MAE {self.best_mae:.4f}"

        prompt = f"""Analyze these hyperparameter optimization experiments and provide insights.

Total experiments: {len(self.experiment_history)}
Best MAE: {self.best_mae:.4f}
Baseline MAE: {baseline_mae:.4f if baseline_mae else 'N/A'}

Top 10 experiments:
{self._format_experiment_history(top_n=10)}

Best hyperparameters:
{json.dumps(self.best_hyperparams, indent=2)}

Questions:
1. What hyperparameters matter most for MAE?
2. What patterns led to good performance?
3. Are there surprising findings?
4. Recommendations for future optimization?

Provide 3-4 key insights in bullet points.
"""

        return self.llm_agent.query(prompt, stream=False)

    def _compute_importance(self) -> Dict[str, float]:
        """
        Compute importance score for each hyperparameter

        Simple correlation-based importance
        """
        if len(self.experiment_history) < 5:
            return {}

        importance = {}

        for param in self.hyperparameter_space.keys():
            # Extract values and MAEs
            values = []
            maes = []

            for hyperparams, mae, _ in self.experiment_history:
                if param in hyperparams:
                    value = hyperparams[param]

                    # Normalize value
                    config = self.hyperparameter_space[param]
                    if config['type'] in ['int', 'float']:
                        normalized = (value - config['min']) / (config['max'] - config['min'])
                    else:
                        normalized = value

                    values.append(normalized)
                    maes.append(mae)

            # Compute correlation (absolute value = importance)
            if len(values) >= 3:
                correlation = np.corrcoef(values, maes)[0, 1]
                importance[param] = abs(correlation) if not np.isnan(correlation) else 0.0

        return importance

    def _build_critique_prompt(self, result: Dict, context: Dict) -> str:
        """Build critique prompt (required by base class)"""
        return f"Evaluate hyperparameter optimization result: {result}"

    def check_convergence(self) -> bool:
        """Check if optimization has converged"""
        # Standard checks
        if super().check_convergence():
            return True

        # Check if MAE hasn't improved in last 5 iterations
        if len(self.experiment_history) >= 5:
            recent_maes = [mae for _, mae, _ in self.experiment_history[-5:]]
            if all(mae >= self.best_mae for mae in recent_maes):
                self.log("No improvement in last 5 experiments, converged", "warning")
                return True

        return False


if __name__ == "__main__":
    print("Meta-Learning Agent - Example Usage")
    print("="*60)

    # Mock experiment runner
    def mock_experiment(hyperparams):
        """Simulate training with given hyperparameters"""
        # True optimal: top_similar=15, diversity_weight=0.3, quality_threshold=60
        optimal = {'top_similar': 15, 'diversity_weight': 0.3, 'quality_threshold': 60}

        # Compute distance from optimal
        distance = 0
        distance += abs(hyperparams.get('top_similar', 20) - optimal['top_similar']) / 50.0
        distance += abs(hyperparams.get('diversity_weight', 0.15) - optimal['diversity_weight']) / 0.5
        distance += abs(hyperparams.get('quality_threshold', 50) - optimal['quality_threshold']) / 80.0

        # MAE is worse when further from optimal
        base_mae = 0.300
        mae = base_mae + distance * 0.2

        # Add small noise
        import random
        mae += random.uniform(-0.01, 0.01)

        return mae

    # Create agent
    agent = MetaLearningAgent(
        experiment_runner=mock_experiment,
        llm_agent=None,
        verbose=True,
        max_iterations=15,
        optimization_method='bayesian'
    )

    # Define search space
    search_space = {
        'top_similar': {'type': 'int', 'min': 5, 'max': 50},
        'diversity_weight': {'type': 'float', 'min': 0.0, 'max': 0.5},
        'quality_threshold': {'type': 'int', 'min': 30, 'max': 80}
    }

    # Run agent
    result = agent.solve({
        'hyperparameter_space': search_space,
        'initial_hyperparams': {'top_similar': 20, 'diversity_weight': 0.15, 'quality_threshold': 50},
        'baseline_mae': 0.320
    })

    print("\n" + "="*60)
    print("RESULTS:")
    print("="*60)
    print(f"Best hyperparameters: {result['best_hyperparams']}")
    print(f"Best MAE: {result['best_mae']:.4f}")
    print(f"Improvement: {result['improvement_from_baseline']:.4f} ({result['improvement_pct']:.2f}%)")
    print(f"Experiments run: {result['experiment_count']}")
    print(f"\nHyperparameter importance:")
    for param, importance in result['hyperparameter_importance'].items():
        print(f"  {param}: {importance:.3f}")
    print("\n✓ Test complete!")
