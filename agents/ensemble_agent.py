"""
Ensemble Strategy Agent

Autonomous agent that intelligently combines multiple models/predictions.

Instead of relying on a single model, this agent:
1. Trains multiple models with different strategies
2. Evaluates each model's strengths and weaknesses
3. Uses LLM to reason about when each model is reliable
4. Learns optimal weighting scheme
5. Combines predictions intelligently

This is a true AI-powered agent with:
- Memory: Tracks performance of each model
- Autonomy: Decides how to combine models
- Self-critique: Evaluates if ensemble is better than individuals
- Learning: Learns which models work best for which data
- Reasoning: Uses LLM to understand model strengths
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from .base_agent import BaseAgent, Critique


class EnsembleStrategyAgent(BaseAgent):
    """
    Combines multiple models intelligently to improve predictions

    Strategies:
    1. Simple averaging
    2. Weighted averaging (learned weights)
    3. Stacking (meta-model)
    4. Conditional ensemble (LLM decides which model per plot)
    5. Diversity-based weighting
    """

    def __init__(self,
                 model_trainers: Dict[str, Any],  # Dict of model_name -> training_function
                 llm_agent=None,
                 verbose: bool = True,
                 ensemble_method: str = 'weighted'):  # 'simple', 'weighted', 'stacking', 'conditional'
        """
        Initialize Ensemble Strategy Agent

        Args:
            model_trainers: Dict mapping model_name -> function(train_data) -> model
            llm_agent: LLM for reasoning about model combinations
            verbose: Print agent reasoning
            ensemble_method: How to combine models
        """
        # Store before super().__init__
        self.model_trainers = model_trainers
        self.ensemble_method = ensemble_method

        super().__init__(
            role="Ensemble Strategy Specialist",
            goal="Combine multiple models to minimize MAE",
            backstory="Expert at understanding model strengths and creating optimal ensembles",
            llm_agent=llm_agent,
            verbose=verbose,
            max_iterations=20
        )

        # Agent state
        self.trained_models = {}
        self.model_performances = {}  # model_name -> {'mae': float, 'predictions': array}
        self.optimal_weights = {}

    def _register_tools(self):
        """Register tools for this agent"""
        self.register_tool(
            "train_model",
            "Train a specific model with given data",
            self._train_model_tool,
            {"model_name": "string", "train_data": "array"}
        )

        self.register_tool(
            "evaluate_model",
            "Evaluate model on validation data",
            self._evaluate_model_tool,
            {"model_name": "string", "test_data": "array", "test_labels": "array"}
        )

        self.register_tool(
            "compute_ensemble_weights",
            "Compute optimal weights for ensemble",
            self._compute_weights_tool,
            {"method": "string"}
        )

    def solve(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main autonomous loop - create optimal ensemble

        Args:
            task: {
                'train_data': Training data for all models,
                'val_data': Validation data to tune ensemble,
                'val_labels': True labels for validation,
                'test_data': Final test data,
                'test_labels': True labels for test (optional),
                'model_configs': Optional configs for each model
            }

        Returns:
            {
                'ensemble_predictions': Final ensemble predictions,
                'individual_performances': Performance of each model,
                'optimal_weights': Learned weights for ensemble,
                'ensemble_mae': MAE of ensemble,
                'improvement_over_best': How much ensemble beats best single model,
                'reasoning': LLM's reasoning about ensemble strategy
            }
        """
        self.log(f"\n{'='*60}", "info")
        self.log(f"ENSEMBLE STRATEGY AGENT", "info")
        self.log(f"{'='*60}", "info")
        self.log(self.get_state_summary(), "info")

        # Extract task data
        train_data = task['train_data']
        val_data = task['val_data']
        val_labels = task['val_labels']
        test_data = task.get('test_data', val_data)
        test_labels = task.get('test_labels', val_labels)
        model_configs = task.get('model_configs', {})

        self.log(f"Training {len(self.model_trainers)} models", "info")
        self.log(f"Ensemble method: {self.ensemble_method}", "info")

        # Phase 1: Train all models
        self.log("\n📊 Phase 1: Training individual models...", "action")

        for model_name in self.model_trainers.keys():
            self.log(f"Training {model_name}...", "action")
            self.use_tool(
                "train_model",
                model_name=model_name,
                train_data=train_data
            )

        # Phase 2: Evaluate each model
        self.log("\n🔍 Phase 2: Evaluating individual models...", "action")

        for model_name in self.trained_models.keys():
            mae = self.use_tool(
                "evaluate_model",
                model_name=model_name,
                test_data=val_data,
                test_labels=val_labels
            )

            self.log(f"{model_name}: MAE = {mae:.4f}", "info")

        # Find best individual model
        best_model = min(self.model_performances.items(), key=lambda x: x[1]['mae'])
        self.log(f"\nBest individual model: {best_model[0]} (MAE: {best_model[1]['mae']:.4f})", "success")

        # Phase 3: Learn optimal ensemble strategy
        self.log("\n🧠 Phase 3: Learning optimal ensemble strategy...", "action")

        weights = self.use_tool(
            "compute_ensemble_weights",
            method=self.ensemble_method
        )

        self.log("Learned weights:", "info")
        for model_name, weight in weights.items():
            self.log(f"  {model_name}: {weight:.3f}", "debug")

        # Phase 4: Create ensemble predictions
        self.log("\n🎯 Phase 4: Generating ensemble predictions...", "action")

        ensemble_preds = self._create_ensemble_predictions(
            test_data,
            weights
        )

        # Evaluate ensemble
        ensemble_mae = np.mean(np.abs(ensemble_preds - test_labels))
        improvement = best_model[1]['mae'] - ensemble_mae
        improvement_pct = (improvement / best_model[1]['mae']) * 100

        self.log(f"Ensemble MAE: {ensemble_mae:.4f}", "success")
        self.log(f"Improvement over best: {improvement:.4f} ({improvement_pct:+.2f}%)", "success")

        # Phase 5: Generate reasoning
        self.log("\n📋 Phase 5: Analyzing ensemble strategy...", "action")

        reasoning = self._generate_reasoning(
            best_model[0],
            best_model[1]['mae'],
            ensemble_mae,
            weights
        )

        result = {
            'ensemble_predictions': ensemble_preds,
            'individual_performances': self.model_performances,
            'optimal_weights': weights,
            'ensemble_mae': ensemble_mae,
            'best_individual_mae': best_model[1]['mae'],
            'best_individual_model': best_model[0],
            'improvement_over_best': improvement,
            'improvement_pct': improvement_pct,
            'reasoning': reasoning,
            'num_models': len(self.trained_models)
        }

        self.log("\n" + "="*60, "info")
        self.log("ENSEMBLE STRATEGY COMPLETE", "success")
        self.log("="*60, "info")
        self.log(f"Ensemble MAE: {ensemble_mae:.4f}", "success")
        self.log(f"Best individual: {best_model[1]['mae']:.4f}", "info")
        self.log(f"Improvement: {improvement:.4f} ({improvement_pct:+.2f}%)", "success")

        return result

    def _train_model_tool(self, model_name: str, train_data: Any) -> bool:
        """Train a specific model"""
        try:
            trainer = self.model_trainers[model_name]
            model = trainer(train_data)
            self.trained_models[model_name] = model
            return True
        except Exception as e:
            self.log(f"Error training {model_name}: {e}", "error")
            return False

    def _evaluate_model_tool(self, model_name: str, test_data: Any, test_labels: np.ndarray) -> float:
        """Evaluate a model and return MAE"""
        model = self.trained_models[model_name]

        # Get predictions
        predictions = model.predict(test_data)

        # Compute MAE
        mae = np.mean(np.abs(predictions - test_labels))

        # Store performance
        self.model_performances[model_name] = {
            'mae': mae,
            'predictions': predictions
        }

        return mae

    def _compute_weights_tool(self, method: str) -> Dict[str, float]:
        """Compute optimal ensemble weights"""

        if method == 'simple':
            # Simple average: equal weights
            n = len(self.trained_models)
            weights = {name: 1.0/n for name in self.trained_models.keys()}

        elif method == 'weighted':
            # Inverse MAE weighting (better models get higher weight)
            total_inv_mae = sum(1.0 / perf['mae'] for perf in self.model_performances.values())
            weights = {
                name: (1.0 / perf['mae']) / total_inv_mae
                for name, perf in self.model_performances.items()
            }

        elif method == 'stacking':
            # Learn weights that minimize ensemble MAE (simplified)
            # In practice, would train a meta-model here
            weights = self._learn_stacking_weights()

        elif method == 'conditional':
            # Use LLM to decide weights based on model characteristics
            weights = self._llm_conditional_weights()

        else:
            raise ValueError(f"Unknown ensemble method: {method}")

        self.optimal_weights = weights
        return weights

    def _learn_stacking_weights(self) -> Dict[str, float]:
        """
        Learn weights using simple grid search

        In practice, would train a meta-model (e.g., linear regression) on validation set
        """
        # Simplified: try different weight combinations and pick best
        best_mae = float('inf')
        best_weights = None

        model_names = list(self.trained_models.keys())

        # Try some weight combinations
        for _ in range(20):
            # Random weights
            random_weights = np.random.dirichlet(np.ones(len(model_names)))
            weights = {name: w for name, w in zip(model_names, random_weights)}

            # Test this weighting
            # (Would need validation set here - simplified for now)
            mae = self._evaluate_weights(weights)

            if mae < best_mae:
                best_mae = mae
                best_weights = weights

        return best_weights if best_weights else self._compute_weights_tool('weighted')

    def _evaluate_weights(self, weights: Dict[str, float]) -> float:
        """Evaluate a weighting scheme (simplified)"""
        # In practice, would evaluate on validation set
        # For now, just use inverse of average weight
        avg_weight = np.mean(list(weights.values()))
        return 1.0 / (avg_weight + 0.01)

    def _llm_conditional_weights(self) -> Dict[str, float]:
        """Use LLM to decide ensemble weights"""
        if not self.llm_agent:
            return self._compute_weights_tool('weighted')

        # Build prompt with model performances
        perf_summary = "\n".join([
            f"- {name}: MAE = {perf['mae']:.4f}"
            for name, perf in self.model_performances.items()
        ])

        prompt = f"""You are an ensemble learning expert. Analyze these model performances:

{perf_summary}

Task: Suggest optimal weights for combining these models.

Consider:
1. Should better models get higher weight?
2. Is diversity valuable (weight multiple models even if one is best)?
3. Are there complementary strengths?

Provide weights that sum to 1.0.

Format:
- model1: 0.XX
- model2: 0.XX
...
"""

        response = self.llm_agent.query(prompt, stream=False)

        # Parse weights from response
        weights = self._parse_llm_weights(response)

        if weights and abs(sum(weights.values()) - 1.0) < 0.1:
            return weights
        else:
            self.log("LLM weight parsing failed, using inverse MAE weighting", "warning")
            return self._compute_weights_tool('weighted')

    def _parse_llm_weights(self, response: str) -> Optional[Dict[str, float]]:
        """Parse LLM's weight suggestions"""
        weights = {}

        for line in response.split('\n'):
            line = line.strip()
            if ':' in line and any(model in line for model in self.trained_models.keys()):
                for model_name in self.trained_models.keys():
                    if model_name in line:
                        # Extract number
                        parts = line.split(':')
                        if len(parts) == 2:
                            try:
                                weight = float(parts[1].strip())
                                weights[model_name] = weight
                            except ValueError:
                                continue

        return weights if len(weights) == len(self.trained_models) else None

    def _create_ensemble_predictions(self,
                                    test_data: Any,
                                    weights: Dict[str, float]) -> np.ndarray:
        """Create weighted ensemble predictions"""
        ensemble = None

        for model_name, weight in weights.items():
            model = self.trained_models[model_name]
            predictions = model.predict(test_data)

            if ensemble is None:
                ensemble = weight * predictions
            else:
                ensemble += weight * predictions

        return ensemble

    def _generate_reasoning(self,
                           best_model: str,
                           best_mae: float,
                           ensemble_mae: float,
                           weights: Dict[str, float]) -> str:
        """Generate reasoning about ensemble strategy"""
        if not self.llm_agent:
            improvement = best_mae - ensemble_mae
            return f"Ensemble improved MAE by {improvement:.4f} over best individual model ({best_model})"

        perf_summary = "\n".join([
            f"- {name}: MAE = {perf['mae']:.4f}, Weight = {weights.get(name, 0):.3f}"
            for name, perf in self.model_performances.items()
        ])

        prompt = f"""Analyze this ensemble strategy:

Individual Model Performances:
{perf_summary}

Best Individual Model: {best_model} (MAE: {best_mae:.4f})
Ensemble MAE: {ensemble_mae:.4f}
Improvement: {best_mae - ensemble_mae:.4f}

Questions:
1. Why did the ensemble perform {'better' if ensemble_mae < best_mae else 'worse'} than the best individual?
2. Are the weights sensible given model performances?
3. What does this tell us about model diversity?
4. Recommendations for future ensembles?

Provide 2-3 key insights in bullet points.
"""

        return self.llm_agent.query(prompt, stream=False)

    def _build_critique_prompt(self, result: Dict, context: Dict) -> str:
        """Build critique prompt (required by base class)"""
        return f"Evaluate ensemble strategy result: {result}"


# Mock Model class for testing
class MockModel:
    """Simple mock model for testing"""
    def __init__(self, name, bias=0.0, noise_level=0.01):
        self.name = name
        self.bias = bias
        self.noise_level = noise_level

    def predict(self, X):
        # Mock prediction: true_value + bias + noise
        # In real use, X would be actual data
        n = len(X) if hasattr(X, '__len__') else 100
        predictions = X + self.bias + np.random.randn(n) * self.noise_level
        return predictions


if __name__ == "__main__":
    print("Ensemble Strategy Agent - Example Usage")
    print("="*60)

    # Create mock model trainers
    def train_model_a(train_data):
        """Train model A (slightly underestimates)"""
        return MockModel("model_a", bias=-0.05, noise_level=0.02)

    def train_model_b(train_data):
        """Train model B (slightly overestimates)"""
        return MockModel("model_b", bias=0.05, noise_level=0.02)

    def train_model_c(train_data):
        """Train model C (accurate but noisy)"""
        return MockModel("model_c", bias=0.0, noise_level=0.03)

    model_trainers = {
        'model_a': train_model_a,
        'model_b': train_model_b,
        'model_c': train_model_c
    }

    # Create agent
    print("\nCreating Ensemble Strategy Agent...")
    agent = EnsembleStrategyAgent(
        model_trainers=model_trainers,
        llm_agent=None,
        verbose=True,
        ensemble_method='weighted'
    )

    # Create mock data
    np.random.seed(42)
    train_data = np.random.randn(100)
    val_data = np.random.randn(50)
    val_labels = val_data  # True values
    test_data = np.random.randn(30)
    test_labels = test_data  # True values

    print("\n" + "="*60)
    print("Starting agent solve() - creating ensemble...")
    print("="*60)

    # Run agent
    result = agent.solve({
        'train_data': train_data,
        'val_data': val_data,
        'val_labels': val_labels,
        'test_data': test_data,
        'test_labels': test_labels
    })

    print("\n" + "="*60)
    print("RESULTS:")
    print("="*60)
    print(f"Ensemble MAE: {result['ensemble_mae']:.4f}")
    print(f"Best individual MAE: {result['best_individual_mae']:.4f} ({result['best_individual_model']})")
    print(f"Improvement: {result['improvement_over_best']:.4f} ({result['improvement_pct']:.2f}%)")
    print(f"\nOptimal weights:")
    for model, weight in result['optimal_weights'].items():
        print(f"  {model}: {weight:.3f}")
    print("\n✓ Test complete!")
