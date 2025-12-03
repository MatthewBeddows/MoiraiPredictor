#!/usr/bin/env python3
"""
Test script for Meta-Learning Agent
"""

from agents import MetaLearningAgent

print("Meta-Learning Agent - Standalone Test")
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

    print(f"    Experiment: MAE = {mae:.4f} with {hyperparams}")
    return mae

# Create agent
print("\nCreating Meta-Learning Agent...")
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

print("\n" + "="*60)
print("Starting agent solve() - autonomous optimization begins...")
print("="*60)

# Run agent
result = agent.solve({
    'hyperparameter_space': search_space,
    'initial_hyperparams': {'top_similar': 20, 'diversity_weight': 0.15, 'quality_threshold': 50},
    'baseline_mae': 0.320
})

print("\n" + "="*60)
print("FINAL RESULTS:")
print("="*60)
print(f"Best hyperparameters found: {result['best_hyperparams']}")
print(f"Best MAE: {result['best_mae']:.4f}")
print(f"Baseline MAE: 0.320")
print(f"Improvement: {result['improvement_from_baseline']:.4f} ({result['improvement_pct']:.2f}%)")
print(f"Experiments run: {result['experiment_count']}")
print(f"Best found at iteration: {result['convergence_iteration']}")

print(f"\nHyperparameter importance:")
for param, importance in sorted(result['hyperparameter_importance'].items(), key=lambda x: x[1], reverse=True):
    print(f"  {param}: {importance:.3f}")

print("\n" + "="*60)
print("✓ Meta-Learning Agent test complete!")
print("="*60)

# Verify it found good hyperparameters
optimal = {'top_similar': 15, 'diversity_weight': 0.3, 'quality_threshold': 60}
found = result['best_hyperparams']

distance = 0
for param in optimal.keys():
    distance += abs(found[param] - optimal[param])

if distance < 20:
    print("\n✅ SUCCESS: Agent found near-optimal hyperparameters!")
else:
    print(f"\n⚠️  PARTIAL: Agent found hyperparameters with distance {distance:.1f} from optimal")
