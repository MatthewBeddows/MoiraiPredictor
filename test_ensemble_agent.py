#!/usr/bin/env python3
from agents import EnsembleStrategyAgent
import numpy as np

# Mock Model class
class MockModel:
    def __init__(self, name, bias=0.0, noise_level=0.01):
        self.name = name
        self.bias = bias
        self.noise_level = noise_level

    def predict(self, X):
        n = len(X) if hasattr(X, '__len__') else 100
        predictions = X + self.bias + np.random.randn(n) * self.noise_level
        return predictions

# Model trainers
def train_model_a(train_data):
    return MockModel("model_a", bias=-0.05, noise_level=0.02)

def train_model_b(train_data):
    return MockModel("model_b", bias=0.05, noise_level=0.02)

def train_model_c(train_data):
    return MockModel("model_c", bias=0.0, noise_level=0.03)

model_trainers = {
    'model_a': train_model_a,
    'model_b': train_model_b,
    'model_c': train_model_c
}

print("Ensemble Strategy Agent - Test")
print("="*60)

agent = EnsembleStrategyAgent(
    model_trainers=model_trainers,
    llm_agent=None,
    verbose=True,
    ensemble_method='weighted'
)

np.random.seed(42)
train_data = np.random.randn(100)
val_data = np.random.randn(50)
val_labels = val_data
test_data = np.random.randn(30)
test_labels = test_data

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
print(f"Best individual: {result['best_individual_mae']:.4f}")
print(f"Improvement: {result['improvement_over_best']:.4f} ({result['improvement_pct']:.2f}%)")
print(f"\nWeights:")
for model, weight in result['optimal_weights'].items():
    print(f"  {model}: {weight:.3f}")
print("\n✓ Test complete!")
