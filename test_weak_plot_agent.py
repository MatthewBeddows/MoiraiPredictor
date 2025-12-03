#!/usr/bin/env python3
"""
Test script for Weak Plot Identifier Agent
"""

from agents import WeakPlotIdentifierAgent

print("Weak Plot Identifier Agent - Standalone Test")
print("=" * 60)

# Mock experiment runner for testing
def mock_experiment(training_ids, test_ids):
    """Simulate training and return mock MAE"""
    # Simulate: plot 5 and plot 12 are "weak" (removing improves MAE)
    weak_plots = {5, 12}
    mae = 0.500  # Base MAE

    # Removing weak plots improves MAE
    for weak in weak_plots:
        if weak not in training_ids:
            mae -= 0.015  # 1.5% improvement per weak plot removed

    # Add some noise
    import random
    mae += random.uniform(-0.002, 0.002)

    return mae

# Mock metadata
metadata = {
    i: {
        'farm': (i % 3) + 1,
        'year': 2020 + (i % 3),
        'growth_pattern': ['peak_middle', 'increasing', 'decreasing'][i % 3],
        'mean_yield': 10.0 + (i % 5),
        'volatility': 0.2 + (i % 10) * 0.05,
        'cv_yield': 0.15 + (i % 8) * 0.05
    }
    for i in range(1, 21)
}

# Create agent (without LLM for testing)
print("\nCreating Weak Plot Identifier Agent...")
agent = WeakPlotIdentifierAgent(
    experiment_runner=mock_experiment,
    llm_agent=None,
    verbose=True,
    max_iterations=20,
    min_improvement_threshold=0.01
)

print("\n" + "=" * 60)
print("Starting agent solve() - autonomous loop begins...")
print("=" * 60)

# Run agent
result = agent.solve({
    'training_ids': list(range(1, 21)),
    'test_ids': list(range(100, 110)),
    'metadata_lookup': metadata
})

print("\n" + "=" * 60)
print("FINAL RESULTS:")
print("=" * 60)
print(f"Weak plots identified: {result['weak_plots']}")
print(f"Clean training set size: {len(result['clean_training_ids'])}")
print(f"Baseline MAE: {result['baseline_mae']:.4f}")
print(f"Final MAE: {result['final_mae']:.4f}")
print(f"Improvement: {result['improvement']:.4f} ({result['improvement_pct']:.2f}%)")
print(f"Experiments run: {result['experiment_count']}")
print(f"\nPatterns learned:")
for key, value in result['patterns_learned'].items():
    print(f"  {key}: {value}")

print("\n" + "=" * 60)
print("✓ Weak Plot Identifier Agent test complete!")
print("=" * 60)

# Verify expected behavior
expected_weak = {5, 12}
found_weak = set(result['weak_plots'])

if found_weak == expected_weak:
    print("\n✅ SUCCESS: Agent correctly identified weak plots 5 and 12!")
elif found_weak.intersection(expected_weak):
    print(f"\n⚠️  PARTIAL: Agent found {found_weak.intersection(expected_weak)}, expected {expected_weak}")
else:
    print(f"\n❌ MISS: Agent found {found_weak}, expected {expected_weak}")
    print("   (Note: Random noise in mock experiment might cause this)")
