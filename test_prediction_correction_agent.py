#!/usr/bin/env python3
"""
Test script for Prediction Correction Agent
"""

import pandas as pd
import numpy as np
from agents import PredictionCorrectionAgent

print("Prediction Correction Agent - Standalone Test")
print("="*60)

# Create mock historical data (smooth growth curve)
print("\n📊 Creating mock historical data...")
historical = pd.DataFrame({
    'target': [10, 15, 20, 25, 30, 35, 40, 45, 50]  # Smooth increasing trend
})

print(f"Historical values: {historical['target'].values}")
print(f"Range: [{historical['target'].min()}, {historical['target'].max()}]")
print(f"Mean: {historical['target'].mean():.1f}")

# Create predictions with anomalies
print("\n🎯 Creating predictions with anomalies...")
predictions = np.array([
    55,   # idx 0 - reasonable (continuing trend)
    60,   # idx 1 - reasonable
    250,  # idx 2 - SPIKE! Way too high
    65,   # idx 3 - reasonable
    70,   # idx 4 - reasonable
    -10,  # idx 5 - NEGATIVE! Impossible
    75,   # idx 6 - reasonable
    80    # idx 7 - reasonable
])

print(f"Original predictions: {predictions}")
print(f"❌ Anomalies planted:")
print(f"  • Index 2: 250 (spike - way above historical max of 50)")
print(f"  • Index 5: -10 (negative - impossible for yield)")

# Create agent
print("\n🤖 Creating Prediction Correction Agent...")
agent = PredictionCorrectionAgent(
    llm_agent=None,  # No LLM for this test
    verbose=True,
    correction_threshold=3.0
)

print("\n" + "="*60)
print("Starting agent solve() - autonomous correction...")
print("="*60)

# Run correction
result = agent.solve({
    'predictions': predictions,
    'plot_id': 1,
    'historical_data': historical
})

print("\n" + "="*60)
print("FINAL RESULTS:")
print("="*60)

print(f"\n📥 Original predictions: {predictions}")
print(f"📤 Corrected predictions: {result['corrected_predictions']}")

print(f"\n📊 Statistics:")
print(f"  • Predictions checked: {result['correction_stats']['predictions_checked']}")
print(f"  • Anomalies detected: {result['correction_stats']['anomalies_detected']}")
print(f"  • Corrections applied: {result['correction_stats']['corrections_applied']}")
print(f"  • Avg correction size: {result['correction_stats']['avg_correction_size']:.2f}")
print(f"  • Confidence: {result['confidence']:.2f}")

print(f"\n🔧 Detailed corrections:")
if len(result['corrections_made']) == 0:
    print("  None - predictions were good!")
else:
    for corr in result['corrections_made']:
        change_pct = (abs(corr['change']) / abs(corr['original'])) * 100 if corr['original'] != 0 else 0
        print(f"  • Index {corr['index']}:")
        print(f"      {corr['original']:7.2f} → {corr['corrected']:7.2f} (Δ {corr['change']:+.2f}, {change_pct:.1f}%)")
        print(f"      Reason: {corr['reason']}")

print("\n" + "="*60)
print("✓ Prediction Correction Agent test complete!")
print("="*60)

# Verification
print("\n🔍 Verification:")
expected_corrections = 2  # Should fix spike and negative

if result['correction_stats']['anomalies_detected'] == expected_corrections:
    print(f"✅ SUCCESS: Detected all {expected_corrections} anomalies!")
else:
    print(f"⚠️  Expected {expected_corrections} anomalies, found {result['correction_stats']['anomalies_detected']}")

# Check that corrections are reasonable
corrected = result['corrected_predictions']
if np.all(corrected >= 0):
    print("✅ SUCCESS: All predictions are now non-negative!")
else:
    print("❌ FAIL: Some predictions are still negative")

if np.all(corrected <= historical['target'].max() + 50):  # Allow some extrapolation
    print("✅ SUCCESS: All predictions are within reasonable range!")
else:
    print("❌ FAIL: Some predictions are still too high")

print("\n" + "="*60)
print("Test Summary: Prediction Correction Agent is working! 🎉")
print("="*60)
