#!/usr/bin/env python3
"""
Test script for Data Explorer Agent
"""

import pandas as pd
import numpy as np
from agents import DataExplorerAgent
from knowledge_graph import AgriculturalKnowledgeGraph

print("Data Explorer Agent - Standalone Test")
print("="*60)

# Create mock agricultural dataset
np.random.seed(42)

# Simulate 20 plots with weekly yield data
n_plots = 20
n_weeks = 20

data_rows = []

for plot_id in range(n_plots):
    # Each plot has different characteristics
    farm = plot_id % 3  # 3 farms
    year = 2020 + (plot_id % 3)  # 3 years

    # Simulate some clusters
    if plot_id < 7:
        # High-yielding plots (cluster 1)
        base_yield = 80 + np.random.randn() * 5
        trend = 2.0
    elif plot_id < 14:
        # Medium-yielding plots (cluster 2)
        base_yield = 50 + np.random.randn() * 5
        trend = 1.0
    else:
        # Low-yielding plots (cluster 3)
        base_yield = 30 + np.random.randn() * 5
        trend = 0.5

    for week in range(n_weeks):
        # Seasonal pattern + trend + noise
        seasonal = 10 * np.sin(2 * np.pi * week / n_weeks)
        target = base_yield + trend * week + seasonal + np.random.randn() * 3

        # Add some anomalies
        if np.random.rand() < 0.05:  # 5% chance of anomaly
            target += np.random.randn() * 20

        # Simulate correlated features
        temp = 15 + 10 * np.sin(2 * np.pi * week / n_weeks) + np.random.randn() * 2
        precip = max(0, 50 - temp + np.random.randn() * 10)  # Inversely correlated with temp
        soil_moisture = 60 + 0.5 * precip + np.random.randn() * 5

        data_rows.append({
            'lookupEncoded': plot_id,
            'FarmEncoded': farm,
            'year': year,
            'week': week,
            'date': pd.Timestamp(f'{year}-01-01') + pd.Timedelta(weeks=week),
            'target': max(0, target),  # Yield can't be negative
            'temperature': temp,
            'precipitation': max(0, precip),
            'soil_moisture': soil_moisture,
            'plant_age': week + 10,
        })

# Create DataFrame
df_train = pd.DataFrame(data_rows)

print(f"\n✓ Created mock dataset: {len(df_train)} rows, {df_train['lookupEncoded'].nunique()} plots")

# Create knowledge graph
print("\nInitializing Knowledge Graph...")
kg = AgriculturalKnowledgeGraph()

# Create agent
print("\nCreating Data Explorer Agent...")
agent = DataExplorerAgent(
    knowledge_graph=kg,
    llm_agent=None,  # No LLM for this test
    verbose=True,
    exploration_depth='thorough'  # Full exploration
)

print("\n" + "="*60)
print("Starting autonomous data exploration...")
print("="*60)

# Run agent
result = agent.solve({
    'df_train': df_train,
    'target_col': 'target',
    'plot_id_col': 'lookupEncoded',
    'feature_cols': ['temperature', 'precipitation', 'soil_moisture', 'plant_age', 'target'],
    'metadata_cols': ['FarmEncoded', 'year']
})

print("\n" + "="*60)
print("FINAL RESULTS:")
print("="*60)

print(f"\n✓ Exploration complete: {result['exploration_complete']}")
print(f"✓ Plots explored: {len(result['plots_explored'])}")
print(f"✓ KG enriched: {result['kg_enriched']}")

print("\n📊 Exploration Statistics:")
for key, value in result['exploration_stats'].items():
    print(f"  • {key}: {value}")

print("\n🎯 Patterns Found:")
for pattern_type, pattern_data in result['patterns_found'].items():
    print(f"  • {pattern_type}: {pattern_data}")

print("\n💡 Recommendations:")
for i, rec in enumerate(result['recommendations'], 1):
    print(f"  {i}. {rec}")

print("\n🔍 Knowledge Graph Summary:")
print(f"  • Total nodes: {len(kg.get_all_plots())}")
print(f"  • Total edges: {kg.G.number_of_edges()}")

# Verify KG has rich metadata
sample_plot = result['plots_explored'][0]
plot_metadata = kg.get_plot_metadata(sample_plot)
print(f"\n📋 Sample Plot Metadata (plot {sample_plot}):")
print(f"  • Total attributes: {len(plot_metadata)}")
print(f"  • Sample attributes: {list(plot_metadata.keys())[:10]}")

print("\n" + "="*60)
print("✓ Data Explorer Agent test complete!")
print("="*60)

# Verify exploration depth
stats = result['exploration_stats']
success = True

if stats['plots_processed'] < 18:  # Should process most plots
    print("\n⚠️  WARNING: Not enough plots processed")
    success = False

if stats['correlations_found'] < 3:  # Should find temp-precip correlation
    print("\n⚠️  WARNING: Not enough correlations found")
    success = False

if stats['clusters_identified'] < 2:  # Should identify clusters
    print("\n⚠️  WARNING: Not enough clusters identified")
    success = False

if success:
    print("\n✅ SUCCESS: Agent thoroughly explored data and enriched KG!")
else:
    print("\n⚠️  PARTIAL: Agent exploration could be more thorough")
