#!/usr/bin/env python3
"""
Analyze and visualize plot similarities in the knowledge graph
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from knowledge_graph import AgriculturalKnowledgeGraph

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
TRAIN_FILE = "AngusTrain_with_target_lags.csv"
KG_FILE = "results/agricultural_kg.pkl"
TARGET_PLOT_ID = 1
FREQ = "W"
OUTPUT_DIR = Path("results/similarity_analysis")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------
# LOAD OR BUILD KNOWLEDGE GRAPH
# -------------------------------------------------
print("Loading Knowledge Graph...")
if Path(KG_FILE).exists():
    kg = AgriculturalKnowledgeGraph.load(KG_FILE)
else:
    print("Building Knowledge Graph from training data...")
    kg = AgriculturalKnowledgeGraph()

    df_train = pd.read_csv(TRAIN_FILE)
    if all(c in df_train.columns for c in ["year", "month", "dayofmonth"]):
        df_train["date"] = pd.to_datetime(
            df_train[["year", "month", "dayofmonth"]].rename(columns={"dayofmonth": "day"})
        )
    else:
        df_train["date"] = pd.to_datetime(df_train["date"])

    num_cols = df_train.select_dtypes(include=[np.number]).columns
    num_cols = [c for c in num_cols if c not in ("lookupEncoded", "FarmEncoded")]

    for plot_id in df_train['lookupEncoded'].unique():
        plot_df = df_train[df_train['lookupEncoded'] == plot_id].copy()
        if len(plot_df) == 0:
            continue

        plot_weekly = plot_df.sort_values("date").set_index("date")[num_cols]
        plot_weekly = plot_weekly.resample(FREQ).mean().ffill().interpolate().dropna()

        if len(plot_weekly) < 4:
            continue

        metadata = {
            'farm': int(plot_df['FarmEncoded'].iloc[0]) if 'FarmEncoded' in plot_df.columns else 0,
            'year': int(plot_df['year'].iloc[0]) if 'year' in plot_df.columns else 2020,
            'crop': 'unknown',
            'soil': 'unknown',
            'location': 'Aberdeen'
        }

        kg.add_plot(plot_id, metadata, plot_weekly)

    kg.save(KG_FILE)

kg.summary()

# -------------------------------------------------
# COMPUTE SIMILARITIES FOR DIFFERENT METHODS
# -------------------------------------------------
print(f"\nAnalyzing similarities to target plot {TARGET_PLOT_ID}...")

methods = ['dtw', 'pearson', 'spearman', 'cosine', 'euclidean']
similarity_results = {}

for method in methods:
    print(f"\nComputing {method.upper()} similarities...")
    similar_plots = kg.find_similar_plots(TARGET_PLOT_ID, method=method)
    similarity_results[method] = similar_plots

    if len(similar_plots) > 0:
        print(f"  Top 5 most similar plots:")
        for plot_id, sim in similar_plots[:5]:
            print(f"    Plot {plot_id}: {sim:.4f}")

# -------------------------------------------------
# VISUALIZE SIMILARITY DISTRIBUTIONS
# -------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle(f'Similarity Score Distributions (Target: Plot {TARGET_PLOT_ID})',
             fontsize=16, fontweight='bold')

for idx, method in enumerate(methods):
    ax = axes[idx // 3, idx % 3]
    scores = [sim for _, sim in similarity_results[method]]

    ax.hist(scores, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(np.mean(scores), color='red', linestyle='--',
               linewidth=2, label=f'Mean: {np.mean(scores):.3f}')
    ax.axvline(np.median(scores), color='green', linestyle='--',
               linewidth=2, label=f'Median: {np.median(scores):.3f}')

    ax.set_title(f'{method.upper()} Similarity', fontsize=12, fontweight='bold')
    ax.set_xlabel('Similarity Score')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True, alpha=0.3)

# Remove extra subplot
axes[1, 2].remove()

plt.tight_layout()
plt.savefig(OUTPUT_DIR / f'similarity_distributions_plot{TARGET_PLOT_ID}.png',
            dpi=300, bbox_inches='tight')
print(f"\n✓ Saved similarity distributions plot")

# -------------------------------------------------
# COMPARE TOP K PLOTS ACROSS METHODS
# -------------------------------------------------
top_k = 10
fig, axes = plt.subplots(1, len(methods), figsize=(20, 4))
fig.suptitle(f'Top {top_k} Most Similar Plots by Method (Target: Plot {TARGET_PLOT_ID})',
             fontsize=16, fontweight='bold')

for idx, method in enumerate(methods):
    ax = axes[idx]
    top_plots = similarity_results[method][:top_k]
    plot_ids = [str(pid) for pid, _ in top_plots]
    scores = [sim for _, sim in top_plots]

    ax.barh(plot_ids, scores, color='steelblue', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Similarity Score')
    ax.set_ylabel('Plot ID')
    ax.set_title(f'{method.upper()}', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()

plt.tight_layout()
plt.savefig(OUTPUT_DIR / f'top{top_k}_similar_plots_plot{TARGET_PLOT_ID}.png',
            dpi=300, bbox_inches='tight')
print(f"✓ Saved top {top_k} similar plots comparison")

# -------------------------------------------------
# VISUALIZE YIELD CURVES OF SIMILAR PLOTS
# -------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(20, 10))
fig.suptitle(f'Yield Curves: Target Plot vs Top 5 Similar Plots by Method',
             fontsize=16, fontweight='bold')

# Get target plot data
target_data = kg.get_plot_data(TARGET_PLOT_ID)['target']

for idx, method in enumerate(methods):
    ax = axes[idx // 3, idx % 3]

    # Plot target
    ax.plot(target_data.values, label=f'Target (Plot {TARGET_PLOT_ID})',
            linewidth=3, color='red', marker='o', markersize=4)

    # Plot top 5 similar
    colors = plt.cm.viridis(np.linspace(0, 0.9, 5))
    for i, (plot_id, sim) in enumerate(similarity_results[method][:5]):
        plot_data = kg.get_plot_data(plot_id)['target']
        ax.plot(plot_data.values, label=f'Plot {plot_id} (sim={sim:.3f})',
                linewidth=2, alpha=0.7, color=colors[i])

    ax.set_title(f'{method.upper()} Similarity', fontsize=12, fontweight='bold')
    ax.set_xlabel('Week')
    ax.set_ylabel('Target Value')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

# Remove extra subplot
axes[1, 2].remove()

plt.tight_layout()
plt.savefig(OUTPUT_DIR / f'yield_curves_comparison_plot{TARGET_PLOT_ID}.png',
            dpi=300, bbox_inches='tight')
print(f"✓ Saved yield curves comparison")

# -------------------------------------------------
# CREATE SIMILARITY MATRIX HEATMAP (DTW)
# -------------------------------------------------
print("\nCreating similarity matrix (this may take a moment)...")
all_plots = kg.get_all_plots()[:20]  # Limit to first 20 plots for visualization

similarity_matrix = np.zeros((len(all_plots), len(all_plots)))

for i, plot1 in enumerate(all_plots):
    for j, plot2 in enumerate(all_plots):
        if i == j:
            similarity_matrix[i, j] = 1.0
        elif i < j:
            sim = kg.compute_yield_curve_similarity(plot1, plot2, method='dtw')
            similarity_matrix[i, j] = sim
            similarity_matrix[j, i] = sim
        else:
            continue

plt.figure(figsize=(14, 12))
sns.heatmap(similarity_matrix,
            xticklabels=all_plots,
            yticklabels=all_plots,
            annot=True,
            fmt='.2f',
            cmap='YlOrRd',
            cbar_kws={'label': 'DTW Similarity'},
            square=True)
plt.title(f'DTW Similarity Matrix (First {len(all_plots)} Plots)',
          fontsize=14, fontweight='bold')
plt.xlabel('Plot ID')
plt.ylabel('Plot ID')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'similarity_matrix_dtw.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved DTW similarity matrix")

# -------------------------------------------------
# SAVE SIMILARITY RESULTS TO CSV
# -------------------------------------------------
for method in methods:
    df_sim = pd.DataFrame(similarity_results[method],
                          columns=['plot_id', 'similarity_score'])
    df_sim['method'] = method
    df_sim['target_plot'] = TARGET_PLOT_ID

    csv_file = OUTPUT_DIR / f'similarities_{method}_plot{TARGET_PLOT_ID}.csv'
    df_sim.to_csv(csv_file, index=False)
    print(f"✓ Saved {method} similarities to CSV")

# -------------------------------------------------
# SUMMARY STATISTICS
# -------------------------------------------------
print("\n" + "="*60)
print("SIMILARITY ANALYSIS SUMMARY")
print("="*60)
print(f"Target Plot: {TARGET_PLOT_ID}")
print(f"Total Plots Analyzed: {len(kg.get_all_plots())}")
print(f"\nSummary Statistics by Method:")
print("-"*60)

summary_data = []
for method in methods:
    scores = [sim for _, sim in similarity_results[method]]
    summary_data.append({
        'method': method,
        'mean': np.mean(scores),
        'median': np.median(scores),
        'std': np.std(scores),
        'min': np.min(scores),
        'max': np.max(scores),
        'q25': np.percentile(scores, 25),
        'q75': np.percentile(scores, 75)
    })

summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))

summary_df.to_csv(OUTPUT_DIR / f'similarity_summary_plot{TARGET_PLOT_ID}.csv', index=False)
print(f"\n✓ Saved summary statistics")

print("\n" + "="*60)
print(f"Analysis complete! Results saved to: {OUTPUT_DIR}")
print("="*60)
