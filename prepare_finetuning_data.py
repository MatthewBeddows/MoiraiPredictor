#!/usr/bin/env python3
"""
Prepare Agricultural Plot Data for Moirai Fine-Tuning

This script converts selected training plots from the knowledge graph into
the uni2ts format required for fine-tuning Moirai models.

Usage:
    python prepare_finetuning_data.py --plot_id 1 --method dtw --top_k 10
"""

import argparse
import os
import pandas as pd
import numpy as np
from pathlib import Path
from knowledge_graph import KnowledgeGraph

def prepare_uni2ts_data(
    plot_id: int,
    similarity_method: str = "dtw",
    top_k: int = 10,
    use_llm: bool = False,
    output_dir: str = "data/finetuning"
):
    """
    Convert agricultural plots to uni2ts format for fine-tuning.

    Args:
        plot_id: Target plot ID
        similarity_method: Similarity metric ('dtw', 'pearson', etc.)
        top_k: Number of similar plots to use
        use_llm: Whether to use LLM for plot selection
        output_dir: Directory to save processed data

    Returns:
        dict with paths to train/val CSVs
    """
    print("="*60)
    print("PREPARING DATA FOR MOIRAI FINE-TUNING")
    print("="*60)

    # Initialize knowledge graph
    print("\n1. Loading knowledge graph...")
    kg = KnowledgeGraph("data/cleaned_weekly_data.csv")
    print(f"   ✓ Loaded {len(kg.weekly_data)} data points")

    # Retrieve similar training plots
    print(f"\n2. Retrieving similar plots to plot {plot_id}...")
    print(f"   Method: {similarity_method}")
    print(f"   Top K: {top_k}")

    # For now, simplified without LLM (can add later)
    training_plots = kg.retrieve_training_data(
        target_plot_id=plot_id,
        similarity_method=similarity_method,
        top_k_similar=top_k,
        use_llm_agent=False,
        llm_agent=None
    )

    print(f"   ✓ Retrieved {len(training_plots)} plots")

    # Convert to uni2ts wide format
    print(f"\n3. Converting to uni2ts format...")

    # Combine all training plots into a single wide DataFrame
    # Each plot becomes a separate column/series
    all_series = []

    for i, (pid, plot_data) in enumerate(training_plots):
        # Resample to weekly if needed
        if 'week' in plot_data.columns:
            plot_data['week'] = pd.to_datetime(plot_data['week'])
            plot_data = plot_data.set_index('week')

        # Extract target column
        series = plot_data['target'].copy()
        series.name = f"plot_{pid}"
        all_series.append(series)

    # Create wide DataFrame with all series
    # Format: timestamp | plot_1 | plot_2 | ... | plot_N
    wide_df = pd.DataFrame({s.name: s for s in all_series})

    # Ensure index is datetime
    if not isinstance(wide_df.index, pd.DatetimeIndex):
        wide_df.index = pd.to_datetime(wide_df.index)

    # Forward fill missing values (uni2ts expects no NaN)
    wide_df = wide_df.fillna(method='ffill').fillna(method='bfill')

    # Split into train/val (80/20)
    split_idx = int(len(wide_df) * 0.8)
    train_df = wide_df.iloc[:split_idx]
    val_df = wide_df.iloc[split_idx:]

    print(f"   ✓ Created wide format with {len(all_series)} series")
    print(f"   ✓ Train: {len(train_df)} timesteps")
    print(f"   ✓ Val: {len(val_df)} timesteps")

    # Save to CSV
    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, f"agricultural_plots_train_{similarity_method}_top{top_k}.csv")
    val_path = os.path.join(output_dir, f"agricultural_plots_val_{similarity_method}_top{top_k}.csv")

    # Save with timestamp index
    train_df.to_csv(train_path)
    val_df.to_csv(val_path)

    print(f"\n4. Saved to disk:")
    print(f"   Train: {train_path}")
    print(f"   Val: {val_path}")

    # Create metadata file
    metadata = {
        'plot_id': plot_id,
        'similarity_method': similarity_method,
        'top_k': top_k,
        'num_series': len(all_series),
        'train_timesteps': len(train_df),
        'val_timesteps': len(val_df),
        'train_path': train_path,
        'val_path': val_path,
        'plots_used': [pid for pid, _ in training_plots]
    }

    import json
    metadata_path = os.path.join(output_dir, f"metadata_{similarity_method}_top{top_k}.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"   Metadata: {metadata_path}")

    print("\n" + "="*60)
    print("✅ DATA PREPARATION COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("1. Run uni2ts data builder:")
    print(f"   python -m uni2ts.data.builder.simple \\")
    print(f"     agricultural_plots_{similarity_method}_top{top_k} \\")
    print(f"     {train_path} \\")
    print(f"     --dataset_type wide")
    print()
    print("2. Run fine-tuning:")
    print(f"   python finetune_moirai.py \\")
    print(f"     --config agricultural_plots_{similarity_method}_top{top_k}")

    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data for Moirai fine-tuning")
    parser.add_argument("--plot_id", type=int, default=1, help="Target plot ID")
    parser.add_argument("--method", type=str, default="dtw", help="Similarity method")
    parser.add_argument("--top_k", type=int, default=10, help="Number of similar plots")
    parser.add_argument("--use_llm", action="store_true", help="Use LLM for selection")
    parser.add_argument("--output_dir", type=str, default="data/finetuning", help="Output directory")

    args = parser.parse_args()

    metadata = prepare_uni2ts_data(
        plot_id=args.plot_id,
        similarity_method=args.method,
        top_k=args.top_k,
        use_llm=args.use_llm,
        output_dir=args.output_dir
    )

    print(f"\n✅ Done! Prepared data for {metadata['num_series']} series")
