# 1. prepare_data_for_finetuning.py
"""
Prepare the KG-retrieved data in Uni2TS format for fine-tuning
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json
from knowledge_graph import AgriculturalKnowledgeGraph
from datasets import Dataset
from datetime import datetime

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
TRAIN_FILE = "AngusTrain_with_target_lags.csv"
PLOT_ID = 1  # Target plot to exclude from training
KG_FILE = "results/agricultural_kg.pkl"
OUTPUT_DIR = Path("data/finetuning")
FREQ = "W"

# Similarity filtering settings
USE_SIMILARITY_FILTERING = True  # Set to False to use all data
SIMILARITY_METHOD = "dtw"  # Options: 'dtw', 'pearson', 'spearman', 'cosine', 'euclidean'
TOP_K_SIMILAR = 10  # Use top K most similar plots
SIMILARITY_THRESHOLD = None  # Or use a threshold instead (e.g., 0.5)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------
# LOAD OR BUILD KG
# -------------------------------------------------
print("Loading Knowledge Graph...")
if Path(KG_FILE).exists():
    kg = AgriculturalKnowledgeGraph.load(KG_FILE)
else:
    print("Building Knowledge Graph...")
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
# RETRIEVE TRAINING DATA (EXCLUDE TARGET)
# -------------------------------------------------
print(f"\nRetrieving training data (excluding plot {PLOT_ID})...")

if USE_SIMILARITY_FILTERING:
    print(f"Using similarity-based filtering:")
    print(f"  Method: {SIMILARITY_METHOD}")
    print(f"  Top K: {TOP_K_SIMILAR}")
    print(f"  Threshold: {SIMILARITY_THRESHOLD}")

    training_plots = kg.retrieve_training_data(
        target_plot_id=PLOT_ID,
        similarity_method=SIMILARITY_METHOD,
        top_k_similar=TOP_K_SIMILAR,
        similarity_threshold=SIMILARITY_THRESHOLD
    )
else:
    print("Using ALL available training data (no similarity filtering)")
    training_plots = kg.retrieve_training_data(target_plot_id=PLOT_ID)

print(f"Retrieved {len(training_plots)} training plots")

# -------------------------------------------------
# CONVERT TO UNI2TS FORMAT
# -------------------------------------------------
print("\nConverting to Uni2TS format...")

# Uni2TS expects data in long format with specific columns
all_records = []

for plot_id, plot_data in training_plots:
    # Get metadata
    metadata = kg.get_plot_metadata(plot_id)
    
    # Each timestamp becomes a record
    for timestamp, value in plot_data['target'].items():
        if pd.notna(value):
            record = {
                'item_id': f"plot_{plot_id}",
                'timestamp': timestamp.isoformat(),
                'target': float(value),
                'freq': FREQ,
                # Add metadata as static features
                'farm': metadata.get('farm', 0),
                'year': metadata.get('year', 2020)
            }
            all_records.append(record)

print(f"Created {len(all_records)} records from {len(training_plots)} plots")

# -------------------------------------------------
# CREATE HUGGINGFACE DATASET
# -------------------------------------------------
print("\nCreating HuggingFace dataset...")

df = pd.DataFrame(all_records)

# Group by item_id to create proper time series format
dataset_dict = {
    'item_id': [],
    'start': [],
    'target': [],
    'freq': []
}

for item_id in df['item_id'].unique():
    item_data = df[df['item_id'] == item_id].sort_values('timestamp')
    
    dataset_dict['item_id'].append(item_id)
    dataset_dict['start'].append(item_data['timestamp'].iloc[0])
    dataset_dict['target'].append(item_data['target'].tolist())
    dataset_dict['freq'].append(FREQ)

dataset = Dataset.from_dict(dataset_dict)

print(f"Dataset created with {len(dataset)} time series")

# Save to disk
dataset.save_to_disk(str(OUTPUT_DIR / "train"))
print(f"✓ Saved to {OUTPUT_DIR / 'train'}")

# -------------------------------------------------
# CREATE VALIDATION SPLIT (USE TARGET PLOT'S HISTORY)
# -------------------------------------------------
print("\nCreating validation split from target plot...")

target_metadata = kg.get_plot_metadata(PLOT_ID)
target_data = kg.get_plot_data(PLOT_ID)

val_records = []
for timestamp, value in target_data['target'].items():
    if pd.notna(value):
        record = {
            'item_id': f"plot_{PLOT_ID}",
            'timestamp': timestamp.isoformat(),
            'target': float(value),
            'freq': FREQ,
            'farm': target_metadata.get('farm', 0),
            'year': target_metadata.get('year', 2023)
        }
        val_records.append(record)

val_df = pd.DataFrame(val_records)
val_dataset_dict = {
    'item_id': [f"plot_{PLOT_ID}"],
    'start': [val_df['timestamp'].iloc[0]],
    'target': [val_df['target'].tolist()],
    'freq': [FREQ]
}

val_dataset = Dataset.from_dict(val_dataset_dict)
val_dataset.save_to_disk(str(OUTPUT_DIR / "val"))
print(f"✓ Validation split saved to {OUTPUT_DIR / 'val'}")

# -------------------------------------------------
# SAVE METADATA
# -------------------------------------------------
metadata = {
    'dataset_name': 'agricultural_kg_finetuning',
    'target_plot': PLOT_ID,
    'num_training_plots': len(training_plots),
    'num_training_records': len(all_records),
    'frequency': FREQ,
    'created_at': datetime.now().isoformat(),
    'similarity_filtering': USE_SIMILARITY_FILTERING,
    'similarity_method': SIMILARITY_METHOD if USE_SIMILARITY_FILTERING else None,
    'top_k_similar': TOP_K_SIMILAR if USE_SIMILARITY_FILTERING else None,
    'similarity_threshold': SIMILARITY_THRESHOLD if USE_SIMILARITY_FILTERING else None
}

with open(OUTPUT_DIR / "metadata.json", 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"\n✓ Data preparation complete!")
print(f"Training data: {OUTPUT_DIR / 'train'}")
print(f"Validation data: {OUTPUT_DIR / 'val'}")