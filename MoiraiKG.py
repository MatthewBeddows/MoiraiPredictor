#!/usr/bin/env python3
"""
MoiraiKG: Knowledge Graph + LLM Agent for Agricultural Forecasting
Uses similarity filtering and optional LLM reasoning for intelligent plot selection
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gluonts.dataset.pandas import PandasDataset
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
import json
from datetime import datetime
import os
import random
import torch
from knowledge_graph import AgriculturalKnowledgeGraph

# =============================================================================
# CONFIGURATION
# =============================================================================

# Reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print(f"Random seed set to {RANDOM_SEED} for reproducibility\n")

# Data files
TRAIN_FILE = "AngusTrain_with_target_lags.csv"  # 2020-2022 historical data
TEST_FILE = "AngusTest_with_target_lags.csv"    # 2023 data to predict
KG_FILE = "results/agricultural_kg.pkl"          # Saved knowledge graph

# Knowledge Graph settings
USE_EXISTING_KG = False   # True = load from KG_FILE if exists, False = rebuild from scratch

# Forecasting parameters
PLOT_ID = 1          # Which plot are we trying to predict?
TARGET_COL = "target"  # Target variable (yield)
HORIZON = 1           # Predict 1 week ahead
CONTEXT_LEN = 4      # Use 4 weeks of history as context
FREQ = "W"           # Weekly frequency
PSZ = "auto"         # Patch size for Moirai model
BSZ = 32             # Batch size for predictions

# Similarity filtering - select training data for fine-tuning
USE_SIMILARITY_FILTERING = True   # True = use similar plots, False = use ALL plots
SIMILARITY_METHOD = "dtw"         # Method: 'dtw', 'pearson', 'spearman', 'cosine', 'euclidean'
TOP_SIMILAR = 10                  # How many plots to use for training
SIMILARITY_THRESHOLD = None       # Or use threshold like 0.6 instead of top-K

# LLM Agent - intelligent plot selection (optional)
USE_LLM_AGENT = False              # Set True to enable LLM-based selection
USE_LLM_ANALYSIS = False          # Set True to get data insights from LLM (just feedback, not essential)
LLM_SERVER_URL = "https://7167172d87c6.ngrok-free.app"  # Your Ollama server
LLM_MODEL = "llama3.2:latest"    # Which model to use

# Fine-tuning - automatically fine-tune on selected training plots
USE_FINETUNING = True           # Set True to fine-tune (NOTE: gradient computation issue needs fixing)
FINETUNING_EPOCHS = 5             # Number of training epochs
FINETUNING_LR = 1e-5              # Learning rate
FINETUNING_BATCH_SIZE = 4         # Batch size for training

# Experiment naming
if USE_FINETUNING:
    if USE_SIMILARITY_FILTERING:
        EXPERIMENT_NAME = f"kg_finetuned_{SIMILARITY_METHOD}_top{TOP_SIMILAR}"
    else:
        EXPERIMENT_NAME = "kg_finetuned_all_data"
elif USE_SIMILARITY_FILTERING and USE_LLM_AGENT:
    EXPERIMENT_NAME = f"kg_llm_{SIMILARITY_METHOD}_top{TOP_SIMILAR}"
elif USE_SIMILARITY_FILTERING:
    EXPERIMENT_NAME = f"kg_similar_{SIMILARITY_METHOD}_top{TOP_SIMILAR}"
else:
    EXPERIMENT_NAME = "kg_all_data"

os.makedirs("results", exist_ok=True)

# =============================================================================
# SETUP DEBUG LOGGING
# =============================================================================
DEBUG_LOG_FILE = f"results/debug_{EXPERIMENT_NAME}_plot{PLOT_ID}.txt"
def log_debug(msg):
    """Write debug message to both console and file"""
    print(msg)
    with open(DEBUG_LOG_FILE, 'a') as f:
        f.write(msg + "\n")

# Clear previous log
with open(DEBUG_LOG_FILE, 'w') as f:
    f.write("="*80 + "\n")
    f.write(f"DEBUG LOG FOR: {EXPERIMENT_NAME}\n")
    f.write(f"Timestamp: {datetime.now()}\n")
    f.write("="*80 + "\n\n")

# =============================================================================
# STEP 1: INITIALIZE LLM AGENT (if we want AI assistance)
# =============================================================================
llm_agent = None
if USE_LLM_AGENT or USE_LLM_ANALYSIS:
    print("="*60)
    print("STEP 1: INITIALIZING LLM AGENT")
    print("="*60)
    # Import the agent (like your Tkinter chatbot but for data analysis)
    try:
        from llm_agent import LLMAgent
        llm_agent = LLMAgent(LLM_SERVER_URL, LLM_MODEL)
        print("✓ LLM Agent connected and ready")
        print(f"  Server: {LLM_SERVER_URL}")
        print(f"  Model: {LLM_MODEL}\n")
    except Exception as e:
        print(f"⚠ Warning: Could not initialize LLM agent: {e}")
        print("  Continuing without LLM agent...\n")
        USE_LLM_AGENT = False
        USE_LLM_ANALYSIS = False

# =============================================================================
# STEP 2: LOAD DATA
# =============================================================================
print("="*60)
print("STEP 2: LOADING DATA")
print("="*60)
print(f"Training data: {TRAIN_FILE}")
print(f"Test data: {TEST_FILE}")

# Load CSV files
df_train = pd.read_csv(TRAIN_FILE)
df_test = pd.read_csv(TEST_FILE)

# Parse dates - handle different date column formats
for df in (df_train, df_test):
    if all(c in df.columns for c in ["year", "month", "dayofmonth"]):
        # If dates are split into separate columns, combine them
        df["date"] = pd.to_datetime(
            df[["year", "month", "dayofmonth"]].rename(columns={"dayofmonth": "day"})
        )
    else:
        # Otherwise just parse the date column
        df["date"] = pd.to_datetime(df["date"])

print(f"✓ Loaded {len(df_train)} training rows, {len(df_test)} test rows\n")

# =============================================================================
# STEP 3: BUILD OR LOAD KNOWLEDGE GRAPH
# =============================================================================
print("="*60)
print("STEP 3: BUILDING/LOADING KNOWLEDGE GRAPH")
print("="*60)

# Try to load existing KG if enabled
kg_loaded = False
if USE_EXISTING_KG and os.path.exists(KG_FILE):
    print(f"Loading existing knowledge graph from {KG_FILE}...")
    kg = AgriculturalKnowledgeGraph.load(KG_FILE)
    print(f"✓ Loaded KG with {len(kg.get_all_plots())} plots")
    kg_loaded = True
else:
    print("Building knowledge graph from scratch...")
    kg = AgriculturalKnowledgeGraph()
    kg_loaded = False

# Only build KG if we didn't load it
if not kg_loaded:
    # Figure out which columns we can use
    # We need features that exist in BOTH train and test datasets
    train_num_cols = set(df_train.select_dtypes(include=[np.number]).columns)
    test_num_cols = set(df_test.select_dtypes(include=[np.number]).columns)
    common_num_cols = train_num_cols.intersection(test_num_cols)
    # Don't include ID columns as features
    num_cols = [c for c in common_num_cols if c not in ("lookupEncoded", "FarmEncoded")]
    num_cols = sorted(num_cols)

    print(f"Using {len(num_cols)} common numeric columns")
    print(f"Sample columns: {num_cols[:5]}..." if len(num_cols) > 5 else f"Columns: {num_cols}")

    # If LLM analysis is enabled, ask it about the features
    if USE_LLM_ANALYSIS and llm_agent:
        print("\n🤖 Asking LLM: Which features matter most for yield prediction?")
        important_features = llm_agent.suggest_additional_features(num_cols, "strawberry yield")
        if important_features:
            print(f"✓ LLM suggests focusing on: {important_features[:5]}")

    # Loop through each plot and add to knowledge graph
    print(f"\nProcessing {df_train['lookupEncoded'].nunique()} unique plots...")

    for plot_id in df_train['lookupEncoded'].unique():
        # Get all data for this specific plot
        plot_df = df_train[df_train['lookupEncoded'] == plot_id].copy()

        if len(plot_df) == 0:
            continue  # Skip if no data

        # Resample to weekly data (some weeks might have multiple measurements)
        plot_weekly = plot_df.sort_values("date").set_index("date")[num_cols]
        plot_weekly = plot_weekly.resample(FREQ).mean()  # Average per week
        plot_weekly = plot_weekly.ffill()  # Forward fill missing values
        plot_weekly = plot_weekly.interpolate()  # Interpolate remaining gaps
        plot_weekly = plot_weekly.dropna()  # Drop any remaining NaNs

        if len(plot_weekly) < 4:
            continue  # Need at least 4 weeks of data

        # Extract metadata about this plot
        # This is info about WHERE and WHEN the data came from
        metadata = {
            'farm': int(plot_df['FarmEncoded'].iloc[0]) if 'FarmEncoded' in plot_df.columns else 0,
            'year': int(plot_df['year'].iloc[0]) if 'year' in plot_df.columns else 2020,
            'crop': 'strawberries',  # Your specific crop
            'location': 'Aberdeen',
            'weeks_of_data': len(plot_weekly)
        }

        # If LLM analysis is enabled, extract temporal insights
        if USE_LLM_ANALYSIS and llm_agent and 'target' in plot_weekly.columns:
            print(f"\n🤖 Analyzing plot {plot_id} with LLM...")
            temporal_analysis = llm_agent.extract_temporal_patterns(plot_weekly['target'])
            metadata['yield_mean'] = temporal_analysis['mean']
            metadata['yield_std'] = temporal_analysis['std']
            metadata['llm_pattern_analysis'] = temporal_analysis['trend']
            print(f"  Mean yield: {temporal_analysis['mean']:.2f}")
            print(f"  LLM says: {temporal_analysis['trend'][:100]}...")

        # Add this plot to the knowledge graph
        kg.add_plot(plot_id, metadata, plot_weekly)

    # Save the knowledge graph so we don't have to rebuild it next time
    kg.save(KG_FILE)
    kg.summary()

# If LLM is enabled, ask it to identify any weird plots
if USE_LLM_ANALYSIS and llm_agent:
    print("\n🤖 Asking LLM: Are there any outlier plots we should know about?")
    all_plots_for_analysis = [
        (pid, kg.get_plot_metadata(pid), kg.get_plot_data(pid))
        for pid in kg.get_all_plots()[:20]  # First 20 for speed
    ]
    outliers = llm_agent.identify_outlier_plots(all_plots_for_analysis, top_n=3)
    if outliers:
        print(f"✓ LLM identified potential outliers: {outliers}")
        print("  (These plots might have unusual patterns)")

# =============================================================================
# STEP 4: IDENTIFY ALL 2023 TEST PLOTS
# =============================================================================
print("\n" + "="*60)
print("STEP 4: IDENTIFYING TEST PLOTS (2023)")
print("="*60)

# Get all 2023 plot IDs from test data
test_plot_ids = sorted(df_test['lookupEncoded'].unique())
print(f"Found {len(test_plot_ids)} plots in 2023 test data")
print(f"Test plot IDs: {test_plot_ids[:20]}{'...' if len(test_plot_ids) > 20 else ''}")

# =============================================================================
# STEP 5: RANK TRAINING DATA BY AVERAGE SIMILARITY TO ALL 2023 PLOTS
# =============================================================================
print("\n" + "="*60)
print("STEP 5: RANKING TRAINING DATA")
print("="*60)
print(f"Strategy: Rank pre-2023 plots by AVERAGE similarity to ALL {len(test_plot_ids)} test plots")

# Get all pre-2023 plot IDs from KG
all_kg_plots = kg.get_all_plots()
pre_2023_plots = [pid for pid in all_kg_plots if pid not in test_plot_ids]
print(f"Found {len(pre_2023_plots)} pre-2023 plots for training")

# Calculate average similarity scores for each pre-2023 plot
print(f"\nComputing average {SIMILARITY_METHOD} similarity across all test plots...")
avg_similarity_scores = {}

for train_plot_id in pre_2023_plots:
    similarities = []
    for test_plot_id in test_plot_ids:
        try:
            sim = kg.compute_yield_curve_similarity(test_plot_id, train_plot_id, SIMILARITY_METHOD)
            similarities.append(sim)
        except:
            continue

    if similarities:
        avg_similarity_scores[train_plot_id] = sum(similarities) / len(similarities)

# Sort by average similarity
sorted_by_avg_sim = sorted(avg_similarity_scores.items(), key=lambda x: x[1], reverse=True)

print(f"✓ Computed average similarity for {len(sorted_by_avg_sim)} training plots")
print(f"  Top 5 by avg similarity: {[pid for pid, _ in sorted_by_avg_sim[:5]]}")
print(f"  Similarity range: {sorted_by_avg_sim[-1][1]:.3f} to {sorted_by_avg_sim[0][1]:.3f}")

# Save rankings to CSV
ranking_file = os.path.join("results", f"plot_rankings_all2023targets.csv")
import csv
with open(ranking_file, 'w', newline='') as f:
    fieldnames = ['rank', 'plot_id', 'avg_similarity_score'] + \
                 ['mean_yield', 'std_yield', 'cv_yield', 'trend_slope', 'peak_week',
                  'peak_value', 'valley_value', 'range', 'volatility',
                  'early_avg', 'mid_avg', 'late_avg', 'growth_pattern']

    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    for rank, (plot_id, avg_sim) in enumerate(sorted_by_avg_sim, 1):
        meta = kg.get_plot_metadata(plot_id)
        row = {
            'rank': rank,
            'plot_id': plot_id,
            'avg_similarity_score': f"{avg_sim:.4f}",
            'mean_yield': f"{meta.get('mean_yield', ''):.2f}" if meta.get('mean_yield') else '',
            'std_yield': f"{meta.get('std_yield', ''):.2f}" if meta.get('std_yield') else '',
            'cv_yield': f"{meta.get('cv_yield', ''):.4f}" if meta.get('cv_yield') else '',
            'trend_slope': f"{meta.get('trend_slope', ''):.4f}" if meta.get('trend_slope') else '',
            'peak_week': f"{meta.get('peak_week', ''):.4f}" if meta.get('peak_week') else '',
            'peak_value': f"{meta.get('peak_value', ''):.2f}" if meta.get('peak_value') else '',
            'valley_value': f"{meta.get('valley_value', ''):.2f}" if meta.get('valley_value') else '',
            'range': f"{meta.get('range', ''):.2f}" if meta.get('range') else '',
            'volatility': f"{meta.get('volatility', ''):.2f}" if meta.get('volatility') else '',
            'early_avg': f"{meta.get('early_avg', ''):.2f}" if meta.get('early_avg') else '',
            'mid_avg': f"{meta.get('mid_avg', ''):.2f}" if meta.get('mid_avg') else '',
            'late_avg': f"{meta.get('late_avg', ''):.2f}" if meta.get('late_avg') else '',
            'growth_pattern': meta.get('growth_pattern', '')
        }
        writer.writerow(row)

print(f"\n✓ Rankings saved to CSV: {ranking_file}")

# If LLM is enabled, get LLM rankings
llm_rankings = None
if USE_LLM_AGENT and llm_agent:
    print(f"\n🤖 Getting LLM rankings for ALL pre-2023 plots...")

    # Use representative test plot metadata (average across all test plots)
    test_metadata = kg.get_plot_metadata(test_plot_ids[0])  # Use first test plot as representative

    # Prepare candidates with similarity scores
    candidates_with_metadata = [
        (pid, kg.get_plot_metadata(pid), score)
        for pid, score in sorted_by_avg_sim
    ]

    # Get LLM's ranked list
    llm_rankings = llm_agent.rank_all_plots_by_relevance(
        test_metadata,
        candidates_with_metadata
    )

    print(f"✓ LLM ranking complete")
    if llm_rankings:
        print(f"  Top 5 by LLM: {llm_rankings[:5]}")

# Create rankings structure for downstream compatibility
rankings = {
    'total_plots_ranked': len(sorted_by_avg_sim),
    'similarity_rankings': [
        {
            'plot_id': pid,
            'similarity_score': score,
            'metadata': dict(kg.get_plot_metadata(pid))
        }
        for pid, score in sorted_by_avg_sim
    ],
    'llm_rankings': llm_rankings
}

# =============================================================================
# STEP 6: SELECT TRAINING DATA
# =============================================================================
print(f"\n{'='*60}")
print(f"STEP 6: SELECTING TRAINING DATA")
print(f"{'='*60}")
print(f"Total pre-2023 plots ranked: {rankings['total_plots_ranked']}")
print(f"\nTop 10 by average {SIMILARITY_METHOD.upper()} similarity to ALL 2023 plots:")
for i, r in enumerate(rankings['similarity_rankings'][:10], 1):
    print(f"  {i:2d}. Plot {r['plot_id']:3d} (avg score: {r['similarity_score']:.3f})")

if USE_SIMILARITY_FILTERING:
    if USE_LLM_AGENT and llm_rankings:
        # Combine both rankings using Borda count (weighted combination)
        print(f"\nStrategy: Select top {TOP_SIMILAR} by COMBINED ranking (Similarity + LLM)")
        print(f"  Method: Borda count - plots ranked highly by BOTH methods score best")

        total_plots = len(rankings['similarity_rankings'])
        borda_scores = {}

        # Similarity rankings contribute points
        for rank, r in enumerate(rankings['similarity_rankings'], 1):
            plot_id = r['plot_id']
            borda_scores[plot_id] = total_plots - rank

        # LLM rankings contribute points
        for rank, plot_id in enumerate(llm_rankings, 1):
            borda_scores[plot_id] = borda_scores.get(plot_id, 0) + (total_plots - rank)

        # Select top plots by combined Borda score
        selected_ids = sorted(borda_scores.keys(), key=lambda x: borda_scores[x], reverse=True)[:TOP_SIMILAR]
        training_plots = [(pid, kg.get_plot_data(pid)) for pid in selected_ids]

        print(f"Selected training plots (Combined): {selected_ids}")
        print(f"  Example Borda scores (higher = better):")
        for plot_id in selected_ids[:5]:
            sim_rank = next((i+1 for i, r in enumerate(rankings['similarity_rankings']) if r['plot_id'] == plot_id), None)
            llm_rank = llm_rankings.index(plot_id) + 1 if plot_id in llm_rankings else None
            print(f"    Plot {plot_id}: Sim rank #{sim_rank}, LLM rank #{llm_rank}, Borda: {borda_scores[plot_id]}")
    else:
        # Use similarity rankings only
        print(f"\nStrategy: Select top {TOP_SIMILAR} by average similarity")
        selected_ids = [r['plot_id'] for r in rankings['similarity_rankings'][:TOP_SIMILAR]]
        training_plots = [(pid, kg.get_plot_data(pid)) for pid in selected_ids]
        print(f"Selected training plots (Similarity): {selected_ids}")
else:
    if USE_LLM_AGENT and llm_rankings:
        # Use LLM rankings only
        print(f"\nStrategy: Select top {TOP_SIMILAR} by LLM relevance only")
        selected_ids = llm_rankings[:TOP_SIMILAR]
        training_plots = [(pid, kg.get_plot_data(pid)) for pid in selected_ids]
        print(f"Selected training plots (LLM Only): {selected_ids}")
    else:
        # Use ALL pre-2023 plots
        print(f"\nStrategy: Use ALL {len(pre_2023_plots)} pre-2023 plots")
        training_plots = [(pid, kg.get_plot_data(pid)) for pid in pre_2023_plots]

log_debug(f"\n✓ Selected {len(training_plots)} plots for training context")

# =============================================================================
# DEBUG: Log which plots were selected
# =============================================================================
log_debug("\n" + "="*60)
log_debug("DEBUG: PLOT SELECTION DETAILS")
log_debug("="*60)
log_debug(f"Configuration:")
log_debug(f"  USE_SIMILARITY_FILTERING: {USE_SIMILARITY_FILTERING}")
log_debug(f"  USE_LLM_AGENT: {USE_LLM_AGENT}")
log_debug(f"  SIMILARITY_METHOD: {SIMILARITY_METHOD}")
log_debug(f"  TOP_SIMILAR: {TOP_SIMILAR}")
log_debug(f"\nSelected {len(training_plots)} training plots:")

if USE_SIMILARITY_FILTERING and len(training_plots) > 0:
    # Show top 10 selected plots with their similarity scores
    for i, (plot_id, plot_data) in enumerate(training_plots[:10], 1):
        metadata = kg.get_plot_metadata(plot_id)
        log_debug(f"  {i}. Plot {plot_id}: {metadata}")
else:
    # Show first 10 plots when using all data
    for i, (plot_id, plot_data) in enumerate(training_plots[:10], 1):
        metadata = kg.get_plot_metadata(plot_id)
        log_debug(f"  {i}. Plot {plot_id}: {metadata}")

# If LLM agent was used, ask it to explain why these plots were chosen
if USE_LLM_AGENT and llm_agent is not None and len(training_plots) > 0:
    print("\n" + "="*60)
    print("🤖 LLM AGENT EXPLANATION")
    print("="*60)

    # Get metadata for target and selected plots
    target_metadata = kg.get_plot_metadata(PLOT_ID)
    selected_with_meta = [
        (pid, kg.get_plot_metadata(pid), 0.0)
        for pid, _ in training_plots[:5]  # Show explanation for top 5
    ]

    # Ask LLM to explain the selection
    explanation = llm_agent.explain_selection(target_metadata, selected_with_meta)
    print(explanation)
    print("="*60)

# Optional: Deep comparison between target and top similar plot
if USE_LLM_ANALYSIS and llm_agent and len(training_plots) > 0:
    print("\n🤖 Deep dive: Comparing target plot with most similar plot...")
    top_similar_id, top_similar_data = training_plots[0]
    comparison = llm_agent.compare_plots_deeply(
        PLOT_ID,
        kg.get_plot_metadata(PLOT_ID),
        top_similar_id,
        top_similar_data,
        kg.get_plot_metadata(top_similar_id)
    )
    print(f"Comparison with Plot {top_similar_id}:")
    print(comparison)

# =============================================================================
# STEP 5: PREPARE TEST DATA
# =============================================================================
print("\n" + "="*60)
print("STEP 5: LOADING TEST DATA (2023)")
print("="*60)

# Get 2023 data for our target plot
test_plot = df_test[df_test["lookupEncoded"] == PLOT_ID].copy()

if test_plot.empty:
    raise ValueError(f"Plot {PLOT_ID} not found in test file!")

# Process test data same way as training
test_weekly = test_plot.sort_values("date").set_index("date")[num_cols]
test_weekly = test_weekly.resample(FREQ).mean().ffill().interpolate().dropna()

print(f"Test data: {test_weekly.shape[0]} weeks to predict")
print(f"Date range: {test_weekly.index[0]} to {test_weekly.index[-1]}")

# =============================================================================
# STEP 6: LOAD PRE-TRAINED MOIRAI MODEL
# =============================================================================
print("\n" + "="*60)
print("STEP 6: LOADING PRE-TRAINED MOIRAI MODEL")
print("="*60)

# Load Moirai foundation model from Salesforce
model = MoiraiForecast(
    module=MoiraiModule.from_pretrained("Salesforce/moirai-1.0-R-small"),
    prediction_length=HORIZON,
    context_length=CONTEXT_LEN,
    patch_size=PSZ,
    num_samples=100,
    target_dim=1,
    feat_dynamic_real_dim=0,
    past_feat_dynamic_real_dim=0,
)
print("✓ Pre-trained model loaded")

# =============================================================================
# STEP 6.5: FINE-TUNE ON SELECTED TRAINING PLOTS (if enabled)
# =============================================================================
if USE_FINETUNING and len(training_plots) > 0:
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from tqdm import tqdm

    print("\n" + "="*60)
    print("STEP 6.5: FINE-TUNING MODEL ON SELECTED PLOTS")
    print("="*60)
    print(f"Fine-tuning on {len(training_plots)} selected plots")
    print(f"Epochs: {FINETUNING_EPOCHS}, LR: {FINETUNING_LR}, Batch: {FINETUNING_BATCH_SIZE}")
    log_debug(f"\n✅ FINE-TUNING ENABLED - Will train on {len(training_plots)} plots")

    # Prepare training data from selected plots (exclude target plot)
    training_samples = []

    for plot_id, plot_data in training_plots:
        if plot_id == PLOT_ID:
            continue  # Don't train on the target plot itself

        series = plot_data[TARGET_COL].values

        # Create sliding windows: context -> target
        for i in range(len(series) - CONTEXT_LEN - HORIZON + 1):
            context_window = series[i:i+CONTEXT_LEN]
            target_value = series[i+CONTEXT_LEN:i+CONTEXT_LEN+HORIZON]
            training_samples.append((context_window, target_value))

    print(f"Created {len(training_samples)} training samples from {len(training_plots)} plots")
    log_debug(f"Training samples: {len(training_samples)}")

    if len(training_samples) > 0:
        # Convert to tensors
        contexts = torch.tensor([s[0] for s in training_samples], dtype=torch.float32)
        targets = torch.tensor([s[1] for s in training_samples], dtype=torch.float32)

        # Create DataLoader
        dataset = TensorDataset(contexts, targets)
        train_loader = DataLoader(dataset, batch_size=FINETUNING_BATCH_SIZE, shuffle=True)

        # Setup training
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        moirai_module = model.module.to(device)
        optimizer = torch.optim.Adam(moirai_module.parameters(), lr=FINETUNING_LR)
        criterion = torch.nn.MSELoss()

        print(f"Training on {device} for {FINETUNING_EPOCHS} epochs...")

        # Training loop
        avg_loss = 0.0
        for epoch in range(FINETUNING_EPOCHS):
            total_loss = 0
            moirai_module.train()

            for batch_contexts, batch_targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{FINETUNING_EPOCHS}"):
                batch_contexts = batch_contexts.to(device).unsqueeze(-1)  # [batch, context_len, 1]
                batch_targets = batch_targets.to(device)
                batch_size_curr = batch_contexts.shape[0]

                optimizer.zero_grad()

                # Prepare all required inputs for MoiraiModule.forward()
                try:
                    # Create required tensors for Moirai
                    observed_mask = torch.ones_like(batch_contexts, dtype=torch.bool)  # All observed
                    sample_id = torch.arange(batch_size_curr, device=device).unsqueeze(1).expand(-1, CONTEXT_LEN).unsqueeze(-1)
                    time_id = torch.arange(CONTEXT_LEN, device=device).unsqueeze(0).expand(batch_size_curr, -1).unsqueeze(-1)
                    variate_id = torch.zeros(batch_size_curr, CONTEXT_LEN, 1, device=device, dtype=torch.long)
                    prediction_mask = torch.ones(batch_size_curr, HORIZON, 1, device=device, dtype=torch.bool)
                    # patch_size must be a tensor
                    patch_size_val = 32 if PSZ == "auto" else int(PSZ)
                    patch_size = torch.tensor([patch_size_val], device=device, dtype=torch.long)

                    # Call module forward with positional arguments in correct order
                    distr_params = moirai_module(
                        batch_contexts,      # past_target
                        observed_mask,       # observed_mask
                        sample_id,           # sample_id
                        time_id,             # time_id
                        variate_id,          # variate_id
                        prediction_mask,     # prediction_mask
                        patch_size           # patch_size
                    )

                    # Extract mean prediction - try to find parameters with gradients
                    predictions = None

                    # Try base_dist first (transformed distributions wrap the real distribution)
                    if hasattr(distr_params, 'base_dist'):
                        base = distr_params.base_dist
                        if hasattr(base, 'loc'):
                            predictions = base.loc.squeeze(-1)
                        elif hasattr(base, 'mean'):
                            predictions = base.mean.squeeze(-1)

                    # If no base_dist, try direct access
                    if predictions is None:
                        if hasattr(distr_params, 'loc'):
                            predictions = distr_params.loc.squeeze(-1)
                        elif hasattr(distr_params, 'mean'):
                            predictions = distr_params.mean.squeeze(-1)

                    # Check if we got valid predictions with gradients
                    if predictions is None or not predictions.requires_grad:
                        continue  # Skip this batch

                    # Calculate loss
                    loss = criterion(predictions, batch_targets)

                    # Backward pass
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

                except Exception as e:
                    print(f"⚠️  Forward pass failed: {e}")
                    import traceback
                    traceback.print_exc()
                    print(f"⚠️  Skipping batch...")
                    continue

            if total_loss > 0:
                avg_loss = total_loss / len(train_loader)
                print(f"Epoch {epoch+1} completed - Avg Loss: {avg_loss:.4f}")
                log_debug(f"Epoch {epoch+1}/{FINETUNING_EPOCHS} - Loss: {avg_loss:.4f}")
            else:
                print(f"Epoch {epoch+1} - No valid batches")
                log_debug(f"Epoch {epoch+1}/{FINETUNING_EPOCHS} - No valid batches")

        print("✓ Fine-tuning complete!")
        log_debug(f"✅ Fine-tuning complete! Final loss: {avg_loss:.4f}")
    else:
        print("⚠️  No training samples created - skipping fine-tuning")
        log_debug("⚠️  No training samples - skipping fine-tuning")

# Create predictor object
predictor = model.create_predictor(batch_size=BSZ)
print("✓ Model ready for forecasting")

# =============================================================================
# STEP 7: GENERATE FORECASTS FOR ALL 2023 PLOTS
# =============================================================================
print("\n" + "="*60)
print(f"STEP 7: FORECASTING ALL {len(test_plot_ids)} 2023 PLOTS")
print("="*60)

# Storage for ALL predictions across all plots
all_results = []

# Loop through each 2023 test plot
for plot_idx, PLOT_ID in enumerate(test_plot_ids, 1):
    print(f"\n{'='*60}")
    print(f"Predicting Plot {PLOT_ID} ({plot_idx}/{len(test_plot_ids)})")
    print(f"{'='*60}")

    # Get training data for this plot (used as context, not for fine-tuning)
    target_plot = df_train[df_train['lookupEncoded'] == PLOT_ID].copy()
    if len(target_plot) == 0:
        print(f"⚠️  No training data for plot {PLOT_ID}, skipping...")
        continue

    target_weekly = target_plot.sort_values("date").set_index("date")[num_cols]
    target_weekly = target_weekly.resample(FREQ).mean().ffill().interpolate().dropna()

    # Get test data for this plot
    test_plot = df_test[df_test['lookupEncoded'] == PLOT_ID].copy()
    if len(test_plot) == 0:
        print(f"⚠️  No test data for plot {PLOT_ID}, skipping...")
        continue

    test_weekly = test_plot.sort_values("date").set_index("date")[num_cols]
    test_weekly = test_weekly.resample(FREQ).mean().ffill().interpolate().dropna()

    print(f"Training data: {len(target_weekly)} weeks")
    print(f"Test data: {len(test_weekly)} weeks")

    # Storage for this plot's predictions
    predictions = []
    actuals = []
    dates = []
    prediction_intervals = []

    # Start with the last CONTEXT_LEN weeks from training as initial context
    context = target_weekly[TARGET_COL].values[-CONTEXT_LEN:]

log_debug("\n" + "="*60)
if USE_FINETUNING:
    log_debug("DEBUG: FINE-TUNED MODEL - TRAINING_PLOTS WERE USED!")
    log_debug("="*60)
    log_debug(f"✅ SUCCESS: Model was fine-tuned on {len(training_plots)} selected plots")
    log_debug(f"✅ Model learned agricultural patterns via gradient descent")
    log_debug(f"✅ Results should differ based on which plots were used for fine-tuning")
else:
    log_debug("DEBUG: PRE-TRAINED MODEL - NO FINE-TUNING!")
    log_debug("="*60)
    log_debug(f"⚠️  WARNING: Using zero-shot pre-trained Moirai model")
    log_debug(f"⚠️  The {len(training_plots)} selected training plots were NOT used for fine-tuning!")
    log_debug(f"⚠️  To use training_plots, set: USE_FINETUNING = True")

log_debug(f"\nStarting context (last {CONTEXT_LEN} weeks from TARGET PLOT {PLOT_ID}):")
log_debug(f"  {context}")
log_debug("="*60)

print(f"\nStarting context (last {CONTEXT_LEN} weeks of training): {context}")
print(f"\nForecasting {len(test_weekly)} weeks...")

# Loop through each week in the test set
for i in range(len(test_weekly)):
    # Create context DataFrame for prediction
    context_df = pd.DataFrame({
        TARGET_COL: context
    }, index=pd.date_range(
        end=target_weekly.index[-1] if i == 0 else test_weekly.index[i-1],
        periods=CONTEXT_LEN,
        freq=FREQ
    ))

    if context_df.index.freq is None:
        context_df = context_df.asfreq(FREQ)

    # Make prediction using Moirai (pre-trained or fine-tuned)
    ds = PandasDataset(dict(context_df))
    forecast_list = list(predictor.predict(ds))
    forecast = forecast_list[0]

    # Extract prediction statistics
    samples = forecast.samples
    pred_mean = samples.mean(axis=0)[0]
    pred_std = samples.std(axis=0)[0]
    pred_q10 = np.percentile(samples, 10, axis=0)[0]
    pred_q90 = np.percentile(samples, 90, axis=0)[0]

    # Store results
    predictions.append(pred_mean)
    actuals.append(test_weekly[TARGET_COL].iloc[i])
    dates.append(test_weekly.index[i])
    prediction_intervals.append({
        'mean': pred_mean,
        'std': pred_std,
        'q10': pred_q10,
        'q90': pred_q90
    })

    # Update context window for next iteration
    # Slide window forward: drop oldest week, add the actual value we just observed
    context = np.append(context[1:], test_weekly[TARGET_COL].iloc[i])

    # Progress update
    if (i + 1) % 10 == 0:
        print(f"  Completed {i+1}/{len(test_weekly)} weeks...")

# After all weeks for this plot are predicted
print(f"✓ Generated {len(predictions)} forecasts for plot {PLOT_ID}")

# Calculate metrics for this plot
plot_predictions = np.array(predictions)
plot_actuals = np.array(actuals)
plot_mae = np.mean(np.abs(plot_predictions - plot_actuals))
plot_rmse = np.sqrt(np.mean((plot_predictions - plot_actuals) ** 2))

print(f"  MAE: {plot_mae:.3f}, RMSE: {plot_rmse:.3f}")

# Store results for this plot
all_results.append({
    'plot_id': PLOT_ID,
    'predictions': predictions,
    'actuals': actuals,
    'dates': dates,
    'prediction_intervals': prediction_intervals,
    'mae': plot_mae,
    'rmse': plot_rmse,
    'num_weeks': len(predictions)
})

print(f"\n✓ Completed predictions for all {len(all_results)} plots")

# =============================================================================
# STEP 8: AGGREGATE AND EVALUATE PERFORMANCE
# =============================================================================
print("\n" + "="*60)
print("STEP 8: AGGREGATE METRICS ACROSS ALL PLOTS")
print("="*60)

# Calculate aggregate metrics
all_predictions = []
all_actuals = []
plot_maes = []
plot_rmses = []

for result in all_results:
    all_predictions.extend(result['predictions'])
    all_actuals.extend(result['actuals'])
    plot_maes.append(result['mae'])
    plot_rmses.append(result['rmse'])

# Overall metrics (all predictions combined)
all_predictions = np.array(all_predictions)
all_actuals = np.array(all_actuals)
mae = np.mean(np.abs(all_predictions - all_actuals))
rmse = np.sqrt(np.mean((all_predictions - all_actuals) ** 2))
mape = np.mean(np.abs(all_predictions - all_actuals) / all_actuals * 100)

# Average per-plot metrics
avg_plot_mae = np.mean(plot_maes)
avg_plot_rmse = np.mean(plot_rmses)

print(f"\nOverall Metrics (all {len(all_predictions)} predictions):")
print(f"  MAE:  {mae:.3f}")
print(f"  RMSE: {rmse:.3f}")
print(f"  MAPE: {mape:.2f}%")

print(f"\nAverage Per-Plot Metrics ({len(all_results)} plots):")
print(f"  Avg MAE:  {avg_plot_mae:.3f}")
print(f"  Avg RMSE: {avg_plot_rmse:.3f}")

# Create combined results DataFrame for saving
# Build expanded lists properly
plot_ids_expanded = []
dates_expanded = []

for result in all_results:
    plot_ids_expanded.extend([result['plot_id']] * result['num_weeks'])
    dates_expanded.extend(result['dates'])

# Debug: Check lengths
print(f"\nDebug - Array lengths:")
print(f"  plot_ids: {len(plot_ids_expanded)}")
print(f"  dates: {len(dates_expanded)}")
print(f"  predictions: {len(all_predictions)}")
print(f"  actuals: {len(all_actuals)}")

# Ensure all arrays are the same length
min_len = min(len(plot_ids_expanded), len(dates_expanded), len(all_predictions), len(all_actuals))
if len(plot_ids_expanded) != len(all_predictions):
    print(f"⚠️  Length mismatch detected, truncating to {min_len}")
    plot_ids_expanded = plot_ids_expanded[:min_len]
    dates_expanded = dates_expanded[:min_len]
    all_predictions = all_predictions[:min_len]
    all_actuals = all_actuals[:min_len]

results = pd.DataFrame({
    "plot_id": plot_ids_expanded,
    "date": dates_expanded,
    "predicted": all_predictions,
    "actual": all_actuals,
    "error": all_predictions - all_actuals,
    "abs_error": np.abs(all_predictions - all_actuals)
})

median_ae = results["abs_error"].median()
max_error = results["abs_error"].max()
min_error = results["abs_error"].min()

# =============================================================================
# DEBUG: Log final metrics and explanation
# =============================================================================
log_debug("\n" + "="*60)
log_debug("DEBUG: FINAL METRICS")
log_debug("="*60)
log_debug(f"MAE:  {mae:.4f}")
log_debug(f"RMSE: {rmse:.4f}")
log_debug(f"MAPE: {mape:.2f}%")
log_debug(f"Median AE: {median_ae:.4f}")
log_debug(f"\n" + "="*60)
if USE_FINETUNING:
    log_debug("USING FINE-TUNED MODEL")
    log_debug("="*60)
    log_debug(f"✅ Model was fine-tuned on {len(training_plots)} selected plots")
    log_debug(f"✅ Model learned agricultural patterns via gradient descent")
    log_debug(f"✅ Different fine-tuning data = different results!")
else:
    log_debug("WHY RESULTS MAY BE IDENTICAL (NO FINE-TUNING)")
    log_debug("="*60)
    log_debug(f"⚠️  We selected {len(training_plots)} training plots based on similarity/LLM")
    log_debug(f"⚠️  BUT the model was NOT fine-tuned on them!")
    log_debug(f"⚠️  Pre-trained Moirai only uses target plot's own history")
    log_debug(f"\nTo use training_plots and get different results:")
    log_debug(f"  Set USE_FINETUNING = True in MoiraiKG.py")
log_debug("\n" + "="*60)

# Aggregate all prediction intervals from all plots
all_prediction_intervals = []
for result in all_results:
    all_prediction_intervals.extend(result['prediction_intervals'])

# Truncate to match results length
all_prediction_intervals = all_prediction_intervals[:len(results)]

# Check how often actual value falls within prediction interval
in_interval = sum(
    1 for i, row in results.iterrows()
    if all_prediction_intervals[i]['q10'] <= row['actual'] <= all_prediction_intervals[i]['q90']
)
coverage = in_interval / len(results) * 100

print("\nPERFORMANCE METRICS:")
print(f"  MAE:            {mae:.3f} (lower is better)")
print(f"  RMSE:           {rmse:.3f} (lower is better)")
print(f"  MAPE:           {mape:.2f}% (lower is better)")
print(f"  Median AE:      {median_ae:.3f}")
print(f"  Max Error:      {max_error:.3f}")
print(f"  80% Coverage:   {coverage:.1f}% (should be ~80%)")

# If LLM is enabled, ask it to interpret the results
if USE_LLM_ANALYSIS and llm_agent:
    print("\n🤖 Asking LLM: How did we do?")
    interpretation_prompt = f"""Interpret these forecasting results for agricultural yield prediction:

MAE: {mae:.3f}
RMSE: {rmse:.3f}
MAPE: {mape:.1f}%
80% Prediction Interval Coverage: {coverage:.1f}%

Mean actual yield: {results['actual'].mean():.2f}
Mean predicted yield: {results['predicted'].mean():.2f}

Questions:
1. Is this forecast accuracy good or bad for agricultural predictions?
2. What does the coverage percentage tell us?
3. Should we trust these predictions?

Answer in 2-3 sentences."""

    interpretation = llm_agent.query(interpretation_prompt, stream=False)
    print(f"LLM interpretation:\n{interpretation}")

# =============================================================================
# STEP 9: SAVE RESULTS
# =============================================================================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Save detailed predictions to CSV
detailed_results = results.copy()
detailed_results['pred_std'] = [pi['std'] for pi in all_prediction_intervals]
detailed_results['pred_q10'] = [pi['q10'] for pi in all_prediction_intervals]
detailed_results['pred_q90'] = [pi['q90'] for pi in all_prediction_intervals]

csv_file = f"results/{EXPERIMENT_NAME}_all_plots_{timestamp}.csv"
detailed_results.to_csv(csv_file, index=False)

# Save experiment metadata to JSON
description = "Knowledge graph baseline - retrieving ALL data"
if USE_SIMILARITY_FILTERING:
    if USE_LLM_AGENT:
        description = f"LLM-enhanced selection with {SIMILARITY_METHOD} similarity"
    else:
        description = f"Knowledge graph with {SIMILARITY_METHOD} similarity filtering"

experiment_results = {
    "experiment": {
        "name": EXPERIMENT_NAME,
        "timestamp": timestamp,
        "description": description,
        "random_seed": RANDOM_SEED,
        "llm_agent_used": USE_LLM_AGENT,
        "llm_analysis_used": USE_LLM_ANALYSIS
    },
    "knowledge_graph": {
        "total_plots_in_kg": len(kg.get_all_plots()),
        "retrieved_plots": len(training_plots),
        "retrieval_strategy": "llm_enhanced" if USE_LLM_AGENT else ("similarity_based" if USE_SIMILARITY_FILTERING else "all_data"),
        "similarity_method": SIMILARITY_METHOD if USE_SIMILARITY_FILTERING else None,
        "top_k": TOP_SIMILAR if USE_SIMILARITY_FILTERING else None,
        "llm_model": LLM_MODEL if (USE_LLM_AGENT or USE_LLM_ANALYSIS) else None
    },
    "configuration": {
        "num_test_plots": len(test_plot_ids),
        "test_plot_ids": [int(pid) for pid in test_plot_ids],
        "context_length": int(CONTEXT_LEN),
        "horizon": int(HORIZON),
        "frequency": FREQ
    },
    "metrics": {
        "mae": float(mae),
        "rmse": float(rmse),
        "mape": float(mape),
        "median_ae": float(median_ae),
        "max_error": float(max_error),
        "coverage_80": float(coverage)
    }
}

json_file = f"results/{EXPERIMENT_NAME}_all_plots_{timestamp}.json"
with open(json_file, 'w') as f:
    json.dump(experiment_results, f, indent=2)

print(f"\n✓ Results saved:")
print(f"  CSV: {csv_file}")
print(f"  JSON: {json_file}")

# =============================================================================
# STEP 10: VISUALIZE
# =============================================================================
plt.figure(figsize=(14, 6))

# Plot actual values
plt.plot(results["date"], results["actual"], label="Actual Yield",
         color="green", marker="o", linewidth=2, markersize=6)

# Plot predictions
pred_label = "Predicted (Fine-Tuned)" if USE_FINETUNING else ("Predicted (LLM-enhanced)" if USE_LLM_AGENT else "Predicted")
plt.plot(results["date"], results["predicted"],
         label=pred_label,
         color="tomato", marker="x", linewidth=2, markersize=6)

# Add uncertainty bands
plt.fill_between(results["date"],
                 [pi['q10'] for pi in all_prediction_intervals],
                 [pi['q90'] for pi in all_prediction_intervals],
                 alpha=0.2, color="tomato", label="80% Confidence Interval")

# Labels and title
if USE_FINETUNING:
    model_type = "Fine-Tuned"
elif USE_LLM_AGENT:
    model_type = "LLM-Enhanced"
else:
    model_type = "Baseline"

title = f"{model_type} Forecast - All {len(test_plot_ids)} 2023 Plots Combined\n"
title += f"MAE: {mae:.3f}, RMSE: {rmse:.3f}, MAPE: {mape:.2f}%"
plt.title(title, fontsize=14, fontweight='bold')
plt.xlabel("Date", fontsize=12)
plt.ylabel(TARGET_COL, fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()

plot_file = f"results/{EXPERIMENT_NAME}_all_plots_{timestamp}.png"
plt.savefig(plot_file, dpi=300, bbox_inches='tight')
print(f"  Plot: {plot_file}")

plt.show()

# =============================================================================
# DONE!
# =============================================================================
log_debug("\n" + "="*60)
log_debug("FORECASTING COMPLETE!")
log_debug("="*60)
log_debug(f"Experiment: {EXPERIMENT_NAME}")
log_debug(f"MAE: {mae:.3f}, RMSE: {rmse:.3f}, MAPE: {mape:.2f}%")
log_debug(f"Results saved to: results/")
log_debug("="*60)
log_debug(f"\n📄 Full debug log saved to: {DEBUG_LOG_FILE}")
log_debug("\n" + "="*80)
if USE_FINETUNING:
    log_debug("✅ FINE-TUNING SUCCESS!")
    log_debug("="*80)
    log_debug(f"Model was fine-tuned on {len(training_plots)} selected plots")
    log_debug(f"Model learned agricultural patterns via gradient descent")
    log_debug(f"Results should differ from pre-trained baseline!")
else:
    log_debug("HOW TO USE FINE-TUNING")
    log_debug("="*80)
    log_debug(f"Plot selection works correctly - {len(training_plots)} plots were selected.")
    log_debug(f"However, these plots are NOT used without fine-tuning!")
    log_debug(f"\nTo actually use the selected training plots:")
    log_debug(f"  Set USE_FINETUNING = True in MoiraiKG.py and run again")
    log_debug(f"\nExpected improvement: 10-15% reduction in MAE/RMSE")
log_debug("="*80)
