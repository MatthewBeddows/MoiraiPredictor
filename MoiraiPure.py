# moirai_one_plot.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
import json
from datetime import datetime
import os
import random
import torch

# -------------------------------------------------
# SET RANDOM SEEDS FOR REPRODUCIBILITY
# -------------------------------------------------
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print(f"Random seed set to {RANDOM_SEED} for reproducibility\n")

# -------------------------------------------------
# 1. CONFIG
# -------------------------------------------------
TRAIN_FILE   = "AngusTrain.csv"
TEST_FILE    = "AngusTest.csv"
PLOT_ID      = 1                # change to any plot with 2023 data
TARGET_COL   = "target"
HORIZON      = 1                # 1 week ahead forecast
CONTEXT_LEN  = 4                # 4 week historical window
FREQ         = "W"              # weekly
PSZ          = "auto"           # patch size
BSZ          = 32               # batch size
EXPERIMENT_NAME = "baseline_moirai_no_kg"  # For tracking

# Create results directory if it doesn't exist
os.makedirs("results", exist_ok=True)

# -------------------------------------------------
# 2. LOAD & DATE
# -------------------------------------------------
print("Loading CSVs...")
df_train = pd.read_csv(TRAIN_FILE)
df_test  = pd.read_csv(TEST_FILE)

for df in (df_train, df_test):
    if all(c in df.columns for c in ["year", "month", "dayofmonth"]):
        df["date"] = pd.to_datetime(
            df[["year", "month", "dayofmonth"]].rename(columns={"dayofmonth": "day"})
        )
    else:
        df["date"] = pd.to_datetime(df["date"])

# -------------------------------------------------
# 3. ONE PLOT - SEPARATE TRAIN AND TEST
# -------------------------------------------------
train_plot = df_train[df_train["lookupEncoded"] == PLOT_ID].copy()
test_plot  = df_test[df_test["lookupEncoded"]  == PLOT_ID].copy()

if test_plot.empty:
    raise ValueError(f"Plot {PLOT_ID} not in test file!")

print(f"Plot {PLOT_ID}: {len(train_plot)} train, {len(test_plot)} test")

# -------------------------------------------------
# 4. WEEKLY SERIES - TRAIN ONLY (PRE-2023)
# -------------------------------------------------
num_cols = train_plot.select_dtypes(include=[np.number]).columns
num_cols = [c for c in num_cols if c not in ("lookupEncoded", "FarmEncoded")]

# Create weekly series from TRAIN data only
train_weekly = train_plot.sort_values("date").set_index("date")[num_cols]
train_weekly = train_weekly.resample(FREQ).mean().ffill().interpolate().dropna()

print(f"Train weekly series: {train_weekly.shape[0]} weeks")
print(f"Train date range: {train_weekly.index[0]} to {train_weekly.index[-1]}")

# Create weekly series from TEST data (2023)
test_weekly = test_plot.sort_values("date").set_index("date")[num_cols]
test_weekly = test_weekly.resample(FREQ).mean().ffill().interpolate().dropna()

print(f"Test weekly series: {test_weekly.shape[0]} weeks")
print(f"Test date range: {test_weekly.index[0]} to {test_weekly.index[-1]}")

# -------------------------------------------------
# 5. LOAD MODEL (NO FINE-TUNING - USING PRE-TRAINED)
# -------------------------------------------------
print("\nLoading pre-trained MOIRAI-small (no fine-tuning)...")
start_time = datetime.now()

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

predictor = model.create_predictor(batch_size=BSZ)
model_load_time = (datetime.now() - start_time).total_seconds()

# -------------------------------------------------
# 6. ROLLING FORECAST FOR 2023
# -------------------------------------------------
print(f"\nGenerating rolling {HORIZON}-week forecasts with {CONTEXT_LEN}-week context...")
print("MODE: Week-by-week deployment simulation (Scenario A)")
print("Each forecast incorporates actual values from previous weeks")
print("NOTE: Using PRE-TRAINED model, NOT fine-tuned\n")
forecast_start_time = datetime.now()

predictions = []
actuals = []
dates = []
prediction_intervals = []

# Get the last CONTEXT_LEN weeks from training data as initial context
context = train_weekly[TARGET_COL].values[-CONTEXT_LEN:]

for i in range(len(test_weekly)):
    # Create a temporary dataset with current context
    context_df = pd.DataFrame({
        TARGET_COL: context
    }, index=pd.date_range(
        end=train_weekly.index[-1] if i == 0 else test_weekly.index[i-1],
        periods=CONTEXT_LEN,
        freq=FREQ
    ))
    
    if context_df.index.freq is None:
        context_df = context_df.asfreq(FREQ)
    
    # Convert to GluonTS dataset
    ds = PandasDataset(dict(context_df))
    
    # Generate forecast
    forecast_list = list(predictor.predict(ds))
    forecast = forecast_list[0]
    
    # Extract prediction and confidence intervals
    samples = forecast.samples  # Shape: (num_samples, horizon)
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
    
    # SCENARIO A: Update context with ACTUAL value
    # (Realistic deployment: we'd have this week's actual before predicting next week)
    context = np.append(context[1:], test_weekly[TARGET_COL].iloc[i])
    
    if (i + 1) % 5 == 0:
        print(f"  Processed {i+1}/{len(test_weekly)} weeks...")

forecast_time = (datetime.now() - forecast_start_time).total_seconds()
print(f"✓ Completed {len(predictions)} forecasts in {forecast_time:.2f}s")

# -------------------------------------------------
# 7. CALCULATE METRICS
# -------------------------------------------------
results = pd.DataFrame({
    "date": dates,
    "predicted": predictions,
    "actual": actuals,
    "error": np.array(predictions) - np.array(actuals),
    "abs_error": np.abs(np.array(predictions) - np.array(actuals)),
    "pct_error": np.abs(np.array(predictions) - np.array(actuals)) / np.array(actuals) * 100
})

mae = results["abs_error"].mean()
rmse = np.sqrt((results["error"] ** 2).mean())
mape = results["pct_error"].mean()
median_ae = results["abs_error"].median()
max_error = results["abs_error"].max()
min_error = results["abs_error"].min()

# Calculate coverage of prediction intervals
in_interval = sum(
    1 for i, row in results.iterrows() 
    if prediction_intervals[i]['q10'] <= row['actual'] <= prediction_intervals[i]['q90']
)
coverage = in_interval / len(results) * 100

print("\n" + "="*60)
print(f"PREDICTIONS vs 2023 ACTUALS - Plot {PLOT_ID}")
print("="*60)
print(results.round(3))

print(f"\n{'='*60}")
print("METRICS")
print(f"{'='*60}")
print(f"MAE:           {mae:.3f}")
print(f"RMSE:          {rmse:.3f}")
print(f"MAPE:          {mape:.2f}%")
print(f"Median AE:     {median_ae:.3f}")
print(f"Max Error:     {max_error:.3f}")
print(f"Min Error:     {min_error:.3f}")
print(f"80% Coverage:  {coverage:.1f}%")

# -------------------------------------------------
# 8. SAVE COMPREHENSIVE RESULTS
# -------------------------------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Save detailed predictions CSV
detailed_results = results.copy()
detailed_results['pred_std'] = [pi['std'] for pi in prediction_intervals]
detailed_results['pred_q10'] = [pi['q10'] for pi in prediction_intervals]
detailed_results['pred_q90'] = [pi['q90'] for pi in prediction_intervals]

csv_file = f"results/{EXPERIMENT_NAME}_plot{PLOT_ID}_{timestamp}.csv"
detailed_results.to_csv(csv_file, index=False)
print(f"\n✓ Detailed results saved → {csv_file}")

# Save experiment metadata and metrics JSON
experiment_results = {
    "experiment": {
        "name": EXPERIMENT_NAME,
        "timestamp": timestamp,
        "description": "Baseline Moirai forecast without knowledge graph (Scenario A: week-by-week with actual values in context)",
        "forecast_mode": "rolling_with_actuals",
        "random_seed": RANDOM_SEED,
        "fine_tuned": False
    },
    "configuration": {
        "plot_id": PLOT_ID,
        "target_column": TARGET_COL,
        "horizon": HORIZON,
        "context_length": CONTEXT_LEN,
        "frequency": FREQ,
        "patch_size": PSZ,
        "batch_size": BSZ,
        "num_samples": 100,
        "model": "Salesforce/moirai-1.0-R-small"
    },
    "data": {
        "train_weeks": len(train_weekly),
        "test_weeks": len(test_weekly),
        "train_date_range": {
            "start": str(train_weekly.index[0]),
            "end": str(train_weekly.index[-1])
        },
        "test_date_range": {
            "start": str(test_weekly.index[0]),
            "end": str(test_weekly.index[-1])
        }
    },
    "metrics": {
        "mae": float(mae),
        "rmse": float(rmse),
        "mape": float(mape),
        "median_ae": float(median_ae),
        "max_error": float(max_error),
        "min_error": float(min_error),
        "prediction_interval_coverage_80": float(coverage)
    },
    "performance": {
        "model_load_time_seconds": model_load_time,
        "total_forecast_time_seconds": forecast_time,
        "avg_time_per_forecast": forecast_time / len(predictions)
    },
    "predictions_summary": {
        "num_forecasts": len(predictions),
        "mean_predicted": float(np.mean(predictions)),
        "mean_actual": float(np.mean(actuals)),
        "std_predicted": float(np.std(predictions)),
        "std_actual": float(np.std(actuals))
    }
}

json_file = f"results/{EXPERIMENT_NAME}_plot{PLOT_ID}_{timestamp}.json"
with open(json_file, 'w') as f:
    json.dump(experiment_results, f, indent=2)
print(f"✓ Experiment metadata saved → {json_file}")

# -------------------------------------------------
# 9. PLOT
# -------------------------------------------------
plt.figure(figsize=(14, 6))

plt.plot(results["date"], results["actual"], label="Actual", 
         color="green", marker="o", linewidth=2, markersize=6)
plt.plot(results["date"], results["predicted"], label="Predicted", 
         color="tomato", marker="x", linewidth=2, markersize=6)

# Add prediction intervals
plt.fill_between(
    results["date"],
    [pi['q10'] for pi in prediction_intervals],
    [pi['q90'] for pi in prediction_intervals],
    alpha=0.2, color="tomato", label="80% Prediction Interval"
)

plt.title(f"{EXPERIMENT_NAME.upper()} - Plot {PLOT_ID} (2023)\nMAE: {mae:.3f}, RMSE: {rmse:.3f}, MAPE: {mape:.2f}%", 
          fontsize=14, fontweight='bold')
plt.xlabel("Date", fontsize=12)
plt.ylabel(TARGET_COL, fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plot_file = f"results/{EXPERIMENT_NAME}_plot{PLOT_ID}_{timestamp}.png"
plt.savefig(plot_file, dpi=300, bbox_inches='tight')
plt.show()

print(f"✓ Plot saved → {plot_file}")

# -------------------------------------------------
# 10. CREATE COMPARISON SUMMARY
# -------------------------------------------------
summary_file = "results/experiment_summary.csv"

summary_row = {
    "timestamp": timestamp,
    "experiment": EXPERIMENT_NAME,
    "plot_id": PLOT_ID,
    "mae": mae,
    "rmse": rmse,
    "mape": mape,
    "median_ae": median_ae,
    "coverage_80": coverage,
    "forecast_time_s": forecast_time,
    "model": "moirai-1.0-R-small",
    "context_length": CONTEXT_LEN,
    "horizon": HORIZON,
    "mode": "rolling_with_actuals",
    "random_seed": RANDOM_SEED,
    "fine_tuned": False
}

try:
    summary_df = pd.read_csv(summary_file)
    summary_df = pd.concat([summary_df, pd.DataFrame([summary_row])], ignore_index=True)
except FileNotFoundError:
    summary_df = pd.DataFrame([summary_row])

summary_df.to_csv(summary_file, index=False)
print(f"✓ Summary updated → {summary_file}")

print("\n" + "="*60)
print("BASELINE EXPERIMENT COMPLETE (Scenario A)")
print("="*60)
print(f"Model: Pre-trained Moirai (NOT fine-tuned)")
print(f"Random seed: {RANDOM_SEED} (results should be reproducible)")
print(f"This simulates realistic week-by-week deployment")
print(f"\nWhen you implement the knowledge graph approach, use:")
print(f"  EXPERIMENT_NAME = 'kg_retrieval_moirai'")
print(f"Then compare results in: {summary_file}")