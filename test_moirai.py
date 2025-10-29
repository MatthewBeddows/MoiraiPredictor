# moirai_one_plot.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

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
# 5. LOAD MODEL
# -------------------------------------------------
print("\nLoading MOIRAI-small...")

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

# -------------------------------------------------
# 6. ROLLING FORECAST FOR 2023
# -------------------------------------------------
print(f"\nGenerating rolling {HORIZON}-week forecasts with {CONTEXT_LEN}-week context...")

predictions = []
actuals = []
dates = []

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
    
    # Extract prediction
    samples = forecast.samples
    pred_mean = samples.mean(axis=0)[0]  # Get first horizon value
    
    # Store results
    predictions.append(pred_mean)
    actuals.append(test_weekly[TARGET_COL].iloc[i])
    dates.append(test_weekly.index[i])
    
    # Update context: drop oldest, add actual
    context = np.append(context[1:], test_weekly[TARGET_COL].iloc[i])
    
    if (i + 1) % 5 == 0:
        print(f"  Processed {i+1}/{len(test_weekly)} weeks...")

print(f"✓ Completed {len(predictions)} forecasts")

# -------------------------------------------------
# 7. RESULTS
# -------------------------------------------------
results = pd.DataFrame({
    "date": dates,
    "predicted": predictions,
    "actual": actuals,
    "error": np.array(predictions) - np.array(actuals)
}).round(3)

print("\n" + "="*60)
print(f"PREDICTIONS vs 2023 ACTUALS - Plot {PLOT_ID}")
print("="*60)
print(results)

# Calculate metrics
mae = np.abs(results["error"]).mean()
rmse = np.sqrt((results["error"] ** 2).mean())
mape = (np.abs(results["error"] / results["actual"]).mean() * 100)

print(f"\n{'='*60}")
print("METRICS")
print(f"{'='*60}")
print(f"MAE:  {mae:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"MAPE: {mape:.2f}%")

# -------------------------------------------------
# 8. PLOT - PREDICTIONS VS ACTUALS ONLY
# -------------------------------------------------
plt.figure(figsize=(14, 6))

plt.plot(results["date"], results["actual"], label="Actual", 
         color="green", marker="o", linewidth=2, markersize=6)
plt.plot(results["date"], results["predicted"], label="Predicted", 
         color="tomato", marker="x", linewidth=2, markersize=6)
plt.title(f"MOIRAI 1-Week Rolling Forecast - Plot {PLOT_ID} (2023)", 
          fontsize=14, fontweight='bold')
plt.xlabel("Date", fontsize=12)
plt.ylabel(TARGET_COL, fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"moirai_plot_{PLOT_ID}_rolling.png", dpi=300, bbox_inches='tight')
plt.show()

print(f"\n✓ Plot saved as moirai_plot_{PLOT_ID}_rolling.png")

# -------------------------------------------------
# 9. SAVE
# -------------------------------------------------
out_file = f"moirai_plot_{PLOT_ID}_rolling.csv"
results.to_csv(out_file, index=False)
print(f"✓ Results saved → {out_file}")