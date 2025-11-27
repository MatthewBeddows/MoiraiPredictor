#!/usr/bin/env python3
"""
Auto-run script for creating target rolling features

Just run: python create_target_rolling_features_auto.py

Automatically finds and processes:
- AngusTrain.csv → AngusTrain_with_target_lags.csv
- AngusTest.csv → AngusTest_with_target_lags.csv

No arguments needed!
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os


def create_target_rolling_features(df, plot_id_col='lookupEncoded', lag_weeks=[1, 2, 3, 4], rolling_windows=[2, 4, 8]):
    """
    Create lagged and rolling features for 'target' column only.
    Features are calculated per-plot (no cross-plot contamination).
    """
    
    print(f"\n{'='*70}")
    print("🎯 CREATING TARGET ROLLING FEATURES (PER-PLOT)")
    print(f"{'='*70}")
    
    # Check requirements
    if 'target' not in df.columns:
        print("⚠️  Warning: 'target' column not found in DataFrame!")
        print("   Skipping feature creation for this file.")
        return df
    
    if plot_id_col not in df.columns:
        raise ValueError(f"❌ '{plot_id_col}' column not found in DataFrame!")
    
    n_plots = df[plot_id_col].nunique()
    print(f"\n📊 Found {n_plots} unique plots in '{plot_id_col}'")
    print(f"   Will create features separately for each plot (no mixing)")
    
    # Sort by plot and date to ensure correct ordering
    if 'date' in df.columns:
        df = df.sort_values([plot_id_col, 'date'])
        print(f"✓ Sorted by {plot_id_col} and date")
    else:
        df = df.sort_values(plot_id_col)
        print(f"✓ Sorted by {plot_id_col}")
    
    # ========================================================================
    # 1. CREATE LAG FEATURES (Past values)
    # ========================================================================
    print(f"\n⏮️  Creating lag features...")
    
    for lag in lag_weeks:
        col_name = f'target_lag{lag}w'
        
        # Shift within each plot group
        df[col_name] = df.groupby(plot_id_col)['target'].shift(lag)
        
        n_valid = df[col_name].notna().sum()
        print(f"   ✓ {col_name}: {n_valid} valid values")
    
    # ========================================================================
    # 2. CREATE ROLLING AGGREGATIONS (Trends)
    # ========================================================================
    print(f"\n📊 Creating rolling aggregations...")
    
    for window in rolling_windows:
        # Rolling mean
        col_name = f'target_roll{window}w_mean'
        df[col_name] = (
            df.groupby(plot_id_col)['target']
            .transform(lambda x: x.rolling(window, min_periods=1).mean())
        )
        n_valid = df[col_name].notna().sum()
        print(f"   ✓ {col_name}: {n_valid} valid values")
        
        # Rolling std
        col_name = f'target_roll{window}w_std'
        df[col_name] = (
            df.groupby(plot_id_col)['target']
            .transform(lambda x: x.rolling(window, min_periods=1).std())
        )
        n_valid = df[col_name].notna().sum()
        print(f"   ✓ {col_name}: {n_valid} valid values")
        
        # Rolling min
        col_name = f'target_roll{window}w_min'
        df[col_name] = (
            df.groupby(plot_id_col)['target']
            .transform(lambda x: x.rolling(window, min_periods=1).min())
        )
        n_valid = df[col_name].notna().sum()
        print(f"   ✓ {col_name}: {n_valid} valid values")
        
        # Rolling max
        col_name = f'target_roll{window}w_max'
        df[col_name] = (
            df.groupby(plot_id_col)['target']
            .transform(lambda x: x.rolling(window, min_periods=1).max())
        )
        n_valid = df[col_name].notna().sum()
        print(f"   ✓ {col_name}: {n_valid} valid values")
    
    # ========================================================================
    # 3. CREATE DIFFERENCE FEATURES (Changes)
    # ========================================================================
    print(f"\n📈 Creating difference features...")
    
    # Week-over-week change
    df['target_diff1w'] = df.groupby(plot_id_col)['target'].diff(1)
    n_valid = df['target_diff1w'].notna().sum()
    print(f"   ✓ target_diff1w: {n_valid} valid values")
    
    # Percent change
    df['target_pct_change1w'] = df.groupby(plot_id_col)['target'].pct_change(1)
    n_valid = df['target_pct_change1w'].notna().sum()
    print(f"   ✓ target_pct_change1w: {n_valid} valid values")
    
    # ========================================================================
    # 4. HANDLE MISSING VALUES
    # ========================================================================
    print(f"\n🔧 Handling missing values...")
    
    # Count missing before
    new_cols = [col for col in df.columns if 'lag' in col or 'roll' in col or 'diff' in col or 'pct_change' in col]
    n_missing_before = df[new_cols].isnull().sum().sum()
    
    # Fill NaN with 0 (at start of each plot's history)
    df[new_cols] = df[new_cols].fillna(0)
    
    n_missing_after = df[new_cols].isnull().sum().sum()
    print(f"   Missing values: {n_missing_before} → {n_missing_after}")
    
    return df


def find_files():
    """
    Search for AngusTrain.csv and AngusTest.csv in current directory
    """
    current_dir = Path.cwd()
    
    train_file = None
    test_file = None
    
    # Look in current directory
    if (current_dir / "AngusTrain.csv").exists():
        train_file = current_dir / "AngusTrain.csv"
    
    if (current_dir / "AngusTest.csv").exists():
        test_file = current_dir / "AngusTest.csv"
    
    # Also check common subdirectories
    for subdir in ['data', 'Data', 'DATA', '.']:
        subdir_path = current_dir / subdir
        if subdir_path.exists():
            if (subdir_path / "AngusTrain.csv").exists():
                train_file = subdir_path / "AngusTrain.csv"
            if (subdir_path / "AngusTest.csv").exists():
                test_file = subdir_path / "AngusTest.csv"
    
    return train_file, test_file


def process_file(input_file, plot_col='lookupEncoded', lag_weeks=[1, 2, 3, 4], rolling_windows=[2, 4, 8]):
    """
    Process a single file: load, engineer features, save
    """
    
    print(f"\n{'='*70}")
    print(f"📂 Processing: {input_file.name}")
    print(f"{'='*70}")
    
    # Load data
    print(f"\n📂 Loading data...")
    df = pd.read_csv(input_file)
    
    # Create date if needed
    if 'date' not in df.columns and all(col in df.columns for col in ['year', 'month', 'dayofmonth']):
        print("   Creating date column from year/month/dayofmonth...")
        df['date'] = pd.to_datetime(
            df[['year', 'month', 'dayofmonth']].rename(columns={'dayofmonth': 'day'})
        )
    
    print(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")
    if plot_col in df.columns:
        print(f"   Unique plots: {df[plot_col].nunique()}")
    
    # Create features
    df_with_features = create_target_rolling_features(
        df,
        plot_id_col=plot_col,
        lag_weeks=lag_weeks,
        rolling_windows=rolling_windows
    )
    
    # Generate output filename
    output_file = input_file.parent / f"{input_file.stem}_with_target_lags.csv"
    
    # Summary
    n_new_cols = len(df_with_features.columns) - len(df.columns)
    
    print(f"\n{'='*70}")
    print("📊 SUMMARY")
    print(f"{'='*70}")
    print(f"Original columns: {len(df.columns)}")
    print(f"New columns:      {n_new_cols}")
    print(f"Total columns:    {len(df_with_features.columns)}")
    
    # List new columns
    new_cols = [col for col in df_with_features.columns if col not in df.columns]
    print(f"\n📋 New features created:")
    for col in new_cols:
        print(f"   • {col}")
    
    # Save
    print(f"\n💾 Saving to {output_file.name}...")
    df_with_features.to_csv(output_file, index=False)
    
    file_size = output_file.stat().st_size / (1024 * 1024)
    print(f"   ✅ Saved {len(df_with_features)} rows, {len(df_with_features.columns)} columns")
    print(f"   File size: {file_size:.2f} MB")
    
    # Show sample
    print(f"\n📋 Sample of data with new features (first 5 rows):")
    sample_cols = [plot_col, 'target', 'target_lag1w', 'target_lag2w', 'target_roll4w_mean']
    sample_cols = [col for col in sample_cols if col in df_with_features.columns]
    print(df_with_features[sample_cols].head())
    
    return output_file


def main():
    """
    Main execution - auto-find and process files
    """
    
    print("\n" + "="*70)
    print("🌾 AUTO-RUN: TARGET ROLLING FEATURES")
    print("="*70)
    print("\nSearching for AngusTrain.csv and AngusTest.csv...")
    
    # Find files
    train_file, test_file = find_files()
    
    if not train_file and not test_file:
        print("\n❌ Error: Could not find AngusTrain.csv or AngusTest.csv")
        print("\nSearched in:")
        print(f"  • {Path.cwd()}")
        print(f"  • {Path.cwd() / 'data'}")
        print(f"  • {Path.cwd() / 'Data'}")
        print("\nPlease ensure the CSV files are in one of these locations.")
        return
    
    # Show what we found
    print(f"\n✅ Found files:")
    if train_file:
        print(f"   • {train_file}")
    if test_file:
        print(f"   • {test_file}")
    
    # Configuration
    plot_col = 'lookupEncoded'
    lag_weeks = [1, 2, 3, 4]
    rolling_windows = [2, 4, 8]
    
    print(f"\n⚙️  Configuration:")
    print(f"   • Plot ID column: {plot_col}")
    print(f"   • Lag periods: {lag_weeks} weeks")
    print(f"   • Rolling windows: {rolling_windows} weeks")
    
    output_files = []
    
    # Process training file
    if train_file:
        try:
            output = process_file(train_file, plot_col, lag_weeks, rolling_windows)
            output_files.append(output)
        except Exception as e:
            print(f"\n❌ Error processing {train_file.name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Process test file
    if test_file:
        try:
            output = process_file(test_file, plot_col, lag_weeks, rolling_windows)
            output_files.append(output)
        except Exception as e:
            print(f"\n❌ Error processing {test_file.name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Final summary
    print("\n" + "="*70)
    print("✅ ALL FILES PROCESSED!")
    print("="*70)
    
    if output_files:
        print("\n📦 Output files created:")
        for f in output_files:
            print(f"   ✓ {f.name}")
        
        print("\n🚀 Next steps:")
        print("   1. Verify the output files look correct")
        print("   2. Use the engineered features in your GNN model")
        print("   3. Train on ALL plots, predict for specific plots")
        
        print("\n💡 Example usage:")
        print("   import pandas as pd")
        print(f"   df = pd.read_csv('{output_files[0].name}')")
        print("   print(df.columns)  # See all the new features")
        print("   print(df[['lookupEncoded', 'target', 'target_lag1w']].head())")
    else:
        print("\n⚠️  No files were processed successfully.")


if __name__ == "__main__":
    main()