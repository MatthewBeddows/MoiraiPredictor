#!/usr/bin/env python3
"""
Fine-Tune Moirai Model on Agricultural Plots

This script wraps the uni2ts cli.finetune command to fine-tune Moirai
on selected agricultural training plots.

Usage:
    # Prepare data first
    python prepare_finetuning_data.py --plot_id 1 --method dtw --top_k 10

    # Then fine-tune
    python finetune_moirai.py --config agricultural_plots_dtw_top10

Alternative: Direct fine-tuning (will prepare data automatically)
    python finetune_moirai.py --plot_id 1 --method dtw --top_k 10 --auto_prepare
"""

import argparse
import os
import subprocess
import json
from pathlib import Path
from datetime import datetime

def run_uni2ts_data_builder(train_csv_path: str, dataset_name: str):
    """
    Run uni2ts data builder to convert CSV to uni2ts format.

    Args:
        train_csv_path: Path to training CSV file
        dataset_name: Name for the dataset
    """
    print("\n" + "="*60)
    print("STEP 1: CONVERTING DATA TO UNI2TS FORMAT")
    print("="*60)

    cmd = [
        "python", "-m", "uni2ts.data.builder.simple",
        dataset_name,
        train_csv_path,
        "--dataset_type", "wide"
    ]

    print(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        print("✓ Data conversion complete!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during data conversion:")
        print(e.stderr)
        return False
    except FileNotFoundError:
        print("❌ Error: uni2ts not found. Is it installed?")
        print("Install with: pip install uni2ts")
        return False


def run_finetuning(
    dataset_name: str,
    model_name: str = "moirai_1.0_R_small",
    context_length: int = 4,
    prediction_length: int = 1,
    epochs: int = 5,
    learning_rate: float = 1e-5,
    batch_size: int = 32,
    run_name: str = None
):
    """
    Run Moirai fine-tuning using uni2ts CLI.

    Args:
        dataset_name: Name of the dataset (from data builder)
        model_name: Moirai model variant
        context_length: Context window size
        prediction_length: Forecast horizon
        epochs: Number of training epochs
        learning_rate: Learning rate
        batch_size: Batch size
        run_name: Experiment name (auto-generated if None)

    Returns:
        Path to checkpoint directory
    """
    print("\n" + "="*60)
    print("STEP 2: FINE-TUNING MOIRAI MODEL")
    print("="*60)

    if run_name is None:
        run_name = f"moirai_finetuned_{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print(f"Configuration:")
    print(f"  Model: {model_name}")
    print(f"  Dataset: {dataset_name}")
    print(f"  Context Length: {context_length}")
    print(f"  Prediction Length: {prediction_length}")
    print(f"  Epochs: {epochs}")
    print(f"  Learning Rate: {learning_rate}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Run Name: {run_name}")

    # Build command
    cmd = [
        "python", "-m", "cli.finetune",
        f"run_name={run_name}",
        f"model={model_name}",
        f"model.context_length={context_length}",
        f"model.prediction_length={prediction_length}",
        f"data={dataset_name}",
        f"val_data={dataset_name}",  # Use same dataset for validation
        f"trainer.max_epochs={epochs}",
        f"optimizer.lr={learning_rate}",
        f"data.batch_size={batch_size}",
    ]

    print(f"\nRunning: {' '.join(cmd)}")
    print("\nThis may take several minutes...")

    try:
        # Run fine-tuning (this will take a while)
        result = subprocess.run(cmd, check=True, text=True)
        print("\n✓ Fine-tuning complete!")

        # Find checkpoint directory
        checkpoint_dir = Path("outputs") / run_name / "checkpoints"

        if checkpoint_dir.exists():
            print(f"\n✓ Checkpoint saved to: {checkpoint_dir}")
            return str(checkpoint_dir / "last.ckpt")
        else:
            print(f"⚠️  Warning: Checkpoint directory not found at {checkpoint_dir}")
            return None

    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error during fine-tuning:")
        print(f"Command failed with exit code {e.returncode}")
        return None
    except FileNotFoundError:
        print("❌ Error: cli.finetune not found.")
        print("Make sure you're in the uni2ts repository directory or have it installed correctly.")
        return None


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Moirai on agricultural plots")

    # Dataset options
    parser.add_argument("--config", type=str, help="Dataset config name (e.g., agricultural_plots_dtw_top10)")
    parser.add_argument("--auto_prepare", action="store_true", help="Automatically prepare data before fine-tuning")

    # Data preparation options (if --auto_prepare)
    parser.add_argument("--plot_id", type=int, default=1, help="Target plot ID")
    parser.add_argument("--method", type=str, default="dtw", help="Similarity method")
    parser.add_argument("--top_k", type=int, default=10, help="Number of similar plots")

    # Fine-tuning options
    parser.add_argument("--model", type=str, default="moirai_1.0_R_small", help="Moirai model variant")
    parser.add_argument("--context_length", type=int, default=4, help="Context window size")
    parser.add_argument("--prediction_length", type=int, default=1, help="Forecast horizon")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--run_name", type=str, help="Experiment name")

    args = parser.parse_args()

    # Step 0: Auto-prepare data if requested
    if args.auto_prepare:
        print("="*60)
        print("AUTO-PREPARING DATA")
        print("="*60)

        from prepare_finetuning_data import prepare_uni2ts_data

        metadata = prepare_uni2ts_data(
            plot_id=args.plot_id,
            similarity_method=args.method,
            top_k=args.top_k,
            use_llm=False,
            output_dir="data/finetuning"
        )

        train_csv = metadata['train_path']
        dataset_name = f"agricultural_plots_{args.method}_top{args.top_k}"

    elif args.config:
        # Use existing prepared data
        dataset_name = args.config
        metadata_path = f"data/finetuning/metadata_{args.config.replace('agricultural_plots_', '')}.json"

        if os.path.exists(metadata_path):
            with open(metadata_path) as f:
                metadata = json.load(f)
            train_csv = metadata['train_path']
        else:
            print(f"❌ Error: Metadata file not found: {metadata_path}")
            print("Run prepare_finetuning_data.py first or use --auto_prepare")
            return

    else:
        print("❌ Error: Must specify either --config or --auto_prepare")
        print("Examples:")
        print("  python finetune_moirai.py --config agricultural_plots_dtw_top10")
        print("  python finetune_moirai.py --auto_prepare --plot_id 1 --method dtw --top_k 10")
        return

    # Step 1: Convert to uni2ts format
    success = run_uni2ts_data_builder(train_csv, dataset_name)
    if not success:
        print("\n❌ Data conversion failed. Cannot proceed with fine-tuning.")
        return

    # Step 2: Run fine-tuning
    checkpoint_path = run_finetuning(
        dataset_name=dataset_name,
        model_name=args.model,
        context_length=args.context_length,
        prediction_length=args.prediction_length,
        epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        run_name=args.run_name
    )

    # Summary
    print("\n" + "="*60)
    print("FINE-TUNING SUMMARY")
    print("="*60)

    if checkpoint_path:
        print(f"✅ SUCCESS!")
        print(f"\nCheckpoint: {checkpoint_path}")
        print(f"\nTo use this checkpoint in MoiraiKG.py:")
        print(f"  1. Set USE_FINETUNED_MODEL = True")
        print(f"  2. Set FINETUNED_CHECKPOINT = \"{checkpoint_path}\"")
        print(f"  3. Run: python MoiraiKG.py")
    else:
        print("❌ Fine-tuning failed or checkpoint not found.")
        print("Check the error messages above for details.")

    print("="*60)


if __name__ == "__main__":
    main()
