"""
Fine-tuning utilities and data augmentation helpers for license plate OCR models.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def get_default_albumentations_pipeline():
    """
    Return a comprehensive Albumentations pipeline tailored for license plate OCR robustness.
    Includes motion blur, lighting variation, noise, and perspective distortion.
    """
    try:
        import albumentations as A  # type: ignore[import-not-found,import-untyped] # noqa: PLC0415
    except ImportError as err:
        msg = (
            "albumentations is required for data augmentation. "
            "Install it with: pip install albumentations"
        )
        raise ImportError(msg) from err

    return A.Compose(
        [
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.MotionBlur(blur_limit=5, p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.Perspective(scale=(0.04, 0.08), p=0.4),
            A.ImageCompression(quality_lower=50, quality_upper=95, p=0.3),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=20,
                p=0.3,
            ),
        ]
    )


def generate_training_command(
    train_annotations: str | Path,
    val_annotations: str | Path,
    model_config_file: str | Path,
    augmentation_path: str | Path | None = None,
    batch_size: int = 64,
    epochs: int = 100,
    output_dir: str | Path = "./trained_ocr_model",
) -> str:
    """
    Generate the CLI command for fine-tuning with fast-plate-ocr.
    """
    cmd_parts = [
        "KERAS_BACKEND=tensorflow",
        "fast_plate_ocr",
        "train",
        f"--annotations {train_annotations}",
        f"--val-annotations {val_annotations}",
        f"--model-config-file {model_config_file}",
        f"--batch-size {batch_size}",
        f"--epochs {epochs}",
        f"--output-dir {output_dir}",
    ]
    if augmentation_path:
        cmd_parts.append(f"--augmentation-path {augmentation_path}")
    return " ".join(cmd_parts)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fine-tuning helper and instructions for anpr-ocr models."
    )
    parser.add_argument(
        "--train-csv",
        help="Path to training CSV file (columns: image_path, plate)",
    )
    parser.add_argument(
        "--val-csv",
        help="Path to validation CSV file (columns: image_path, plate)",
    )
    parser.add_argument(
        "--config",
        help="Path to fast-plate-ocr YAML config file",
        default="model_config.yaml",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)",
    )
    parser.add_argument(
        "--output-dir",
        default="./trained_ocr_model",
        help="Directory to save checkpoints",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("ANPR-OCR MODEL FINE-TUNING HELPER")
    print("=" * 60)

    if not args.train_csv or not args.val_csv:
        cmd_example = generate_training_command(
            "train.csv",
            "val.csv",
            args.config,
            batch_size=args.batch_size,
            epochs=args.epochs,
            output_dir=args.output_dir,
        )
        print("\nTo train a custom OCR model on your regional plates:")
        print("1. Prepare CSV files (train.csv and val.csv) with header: image_path,plate")
        print("2. Ensure fast-plate-ocr[train] is installed:")
        print("   pip install fast-plate-ocr[train] albumentations\n")
        print("3. Example Training Command:")
        print(f"   {cmd_example}\n")
        print("4. Export to ONNX once finished:")
        print(
            f"   fast_plate_ocr export --model-path {args.output_dir}/best_model.keras "
            f"--output-path ./custom_plate_ocr.onnx\n"
        )
        return 0

    cmd = generate_training_command(
        args.train_csv,
        args.val_csv,
        args.config,
        batch_size=args.batch_size,
        epochs=args.epochs,
        output_dir=args.output_dir,
    )
    print(f"\nGenerated Command:\n{cmd}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
