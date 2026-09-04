"""Command-line showcase for running anpr-ocr on still images or directories."""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict, cast

import cv2
from fast_plate_ocr.inference.hub import OcrModel
from open_image_models.detection.core.hub import PlateDetectorModel

from anpr_ocr import ALPR, ALPRResult


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IMAGE = PROJECT_ROOT / "assets" / "test2.png"
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "test2-result.png"
DEFAULT_BATCH_OUTPUT = PROJECT_ROOT / "artifacts" / "batch_results"
KNOWN_SAMPLE_PLATE = "LBO2APF"
SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


class ResultPayload(TypedDict):
    """Serializable representation of one showcase result."""

    plate: str | None
    ocr_confidence: float | None
    detector_confidence: float
    bounding_box: dict[str, int]


def _confidence(result: ALPRResult) -> float | None:
    """Return one display confidence for an OCR result."""
    if result.ocr is None or result.ocr.confidence is None:
        return None
    confidence = result.ocr.confidence
    if isinstance(confidence, list):
        return sum(confidence) / len(confidence) if confidence else None
    return confidence


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run anpr-ocr on a single image, multiple images, or a dataset directory."
    )
    parser.add_argument(
        "images",
        nargs="*",
        type=Path,
        default=[DEFAULT_IMAGE],
        help="Image file(s) or directory to analyze (default: assets/test2.png).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help=(
            "Output path for single image (e.g. artifacts/result.png) "
            "or directory for batch (default: artifacts/test2-result.png "
            "or artifacts/batch_results/)."
        ),
    )
    parser.add_argument(
        "--expected",
        help="Optional known plate to validate in single-image mode, e.g. --expected LBO2APF.",
    )
    parser.add_argument(
        "--detector",
        default="yolo-v9-s-608-license-plate-end2end",
        help="Detector model name (default: yolo-v9-s-608-license-plate-end2end).",
    )
    parser.add_argument(
        "--ocr",
        default="cct-s-v2-global-model",
        help="OCR model name (default: cct-s-v2-global-model).",
    )
    parser.add_argument(
        "--crop-margin",
        type=float,
        default=0.05,
        help="Padding margin around detected bounding boxes (default: 0.05).",
    )
    parser.add_argument(
        "--enhance-contrast",
        action="store_true",
        help="Apply CLAHE contrast enhancement before OCR.",
    )
    parser.add_argument(
        "--syntax",
        help="Optional syntax mask pattern (e.g. LLDDLLDDDD).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Also print machine-readable results as JSON.",
    )
    return parser


def _result_payload(results: list[ALPRResult]) -> list[ResultPayload]:
    return [
        {
            "plate": result.ocr.text if result.ocr else None,
            "ocr_confidence": _confidence(result),
            "detector_confidence": result.detection.confidence,
            "bounding_box": {
                "x1": result.detection.bounding_box.x1,
                "y1": result.detection.bounding_box.y1,
                "x2": result.detection.bounding_box.x2,
                "y2": result.detection.bounding_box.y2,
            },
        }
        for result in results
    ]


def run_single_demo(
    image_path: Path,
    output_path: Path,
    expected: str | None,
    as_json: bool,
    alpr: ALPR,
    detector_model: str,
    ocr_model: str | None,
) -> int:
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    print("\nANPR-OCR SHOWCASE (Single Image)")
    print("=" * 52)
    print(f"Input:    {image_path}")
    print(f"Detector: {detector_model}")
    print(f"OCR:      {ocr_model}")

    inference_started = time.perf_counter()
    drawn = alpr.draw_predictions(str(image_path))
    inference_ms = (time.perf_counter() - inference_started) * 1000

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), drawn.image):
        raise RuntimeError(f"Could not write annotated image: {output_path}")

    payload = _result_payload(drawn.results)
    print(f"Inference: {inference_ms:.0f} ms")
    print(f"Detections: {len(payload)}")
    print("-" * 52)
    if payload:
        for index, item in enumerate(payload, start=1):
            plate = item["plate"] or "(OCR did not return text)"
            ocr_confidence = item["ocr_confidence"]
            detector_confidence = item["detector_confidence"]
            ocr_display = "n/a" if ocr_confidence is None else f"{ocr_confidence * 100:.1f}%"
            print(f"Plate {index}: {plate}")
            print(f"  OCR confidence:      {ocr_display}")
            print(f"  Detector confidence: {detector_confidence * 100:.1f}%")
    else:
        print("No license plates detected.")

    exit_code = 0
    if expected is not None:
        recognized = {str(item["plate"]).upper() for item in payload if item["plate"]}
        status = "PASS" if expected.upper() in recognized else "REVIEW"
        exit_code = 0 if status == "PASS" else 1
        print("-" * 52)
        print(f"Validation: {status} (expected {expected.upper()})")
    print(f"Annotated image: {output_path}")

    if as_json:
        print(json.dumps(payload, indent=2))
    return exit_code


def run_batch_demo(
    image_paths: list[Path],
    output_dir: Path,
    as_json: bool,
    alpr: ALPR,
    detector_model: str,
    ocr_model: str | None,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "annotated_images"
    images_dir.mkdir(parents=True, exist_ok=True)

    print("\nANPR-OCR BATCH PROCESSING")
    print("=" * 70)
    print(f"Total Images: {len(image_paths)}")
    print(f"Output Directory: {output_dir}")
    print(f"Detector: {detector_model} | OCR: {ocr_model}")
    print("-" * 70)

    header = f"{'Image File':<25} | {'Detections':<10} | {'Plates Detected':<28}"
    print(header)
    print("-" * 70)

    all_batch_results: list[dict] = []
    total_plates = 0
    start_time = time.perf_counter()

    for _idx, img_path in enumerate(image_paths, start=1):
        if not img_path.is_file():
            continue
        try:
            drawn = alpr.draw_predictions(str(img_path))
            out_img_path = images_dir / f"annotated_{img_path.name}"
            cv2.imwrite(str(out_img_path), drawn.image)

            payload = _result_payload(drawn.results)
            plates_str = ", ".join(
                f"{p['plate'] or '?'} ({((p['ocr_confidence'] or 0) * 100):.0f}%)" for p in payload
            )
            total_plates += len(payload)

            display_name = img_path.name[:23] if len(img_path.name) > 23 else img_path.name
            print(f"{display_name:<25} | {len(payload):<10} | {plates_str:<28}")

            all_batch_results.append(
                {
                    "file_name": img_path.name,
                    "file_path": str(img_path),
                    "annotated_path": str(out_img_path),
                    "num_detections": len(payload),
                    "results": payload,
                }
            )
        except Exception as err:
            print(f"{img_path.name:<25} | ERROR: {err}")

    elapsed = time.perf_counter() - start_time
    avg_ms = (elapsed / len(image_paths) * 1000) if image_paths else 0.0

    print("=" * 70)
    print(f"Summary: {len(image_paths)} images processed, {total_plates} plates detected.")
    print(f"Total time: {elapsed:.2f}s ({avg_ms:.1f} ms/image)")

    # 1. Save JSON Report
    json_report_path = output_dir / "results.json"
    with open(json_report_path, "w", encoding="utf-8") as f:
        json.dump(all_batch_results, f, indent=2)

    # 2. Save CSV Report
    csv_report_path = output_dir / "results.csv"
    with open(csv_report_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["file_name", "plate_index", "plate_text", "ocr_confidence", "detector_confidence"]
        )
        for item in all_batch_results:
            if not item["results"]:
                writer.writerow([item["file_name"], 0, "", "", ""])
            for p_idx, p in enumerate(item["results"], start=1):
                writer.writerow(
                    [
                        item["file_name"],
                        p_idx,
                        p["plate"] or "",
                        f"{p['ocr_confidence']:.4f}" if p["ocr_confidence"] is not None else "",
                        f"{p['detector_confidence']:.4f}",
                    ]
                )

    print("\nSaved Reports:")
    print(f"  - Annotated images: {images_dir}")
    print(f"  - JSON Report:      {json_report_path}")
    print(f"  - CSV Report:       {csv_report_path}\n")

    if as_json:
        print(json.dumps(all_batch_results, indent=2))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    # Expand directories or lists of files
    resolved_paths: list[Path] = []
    for item in args.images:
        path = item.resolve()
        if path.is_dir():
            resolved_paths.extend(
                child
                for child in sorted(path.iterdir())
                if child.suffix.lower() in SUPPORTED_IMAGE_EXTS
            )
        elif path.is_file():
            resolved_paths.append(path)
        elif "*" in str(item) or "?" in str(item):
            # Glob support
            resolved_paths.extend(
                sorted(
                    p
                    for p in Path().glob(str(item))
                    if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTS
                )
            )

    if not resolved_paths:
        print(f"Error: No valid images found matching: {args.images}")
        return 2

    # Initialize ALPR system once for all images
    print("Loading detector and OCR models...")
    init_started = time.perf_counter()
    alpr = ALPR(
        detector_model=cast(PlateDetectorModel, args.detector),
        ocr_model=cast(OcrModel, args.ocr),
        crop_margin=args.crop_margin,
        enhance_contrast=args.enhance_contrast,
        syntax_pattern=args.syntax,
    )
    init_ms = (time.perf_counter() - init_started) * 1000
    print(f"Models loaded in {init_ms:.0f} ms")

    # Determine single vs batch mode
    is_batch = len(resolved_paths) > 1 or (len(args.images) == 1 and args.images[0].is_dir())

    if is_batch:
        output_dir = args.output.resolve() if args.output else DEFAULT_BATCH_OUTPUT
        return run_batch_demo(
            resolved_paths,
            output_dir,
            args.json,
            alpr,
            args.detector,
            args.ocr,
        )

    # Single-image mode (backwards compatible)
    single_image = resolved_paths[0]
    expected = args.expected
    if expected is None and single_image == DEFAULT_IMAGE.resolve():
        expected = KNOWN_SAMPLE_PLATE

    output_path = args.output.resolve() if args.output else DEFAULT_OUTPUT
    if output_path.is_dir() or output_path.suffix == "":
        output_path = output_path / f"result_{single_image.name}"

    return run_single_demo(
        single_image,
        output_path,
        expected,
        args.json,
        alpr,
        args.detector,
        args.ocr,
    )


if __name__ == "__main__":
    raise SystemExit(main())
