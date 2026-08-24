"""Command-line showcase for running anpr-ocr on a still image."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import TypedDict

import cv2

from anpr_ocr import ALPR, ALPRResult


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IMAGE = PROJECT_ROOT / "assets" / "test2.png"
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "test2-result.png"
KNOWN_SAMPLE_PLATE = "LBO2APF"


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
        description="Run a polished anpr-ocr demo on an image and save an annotated result."
    )
    parser.add_argument(
        "image",
        nargs="?",
        type=Path,
        default=DEFAULT_IMAGE,
        help="Image to analyze (default: assets/test2.png).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Annotated image output path (default: artifacts/test2-result.png).",
    )
    parser.add_argument(
        "--expected",
        help="Optional known plate to validate, e.g. --expected LBO2APF.",
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


def run_demo(image_path: Path, output_path: Path, expected: str | None, as_json: bool) -> int:
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    print("\nANPR-OCR SHOWCASE")
    print("=" * 52)
    print(f"Input:  {image_path}")
    print("Loading detector and OCR models...")
    init_started = time.perf_counter()
    alpr = ALPR(
        detector_model="yolo-v9-t-384-license-plate-end2end",
        ocr_model="cct-xs-v2-global-model",
    )
    init_ms = (time.perf_counter() - init_started) * 1000

    inference_started = time.perf_counter()
    drawn = alpr.draw_predictions(str(image_path))
    inference_ms = (time.perf_counter() - inference_started) * 1000

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), drawn.image):
        raise RuntimeError(f"Could not write annotated image: {output_path}")

    payload = _result_payload(drawn.results)
    print(f"Models ready in {init_ms:.0f} ms")
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
        print(json.dumps(payload))
    return exit_code


def main() -> int:
    args = _build_parser().parse_args()
    expected = args.expected
    if expected is None and args.image.resolve() == DEFAULT_IMAGE.resolve():
        expected = KNOWN_SAMPLE_PLATE
    try:
        return run_demo(args.image.resolve(), args.output.resolve(), expected, args.json)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"\nShowcase failed: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
