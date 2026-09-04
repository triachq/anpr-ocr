"""
PlateLogger module for logging recognized license plates with peak-score deduplication,
metadata extraction (timestamps, state/region), and CSV / snapshot exports.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from anpr_ocr.utils import INDIAN_STATE_NAMES, get_state_name


def is_similar_plate(p1: str, p2: str, threshold: float = 0.75) -> bool:
    """
    Check if two plate strings represent the same physical vehicle
    (e.g. prefix truncation or slight OCR character confusion across frames).
    """
    if not p1 or not p2:
        return False
    # If one is a direct substring/prefix of the other
    if p1.startswith(p2) or p2.startswith(p1):
        return True
    # If character sequence similarity meets threshold
    return SequenceMatcher(None, p1, p2).ratio() >= threshold


@dataclass
class VehicleRecord:
    """A finalized log record for a unique vehicle at its peak recognition score."""

    plate_number: str
    confidence: float
    state_region: str
    video_time: str
    frame_index: int
    detected_at: str
    frames_observed: int = 1
    snapshot_path: str | None = None
    crop_image: np.ndarray | None = None
    first_frame: int = 0
    last_frame: int = 0


class PlateLogger:
    """
    Captures license plate observations across video frames, tracks each vehicle,
    and isolates only the single highest-accuracy (peak score) reading per car.
    """

    def __init__(
        self,
        output_csv: str | Path | None = None,
        snapshots_dir: str | Path | None = None,
        min_conf: float = 0.50,
        min_chars: int = 4,
        max_frame_gap: int = 75,
    ) -> None:
        """
        Parameters:
            output_csv: File path to save the CSV log.
            snapshots_dir: Directory path to save peak crop snapshot images.
            min_conf: Minimum peak confidence threshold to include in final log.
            min_chars: Minimum character count to accept as a valid plate.
            max_frame_gap: Maximum frame distance between observations before
                considering it a different vehicle (default: 75 frames = ~3 seconds).
        """
        self.output_csv = Path(output_csv) if output_csv else None
        self.snapshots_dir = Path(snapshots_dir) if snapshots_dir else None
        self.min_conf = min_conf
        self.min_chars = min_chars
        self.max_frame_gap = max_frame_gap

        # Active vehicle events: list of VehicleRecord
        self.events: list[VehicleRecord] = []

    def observe(
        self,
        plate_text: str,
        confidence: float,
        bounding_box: Any,
        frame_idx: int,
        frame_bgr: np.ndarray,
        fps: float = 25.0,
    ) -> None:
        """
        Record a plate observation from a video frame. If the vehicle is already
        being tracked, updates the record only if the new reading has a higher
        accuracy or more complete plate sequence.
        """
        clean_text = plate_text.strip().replace(" ", "").upper()
        if len(clean_text) < self.min_chars or confidence < 0.35:
            return

        sec = frame_idx / max(fps, 1.0)
        time_str = f"{int(sec // 60):02d}:{sec % 60:05.2f}"
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Extract plate crop
        b = bounding_box
        x1, y1 = max(0, b.x1), max(0, b.y1)
        x2, y2 = min(frame_bgr.shape[1], b.x2), min(frame_bgr.shape[0], b.y2)
        crop = frame_bgr[y1:y2, x1:x2].copy() if (x2 > x1 and y2 > y1) else None

        state = get_state_name(clean_text) or "Other / International"

        # Check if this detection matches an active vehicle event
        matched: VehicleRecord | None = None
        for ev in self.events:
            if (frame_idx - ev.last_frame) <= self.max_frame_gap:
                if is_similar_plate(ev.plate_number, clean_text):
                    matched = ev
                    break

        if matched is not None:
            matched.last_frame = frame_idx
            matched.frames_observed += 1

            # PEAK ACCURACY CRITERION:
            # Update candidate if new frame has longer plate length (e.g. 10 chars vs 8)
            # OR higher confidence score on equal plate length.
            curr_len = len(matched.plate_number)
            new_len = len(clean_text)
            if new_len > curr_len or (new_len == curr_len and confidence > matched.confidence):
                matched.plate_number = clean_text
                matched.confidence = confidence
                matched.video_time = time_str
                matched.frame_index = frame_idx
                matched.detected_at = now_str
                matched.state_region = state
                if crop is not None:
                    matched.crop_image = crop
        else:
            # New unique vehicle event
            new_ev = VehicleRecord(
                plate_number=clean_text,
                confidence=confidence,
                state_region=state,
                video_time=time_str,
                frame_index=frame_idx,
                detected_at=now_str,
                frames_observed=1,
                crop_image=crop,
                first_frame=frame_idx,
                last_frame=frame_idx,
            )
            self.events.append(new_ev)

    def finalize(self) -> list[VehicleRecord]:
        """
        Filter and finalize all logged vehicle events, sorting by time of first appearance.
        Returns only genuine vehicles meeting the confidence and consistency thresholds.
        """
        valid: list[VehicleRecord] = []
        for ev in self.events:
            # Must meet minimum confidence and not be a 1-frame spurious blur
            if ev.confidence >= self.min_conf:
                if ev.frames_observed >= 2 or ev.confidence >= 0.85:
                    valid.append(ev)

        valid.sort(key=lambda x: x.first_frame)
        return valid

    def save_snapshots(self, output_dir: str | Path | None = None) -> list[str]:
        """
        Save high-resolution plate crop snapshot images for each finalized vehicle.
        """
        target_dir = Path(output_dir) if output_dir else self.snapshots_dir
        if not target_dir:
            return []

        target_dir.mkdir(parents=True, exist_ok=True)
        saved_paths: list[str] = []

        for idx, ev in enumerate(self.finalize(), 1):
            if ev.crop_image is not None and ev.crop_image.size > 0:
                conf_pct = int(round(ev.confidence * 100))
                fname = f"vehicle_{idx:02d}_{ev.plate_number}_{conf_pct}pct.jpg"
                save_path = target_dir / fname
                cv2.imwrite(str(save_path), ev.crop_image)
                ev.snapshot_path = str(save_path)
                saved_paths.append(str(save_path))

        return saved_paths

    def export_csv(self, file_path: str | Path | None = None) -> Path | None:
        """
        Write finalized peak-score records to a CSV spreadsheet.
        """
        target = Path(file_path) if file_path else self.output_csv
        if not target:
            return None

        target.parent.mkdir(parents=True, exist_ok=True)
        finalized = self.finalize()

        # Save snapshots if directory is configured
        if self.snapshots_dir:
            self.save_snapshots()

        with open(target, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Vehicle #",
                "Plate Number",
                "Peak Confidence",
                "State / Region",
                "Video Timestamp",
                "Peak Frame",
                "Detected At (Real Time)",
                "Frames Observed",
                "Snapshot File",
            ])
            for idx, ev in enumerate(finalized, 1):
                writer.writerow([
                    idx,
                    ev.plate_number,
                    f"{ev.confidence * 100:.1f}%",
                    ev.state_region,
                    ev.video_time,
                    ev.frame_index,
                    ev.detected_at,
                    ev.frames_observed,
                    ev.snapshot_path or "N/A",
                ])

        return target

    def summary_table(self) -> str:
        """
        Generate a formatted ASCII summary table of finalized unique vehicles.
        """
        finalized = self.finalize()
        if not finalized:
            return "  No license plates recognized with sufficient confidence."

        lines = [
            "=" * 92,
            f"  FINAL VEHICLE LOG -- {len(finalized)} Unique Vehicle(s) Detected at Peak Accuracy",
            "=" * 92,
            f"{'#':<3} | {'Plate Number':<14} | {'Peak Conf':<10} | {'State / Region':<22} | {'Video Time':<10} | {'Frames':<6}",
            "-" * 92,
        ]
        for idx, ev in enumerate(finalized, 1):
            conf_str = f"{ev.confidence * 100:.1f}%"
            lines.append(
                f"{idx:<3} | {ev.plate_number:<14} | {conf_str:<10} | {ev.state_region:<22} | {ev.video_time:<10} | {ev.frames_observed:<6}"
            )
        lines.append("=" * 92)
        return "\n".join(lines)
