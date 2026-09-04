"""
Image preprocessing and post-processing utility functions for ANPR.
"""

from __future__ import annotations

import cv2
import numpy as np

# Ambiguity mappings between visually similar characters
LETTER_TO_DIGIT: dict[str, str] = {
    "O": "0",
    "D": "0",
    "Q": "0",
    "I": "1",
    "L": "1",
    "Z": "2",
    "E": "3",
    "A": "4",
    "S": "5",
    "G": "6",
    "B": "8",
}

DIGIT_TO_LETTER: dict[str, str] = {
    "0": "O",
    "1": "I",
    "2": "Z",
    "3": "E",
    "4": "A",
    "5": "S",
    "6": "G",
    "8": "B",
}


def pad_bounding_box(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    img_width: int,
    img_height: int,
    margin_x: float = 0.05,
    margin_y: float = 0.05,
) -> tuple[int, int, int, int]:
    """
    Expand a bounding box with relative horizontal and vertical margins,
    preventing edge character truncation.

    Parameters:
        x1, y1, x2, y2: Initial bounding box coordinates.
        img_width: Image width.
        img_height: Image height.
        margin_x: Fractional horizontal margin (e.g. 0.05 = 5%).
        margin_y: Fractional vertical margin (e.g. 0.05 = 5%).

    Returns:
        tuple[int, int, int, int]: Padded coordinates clamped to image dimensions.
    """
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)

    pad_w = round(box_w * margin_x)
    pad_h = round(box_h * margin_y)

    padded_x1 = max(0, x1 - pad_w)
    padded_y1 = max(0, y1 - pad_h)
    padded_x2 = min(img_width, x2 + pad_w)
    padded_y2 = min(img_height, y2 + pad_h)

    return padded_x1, padded_y1, padded_x2, padded_y2


def enhance_plate_image(
    cropped_plate: np.ndarray,
    enhance_contrast: bool = True,
    min_width: int = 94,
) -> np.ndarray:
    """
    Enhance a cropped license plate image for improved OCR character readability.

    Parameters:
        cropped_plate: Cropped BGR plate image.
        enhance_contrast: Whether to apply CLAHE contrast enhancement.
        min_width: Minimum width for upscaling small crops (0 to disable).

    Returns:
        np.ndarray: Enhanced BGR plate image.
    """
    if cropped_plate is None or cropped_plate.size == 0:
        return cropped_plate

    img = cropped_plate.copy()
    h, w = img.shape[:2]

    # 1. Bicubic upscaling for small crops to prevent pixelation blur
    if min_width > 0 and w < min_width:
        scale = min_width / float(w)
        new_w = min_width
        new_h = max(1, round(h * scale))
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        # Apply mild unsharp mask to restore edge sharpness after upscaling
        blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=1.5)
        img = cv2.addWeighted(img, 1.4, blurred, -0.4, 0)

    # 2. Contrast Limited Adaptive Histogram Equalization (CLAHE) on luminance
    if enhance_contrast:
        if len(img.shape) == 3 and img.shape[2] == 3:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced_l = clahe.apply(l_channel)
            lab_enhanced = cv2.merge((enhanced_l, a_channel, b_channel))
            img = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        elif len(img.shape) == 2:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img = clahe.apply(img)

    return img


# Indian 2-letter state/UT and special codes
INDIAN_STATE_CODES: frozenset[str] = frozenset({
    "AN", "AP", "AR", "AS", "BR", "CH", "CG", "DD", "DH", "DL", "DN",
    "GA", "GJ", "HP", "HR", "JH", "JK", "KA", "KL", "LA", "LD", "MH",
    "ML", "MN", "MP", "MZ", "NL", "OD", "PB", "PY", "RJ", "SK", "TN",
    "TR", "TS", "UK", "UP", "WB", "IN", "BH",
})


INDIAN_STATE_NAMES: dict[str, str] = {
    "AN": "Andaman and Nicobar",
    "AP": "Andhra Pradesh",
    "AR": "Arunachal Pradesh",
    "AS": "Assam",
    "BR": "Bihar",
    "CH": "Chandigarh",
    "CG": "Chhattisgarh",
    "DD": "Daman and Diu",
    "DH": "Dadra and Nagar Haveli",
    "DL": "Delhi",
    "DN": "Dadra and Nagar Haveli and Daman and Diu",
    "GA": "Goa",
    "GJ": "Gujarat",
    "HP": "Himachal Pradesh",
    "HR": "Haryana",
    "JH": "Jharkhand",
    "JK": "Jammu and Kashmir",
    "KA": "Karnataka",
    "KL": "Kerala",
    "LA": "Ladakh",
    "LD": "Lakshadweep",
    "MH": "Maharashtra",
    "ML": "Meghalaya",
    "MN": "Manipur",
    "MP": "Madhya Pradesh",
    "MZ": "Mizoram",
    "NL": "Nagaland",
    "OD": "Odisha",
    "PB": "Punjab",
    "PY": "Puducherry",
    "RJ": "Rajasthan",
    "SK": "Sikkim",
    "TN": "Tamil Nadu",
    "TR": "Tripura",
    "TS": "Telangana",
    "UK": "Uttarakhand",
    "UP": "Uttar Pradesh",
    "WB": "West Bengal",
    "IN": "India (Government/Defense)",
    "BH": "Bharat Series",
}


def get_state_name(plate_text: str) -> str:
    """Extract and resolve the state or UT name from an Indian license plate prefix."""
    if not plate_text or len(plate_text) < 2:
        return ""
    prefix = plate_text[:2].upper()
    return INDIAN_STATE_NAMES.get(prefix, "")
STATE_PREFIX_CORRECTIONS: dict[str, str] = {
    "0L": "DL", "OL": "DL", "QL": "DL", "D1": "DL",
    "8R": "BR",
    "7N": "TN",
    "7S": "TS", "1S": "TS",
    "1H": "JH", "IH": "JH",
    "1N": "IN",
    "0D": "OD",
    # Karnataka (KA)
    "K1": "KA", "K4": "KA", "KI": "KA", "KO": "KA", "K0": "KA",
    # Haryana (HR)
    "4R": "HR", "HB": "HR",
    # Maharashtra (MH)
    "M8": "MH", "NH": "MH",
    # Uttar Pradesh (UP)
    "0P": "UP", "OP": "UP",
    # Andhra Pradesh (AP)
    "4P": "AP",
    # Gujarat (GJ)
    "6J": "GJ", "CJ": "GJ",
    # Rajasthan (RJ)
    "8J": "RJ",
    # Punjab (PB)
    "P8": "PB",
    # West Bengal (WB)
    "W8": "WB",
}


def compute_box_iou(b1, b2) -> float:
    """
    Compute Intersection-over-Union (IoU) between two bounding box objects.
    Both b1 and b2 are expected to have x1, y1, x2, y2 attributes.
    """
    xA = max(b1.x1, b2.x1)
    yA = max(b1.y1, b2.y1)
    xB = min(b1.x2, b2.x2)
    yB = min(b1.y2, b2.y2)
    inter = max(0, xB - xA) * max(0, yB - yA)
    area1 = max(1, (b1.x2 - b1.x1) * (b1.y2 - b1.y1))
    area2 = max(1, (b2.x2 - b2.x1) * (b2.y2 - b2.y1))
    union = area1 + area2 - inter
    return inter / max(union, 1)


def is_two_row_plate(width: int, height: int) -> bool:
    """
    Check whether a plate crop is likely a square / two-row license plate.
    Standard plates have aspect ratio ~3.0 - 5.0.
    Two-row plates (auto-rickshaws, two-wheelers) typically have aspect ratio < 2.2.
    """
    if height <= 0:
        return False
    aspect = width / float(height)
    return aspect < 2.2 and height >= 24


def split_two_row_crop(
    img: np.ndarray,
    top_fraction: float = 0.52,
    bot_fraction: float = 0.48,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Vertically split a two-row license plate into top row and bottom row crops.
    A slight vertical overlap prevents cutting character ascenders / descenders.
    """
    h = img.shape[0]
    top_end = max(1, int(h * top_fraction))
    bot_start = min(h - 1, int(h * bot_fraction))
    return img[:top_end, :], img[bot_start:, :]


from collections import deque
from dataclasses import dataclass
from typing import Any


@dataclass
class TrackedPlate:
    """State for a single tracked license plate across video frames."""

    track_id: int
    box: Any
    display_text: str
    display_conf: float
    last_seen: int
    readings: deque


class PlateTracker:
    """
    Intersection-over-Union (IoU) multi-object license plate tracker with
    rolling-window consensus voting. Prevents zombie track hopping across vehicles.
    """

    def __init__(self, max_unseen_frames: int = 8, window_size: int = 5) -> None:
        self.max_unseen_frames = max_unseen_frames
        self.window_size = window_size
        self.tracks: dict[int, TrackedPlate] = {}
        self.next_id: int = 0

    def update(
        self,
        detections: list[tuple[Any, str, float]],
        frame_idx: int,
    ) -> list[tuple[Any, str, float, int]]:
        """
        Update tracker with detections from the current frame.

        Parameters:
            detections: List of (bounding_box, text, confidence) tuples.
            frame_idx: Index of the current video frame.

        Returns:
            List of (bounding_box, smoothed_text, smoothed_confidence, track_id).
        """
        # 1. Prune expired tracks
        for tid in list(self.tracks.keys()):
            if frame_idx - self.tracks[tid].last_seen > self.max_unseen_frames:
                del self.tracks[tid]

        # 2. Match detections to existing tracks
        updated_results: list[tuple[Any, str, float, int]] = []
        assigned_tracks: set[int] = set()

        for box, text, conf in detections:
            bw = box.x2 - box.x1
            bh = box.y2 - box.y1
            cx, cy = (box.x1 + box.x2) / 2.0, (box.y1 + box.y2) / 2.0
            max_c_dist = max(bw, bh) * 1.5

            best_iou = 0.0
            best_match_id = None

            for tid, track in self.tracks.items():
                if tid in assigned_tracks:
                    continue
                tb = track.box
                iou = compute_box_iou(box, tb)
                tcx, tcy = (tb.x1 + tb.x2) / 2.0, (tb.y1 + tb.y2) / 2.0
                cdist = ((cx - tcx) ** 2 + (cy - tcy) ** 2) ** 0.5

                if iou > 0.15 or (cdist < max_c_dist and iou > 0.05):
                    if iou > best_iou:
                        best_iou = iou
                        best_match_id = tid

            if best_match_id is not None:
                track = self.tracks[best_match_id]
                track.box = box
                track.last_seen = frame_idx
                track.readings.append((text, conf))
                assigned_tracks.add(best_match_id)

                # Rolling window consensus: prefer longer valid text or highest confidence in window
                best_t, best_c = max(track.readings, key=lambda x: (len(x[0]), x[1]))
                track.display_text = best_t
                track.display_conf = best_c
                updated_results.append((box, track.display_text, track.display_conf, best_match_id))
            else:
                new_id = self.next_id
                self.next_id += 1
                rd: deque = deque([(text, conf)], maxlen=self.window_size)
                self.tracks[new_id] = TrackedPlate(
                    track_id=new_id,
                    box=box,
                    display_text=text,
                    display_conf=conf,
                    last_seen=frame_idx,
                    readings=rd,
                )
                assigned_tracks.add(new_id)
                updated_results.append((box, text, conf, new_id))

        return updated_results


def heal_indian_plate(text: str) -> str:
    """
    Auto-heal common character recognition confusions for Indian license plates
    using known state prefixes and RTO format conventions.

    Parameters:
        text: Raw recognized alphanumeric plate text.

    Returns:
        str: Corrected plate string.
    """
    clean = text.replace(" ", "").upper()

    # Handle 11-character plates caused by vertical bleed-through artifact between row 1 and row 2
    # e.g. KA01A1L6528 -> rogue digit '1' between series letters 'A' and 'L' -> KA01AL6528
    if len(clean) == 11:
        prefix = clean[:2]
        p_corr = STATE_PREFIX_CORRECTIONS.get(prefix, prefix)
        c0 = DIGIT_TO_LETTER.get(clean[0], clean[0])
        c1 = DIGIT_TO_LETTER.get(clean[1], clean[1])
        if p_corr in INDIAN_STATE_CODES or (c0 + c1) in INDIAN_STATE_CODES:
            if sum(c.isdigit() for c in clean[-4:]) >= 3:
                mid = clean[4:7]
                for i, c in enumerate(mid):
                    if c.isdigit():
                        clean = clean[:4 + i] + clean[4 + i + 1:]
                        break

    if len(clean) not in (7, 8, 9, 10):
        return clean

    chars = list(clean)

    # 1. Validate / heal state prefix (positions 0 and 1)
    prefix = "".join(chars[:2])
    if prefix in STATE_PREFIX_CORRECTIONS:
        corr = STATE_PREFIX_CORRECTIONS[prefix]
        chars[0], chars[1] = corr[0], corr[1]
    elif prefix not in INDIAN_STATE_CODES:
        c0 = DIGIT_TO_LETTER.get(chars[0], chars[0])
        c1 = DIGIT_TO_LETTER.get(chars[1], chars[1])
        if c0 + c1 in INDIAN_STATE_CODES:
            chars[0], chars[1] = c0, c1
        elif c0 + chars[1] in INDIAN_STATE_CODES:
            chars[0] = c0
        elif chars[0] + c1 in INDIAN_STATE_CODES:
            chars[1] = c1

    # 2. If state prefix is confirmed, enforce strict position rules
    if "".join(chars[:2]) in INDIAN_STATE_CODES:
        # Positions 2 & 3 must always be digits (RTO code, e.g. MH 12, TS 07)
        if len(chars) > 2:
            chars[2] = LETTER_TO_DIGIT.get(chars[2], chars[2])
        if len(chars) > 3:
            chars[3] = LETTER_TO_DIGIT.get(chars[3], chars[3])

        # 10-char format: LLDDLLDDDD (e.g. MH12DE1433, TS07UA7927)
        if len(chars) == 10:
            chars[4] = DIGIT_TO_LETTER.get(chars[4], chars[4])
            chars[5] = DIGIT_TO_LETTER.get(chars[5], chars[5])
            for i in range(6, 10):
                chars[i] = LETTER_TO_DIGIT.get(chars[i], chars[i])
        # 9-char format: LLDDLDDDD (e.g. TN45Q3566, JH02X7774)
        elif len(chars) == 9:
            chars[4] = DIGIT_TO_LETTER.get(chars[4], chars[4])
            for i in range(5, 9):
                chars[i] = LETTER_TO_DIGIT.get(chars[i], chars[i])
        # 8-char format: LLDDDDDD (e.g. LA020749)
        elif len(chars) == 8:
            for i in range(4, 8):
                chars[i] = LETTER_TO_DIGIT.get(chars[i], chars[i])
        # 7-char format: LLDDDDD (e.g. IN03044)
        elif len(chars) == 7:
            for i in range(2, 7):
                chars[i] = LETTER_TO_DIGIT.get(chars[i], chars[i])

    return "".join(chars)


def disambiguate_plate(
    text: str,
    pattern_mask: str | list[str] | None = None,
    custom_letter_to_digit: dict[str, str] | None = None,
    custom_digit_to_letter: dict[str, str] | None = None,
) -> str:
    """
    Disambiguate visually similar characters (e.g. 0 vs O, 1 vs I, 8 vs B)
    based on a regional syntax mask, scoring candidate masks to select the best fit.

    Parameters:
        text: Raw recognized text.
        pattern_mask: Syntax mask or list of masks. When multiple masks match the text
            length, the mask with the highest match score is selected.
        custom_letter_to_digit: Optional mapping overrides for letter -> digit.
        custom_digit_to_letter: Optional mapping overrides for digit -> letter.

    Returns:
        str: Disambiguated license plate string.
    """
    if not text:
        return text

    clean_text = text.replace(" ", "").upper()

    # Apply Indian plate healing first if eligible
    clean_text = heal_indian_plate(clean_text)

    if not pattern_mask:
        return clean_text

    # Normalise to a list of masks
    masks: list[str] = (
        [pattern_mask] if isinstance(pattern_mask, str) else list(pattern_mask)
    )

    # Filter candidate masks matching length
    candidates = [
        m.replace(" ", "").upper()
        for m in masks
        if len(m.replace(" ", "")) == len(clean_text)
    ]
    if not candidates:
        return clean_text

    let_to_dig = custom_letter_to_digit or LETTER_TO_DIGIT
    dig_to_let = custom_digit_to_letter or DIGIT_TO_LETTER

    # Score each candidate: prefer masks where characters already match or can be converted
    def _score_candidate(cand: str) -> int:
        score = 0
        for char, exp in zip(clean_text, cand, strict=True):
            if exp == "D":
                if char.isdigit():
                    score += 2
                elif char in let_to_dig:
                    score += 1
            elif exp == "L":
                if char.isalpha():
                    score += 2
                elif char in dig_to_let:
                    score += 1
            else:
                score += 2
        return score

    best_mask = max(candidates, key=_score_candidate)

    result_chars: list[str] = []
    for char, expected_type in zip(clean_text, best_mask, strict=True):
        if expected_type == "D":
            result_chars.append(let_to_dig.get(char, char))
        elif expected_type == "L":
            result_chars.append(dig_to_let.get(char, char))
        else:
            result_chars.append(char)

    disambiguated = "".join(result_chars)
    return heal_indian_plate(disambiguated)
