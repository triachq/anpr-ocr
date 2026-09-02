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


def disambiguate_plate(
    text: str,
    pattern_mask: str | None = None,
    custom_letter_to_digit: dict[str, str] | None = None,
    custom_digit_to_letter: dict[str, str] | None = None,
) -> str:
    """
    Disambiguate visually similar characters (e.g. 0 vs O, 1 vs I, 8 vs B)
    based on a regional syntax mask.

    Parameters:
        text: Raw recognized text.
        pattern_mask: Syntax mask where:
            'L' indicates an alphabetic letter (A-Z)
            'D' indicates a numeric digit (0-9)
            '?' indicates any character (no conversion)
            Example: 'LLDDLLDDDD' (e.g. MH12DE1433)
        custom_letter_to_digit: Optional mapping overrides for letter -> digit.
        custom_digit_to_letter: Optional mapping overrides for digit -> letter.

    Returns:
        str: Disambiguated license plate string.
    """
    if not text or not pattern_mask:
        return text

    clean_text = text.replace(" ", "").upper()
    mask = pattern_mask.replace(" ", "").upper()

    if len(clean_text) != len(mask):
        return text

    let_to_dig = custom_letter_to_digit or LETTER_TO_DIGIT
    dig_to_let = custom_digit_to_letter or DIGIT_TO_LETTER

    result_chars: list[str] = []
    for char, expected_type in zip(clean_text, mask, strict=True):
        if expected_type == "D":
            result_chars.append(let_to_dig.get(char, char))
        elif expected_type == "L":
            result_chars.append(dig_to_let.get(char, char))
        else:
            result_chars.append(char)

    return "".join(result_chars)
