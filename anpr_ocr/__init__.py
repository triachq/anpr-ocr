"""
anpr-ocr package.
"""

from anpr_ocr.alpr import ALPR, ALPRResult, DrawPredictionsResult, VideoResult
from anpr_ocr.base import BaseDetector, BaseOCR, DetectionResult, OcrResult
from anpr_ocr.logger import PlateLogger, VehicleRecord
from anpr_ocr.utils import (
    PlateTracker,
    disambiguate_plate,
    enhance_plate_image,
    heal_indian_plate,
    pad_bounding_box,
    vote_consensus_plate,
)

__all__ = [
    "ALPR",
    "ALPRResult",
    "BaseDetector",
    "BaseOCR",
    "DetectionResult",
    "DrawPredictionsResult",
    "OcrResult",
    "PlateLogger",
    "PlateTracker",
    "VehicleRecord",
    "VideoResult",
    "disambiguate_plate",
    "enhance_plate_image",
    "heal_indian_plate",
    "pad_bounding_box",
    "vote_consensus_plate",
]
