"""
anpr-ocr package.
"""

from anpr_ocr.alpr import ALPR, ALPRResult, DrawPredictionsResult, VideoResult
from anpr_ocr.base import BaseDetector, BaseOCR, DetectionResult, OcrResult
from anpr_ocr.logger import PlateLogger, VehicleRecord
from anpr_ocr.utils import disambiguate_plate, enhance_plate_image, pad_bounding_box

__all__ = [
    "ALPR",
    "ALPRResult",
    "BaseDetector",
    "BaseOCR",
    "DetectionResult",
    "DrawPredictionsResult",
    "OcrResult",
    "PlateLogger",
    "VehicleRecord",
    "VideoResult",
    "disambiguate_plate",
    "enhance_plate_image",
    "pad_bounding_box",
]

