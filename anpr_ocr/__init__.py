"""
anpr-ocr package.
"""

from anpr_ocr.alpr import ALPR, ALPRResult, DrawPredictionsResult
from anpr_ocr.base import BaseDetector, BaseOCR, DetectionResult, OcrResult

__all__ = [
    "ALPR",
    "ALPRResult",
    "BaseDetector",
    "BaseOCR",
    "DetectionResult",
    "DrawPredictionsResult",
    "OcrResult",
]
