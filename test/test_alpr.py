"""
Test ALPR end-to-end.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest
from fast_plate_ocr.inference.hub import OcrModel
from open_image_models.detection.core.hub import PlateDetectorModel

from anpr_ocr.alpr import ALPR
from anpr_ocr.utils import disambiguate_plate, enhance_plate_image, pad_bounding_box

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"



@pytest.fixture(scope="module", name="alpr")
def alpr_fixture() -> ALPR:
    return ALPR(
        detector_model="yolo-v9-t-384-license-plate-end2end",
        ocr_model="cct-xs-v2-global-model",
    )


@pytest.mark.parametrize(
    "img_path, expected_plates", [(ASSETS_DIR / "test_image.png", {"5AU5341"})]
)
@pytest.mark.parametrize("detector_model", ["yolo-v9-t-384-license-plate-end2end"])
@pytest.mark.parametrize(
    "ocr_model",
    [
        "cct-s-v2-global-model",
        "cct-xs-v2-global-model",
        "cct-s-v1-global-model",
        "cct-xs-v1-global-model",
        "global-plates-mobile-vit-v2-model",
        "european-plates-mobile-vit-v2-model",
    ],
)
def test_default_alpr(
    img_path: Path,
    expected_plates: set[str],
    detector_model: PlateDetectorModel,
    ocr_model: OcrModel,
) -> None:
    # pylint: disable=too-many-locals
    im = cv2.imread(str(img_path))
    assert im is not None, "Failed to load test image"
    alpr = ALPR(
        detector_model=detector_model,
        ocr_model=ocr_model,
    )
    actual_result = alpr.predict(im)
    actual_plates = {x.ocr.text for x in actual_result if x.ocr is not None}
    assert actual_plates == expected_plates

    for res in actual_result:
        bbox = res.detection.bounding_box
        height, width = im.shape[:2]
        x1, y1 = max(bbox.x1, 0), max(bbox.y1, 0)
        x2, y2 = min(bbox.x2, width), min(bbox.y2, height)

        assert 0 <= x1 < width, f"x1 coordinate {x1} out of bounds (0, {width})"
        assert 0 <= x2 <= width, f"x2 coordinate {x2} out of bounds (0, {width})"
        assert 0 <= y1 < height, f"y1 coordinate {y1} out of bounds (0, {height})"
        assert 0 <= y2 <= height, f"y2 coordinate {y2} out of bounds (0, {height})"
        assert x1 < x2, f"x1 ({x1}) should be less than x2 ({x2})"
        assert y1 < y2, f"y1 ({y1}) should be less than y2 ({y2})"

        if res.ocr is not None:
            conf = res.ocr.confidence
            if isinstance(conf, list):
                assert all(0.0 <= x <= 1.0 for x in conf)
            elif isinstance(conf, float):
                assert 0.0 <= conf <= 1.0
            else:
                raise TypeError(f"Unexpected type for confidence: {type(conf).__name__}")


@pytest.mark.parametrize("img_path", [ASSETS_DIR / "test_image.png"])
def test_draw_predictions(img_path: Path, alpr: ALPR) -> None:
    im = cv2.imread(str(img_path))
    assert im is not None, "Failed to load test image"
    h, w, c = im.shape

    # ndarray input
    drawn_nd = alpr.draw_predictions(im.copy())
    assert isinstance(drawn_nd.image, np.ndarray)
    assert drawn_nd.image.shape == (h, w, c)
    assert drawn_nd.results

    diff_nd = cv2.absdiff(drawn_nd.image, im)
    assert int(diff_nd.sum()) > 0

    # string path input
    drawn_path = alpr.draw_predictions(str(img_path))
    assert isinstance(drawn_path.image, np.ndarray)
    assert drawn_path.image.shape == (h, w, c)
    assert drawn_path.results

    diff_path = cv2.absdiff(drawn_path.image, im)
    assert int(diff_path.sum()) > 0


def test_pad_bounding_box() -> None:
    x1, y1, x2, y2 = pad_bounding_box(10, 20, 110, 70, 200, 200, margin_x=0.1, margin_y=0.1)
    assert x1 == 0  # 10 - 10
    assert y1 == 15  # 20 - 5
    assert x2 == 120  # 110 + 10
    assert y2 == 75  # 70 + 5

    # Test boundaries clamping
    x1, y1, x2, y2 = pad_bounding_box(0, 0, 50, 50, 100, 100, margin_x=0.2, margin_y=0.2)
    assert x1 == 0
    assert y1 == 0
    assert x2 == 60
    assert y2 == 60


def test_disambiguate_plate() -> None:
    # 'LLDDLLDDDD' pattern
    raw = "MHIZDEI433"
    fixed = disambiguate_plate(raw, "LLDDLLDDDD")
    assert fixed == "MH12DE1433"

    # Digits to letters
    raw2 = "0O12AB34"
    fixed2 = disambiguate_plate(raw2, "LLDDLLDD")
    assert fixed2 == "OO12AB34"


def test_enhance_plate_image() -> None:
    dummy_img = np.zeros((20, 40, 3), dtype=np.uint8)
    enhanced = enhance_plate_image(dummy_img, enhance_contrast=True, min_width=100)
    assert enhanced.shape[1] == 100
    assert enhanced.shape[0] == 50  # 20 * (100 / 40)


