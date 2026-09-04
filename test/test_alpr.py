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
from anpr_ocr.cli import _create_main_parser
from anpr_ocr.logger import PlateLogger
from anpr_ocr.utils import (
    PlateTracker,
    disambiguate_plate,
    enhance_plate_image,
    heal_indian_plate,
    is_two_row_plate,
    pad_bounding_box,
    split_two_row_crop,
    vote_consensus_plate,
)

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


def test_heal_indian_plate() -> None:
    # 10-character standard
    assert heal_indian_plate("MH12DE1433") == "MH12DE1433"
    assert heal_indian_plate("0L01AB1234") == "DL01AB1234"
    assert heal_indian_plate("M812AB1234") == "MH12AB1234"
    assert heal_indian_plate("KA-01-AB-1234") == "KA01AB1234"
    # 9-character
    assert heal_indian_plate("TN45Q3566") == "TN45Q3566"
    # 11-character artifact healing
    assert heal_indian_plate("KA01A1L6528") == "KA01AL6528"


def test_bharat_series_plate() -> None:
    # Format: YY BH NNNN AA
    assert heal_indian_plate("21BH1234AA") == "21BH1234AA"
    assert heal_indian_plate("228H1234AB") == "22BH1234AB"
    assert heal_indian_plate("23BH5678A") == "23BH5678A"


def test_vote_consensus_plate() -> None:
    # Test glare / transient confusion correction
    readings = [
        ("MH12DE1433", 0.95),
        ("MH120E1433", 0.70),
        ("MH12DE1433", 0.92),
    ]
    voted, conf = vote_consensus_plate(readings)
    assert voted == "MH12DE1433"
    assert conf > 0.80


def test_two_row_plate_detection_and_split() -> None:
    assert is_two_row_plate(100, 70) is True
    assert is_two_row_plate(300, 70) is False

    dummy = np.zeros((60, 100, 3), dtype=np.uint8)
    top, bot = split_two_row_crop(dummy)
    assert top.shape[0] > 0 and top.shape[1] == 100
    assert bot.shape[0] > 0 and bot.shape[1] == 100


def test_plate_tracker() -> None:
    class DummyBox:
        def __init__(self, x1: int, y1: int, x2: int, y2: int) -> None:
            self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2

    tracker = PlateTracker(max_unseen_frames=5, window_size=3)
    b1 = DummyBox(10, 10, 100, 50)
    dets1 = [(b1, "MH12DE1433", 0.90)]
    res1 = tracker.update(dets1, frame_idx=0)
    assert len(res1) == 1
    assert res1[0][1] == "MH12DE1433"
    tid = res1[0][3]

    # Next frame, slight shift
    b2 = DummyBox(12, 11, 102, 51)
    dets2 = [(b2, "MH12DE1433", 0.95)]
    res2 = tracker.update(dets2, frame_idx=1)
    assert len(res2) == 1
    assert res2[0][3] == tid


def test_plate_logger_and_exports(tmp_path: Path) -> None:
    class DummyBox:
        def __init__(self, x1: int, y1: int, x2: int, y2: int) -> None:
            self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2

    csv_file = tmp_path / "test_log.csv"
    json_file = tmp_path / "test_log.json"
    snaps_dir = tmp_path / "snapshots"

    logger = PlateLogger(
        output_csv=csv_file,
        snapshots_dir=snaps_dir,
        min_conf=0.40,
        min_chars=4,
    )
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    box = DummyBox(20, 20, 120, 60)

    logger.observe("MH12DE1433", 0.80, box, frame_idx=0, frame_bgr=frame, fps=30.0)
    logger.observe("MH12DE1433", 0.95, box, frame_idx=1, frame_bgr=frame, fps=30.0)

    finalized = logger.finalize()
    assert len(finalized) == 1
    assert finalized[0].plate_number == "MH12DE1433"
    assert finalized[0].confidence == 0.95

    # Export CSV & JSON
    out_csv = logger.export_csv()
    assert out_csv is not None and out_csv.is_file()

    out_json = logger.export_json(json_file)
    assert out_json is not None and out_json.is_file()

    table = logger.summary_table()
    assert "MH12DE1433" in table


def test_cli_parser() -> None:
    parser = _create_main_parser()
    args_img = parser.parse_args(["image", "test.jpg"])
    assert args_img.command == "image"
    assert args_img.images == ["test.jpg"]

    args_vid = parser.parse_args(["video", "test.mp4", "--frame-skip", "2"])
    assert args_vid.command == "video"
    assert args_vid.frame_skip == 2

    args_stream = parser.parse_args(["stream", "0"])
    assert args_stream.command == "stream"
    assert args_stream.source == "0"
