## 🛠️ Customization, Accuracy & Fine-Tuning

**anpr-ocr** is designed to be highly customizable and extensible. You can easily adjust pre-processing, post-processing disambiguation, swap in custom OCR engines, or fine-tune models on your regional datasets.

---

### 🎯 Improving Accuracy via Built-in Pipeline Features

#### 1. Bounding Box Crop Margin (`crop_margin`)
By default, detector bounding boxes may closely clip license plate edges, causing edge characters (like leading '1', 'I', or trailing digits) to be truncated. You can adjust the crop margin (e.g., 5% padding):

```python
from anpr_ocr import ALPR

# Apply 5% padding around detected plates before OCR
alpr = ALPR(crop_margin=0.05)
```

#### 2. Contrast Enhancement (`enhance_contrast`)
For low-contrast images, shadows, or glaring sunlight, enable CLAHE (Contrast Limited Adaptive Histogram Equalization):

```python
alpr = ALPR(enhance_contrast=True, min_plate_width=94)
```

#### 3. Character Disambiguation via Syntax Masks (`syntax_pattern`)
If your jurisdiction follows a known syntax (e.g., `MH12DE1433` -> 2 Letters, 2 Digits, 2 Letters, 4 Digits), supply a syntax pattern to automatically resolve ambiguous characters (`0` vs `O`, `1` vs `I`, `8` vs `B`, `5` vs `S`, `2` vs `Z`):

```python
# 'L' = Letter, 'D' = Digit, '?' = Wildcard
alpr = ALPR(syntax_pattern="LLDDLLDDDD")
```

---

### 🏋️ Fine-Tuning on Custom Plate Datasets

To fine-tune a model on your specific country's license plates:

#### Step 1: Install Training Dependencies
```shell
pip install fast-plate-ocr[train] albumentations
```

#### Step 2: Prepare Dataset CSVs
Create `train.csv` and `val.csv` with the following columns:
```csv
image_path,plate
datasets/plate_001.jpg,MH12DE1433
datasets/plate_002.jpg,DL08CA9821
```

#### Step 3: Train with Albumentations Augmentations
```shell
KERAS_BACKEND=tensorflow fast_plate_ocr train \
  --annotations train.csv \
  --val-annotations val.csv \
  --model-config-file model_config.yaml \
  --batch-size 64 \
  --epochs 100 \
  --output-dir ./trained_ocr_model
```

#### Step 4: Export to ONNX & Load in ALPR
```shell
fast_plate_ocr export --model-path ./trained_ocr_model/best_model.keras --output-path ./custom_plate_ocr.onnx
```

Load your fine-tuned model:
```python
alpr = ALPR(
    ocr_model_path="custom_plate_ocr.onnx",
    ocr_config_path="model_config.yaml",
)
```

---

### 🔌 Using Custom OCR Engines

#### Using PaddleOCR (PP-OCRv4)
```python
from paddleocr import PaddleOCR
from anpr_ocr.alpr import ALPR, BaseOCR, OcrResult


class PaddleOCRBackend(BaseOCR):
    def __init__(self) -> None:
        self.engine = PaddleOCR(use_angle_cls=True, lang="en")

    def predict(self, cropped_plate):
        if cropped_plate is None or cropped_plate.size == 0:
            return None
        result = self.engine.ocr(cropped_plate, cls=True)
        if not result or not result[0]:
            return None
        texts = [line[1][0] for line in result[0]]
        confs = [line[1][1] for line in result[0]]
        return OcrResult(text="".join(texts), confidence=float(sum(confs) / len(confs)))


alpr = ALPR(ocr=PaddleOCRBackend())
```

#### Using Tesseract OCR
```python
import re
from statistics import mean
import numpy as np
import pytesseract
from anpr_ocr.alpr import ALPR, BaseOCR, OcrResult


class PytesseractOCR(BaseOCR):
    def predict(self, cropped_plate: np.ndarray) -> OcrResult | None:
        if cropped_plate is None or cropped_plate.size == 0:
            return None
        data = pytesseract.image_to_data(
            cropped_plate,
            lang="eng",
            config="--oem 3 --psm 6",
            output_type=pytesseract.Output.DICT,
        )
        plate_text = " ".join(data["text"]).strip()
        plate_text = re.sub(r"[^A-Za-z0-9]", "", plate_text)
        avg_confidence = mean(conf for conf in data["conf"] if conf > 0) / 100.0 if data["conf"] else 0.0
        return OcrResult(text=plate_text, confidence=avg_confidence)


alpr = ALPR(ocr=PytesseractOCR())
```
