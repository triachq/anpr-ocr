# Reference

This page shows the public API of anpr-ocr.

## At a Glance

- Use `ALPR.predict()` to get structured ALPR results
- Use `ALPR.draw_predictions()` to get an annotated image and the same ALPR results
- `BoundingBox` and `DetectionResult` come from `open-image-models`

## Imports

```python
from anpr_ocr import ALPR, ALPRResult, DrawPredictionsResult, OcrResult
```

## Common Inputs

- A NumPy image in BGR format
- A string path to an image file

## Common Returns

- `ALPR.predict(...)` returns `list[ALPRResult]`
- `ALPR.draw_predictions(...)` returns `DrawPredictionsResult`

`ALPRResult` contains:

- `detection`: box, label, and detection confidence
- `ocr`: recognized text and OCR confidence, or `None`

`DrawPredictionsResult` contains:

- `image`: the image with boxes and text drawn on it
- `results`: the same ALPR results used for drawing

## Available Models

See the available detection models in [open-image-models](https://pypi.org/project/open-image-models/)
and OCR models in [fast-plate-ocr](https://pypi.org/project/fast-plate-ocr/).

## Main Class

::: anpr_ocr.alpr.ALPR
    options:
      show_root_heading: true
      show_root_toc_entry: false

## Result Types

::: anpr_ocr.alpr.ALPRResult
    options:
      show_root_heading: true
      show_root_toc_entry: false

::: anpr_ocr.alpr.DrawPredictionsResult
    options:
      show_root_heading: true
      show_root_toc_entry: false

::: anpr_ocr.base.OcrResult
    options:
      show_root_heading: true
      show_root_toc_entry: false

## Interfaces

::: anpr_ocr.base.BaseDetector
    options:
      show_root_heading: true
      show_root_toc_entry: false

::: anpr_ocr.base.BaseOCR
    options:
      show_root_heading: true
      show_root_toc_entry: false

## External Types

See [`BoundingBox`][open_image_models.detection.core.base.BoundingBox]
and [`DetectionResult`][open_image_models.detection.core.base.DetectionResult].
