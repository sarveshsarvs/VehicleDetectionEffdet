# EfficientDet Vehicle Detection

This project performs object detection on images using a pre-trained EfficientDet model. It supports both TensorFlow and PyTorch backends, enabling detection of vehicles such as cars, bikes, trucks, rickshaws, carts, and ambulances. The script loads a model, preprocesses an input image, runs inference, filters predictions by confidence threshold, and visualizes the detections with bounding boxes.

## Features
- Supports both **TensorFlow** and **PyTorch** inference pipelines.
- Detects vehicles: `car`, `bike`, `truck`, `rickshaw`, `cart`, `ambulance`.
- Preprocessing includes resizing, RGB conversion, and ImageNet-style normalization.
- Configurable confidence threshold for filtering predictions.
- Visualization using Matplotlib with labeled bounding boxes.
- Simple plug-and-play structure for running inference on new images.

## Requirements
- Python 3.8+
- Install dependencies:
```bash
pip install tensorflow torch torchvision pillow matplotlib numpy
```
- Pre-trained model (EfficientDet) in either:
  - TensorFlow SavedModel format
  - PyTorch checkpoint

## Project Structure
```
project/
│── efficientdet_inference_tf.py       # TensorFlow inference script
│── efficientdet_inference_torch.py    # PyTorch inference script
│── classes.json                       # Class names file
│── README.md
```

## Usage

### 1. TensorFlow Inference
```bash
python efficientdet_inference_tf.py
```
**Script Highlights**:
- Loads a TensorFlow model via `tf.keras.models.load_model()`.
- Preprocesses image (resize, normalize, batch dimension).
- Runs model inference and extracts:
  - `detection_boxes` (normalized ymin, xmin, ymax, xmax)
  - `detection_scores`
  - `detection_classes`
- Converts boxes to pixel coordinates and filters by confidence threshold.
- Draws bounding boxes with labels and scores.

**Example**:
```python
MODEL_PATH = '/path/to/efficientdet_tf_model'
IMAGE_PATH = '/path/to/image.jpg'
INPUT_SIZE = 512
```

---

### 2. PyTorch Inference
```bash
python efficientdet_inference_torch.py
```
**Script Highlights**:
- Loads a PyTorch model via `torch.load()` or model-specific loader.
- Preprocesses image with `torchvision.transforms` (resize, tensor, normalize).
- Runs inference under `torch.no_grad()`.
- Parses output:
  - `boxes` (xmin, ymin, xmax, ymax)
  - `scores`
  - `labels`
- Filters predictions by confidence threshold.
- Draws bounding boxes with labels and scores.

**Example**:
```python
MODEL_PATH = '/path/to/efficientdet_torch_model.pth'
IMAGE_PATH = '/path/to/image.jpg'
INPUT_SIZE = 512
```

---

## Visualization
- Bounding boxes are drawn using Matplotlib’s `patches.Rectangle`.
- Label format: `ClassName: ConfidenceScore`.
- Example output:
```
Boxes: [[120, 150, 300, 400], ...]
Scores: [0.92, 0.85, ...]
Labels: [0, 2, ...]  # indices in CLASS_NAMES
Number of vehicles detected: 3
```

---

## Notes
- This project is **inference-only**; training functionality is not included.
- Ensure model format matches the inference script (TF model with TF script, PyTorch model with PyTorch script).
- For best results, use models trained on datasets with your target classes.

---

## License
MIT License — feel free to modify and distribute.