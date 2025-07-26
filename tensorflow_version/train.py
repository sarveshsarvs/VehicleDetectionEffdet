import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from typing import List, Tuple
from dataclasses import dataclass

# Set matplotlib backend
plt.switch_backend('agg')

# Constants
INPUT_SIZE = 512
CONFIDENCE_THRESHOLD = 0.5
CLASS_NAMES = ['car', 'bike', 'truck', 'rickshaw', 'cart', 'ambulance']
COLOR_PALETTE = [(255, 0, 0), (0, 255, 0), (0, 0, 255), 
                 (255, 255, 0), (0, 255, 255), (255, 0, 255)]

@dataclass
class Detection:
    box: List[float]
    score: float
    label: int

def load_model(model_path: str) -> tf.keras.Model:
    """
    Load a saved TensorFlow EfficientDet model.
    """
    print(f"Loading model from: {model_path}")
    model = tf.saved_model.load(model_path)
    print("Model loaded successfully.")
    return model

def preprocess_image(image_path: str, input_size: int) -> Tuple[tf.Tensor, np.ndarray]:
    """
    Load and preprocess the image.
    """
    image = cv2.imread(image_path)
    original = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (input_size, input_size))
    image = image.astype(np.float32) / 255.0
    image = tf.convert_to_tensor(image)
    image = tf.expand_dims(image, axis=0)
    return image, original

def postprocess_predictions(preds: dict, image_shape: Tuple[int, int], threshold: float) -> List[Detection]:
    """
    Convert raw model predictions into structured detections.
    """
    detections = []
    boxes = preds['detection_boxes'][0].numpy()
    scores = preds['detection_scores'][0].numpy()
    labels = preds['detection_classes'][0].numpy().astype(np.int32)
    h, w = image_shape[:2]

    for box, score, label in zip(boxes, scores, labels):
        if score < threshold:
            continue
        ymin, xmin, ymax, xmax = box
        box_pixels = [xmin * w, ymin * h, xmax * w, ymax * h]
        detections.append(Detection(box_pixels, float(score), int(label)))

    return detections

def draw_boxes(image: np.ndarray, detections: List[Detection], class_names: List[str]) -> np.ndarray:
    """
    Draw bounding boxes and class labels on the image.
    """
    for det in detections:
        x1, y1, x2, y2 = map(int, det.box)
        color = COLOR_PALETTE[det.label % len(COLOR_PALETTE)]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label_text = f"{class_names[det.label]}: {det.score:.2f}"
        cv2.putText(image, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, color, 2)
    return image

def save_output(image: np.ndarray, save_path: str) -> None:
    """
    Save the annotated image to disk.
    """
    cv2.imwrite(save_path, image)
    print(f"Saved output image to: {save_path}")

def display_image(image: np.ndarray, window_name: str = "Detections") -> None:
    """
    Display the image using matplotlib.
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('off')
    plt.title(window_name)
    plt.show()

def run_inference(
    model: tf.keras.Model, 
    image_path: str, 
    input_size: int, 
    threshold: float,
    save_output_path: str
) -> None:
    """
    Complete inference pipeline: preprocess, predict, postprocess, draw, save.
    """
    print("Preprocessing...")
    image_tensor, original = preprocess_image(image_path, input_size)

    print("Running inference...")
    infer = model.signatures['serving_default']
    preds = infer(image_tensor)

    print("Processing predictions...")
    detections = postprocess_predictions(preds, original.shape, threshold)
    print(f"Total detections: {len(detections)}")

    print("Drawing boxes...")
    annotated = draw_boxes(original, detections, CLASS_NAMES)

    save_output(annotated, save_output_path)
    display_image(annotated)

def main():
    model_path = "path/to/saved_model"
    image_path = "path/to/image.jpg"
    output_path = "output.jpg"

    model = load_model(model_path)
    run_inference(model, image_path, INPUT_SIZE, CONFIDENCE_THRESHOLD, output_path)

def menu():
    """
    CLI for selecting different inference options.
    """
    print("TensorFlow EfficientDet Test Interface")
    print("====================================")
    print("1. Run inference on single image")
    print("2. Batch process folder")
    print("3. Exit")

    while True:
        choice = input("Enter your choice: ").strip()
        if choice == '1':
            main()
        elif choice == '2':
            batch_process()
        elif choice == '3':
            break
        else:
            print("Invalid input. Try again.")

def batch_process():
    """
    Optional: Batch inference on folder of images.
    """
    model_path = "path/to/saved_model"
    input_folder = "path/to/images"
    output_folder = "output_folder"

    os.makedirs(output_folder, exist_ok=True)
    model = load_model(model_path)

    for file in os.listdir(input_folder):
        if file.lower().endswith(('.jpg', '.png')):
            image_path = os.path.join(input_folder, file)
            output_path = os.path.join(output_folder, f"det_{file}")
            print(f"Processing {file}")
            run_inference(model, image_path, INPUT_SIZE, CONFIDENCE_THRESHOLD, output_path)

if __name__ == "__main__":
    menu()
