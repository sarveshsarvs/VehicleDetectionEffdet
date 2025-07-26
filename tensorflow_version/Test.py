import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# Define your classes
CLASS_NAMES = ['car', 'bike', 'truck', 'rickshaw', 'cart', 'ambulance']

# Load model
def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    print("Model loaded.")
    return model

# Preprocess image
def preprocess_image(image_path, input_size):
    image = Image.open(image_path).convert("RGB").resize((input_size, input_size))
    image_np = np.array(image) / 255.0  # normalize to 0–1
    image_np = (image_np - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]  # normalize
    return tf.expand_dims(image_np.astype(np.float32), axis=0), image

# Draw boxes
def draw_boxes(image, boxes, scores, labels):
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(image)

    for box, score, label in zip(boxes, scores, labels):
        box = [int(coord) for coord in box]
        rect = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1],
                                 linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        plt.text(box[0], box[1], f'{CLASS_NAMES[label]}: {score:.2f}',
                 bbox=dict(facecolor='yellow', alpha=0.5))

    plt.show()

# Process predictions
def process_predictions(predictions, threshold=0.5):
    # predictions: list of dicts with keys 'detection_boxes', 'detection_scores', 'detection_classes'
    boxes, scores, labels = [], [], []

    detection_boxes = predictions['detection_boxes'][0].numpy()
    detection_scores = predictions['detection_scores'][0].numpy()
    detection_classes = predictions['detection_classes'][0].numpy().astype(int)

    for i in range(len(detection_scores)):
        if detection_scores[i] > threshold:
            ymin, xmin, ymax, xmax = detection_boxes[i]
            box = [xmin * 512, ymin * 512, xmax * 512, ymax * 512]
            boxes.append(box)
            scores.append(detection_scores[i])
            labels.append(detection_classes[i])

    return boxes, scores, labels

# Main function
def main():
    MODEL_PATH = '/path/to/efficientdet_tf_model'
    IMAGE_PATH = '/path/to/image.jpg'
    INPUT_SIZE = 512

    model = load_model(MODEL_PATH)
    image_tensor, image_pil = preprocess_image(IMAGE_PATH, INPUT_SIZE)

    # Run inference
    print("Running inference...")
    predictions = model(image_tensor, training=False)
    boxes, scores, labels = process_predictions(predictions, threshold=0.5)

    print("Boxes:\n", boxes)
    print("Scores:\n", scores)
    print("Labels:\n", labels)
    print("No of vehicles:", len(scores))

    # Draw
    draw_boxes(image_pil, boxes, scores, labels)

if __name__ == '__main__':
    main()
