import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from effdet import create_model
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Define the model and loading function
def load_model(model_path):
    # Load EfficientDet-Lite3 model
    model = create_model('tf_efficientdet_lite3', pretrained=True, num_classes=6)
    
    # Load the saved state_dict
    state_dict = torch.load(model_path, map_location='cpu')
    
    # Load the state dict into the model
    model.load_state_dict(state_dict)
    
    model.eval()
    return model

# Preprocess the image
def preprocess_image(image_path, input_size):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

# Draw bounding boxes
def draw_boxes(image, boxes, scores, labels, class_names):
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(image)

    for box, score, label in zip(boxes, scores, labels):
        box = [int(x) for x in box]
        rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.text(box[0], box[1], f'{class_names[label]}: {score:.2f}', bbox=dict(facecolor='yellow', alpha=0.5))

    plt.show()

# Handle reshaping, padding, and concatenation of boxes, scores, and labels
def process_predictions(predictions, threshold=0.5):
    global CLASS_NAMES
    boxes, scores, labels = [], [], []

    # Assuming predictions is a tuple of (bbox_preds, class_preds)
    bbox_preds, class_preds = predictions
    
    for bbox_pred, class_pred in zip(bbox_preds, class_preds):
        if isinstance(bbox_pred, torch.Tensor) and isinstance(class_pred, torch.Tensor):
            bbox_pred = bbox_pred.view(-1, 4).cpu().numpy()
            class_pred = class_pred.view(-1, len(CLASS_NAMES)).cpu().numpy()
            
            # Apply threshold to filter predictions
            for i in range(class_pred.shape[0]):
                score = np.max(class_pred[i])
                if score > threshold:
                    boxes.append(bbox_pred[i])
                    scores.append(score)
                    labels.append(np.argmax(class_pred[i]))
    
    # Convert lists to numpy arrays
    final_boxes = np.array(boxes) if boxes else np.array([])
    final_scores = np.array(scores) if scores else np.array([])
    final_labels = np.array(labels) if labels else np.array([])

    return final_boxes, final_scores, final_labels

def main():
    global CLASS_NAMES
    MODEL_SAVE_PATH = '/home/allan/project/sih/efficientdet.pth'
    IMAGE_PATH = '/home/allan/project/sih/te2s.jpg'
    INPUT_SIZE = 512  # Set input size according to your model configuration
    CLASS_NAMES = ['car', 'bike', 'truck', 'rickshaw', 'cart', 'ambulance']  # Update with your class names

    # Load the model
    model = load_model(MODEL_SAVE_PATH)
    print("Model loaded")
    
    # Preprocess the image
    image_tensor = preprocess_image(IMAGE_PATH, INPUT_SIZE)
    
    # Perform inference
    with torch.no_grad():
        predictions = model(image_tensor)
        print("Inference done")
    
    # Process predictions with a score threshold of 0.5
    boxes, scores, labels = process_predictions(predictions, threshold=217)
    print("Boxes:\n", boxes)
    print("Scores:\n", scores)
    print("Labels:\n", labels)
    print("No of vehicles: ", len(scores))

    # Draw boxes on the image
    image = Image.open(IMAGE_PATH)
    draw_boxes(image, boxes, scores, labels, CLASS_NAMES)

if __name__ == '__main__':
    main()
