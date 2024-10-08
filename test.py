import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from effdet import create_model
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.tree import DecisionTreeRegressor

#define the model and loading function
def load_model(model_path):
    #load model
    model = create_model('tf_efficientdet_lite3', pretrained=True, num_classes=6)
    
    #load state of model
    state_dict = torch.load(model_path, map_location='cpu')
    
    #load into model
    model.load_state_dict(state_dict)
    
    model.eval()
    return model

#preprocess image
def preprocess_image(image_path, input_size):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)  #add bat dim

#draw bboxes
def draw_boxes(image, boxes, scores, labels, class_names):
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(image)

    for box, score, label in zip(boxes, scores, labels):
        box = [int(x) for x in box]
        rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.text(box[0], box[1], f'{class_names[label]}: {score:.2f}', bbox=dict(facecolor='yellow', alpha=0.5))

    plt.show()

#reshape, pading, trimming, parse predictions from model output
def process_predictions(predictions, threshold=0.5):
    global CLASS_NAMES
    boxes, scores, labels = [], [], []


    bbox_preds, class_preds = predictions
    
    for bbox_pred, class_pred in zip(bbox_preds, class_preds):
        if isinstance(bbox_pred, torch.Tensor) and isinstance(class_pred, torch.Tensor):
            bbox_pred = bbox_pred.view(-1, 4).cpu().numpy()
            class_pred = class_pred.view(-1, len(CLASS_NAMES)).cpu().numpy()
            
            #apply treashold
            for i in range(class_pred.shape[0]):
                score = np.max(class_pred[i])
                if score > threshold:
                    boxes.append(bbox_pred[i])
                    scores.append(score)
                    labels.append(np.argmax(class_pred[i]))
    
    #list to numpy array
    final_boxes = np.array(boxes) if boxes else np.array([])
    final_scores = np.array(scores) if scores else np.array([])
    final_labels = np.array(labels) if labels else np.array([])

    return final_boxes, final_scores, final_labels

def main():
    global CLASS_NAMES
    MODEL_SAVE_PATH = '/model/path/efficientdet.pth'
    IMAGE_PATH = '/test/image/path/.jpg'
    INPUT_SIZE = 512
    CLASS_NAMES = ['car', 'bike', 'truck', 'rickshaw', 'cart', 'ambulance']


    model = load_model(MODEL_SAVE_PATH)
    print("Model loaded")
    
 
    image_tensor = preprocess_image(IMAGE_PATH, INPUT_SIZE)
    
    #debug
    with torch.no_grad():
        predictions = model(image_tensor)
        print("Inference done")
    
    #prediction
    boxes, scores, labels = process_predictions(predictions, threshold=214)
    print("Boxes:\n", boxes)
    print("Scores:\n", scores)
    print("Labels:\n", labels)
    print("No of vehicles: ", len(scores))

    #draw boxes
    image = Image.open(IMAGE_PATH)
    draw_boxes(image, boxes, scores, labels, CLASS_NAMES)


if __name__ == '__main__':
    main()

