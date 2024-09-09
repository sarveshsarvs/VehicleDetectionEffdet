import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from effdet import create_model

#define the model and loading function
def load_model(model_path):
    #load EfficientDet-Lite3 model
    model = create_model('tf_efficientdet_lite3', pretrained=True, num_classes=6)
    
    #load the saved state_dict
    state_dict = torch.load(model_path, map_location='cpu')
    
    #load the state dict into the model
    model.load_state_dict(state_dict)
    
    model.eval()
    return model

def preprocess_image(image_path, input_size):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)  #add batch dimension

def draw_boxes(image, boxes, scores, labels, class_names):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(image)

    for box, score, label in zip(boxes, scores, labels):
        box = [int(x) for x in box]
        rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.text(box[0], box[1], f'{class_names[label]}: {score:.2f}', bbox=dict(facecolor='yellow', alpha=0.5))

    plt.show()

def main():

    MODEL_SAVE_PATH = '/home/allan/project/sih/efficientdet_d6_best.pth'
    IMAGE_PATH = '/home/allan/project/sih/Dataset/validation/images/images2.jpeg'
    INPUT_SIZE = 512  #set input size according to your model configuration
    CLASS_NAMES = ['car', 'bike', 'truck', 'rickshaw', 'cart', 'ambulance']  #update with your class names

    #load the model
    model = load_model(MODEL_SAVE_PATH)
    
    #preprocess the image
    image_tensor = preprocess_image(IMAGE_PATH, INPUT_SIZE)
    
    #perform inference
    with torch.no_grad():
        predictions = model(image_tensor)
    
    #extract bounding boxes, scores, and labels from the predictions
    boxes = []
    scores = []
    labels = []


    for pred in predictions:
        if isinstance(pred, torch.Tensor):
            if len(pred.shape) == 4:
                #handle if predictions is a 4D tensor
                num_boxes = pred.shape[1]
                boxes.append(pred[0].reshape(num_boxes, -1).cpu().numpy())
            else:
                boxes.append(pred[0].cpu().numpy())
    
    #convert lists to numpy arrays
    boxes = np.concatenate(boxes, axis=0) if boxes else np.array([])
    scores = np.concatenate(scores, axis=0) if scores else np.array([])
    labels = np.concatenate(labels, axis=0) if labels else np.array([])

    #draw boxes on the image
    image = Image.open(IMAGE_PATH)
    draw_boxes(image, boxes, scores, labels, CLASS_NAMES)

if __name__ == '__main__':
    main()
