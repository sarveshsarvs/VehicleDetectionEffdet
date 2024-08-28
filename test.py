import tensorflow as tf
import numpy as np
import cv2
import os

#constants
MODEL_SAVE_PATH = 'path/to/model'
IMAGE_PATH = 'path/to/test/image.jpg'

#load the trained model
def load_model():
    print("Loading model with existing weights...")
    model = tf.keras.models.load_model(MODEL_SAVE_PATH, compile=False)
    return model

#load and preprocess test image
def preprocess_image(image_path):
    image = cv2.imread(image_path)  #read image using OpenCV
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  #convert image to RGB
    image_resized = tf.image.resize(image_rgb, [512, 512])  #resize to match the input size
    image_normalized = image_resized / 255.0  #normalize to [0, 1]
    image_expanded = tf.expand_dims(image_normalized, axis=0)  #add batch dimension
    return image, image_expanded

#draw bounding boxes and labels on image
def draw_boxes(image, boxes, class_ids, class_names):
    for i in range(boxes.shape[0]):
        box = boxes[i]
        class_id = int(class_ids[i])
        #convert box coordinates from normalized to pixel values to draw on image
        ymin, xmin, ymax, xmax = box
        (height, width, _) = image.shape
        ymin = int(ymin * height)
        xmin = int(xmin * width)
        ymax = int(ymax * height)
        xmax = int(xmax * width)

        #draw the bounding box and label
        color = (0, 255, 0)  #green color
        label = class_names[class_id]
        image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
        image = cv2.putText(image, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return image

#main function
def main():
    model = load_model()  #load the trained model
    image, preprocessed_image = preprocess_image(IMAGE_PATH)  #load and preprocess the image

    #detection
    predictions = model.predict(preprocessed_image)
    boxes, class_ids, scores = predictions[0], predictions[1], predictions[2]

    #filter out boxes with low confidence scores
    confidence_threshold = 0.5
    high_confidence_indices = np.where(scores > confidence_threshold)
    boxes = boxes[high_confidence_indices]
    class_ids = class_ids[high_confidence_indices]

    #class names in dataset
    class_names = ['car', 'bike', 'rickshaw', 'truck', 'cart', 'ambulance']

    #draw boxes and labels on the image
    annotated_image = draw_boxes(image, boxes, class_ids, class_names)

    #calculate total number of detected vehicles
    total_vehicles = len(boxes)
    print(f"Total number of vehicles detected: {total_vehicles}")

    #calculate vehicle density
    (height, width, _) = image.shape
    image_area = height * width #total area
    vehicle_density = total_vehicles / image_area #vehicle density
    print(f"Vehicle density: {vehicle_density:.6f} vehicles per pixel")

    #display image
    cv2.imshow("Annotated Image", annotated_image)
    cv2.waitKey(0)  #wait for a key press to close window
    cv2.destroyAllWindows()  #close the window

if __name__ == "__main__":
    main()
