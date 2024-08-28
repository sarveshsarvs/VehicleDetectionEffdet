import tensorflow as tf
from tensorflow.keras import optimizers, callbacks
from tensorflow.keras.preprocessing import image_dataset_from_directory
import os

#constants
MODEL_SAVE_PATH = 'path/to/save/efficientdet-d7-model'  #path where the trained model will be saved
DATASET_PATH = 'path/to/dataset'  #path to the dataset directory
NUM_CLASSES = 6  #number of classes (car, bike, truck, auto, rickshaw, ambulance)

#load EfficientDet-D7 model
def load_model():
    """
    load or initialize the EfficientDet-D7 model
    - if a model with existing weights is found at MODEL_SAVE_PATH, load it
    - otherwise, load the EfficientDet-D7 model with ImageNet weights and add custom layers
    """
    if os.path.exists(MODEL_SAVE_PATH):
        #if model weights exist, load the model from the path
        print("Loading model with existing weights...")
        model = tf.keras.models.load_model(MODEL_SAVE_PATH, compile=False)
    else:
        #if model weights dont exist, initialize a new model with ImageNet weights
        print("Loading model architecture and ImageNet weights...")
        #load EfficientDet-D7 model with ImageNet weights (excluding top classification layers)
        model = tf.keras.applications.EfficientDetD7(input_shape=(None, None, 3), include_top=False, weights='imagenet')
        #add custom classification layers for the 6 classes output
        x = tf.keras.layers.GlobalAveragePooling2D()(model.output)  #global average pooling layer to reduce feature map dimensions
        x = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)  #dense layer for classification with softmax activation
        model = tf.keras.Model(inputs=model.input, outputs=x)  #ceate a new model with the custom output layer
    return model

#load and preprocess dataset
def preprocess_data(image, label):
    """
    preprocess images for training
    - resize images to 512x512 pixels
    - normalize pixel values to the range [0, 1]
    """
    image = tf.image.resize(image, [512, 512])  #resize images to 512x512 pixels
    image = image / 255.0  #normalize pixel values
    return image, label

#create training dataset
train_dataset = image_dataset_from_directory(
    DATASET_PATH,  #directory where dataset is stored
    label_mode="categorical",  #use categorical labels (one-hot encoded)
    batch_size=16,  #number of images per batch
    image_size=(512, 512),  #resize images to 512x512 pixels
    validation_split=0.2,  #use 20% of the data for validation
    subset="training",  #specify this subset as the training data
    seed=123  #seed for reproducibility (can be anything)
)

#create validation dataset
val_dataset = image_dataset_from_directory(
    DATASET_PATH,  #directory where dataset is stored
    label_mode="categorical",  #use categorical labels (one-hot encoded)
    batch_size=16,  #number of images per batch
    image_size=(512, 512),  #resize images to 512x512 pixels
    validation_split=0.2,  #use 20% of the data for validation
    subset="validation",  #Specify this subset as the validation data
    seed=123  #seed for reproducibility
)

#apply prefetching to improve performance
AUTOTUNE = tf.data.AUTOTUNE  #automatically tune prefetch buffer size
train_dataset = train_dataset.cache().shuffle(1024).prefetch(buffer_size=AUTOTUNE)  #cache and shuffle training dataset, then prefetch
val_dataset = val_dataset.prefetch(buffer_size=AUTOTUNE)  #prefetch validation dataset

#model setup
model = load_model()  #load or initialize the model
optimizer = optimizers.Adam(learning_rate=1e-3)  #adam optimizer with learning rate of 0.001
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])  #compile the model with loss function and metrics

#callbacks for model training
checkpoint_cb = callbacks.ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True)  #save only the best model weights
early_stopping_cb = callbacks.EarlyStopping(patience=10, restore_best_weights=True)  #stop training early if no improvement in 10 epochs
tensorboard_cb = callbacks.TensorBoard(log_dir='logs/fit')  #log training progress for TensorBoard

#train the model
history = model.fit(
    train_dataset,  #training dataset
    validation_data=val_dataset,  #validation dataset
    epochs=100,  #number of epochs to train
    callbacks=[checkpoint_cb, early_stopping_cb, tensorboard_cb],  #list of callbacks
    verbose=1  #verbosity mode
)

#save final weights
model.save_weights(MODEL_SAVE_PATH)  #save model weights after training
