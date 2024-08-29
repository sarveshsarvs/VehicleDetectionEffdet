import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense
from tensorflow.keras import optimizers, callbacks
from tensorflow.keras.preprocessing import image_dataset_from_directory
import os

#constants
MODEL_SAVE_PATH = './DetModel'  #path where the trained model will be saved
DATASET_PATH = '/home/allan/Desktop'  #path to the dataset directory
NUM_CLASSES = 6  #number of classes (car, bike, truck, auto, rickshaw, ambulance)

class CustomHubLayer(tf.keras.layers.Layer):  #fix keras layer and tensorflow layer smth idk issue im malding
    def __init__(self, model_url, **kwargs):  #makes the custom layer with a proper shape
        super(CustomHubLayer, self).__init__(**kwargs)
        self.model_url = model_url
        self.hub_layer = hub.KerasLayer(model_url, trainable=True) #creates a keras layer
    
    def call(self, inputs): #called during forward pass of model. allows custom layer to process data using pretrained model
        return self.hub_layer(inputs) #gets input and passes them through 'self.hub_layer(inputs)

    def compute_output_shape(self, input_shape):    #specifies output shape
        return tf.TensorShape([None, 512, 512, 3])  #defines custom layer output for pretrained model, None indicates that batch size is variable
    

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
        #if model doesnt exist, get it online
        print("Loading model architecture and pre-trained weights...")
        model_url = "https://tfhub.dev/tensorflow/efficientdet/d7/1"
        #add your custom layers for the number of classes
        inputs = Input(shape=(512, 512, 3)) #create new input 512x512 res and RGB(3 color channels)
        x = CustomHubLayer(model_url)(inputs)  #passes input through pre-trained model
        x = GlobalAveragePooling2D()(x)  #add pooling layer
        outputs = Dense(NUM_CLASSES, activation='softmax')(x) #custom output layer
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

#load and preprocess dataset
def preprocess_data(image, label):
    """
    preprocess images for training
    - resize images to 512x512 pixels
    - normalize pixel values to the range [0, 1]
    """
    image = tf.image.resize(image, [512, 512])  #resize images to 512x512 pixels
    image = image / 255.0  #normalize pixel values [0-1]
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
    verbose=1  #verbosity mode (training progress)
)

#save final weights
model.save_weights(MODEL_SAVE_PATH)  #save model weights after training
