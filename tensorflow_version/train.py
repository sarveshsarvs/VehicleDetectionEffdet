import os
import json
import numpy as np
import tensorflow as tf
import keras_cv
from tensorflow import keras
from sklearn.model_selection import train_test_split

# Constants
IMAGE_SIZE = (512, 512)
NUM_CLASSES = 6

# Custom mapping (COCO category_id → label index)
CATEGORY_MAPPING = {
    1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5
}

# Load COCO JSON
def load_coco_dataset(image_dir, annotation_path):
    with open(annotation_path, 'r') as f:
        coco_data = json.load(f)

    image_id_to_filename = {img["id"]: img["file_name"] for img in coco_data["images"]}
    image_id_to_annotations = {img_id: [] for img_id in image_id_to_filename}

    for ann in coco_data["annotations"]:
        if ann["image_id"] in image_id_to_annotations:
            image_id_to_annotations[ann["image_id"]].append(ann)

    dataset = []
    for img_id, file_name in image_id_to_filename.items():
        boxes = []
        labels = []
        for ann in image_id_to_annotations[img_id]:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(CATEGORY_MAPPING[ann["category_id"]])
        dataset.append({
            "image_path": os.path.join(image_dir, file_name),
            "boxes": np.array(boxes, dtype=np.float32),
            "labels": np.array(labels, dtype=np.int32)
        })

    return dataset

# Load and split
dataset = load_coco_dataset("/train/images/directory/path/", "/annotations/path/annotations_coco.json")
train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)

# Preprocessing function
def preprocess_sample(sample):
    image = tf.io.read_file(sample["image_path"])
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32) / 255.0

    boxes = sample["boxes"]
    labels = sample["labels"]
    target = {
        "boxes": tf.convert_to_tensor(boxes, dtype=tf.float32),
        "classes": tf.convert_to_tensor(labels, dtype=tf.int32)
    }

    return image, target

# Convert list of dicts to tf.data.Dataset
def create_dataset(data_list, batch_size=8):
    ds = tf.data.Dataset.from_generator(
        lambda: (item for item in data_list),
        output_signature={
            "image_path": tf.TensorSpec(shape=(), dtype=tf.string),
            "boxes": tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
            "labels": tf.TensorSpec(shape=(None,), dtype=tf.int32)
        }
    )
    ds = ds.map(lambda x: preprocess_sample(x), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.padded_batch(batch_size)
    return ds.prefetch(tf.data.AUTOTUNE)

# Build Datasets
train_ds = create_dataset(train_data, batch_size=8)
val_ds = create_dataset(val_data, batch_size=8)

# Build EfficientDet Model (Lite3)
model = keras_cv.models.EfficientDet(
    label_encoder=None,
    bounding_box_format="xyxy",
    backbone="efficientdet-lite3",
    num_classes=NUM_CLASSES
)

# Compile model with built-in loss
model.compile(
    classification_loss="focal",
    box_loss="smoothl1",
    optimizer=keras.optimizers.Adam(1e-4)
)

# Callbacks
checkpoint_cb = keras.callbacks.ModelCheckpoint(
    "efficientdet_tf_best.h5",
    monitor="val_loss",
    save_best_only=True,
    save_weights_only=True,
    verbose=1
)

# Train
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=100,
    callbacks=[checkpoint_cb]
)
