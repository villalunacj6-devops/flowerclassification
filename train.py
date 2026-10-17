import os
import json
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, models

# 1. Load the Oxford 102 Flowers Dataset
print("Downloading Oxford 102 Flowers dataset...")
(ds_train, ds_val), ds_info = tfds.load(
    'oxford_flowers102',
    split=['train[:80%]', 'train[80%:]+validation'],
    as_supervised=True,
    with_info=True
)

# Extract and save the exact 102 species labels
class_names = ds_info.features['label'].names
num_classes = len(class_names)
print(f"Total Flower Species: {num_classes}")

# Save labels to a JSON file so classify.py can read them dynamically
labels_path = os.path.join(os.path.dirname(__file__), 'flower_labels.json')
with open(labels_path, 'w') as f:
    json.dump(class_names, f)

# 2. Preprocess & Augment Data
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

def preprocess(image, label):
    image = tf.image.resize(image, IMG_SIZE)
    return image, label

ds_train = ds_train.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
ds_val = ds_val.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)

ds_train = ds_train.cache().shuffle(1000).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
ds_val = ds_val.batch(BATCH_SIZE).cache().prefetch(buffer_size=tf.data.AUTOTUNE)

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
])

# 3. Build High-Accuracy EfficientNetV2 Model
base_model = tf.keras.applications.EfficientNetV2B0(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

inputs = layers.Input(shape=(224, 224, 3))
x = data_augmentation(inputs)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(num_classes, activation='softmax')(x)

model = models.Model(inputs, outputs)

# Phase 1: Train classification head
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\n--- Phase 1: Training Classification Head ---")
model.fit(ds_train, validation_data=ds_val, epochs=10)

# Phase 2: Fine-Tuning deep layers
base_model.trainable = True
for layer in base_model.layers[:-40]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\n--- Phase 2: Fine-Tuning Model ---")
model.fit(ds_train, validation_data=ds_val, epochs=10)

# 4. Save Final Model
output_path = os.path.join(os.path.dirname(__file__), 'flower_cnn_model.h5')
model.save(output_path)
print(f"\nModel saved successfully at: {output_path}")
print(f"Species labels saved at: {labels_path}")