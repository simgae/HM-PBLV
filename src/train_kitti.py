import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from src.utils import handle_shape_mismatch

# Load the KITTI dataset
dataset, info = tfds.load('kitti', split='train', with_info=True)

# Define a neural network model for 2D object detection
model = keras.Sequential([
    keras.layers.Input(shape=(128, 128, 3)),
    keras.layers.Conv2D(32, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(4)  # Output layer for bounding box coordinates
])

# Preprocess the dataset for training
def preprocess(data):
    image = data['image']
    bbox = data['objects']['bbox']
    image = tf.image.resize(image, (128, 128))
    bbox = tf.reshape(bbox, [-1, 4])  # Ensure bbox shape is consistent
    print(f"Image shape: {image.shape}, BBox shape: {bbox.shape}")  # Debugging statement
    return image, bbox

train_dataset = dataset.map(preprocess).batch(32)

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the neural network model using the preprocessed dataset
try:
    model.fit(train_dataset, epochs=10)
except tf.errors.InvalidArgumentError as e:
    if 'Shapes' in str(e):
        train_dataset = train_dataset.map(lambda x, y: (x, handle_shape_mismatch(y, 32)))
        model.fit(train_dataset, epochs=10)
