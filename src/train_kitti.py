import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from src.utils import handle_shape_mismatch, normalize_bboxes, convert_bboxes_to_fixed_size_tensor

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
    keras.layers.Dense(40), # 10 boxes x 4 coords 
    keras.layers.Reshape((10, 4))
])

# Preprocess the dataset for training
def preprocess(data):
    image = data['image']
    bbox = data['objects']['bbox']
    image = tf.image.resize(image, (128, 128))
    bbox = tf.reshape(bbox, [-1, 4])  # Ensure bbox shape is consistent
    bbox = handle_shape_mismatch(bbox)  # Handle variable number of bounding boxes
    bbox = normalize_bboxes(bbox, image.shape)  # Normalize bounding box coordinates
    bbox = convert_bboxes_to_fixed_size_tensor(bbox)  # Convert to fixed size tensor
    print(f"Image shape: {image.shape}, BBox shape: {bbox.shape}")  # Debugging statement
    return image, bbox

train_dataset = dataset.map(preprocess).batch(32)

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the neural network model using the preprocessed dataset
model.fit(train_dataset, epochs=10)
