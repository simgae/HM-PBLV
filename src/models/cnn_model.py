import tensorflow_datasets as tfds
from tensorflow import keras

from src.utils import preprocess_dataset

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
    keras.layers.Dense(40)  # Output layer for bounding box coordinates (10 bounding boxes * 4 coordinates each)
])


# Split the dataset into training, validation, and test sets
train_dataset = dataset.take(int(0.8 * len(dataset)))
val_dataset = dataset.skip(int(0.8 * len(dataset))).take(int(0.1 * len(dataset)))
test_dataset = dataset.skip(int(0.9 * len(dataset)))

# Preprocess the validation and test datasets
train_dataset = train_dataset.map(preprocess_dataset).batch(32)
val_dataset = val_dataset.map(preprocess_dataset).batch(32)
test_dataset = test_dataset.map(preprocess_dataset).batch(32)

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the neural network model using the preprocessed dataset
model.fit(train_dataset, epochs=10, validation_data=val_dataset)

# Evaluate the model's performance on the test dataset
test_loss = model.evaluate(test_dataset)
print(f"Test Loss: {test_loss}")
