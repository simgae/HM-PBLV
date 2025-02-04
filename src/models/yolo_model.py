import datetime

import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard

from src.utils import preprocess_dataset


class YoloModel:
    """
    A class used to represent a YOLO model for object detection.

    Attributes
    ----------
    dataset : tf.data.Dataset
        the dataset used for training the model
    info : tfds.core.DatasetInfo
        information about the dataset
    model : keras.Sequential
        the YOLO model architecture
    train_dataset : tf.data.Dataset
        the training dataset
    val_dataset : tf.data.Dataset
        the validation dataset
    test_dataset : tf.data.Dataset
        the test dataset
    """

    def __init__(self):
        """
        Initializes the YOLO model by loading the dataset, defining the model architecture,
        and splitting the dataset into training, validation, and test sets.
        """
        # Load the KITTI dataset
        self.dataset, self.info = tfds.load('kitti', split='train', with_info=True)

        self.model = keras.Sequential([
            keras.layers.Input(shape=(128, 128, 3)),
            # Backbone: CSPDarknet53
            keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(1024, (3, 3), padding='same', activation='relu'),
            # YOLO Head
            keras.layers.Conv2D(512, (1, 1), padding='same', activation='relu'),
            keras.layers.Conv2D(1024, (3, 3), padding='same', activation='relu'),
            keras.layers.Conv2D(512, (1, 1), padding='same', activation='relu'),
            keras.layers.Conv2D(1024, (3, 3), padding='same', activation='relu'),
            keras.layers.Conv2D(512, (1, 1), padding='same', activation='relu'),
            keras.layers.Conv2D((8 + 5) * 3, (1, 1), padding='same', activation='sigmoid'),

            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(40)  # Output layer for bounding box coordinates (10 bounding boxes * 4 coordinates each)
        ])

        # Split the dataset into training, validation, and test sets
        self.train_dataset = self.dataset.take(int(0.8 * len(self.dataset)))
        self.val_dataset = self.dataset.skip(int(0.8 * len(self.dataset))).take(int(0.1 * len(self.dataset)))
        self.test_dataset = self.dataset.skip(int(0.9 * len(self.dataset)))

        # Preprocess the validation and test datasets
        self.train_dataset = self.train_dataset.map(preprocess_dataset).batch(32)
        self.val_dataset = self.val_dataset.map(preprocess_dataset).batch(32)
        self.test_dataset = self.test_dataset.map(preprocess_dataset).batch(32)

        # Compile the model
        self.model.compile(optimizer='adam', loss='mean_squared_error')

    def train_model(self):
        """
        Trains the YOLO model using the preprocessed training dataset and validates it using the validation dataset.
        """
        # Train the neural network model using the preprocessed dataset
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

        self.model.fit(self.train_dataset, epochs=2, validation_data=self.val_dataset, callbacks=[tensorboard_callback])
        self._save_model()

    def evaluate_model(self):
        """
        Evaluates the YOLO model's performance on the test dataset.
        """
        # Evaluate the model's performance on the test dataset
        test_loss = self.model.evaluate(self.test_dataset)
        print(f"Test Loss: {test_loss}")

    def _save_model(self):
        """
        Saves the trained YOLO model to a file.
        """
        self.model.save('yolo_model.keras')
