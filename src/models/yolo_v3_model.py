import datetime

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.losses import MeanSquaredError, CategoricalCrossentropy
from tensorflow.keras.utils import register_keras_serializable
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard

from src.utils import preprocess_dataset, postprocess_predictions


@register_keras_serializable()
def yolo_loss(y_true, y_pred):
    """
    Computes the YOLO loss, which is a combination of mean squared error for bounding box coordinates
    and categorical cross-entropy for class probabilities.

    Parameters
    ----------
    y_true : tensor
        The ground truth values, with the first 40 values being bounding box coordinates and the remaining values being class probabilities.
    y_pred : tensor
        The predicted values, with the first 40 values being bounding box coordinates and the remaining values being class probabilities.

    Returns
    -------
    total_loss : tensor
        The combined loss value.
    """
    # Split the true and predicted values into bounding box coordinates and class probabilities
    y_true_boxes = y_true[:, :40]  # First 40 values are bounding box coordinates
    y_true_classes = y_true[:, 40:]  # Remaining values are class probabilities

    y_pred_boxes = y_pred[:, :40]
    y_pred_classes = y_pred[:, 40:]

    # Calculate the mean squared error for the bounding box coordinates
    mse = MeanSquaredError()
    box_loss = mse(y_true_boxes, y_pred_boxes)

    # Calculate the categorical cross-entropy for the class probabilities
    cce = CategoricalCrossentropy()
    class_loss = cce(y_true_classes, y_pred_classes)

    # Combine the two losses
    total_loss = box_loss + class_loss
    return total_loss

class YoloV3Model:
    """
    A class used to represent a YOLO v3 model for object detection and classification.

    Attributes
    ----------
    dataset : tf.data.Dataset
        the dataset used for training the model
    info : tfds.core.DatasetInfo
        information about the dataset
    model : keras.Sequential
        the YOLO v3 model architecture
    train_dataset : tf.data.Dataset
        the training dataset
    val_dataset : tf.data.Dataset
        the validation dataset
    test_dataset : tf.data.Dataset
        the test dataset
    """

    def __init__(self):
        """
        Initializes the YOLO v3 model by loading the dataset, defining the model architecture,
        and splitting the dataset into training, validation, and test sets.
        """
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
            keras.layers.Conv2D(50, (1, 1), padding='same', activation='sigmoid'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(50)  # Output layer for bounding box coordinates and class probabilities (40 + 10)
        ])

        # Load the KITTI dataset
        self.train_dataset = tfds.load('kitti', split='train')
        self.val_dataset = tfds.load('kitti', split='validation')
        self.test_dataset = tfds.load('kitti', split='test')

        # Preprocess the validation and test datasets
        self.train_dataset = self.train_dataset.map(preprocess_dataset).batch(32)
        self.val_dataset = self.val_dataset.map(preprocess_dataset).batch(32)
        self.test_dataset = self.test_dataset.map(preprocess_dataset).batch(32)

        # Compile the model
        self.model.compile(optimizer='adam', loss=yolo_loss)

    def train_model(self, epochs=2):
        """
        Trains the YOLO v3 model using the preprocessed training dataset and validates it using the validation dataset.

        Parameters
        ----------
        epochs : int
            the number of epochs to train the model for
        """
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

        self.model.fit(self.train_dataset, epochs=epochs, validation_data=self.val_dataset,
                       callbacks=[tensorboard_callback])
        self._save_model()

    def evaluate_model(self):
        """
        Evaluates the YOLO v3 model's performance on the test dataset.
        """
        test_loss = self.model.evaluate(self.test_dataset)
        print(f"Test Loss: {test_loss}")

    def _save_model(self):
        """
        Saves the trained YOLO v3 model to a file.
        """
        self.model.save('yolo_v3_model.keras')

    def _evaluate_image(self, path_to_image: str):
        """
        Evaluates the YOLO v3 model's performance on a single image.

        Parameters
        ----------
        path_to_image : str
            the path to the image file
        """
        self.model = keras.models.load_model('yolo_v3_model.keras')
        image = tf.io.read_file(path_to_image)
        image = tf.image.decode_image(image, channels=3)
        image = tf.image.resize(image, (128, 128))
        image = tf.expand_dims(image, axis=0)

        predictions = self.model.predict(image)

        predictions = postprocess_predictions(predictions[0], image.shape)
        print(predictions)
