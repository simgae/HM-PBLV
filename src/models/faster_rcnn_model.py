import datetime
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard


def preprocess_dataset(example):
    """
    Preprocess an example from the KITTI dataset.

    Parameters:
    -----------
    example : dict
        A dictionary representing a single example from the KITTI dataset. It must include at least the 'image' key.

    Returns:
    --------
    tuple:
        A tuple (image, (class_label, bbox_label)) where:
          - image: The preprocessed image (resized to 128x128 and normalized).
          - class_label: A dummy one-hot vector for classification (50 values: 10 objects x 5 classes).
          - bbox_label: A dummy bounding box regression target for 10 bounding boxes (each with 4 coordinates, flattened to 40 values).
    """
    # Load and preprocess the image
    image = example['image']
    image = tf.image.resize(image, (128, 128))
    image = image / 255.0

    # Dummy classification label: 50 values (10 objects × 5 classes)
    # Each object gets a one-hot vector of length 5, concatenated together
    class_label = tf.concat([tf.one_hot(0, depth=5, dtype=tf.float32) for _ in range(10)], axis=0)
    
    # Dummy bounding box label remains the same
    bbox_label = tf.constant([0.0, 0.0, 0.1, 0.1] * 10, dtype=tf.float32)

    return image, (class_label, bbox_label)


class ProgressCallback(tf.keras.callbacks.Callback):
    def on_train_batch_end(self, batch, logs=None):
        if logs and 'loss' in logs:
            print(f"Completed batch {batch} - Loss: {logs['loss']:.4f}")

    def on_epoch_end(self, epoch, logs=None):
        if logs:
            print(
                f"Epoch {epoch+1} completed - "
                f"Loss: {logs.get('loss'):.4f}, "
                f"Val Loss: {logs.get('val_loss'):.4f}"
            )


class FasterRCNNModel:
    """
    A minimal demonstration of a Faster R-CNN–style model with two output heads:
      - rpn_cls: Classification head for predicting classes for 10 objects using softmax.
      - rpn_reg: Bounding box regression head for predicting 10 bounding boxes (each with 4 coordinates, total 40 values).

    Public Attributes:
    ------------------
      train_dataset : tf.data.Dataset
          Preprocessed and batched training dataset.
      val_dataset : tf.data.Dataset
          Preprocessed and batched validation dataset.
      test_dataset : tf.data.Dataset
          Preprocessed and batched test dataset.
      model : tf.keras.Model
          The compiled Keras model with a ResNet50 backbone.
    """

    def __init__(self):
        # Load the KITTI dataset via TFDS
        self.train_dataset = tfds.load('kitti', split='train')
        self.val_dataset   = tfds.load('kitti', split='validation')
        self.test_dataset  = tfds.load('kitti', split='test')

        # Preprocess & batch the datasets
        self.train_dataset = self.train_dataset.map(preprocess_dataset).batch(32)
        self.val_dataset   = self.val_dataset.map(preprocess_dataset).batch(32)
        self.test_dataset  = self.test_dataset.map(preprocess_dataset).batch(32)

        # Build a simple backbone with ResNet50 (frozen)
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=(128, 128, 3)
        )
        base_model.trainable = False

        # Add flatten and two output heads
        x = base_model.output
        x = Flatten()(x)

        # Output 1: classification head for 10 objects × 5 classes = 50 outputs using softmax activation
        # Reshape to (10, 5) before applying softmax to get probability distribution for each object
        rpn_cls = Dense(50, activation=None, name='rpn_cls_dense')(x)
        rpn_cls = tf.reshape(rpn_cls, (-1, 10, 5))
        rpn_cls = tf.nn.softmax(rpn_cls, axis=-1)
        rpn_cls = tf.reshape(rpn_cls, (-1, 50), name='rpn_cls')

        # Output 2: bounding box regression head for 10 bounding boxes (each with 4 coordinates, total 40 values).
        rpn_bbox = Dense(40, activation='linear', name='rpn_reg')(x)

        # Create and compile the model
        self.model = Model(inputs=base_model.input, outputs=[rpn_cls, rpn_bbox])
        self.model.compile(
            optimizer='adam',
            loss=['categorical_crossentropy', 'mean_squared_error']
        )

    def train_model(self, epochs=2):
        """
        Train the model for a given number of epochs.

        Parameters:
        -----------
        epochs : int
            The number of epochs to train the model.
        """
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
        progress_callback = ProgressCallback()

        self.model.fit(
            self.train_dataset,
            epochs=epochs,
            validation_data=self.val_dataset,
            callbacks=[tensorboard_callback, progress_callback]
        )
        self._save_model()

    def evaluate_model(self):
        """
        Evaluate the model on the test dataset.
        """
        test_loss = self.model.evaluate(self.test_dataset)
        print(f"Test Loss: {test_loss}")

    def _save_model(self):
        """
        Save the trained model to disk.
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.model.save(f'faster_rcnn_model_{timestamp}.keras')