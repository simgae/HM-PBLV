import datetime
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard


def preprocess_dataset(example):
    """
    Preprocess 'example' from the KITTI dataset and return (image, (class_label, bbox_label)).

    For demonstration:
    - We create a dummy 'class_label' (binary: 0) and 'bbox_label' (4 zeros).
    - Real code should extract actual labels from 'example'.
    """
    # Load image
    image = example['image']
    # Resize and normalize
    image = tf.image.resize(image, (128, 128))
    image = image / 255.0

    # Dummy classification label (0 or 1). Here just 0:
    class_label = tf.constant(0.0, dtype=tf.float32)
    # Dummy bounding box label (4 coords: [x1, y1, x2, y2]):
    bbox_label = tf.constant([0.0, 0.0, 0.1, 0.1], dtype=tf.float32)

    # Return the image, plus a tuple with both labels
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
    Minimal demonstration of a Faster R-CNN–style model with two heads:
      - A classification (rpn_cls) output
      - A bounding box regression (rpn_reg) output

    We supply matching label tuples (class_label, bbox_label) to avoid structure mismatch.
    """

    def __init__(self):
        # 1) Load the KITTI dataset via TFDS
        self.train_dataset = tfds.load('kitti', split='train')
        self.val_dataset   = tfds.load('kitti', split='validation')
        self.test_dataset  = tfds.load('kitti', split='test')

        # 2) Preprocess & batch
        self.train_dataset = self.train_dataset.map(preprocess_dataset).batch(32)
        self.val_dataset   = self.val_dataset.map(preprocess_dataset).batch(32)
        self.test_dataset  = self.test_dataset.map(preprocess_dataset).batch(32)

        # 3) Build a simple backbone with ResNet50 (frozen)
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=(128, 128, 3)
        )
        base_model.trainable = False

        # 4) Add flatten and two outputs:
        x = base_model.output
        x = Flatten()(x)

        # Output 1: classification (binary, i.e. 1 neuron with sigmoid)
        rpn_cls = Dense(1, activation='sigmoid', name='rpn_cls')(x)
        # Output 2: bounding box coords (4 floats)
        rpn_bbox = Dense(4, activation='linear', name='rpn_reg')(x)

        # 5) Create and compile the model
        self.model = Model(inputs=base_model.input, outputs=[rpn_cls, rpn_bbox])
        # We have 2 outputs => supply 2 losses in the same order
        self.model.compile(
            optimizer='adam',
            loss=['binary_crossentropy', 'mean_squared_error']
        )

    def train_model(self, epochs=2):
        """
        Train the model for a given number of epochs.
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
        self.model.save('faster_rcnn_model.keras')


if __name__ == '__main__':
    faster_rcnn_model = FasterRCNNModel()
    faster_rcnn_model.train_model(epochs=2)
    faster_rcnn_model.evaluate_model()