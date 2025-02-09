import datetime

import tensorflow_datasets as tfds
from keras.src.losses import binary_crossentropy
from tensorflow.keras.utils import register_keras_serializable
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard

from src.utils import preprocess_dataset

import tensorflow as tf

@register_keras_serializable()
def yolo_v3_loss(y_true, y_pred):
    y_pred = tf.reshape(y_pred, tf.shape(y_true))

    # Calculate the class loss
    pred_class = y_pred[..., 5:]
    object_mask = y_true[..., 4:5]
    true_class_probs = y_true[..., 5:]
    pred_class = tf.reshape(pred_class, tf.shape(true_class_probs))
    object_mask = tf.squeeze(object_mask, -1)
    class_loss = object_mask * binary_crossentropy(true_class_probs, pred_class)

    # Calculate the bbox loss
    pred_box = y_pred[..., 0:4]
    true_box = y_true[..., 0:4]

    # Calculate the overall loss
    box_loss = object_mask * tf.reduce_sum(tf.square(true_box - pred_box), axis=-1)

    return tf.reduce_sum(class_loss) + box_loss


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

    def _create_darknet_conv(self, x, filters, kernel_size, strides=1):
        if strides == 2:
            x = keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(x)
        x = keras.layers.Conv2D(filters,
                                kernel_size,
                                strides=strides,
                                padding='same' if strides == 1 else 'valid',
                                use_bias=False)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.LeakyReLU(alpha=0.1)(x)
        return x

    def _create_residual_block(self, inputs, filters):
        shortcut = inputs
        inputs = self._create_darknet_conv(inputs, filters // 2, 1)
        inputs = self._create_darknet_conv(inputs, filters, 3)
        inputs = keras.layers.Add()([shortcut, inputs])
        return inputs

    def _create_backbone_darknet53(self, inputs):
        x = self._create_darknet_conv(inputs, 32, 3)
        x = self._create_darknet_conv(x, 64, 3, strides=2)
        x = self._create_residual_block(x, 64)

        x = self._create_darknet_conv(x, 128, 3, strides=2)
        for _ in range(2):
            x = self._create_residual_block(x, 128)

        x = self._create_darknet_conv(x, 256, 3, strides=2)
        for _ in range(8):
            x = self._create_residual_block(x, 256)
        route1 = x

        x = self._create_darknet_conv(x, 512, 3, strides=2)
        for _ in range(8):
            x = self._create_residual_block(x, 512)
        route2 = x

        x = self._create_darknet_conv(x, 1024, 3, strides=2)
        for _ in range(4):
            x = self._create_residual_block(x, 1024)
        route3 = x

        return route1, route2, route3

    def _create_detection_head(self, inputs, out_filters):

        # Create the detection head with convolutional layers
        inputs = self._create_darknet_conv(inputs, 512, 1)
        inputs = self._create_darknet_conv(inputs, 1024, 3)
        inputs = self._create_darknet_conv(inputs, 512, 1)
        inputs = self._create_darknet_conv(inputs, 1024, 3)
        inputs = self._create_darknet_conv(inputs, 512, 1)
        out = self._create_darknet_conv(inputs, out_filters, 1)

        return inputs, out

    def build_yolo_v3_model(self):
        inputs = keras.layers.Input(shape=self.image_shape)
        route1, route2, route3 = self._create_backbone_darknet53(inputs)

        # Large scale
        x, y1 = self._create_detection_head(route3, 3 * (self.num_classes + 5))

        # Medium scale
        x = self._create_darknet_conv(x, 256, 1)
        x = keras.layers.UpSampling2D(2)(x)
        x = keras.layers.Concatenate(axis=-1)([x, route2])
        x, y2 = self._create_detection_head(x, 3 * (self.num_classes + 5))

        # Small scale
        x = self._create_darknet_conv(x, 128, 1)
        x = keras.layers.UpSampling2D(2)(x)
        x = keras.layers.Concatenate(axis=-1)([x, route1])
        _, y3 = self._create_detection_head(x, 3 * (self.num_classes + 5))

        return keras.models.Model(inputs, [y1, y2, y3])

    def __init__(self):
        """
        Initializes the YOLO v3 model by loading the dataset, defining the model architecture,
        and splitting the dataset into training, validation, and test sets.
        """
        self.num_classes = 3
        self.image_shape = (416, 416, 3)

        self.model = self.build_yolo_v3_model()

        self.model.summary()

        # Load the KITTI dataset
        self.train_dataset = tfds.load('kitti', split='train')
        self.val_dataset = tfds.load('kitti', split='validation')
        self.test_dataset = tfds.load('kitti', split='test')

        # Preprocess the validation and test datasets
        self.train_dataset = self.train_dataset.map(preprocess_dataset).batch(32)
        self.val_dataset = self.val_dataset.map(preprocess_dataset).batch(32)
        self.test_dataset = self.test_dataset.map(preprocess_dataset).batch(32)

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

        self.model.compile(optimizer='adam',
                           loss=[yolo_v3_loss, yolo_v3_loss, yolo_v3_loss])

        self.model.fit(self.train_dataset, epochs=epochs, validation_data=self.val_dataset,
                       callbacks=[tensorboard_callback])

        self._save_model()

    def evaluate_model(self):
        """
        Evaluates the YOLO v3 model's performance on the test dataset.
        """
        test_loss = self.model.evaluate(self.test_dataset)
        print(f"Test Loss: {test_loss}")

    def load_model(self):
        """
        Loads a trained YOLO v3 model from a file.
        """
        self.model = keras.models.load_model('yolo_v3_model.keras',
                                             custom_objects={'yolo_v3_loss': yolo_v3_loss})

    def evaluate_image(self, image_path):
        """
        Evaluates the YOLO v3 model's performance on a single image.

        Parameters
        ----------
        image_path : str
            the path to the image
        """
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, (416, 416))
        image = tf.expand_dims(image, 0)

        # predict bbox and class
        predictions = self.model.predict(image)

        # add prediction to image
        bbox = predictions[0][0][..., :4]
        class_probs = predictions[0][0][..., 5:]

        # define colors
        colors = tf.constant([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

        # add bbox to image
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)


        bbox = tf.reshape(bbox, [-1, 4])  # Flatten the bbox tensor
        bbox = tf.expand_dims(bbox, 0)  # Add batch dimension

        # ensure bbox coordinates are in the correct order
        bbox = tf.stack([
            tf.minimum(bbox[..., 0], bbox[..., 2]),  # y_min
            tf.minimum(bbox[..., 1], bbox[..., 3]),  # x_min
            tf.maximum(bbox[..., 0], bbox[..., 2]),  # y_max
            tf.maximum(bbox[..., 1], bbox[..., 3])  # x_max
        ], axis=-1)

        # draw the bounding box on the image
        image_with_boxes = tf.image.draw_bounding_boxes(
            image,
            bbox,
            colors
        )

        # convert image to uint8
        image_with_boxes = tf.cast(image_with_boxes, tf.uint8)

        # save image
        tf.io.write_file('output.jpg', tf.image.encode_jpeg(tf.squeeze(image_with_boxes, 0)))

    def _save_model(self):
        """
        Saves the trained YOLO v3 model to a file.
        """
        self.model.save('yolo_v3_model.keras')
