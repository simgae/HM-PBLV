import datetime

import tensorflow_datasets as tfds
from keras.src.losses import binary_crossentropy
from tensorflow.keras.utils import register_keras_serializable
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard

from src.utils import preprocess_dataset

import tensorflow as tf

def calculate_iou(box1, box2):
    # Calculate the intersection coordinates
    x1 = tf.maximum(box1[0], box2[0])
    y1 = tf.maximum(box1[1], box2[1])
    x2 = tf.minimum(box1[2], box2[2])
    y2 = tf.minimum(box1[3], box2[3])

    # Calculate the area of the intersection
    intersection_area = tf.maximum(0.0, x2 - x1) * tf.maximum(0.0, y2 - y1)

    # Calculate the area of both bounding boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Calculate the union area
    union_area = box1_area + box2_area - intersection_area

    # Calculate the IoU
    iou = intersection_area / union_area

    return iou

def filter_boxes(boxes, threshold=0.5):
    filtered_boxes = []
    for i in range(len(boxes)):
        discard = False
        for j in range(i + 1, len(boxes)):
            iou = calculate_iou(boxes[i], boxes[j])
            if iou < threshold:
                discard = True
                break
        if not discard:
            filtered_boxes.append(boxes[i])
    return filtered_boxes


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
    box_loss = box_loss * y_pred[..., 4]

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
        try:
            self.model = keras.models.load_model('./src/yolo_v3_model.keras',
                                                 custom_objects={'yolo_v3_loss': yolo_v3_loss})
        except ValueError as e:
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
        # define colors
        colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]

        original_image = tf.io.read_file(image_path)
        original_image = tf.image.decode_jpeg(original_image, channels=3)

        # resize image
        image = tf.image.resize(original_image, (416, 416))
        image = tf.expand_dims(image, 0)

        # predict bbox and class
        predictions = self.model.predict(image)

        # convert image to float32
        image = tf.image.convert_image_dtype(original_image, dtype=tf.float32)

        # add prediction to image
        bbox = predictions[0][0][..., :4]
        objectiveness = predictions[0][0][..., 4]

        # get bbox with highest objectiveness
        bbox = tf.expand_dims(tf.gather(bbox, tf.argmax(objectiveness)), 0)

        # reshape bbox tensor
        bbox = tf.reshape(bbox, [-1, 4])

        # filter boxes with iou threshold
        bbox = filter_boxes(bbox, threshold=0.5)

        # add half of the image width and height to the x and y coordinates
        # because the bbox coordinates are normalized to the center of the image
        bbox += tf.constant([0.5, 0.5, 0.5, 0.5])

        bbox = tf.stack([
            tf.minimum(tf.convert_to_tensor(bbox)[:, 0], tf.convert_to_tensor(bbox)[:, 2]),
            tf.minimum(tf.convert_to_tensor(bbox)[:, 1], tf.convert_to_tensor(bbox)[:, 3]),
            tf.maximum(tf.convert_to_tensor(bbox)[:, 0], tf.convert_to_tensor(bbox)[:, 2]),
            tf.maximum(tf.convert_to_tensor(bbox)[:, 1], tf.convert_to_tensor(bbox)[:, 3]),
        ], axis=-1)

        # draw the bounding box on the image
        image = tf.image.draw_bounding_boxes(
            tf.expand_dims(image, 0),
            tf.expand_dims(bbox, 0),
            colors
        )

        image = tf.squeeze(image, 0)

        encoded_image = tf.image.encode_png(tf.image.convert_image_dtype(image, dtype=tf.uint8))

        # save image
        tf.io.write_file('output_yolo_v3.jpg', encoded_image)

    def _save_model(self):
        """
        Saves the trained YOLO v3 model to a file.
        """
        self.model.save('yolo_v3_model.keras')
