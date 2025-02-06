import datetime

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.utils import register_keras_serializable
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard

from src.utils import preprocess_dataset


def compute_iou(pred_xy, pred_wh, true_xy, true_wh):
    """
    Compute IOU between predicted boxes and ground-truth boxes for each anchor cell.

    pred_xy: shape [batch, gy, gx, anchors, 2]
    pred_wh: shape [batch, gy, gx, anchors, 2]
    true_xy: shape [batch, gy, gx, anchors, 2]
    true_wh: shape [batch, gy, gx, anchors, 2]

    returns iou_scores: shape [batch, gy, gx, anchors]
    """
    # Convert center+wh to top-left, bottom-right
    pred_box_min = pred_xy - pred_wh / 2.0
    pred_box_max = pred_xy + pred_wh / 2.0
    true_box_min = true_xy - true_wh / 2.0
    true_box_max = true_xy + true_wh / 2.0

    # Intersection
    intersect_min = tf.maximum(pred_box_min, true_box_min)
    intersect_max = tf.minimum(pred_box_max, true_box_max)
    intersect_wh = tf.maximum(intersect_max - intersect_min, 0.0)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]

    # Union
    pred_area = pred_wh[..., 0] * pred_wh[..., 1]
    true_area = true_wh[..., 0] * true_wh[..., 1]
    union_area = pred_area + true_area - intersect_area
    return intersect_area / tf.maximum(union_area, 1e-9)


@register_keras_serializable()
def yolo_v3_loss(y_true, y_pred, anchors, num_classes, ignore_thresh=0.5):
    """
    Closer to the standard YOLOv3 loss, including anchor offsets and confidence mask.

    y_true: Ground truth with shape [batch_size, grid_h, grid_w, num_anchors, 5 + num_classes]
        Each anchor in each cell has:
            [x, y, w, h, objectness, class_onehot...]
    y_pred: Model prediction of same shape
        Each anchor in each cell has:
            [tx, ty, tw, th, confidence, class_probs...]
    anchors: List of anchor sizes for this scale, e.g. [(w1,h1),(w2,h2),(w3,h3)]
    num_classes: Total number of classes
    ignore_thresh: Confidence threshold for ignoring objectness loss

    Returns scalar loss tensor.
    """

    # Separate out predictions
    pred_xy = y_pred[..., 0:2]  # tx, ty
    pred_wh = y_pred[..., 2:4]  # tw, th
    pred_conf = y_pred[..., 4:5]
    pred_class = y_pred[..., 5:]

    # Separate out ground truth
    true_xy = y_true[..., 0:2]
    true_wh = y_true[..., 2:4]
    object_mask = y_true[..., 4:5]
    true_class_probs = y_true[..., 5:]

    # Compute grid shape
    grid_shape = tf.shape(y_true)[1:3]  # (gy, gx)
    grid_shape_f = tf.cast(grid_shape, tf.float32)

    # Compute absolute box coords from network outputs
    # YOLOv3: b_xy = (sigmoid(tx) + cx) / grid_w, b_wh = anchors * exp(tw) / grid_w
    grid_y = tf.range(0, grid_shape[0], dtype=tf.int32)
    grid_x = tf.range(0, grid_shape[1], dtype=tf.int32)
    grid_x, grid_y = tf.meshgrid(grid_x, grid_y)
    grid = tf.cast(tf.stack([grid_x, grid_y], axis=-1), tf.float32)

    # Expand/reshape for broadcasting
    grid = tf.expand_dims(grid, 2)  # shape [gy, gx, 1, 2]
    anchors_tf = tf.cast(anchors, tf.float32)  # shape [num_anchors, 2]

    pred_xy_sig = tf.sigmoid(pred_xy) + grid  # offset by grid
    pred_xy_sig = pred_xy_sig / grid_shape_f[::-1]  # normalize by input size
    pred_wh_exp = tf.exp(pred_wh) * anchors_tf  # scale by anchors
    pred_wh_exp = pred_wh_exp / grid_shape_f[::-1]

    # Build iou mask for ignoring predicted boxes that overlap ground truth
    # with high IOU but shouldn’t contribute to conf loss
    iou_scores = compute_iou(pred_xy_sig, pred_wh_exp, true_xy, true_wh)
    best_ious = tf.reduce_max(iou_scores, axis=-1, keepdims=True)
    ignore_mask = tf.cast(best_ious < ignore_thresh, tf.float32)

    # Box loss: MSE for center + size, weighted by object mask
    box_loss_scale = 2.0 - (true_wh[..., 0] * true_wh[..., 1])
    xy_loss = object_mask * box_loss_scale * tf.square(true_xy - (tf.sigmoid(pred_xy) + grid) / grid_shape_f[::-1])
    wh_loss = object_mask * box_loss_scale * tf.square(true_wh - tf.exp(pred_wh) * anchors_tf / grid_shape_f[::-1])
    box_loss = tf.reduce_sum(xy_loss + wh_loss)

    # Confidence loss: BCE on object_mask, ignoring boxes with big IOU
    conf_obj_loss = object_mask * tf.keras.losses.binary_crossentropy(object_mask, tf.sigmoid(pred_conf))
    conf_noobj_loss = (1 - object_mask) * ignore_mask * tf.keras.losses.binary_crossentropy(object_mask,
                                                                                            tf.sigmoid(pred_conf))
    conf_loss = tf.reduce_sum(conf_obj_loss + conf_noobj_loss)

    # Class loss: BCE for class probabilities, multiplied by object mask
    class_loss = object_mask * tf.keras.losses.binary_crossentropy(true_class_probs, tf.sigmoid(pred_class))
    class_loss = tf.reduce_sum(class_loss)

    total_loss = box_loss + conf_loss + class_loss
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

    def _save_model(self):
        """
        Saves the trained YOLO v3 model to a file.
        """
        self.model.save('yolo_v3_model.keras')
