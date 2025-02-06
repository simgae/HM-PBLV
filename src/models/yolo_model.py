import tensorflow as tf
import keras
from keras import layers, Model

class YoloV3(keras.Model):
    """
    A class used to represent a YOLOv3 model for object detection.
    """

    def __init__(self, num_classes=3):
        """
        Initializes YOLOv3 by loading KITTI, creating the model architecture,
        and splitting the dataset into train, val, and test.
        """
        super().__init__()
        self.num_classes = num_classes

        # Build YOLOv3 model
        self.yolo_model = self._build_yolov3(self.num_classes)

    def call(self, inputs):
        return self.yolo_model(inputs)

    def _darknet_conv(self, x, filters, kernel_size, strides=1):
        if strides == 2:
            x = layers.ZeroPadding2D(((1, 0), (1, 0)))(x)
        x = layers.Conv2D(filters,
                          kernel_size,
                          strides=strides,
                          padding='same' if strides == 1 else 'valid',
                          use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.1)(x)
        return x

    def _residual_block(self, x, filters):
        shortcut = x
        x = self._darknet_conv(x, filters//2, 1)
        x = self._darknet_conv(x, filters, 3)
        x = layers.Add()([shortcut, x])
        return x

    def _darknet53_backbone(self, inputs):
        x = self._darknet_conv(inputs, 32, 3)
        x = self._darknet_conv(x, 64, 3, strides=2)
        x = self._residual_block(x, 64)

        x = self._darknet_conv(x, 128, 3, strides=2)
        for _ in range(2):
            x = self._residual_block(x, 128)

        x = self._darknet_conv(x, 256, 3, strides=2)
        for _ in range(8):
            x = self._residual_block(x, 256)
        route1 = x

        x = self._darknet_conv(x, 512, 3, strides=2)
        for _ in range(8):
            x = self._residual_block(x, 512)
        route2 = x

        x = self._darknet_conv(x, 1024, 3, strides=2)
        for _ in range(4):
            x = self._residual_block(x, 1024)
        route3 = x

        return route1, route2, route3

    def _yolo_head(self, x, out_filters):
        x = self._darknet_conv(x, 512, 1)
        x = self._darknet_conv(x, 1024, 3)
        x = self._darknet_conv(x, 512, 1)
        x = self._darknet_conv(x, 1024, 3)
        x = self._darknet_conv(x, 512, 1)
        out = self._darknet_conv(x, out_filters, 1)
        return x, out

    def _build_yolov3(self, num_classes):
        inputs = layers.Input([416, 416, 3])
        route1, route2, route3 = self._darknet53_backbone(inputs)

        # Large scale
        x, y1 = self._yolo_head(route3, 3 * (num_classes + 5))

        # Medium scale
        x = self._darknet_conv(x, 256, 1)
        x = layers.UpSampling2D(2)(x)
        x = tf.concat([x, route2], axis=-1)
        x, y2 = self._yolo_head(x, 3 * (num_classes + 5))

        # Small scale
        x = self._darknet_conv(x, 128, 1)
        x = layers.UpSampling2D(2)(x)
        x = tf.concat([x, route1], axis=-1)
        _, y3 = self._yolo_head(x, 3 * (num_classes + 5))

        return Model(inputs, [y1, y2, y3])

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
    
    pred_xy_sig = tf.sigmoid(pred_xy) + grid      # offset by grid
    pred_xy_sig = pred_xy_sig / grid_shape_f[::-1]  # normalize by input size
    pred_wh_exp = tf.exp(pred_wh) * anchors_tf    # scale by anchors
    pred_wh_exp = pred_wh_exp / grid_shape_f[::-1]
    
    # Build iou mask for ignoring predicted boxes that overlap ground truth
    # with high IOU but shouldn’t contribute to conf loss
    iou_scores = compute_iou(pred_xy_sig, pred_wh_exp, true_xy, true_wh)
    best_ious = tf.reduce_max(iou_scores, axis=-1, keepdims=True)
    ignore_mask = tf.cast(best_ious < ignore_thresh, tf.float32)
    
    # Box loss: MSE for center + size, weighted by object mask
    box_loss_scale = 2.0 - (true_wh[...,0] * true_wh[...,1])
    xy_loss = object_mask * box_loss_scale * tf.square(true_xy - (tf.sigmoid(pred_xy) + grid) / grid_shape_f[::-1])
    wh_loss = object_mask * box_loss_scale * tf.square(true_wh - tf.exp(pred_wh) * anchors_tf / grid_shape_f[::-1])
    box_loss = tf.reduce_sum(xy_loss + wh_loss)
    
    # Confidence loss: BCE on object_mask, ignoring boxes with big IOU
    conf_obj_loss = object_mask * tf.keras.losses.binary_crossentropy(object_mask, tf.sigmoid(pred_conf))
    conf_noobj_loss = (1 - object_mask) * ignore_mask * tf.keras.losses.binary_crossentropy(object_mask, tf.sigmoid(pred_conf))
    conf_loss = tf.reduce_sum(conf_obj_loss + conf_noobj_loss)
    
    # Class loss: BCE for class probabilities, multiplied by object mask
    class_loss = object_mask * tf.keras.losses.binary_crossentropy(true_class_probs, tf.sigmoid(pred_class))
    class_loss = tf.reduce_sum(class_loss)
    
    total_loss = box_loss + conf_loss + class_loss
    return total_loss

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


