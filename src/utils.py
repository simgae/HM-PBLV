import tensorflow as tf

def handle_shape_mismatch(bboxes, max_bboxes=10):
    """
    Handle variable numbers of bounding boxes by padding or truncating to a fixed size.
    """
    bboxes = bboxes[:max_bboxes]  # Truncate to max_bboxes
    padding = [[0, max_bboxes - tf.shape(bboxes)[0]], [0, 0]]  # Padding for bboxes
    bboxes = tf.pad(bboxes, padding)
    return bboxes

def normalize_bboxes(bboxes, image_shape):
    """
    Normalize bounding box coordinates to be between 0 and 1.
    """
    height, width = image_shape[0], image_shape[1]
    bboxes = tf.cast(bboxes, tf.float32)
    bboxes = tf.stack([
        bboxes[:, 0] / height,
        bboxes[:, 1] / width,
        bboxes[:, 2] / height,
        bboxes[:, 3] / width
    ], axis=-1)
    return bboxes

def convert_bboxes_to_fixed_size_tensor(bboxes, max_bboxes=10):
    """
    Convert bounding boxes to a fixed size tensor.
    """
    bboxes = handle_shape_mismatch(bboxes, max_bboxes)
    return bboxes
