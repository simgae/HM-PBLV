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

def preprocess_dataset(data):
    """
    Preprocess the input data by resizing the image, normalizing the bounding boxes, and extracting class labels.

    Args:
        data (dict): A dictionary containing the image, bounding box data, and class labels.

    Returns:
        tuple: A tuple containing the preprocessed image, bounding boxes, and class labels.
    """
    image = data['image']
    bbox = data['objects']['bbox']
    class_labels = data['objects']['type']  # Extract class labels

    image = tf.image.resize(image, (128, 128))
    bbox = tf.reshape(bbox, [-1, 4])  # Ensure bbox shape is consistent
    bbox = handle_shape_mismatch(bbox)  # Handle variable number of bounding boxes
    bbox = normalize_bboxes(bbox, image.shape)  # Normalize bounding box coordinates
    bbox = convert_bboxes_to_fixed_size_tensor(bbox)  # Convert to fixed size tensor
    bbox = tf.reshape(bbox, [-1])  # Flatten the bounding boxes to match the model output shape

    class_labels = tf.one_hot(class_labels, depth=3)  # Convert class labels to one-hot encoding
    class_labels = tf.reduce_sum(class_labels, axis=0)  # Sum one-hot vectors to handle multiple objects

    print(f"Image shape: {image.shape}, BBox shape: {bbox.shape}, Class labels shape: {class_labels.shape}")  # Debugging statement
    return image, tf.concat([bbox, class_labels], axis=0)  # Concatenate bbox and class labels