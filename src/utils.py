import tensorflow as tf
import tensorflow_datasets as tfds

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
    class_labels = data['objects']['type']

    image = tf.image.resize(image, (416, 416))  # Resize image to 416x416
    image = tf.cast(image, tf.float32) / 255.0  # Normalize image
    bbox = tf.reshape(bbox, [-1, 4])  # Ensure bbox shape is consistent
    bbox = handle_shape_mismatch(bbox)  # Handle variable number of bounding boxes

    # Normalization not necessary for kitti dataset - already normalized
    # bbox = normalize_bboxes(bbox, image.shape)

    bbox = convert_bboxes_to_fixed_size_tensor(bbox)  # Convert to fixed size tensor
    bbox = tf.reshape(bbox, [-1])  # Flatten the bounding boxes to match the model output shape

    # Ensure the class labels are also in the correct shape
    # Add +1 to differentiate between 0 classification and padding
    class_labels = tf.reshape(class_labels+1, [-1])
    padding = tf.maximum(0, 10 - tf.shape(class_labels)[0])  # Ensure padding is non-negative
    class_labels = tf.pad(class_labels, [[0, padding]], constant_values=0)  # Pad class labels to fixed size
    class_labels = class_labels[:10]  # Ensure class labels have exactly 10 elements

    return image, tf.concat([bbox, tf.cast(class_labels, tf.float32)], axis=0)  # Concatenate bbox and class labels


def postprocess_predictions(predictions, image_shape):
    """
    Postprocess the model predictions by extracting bounding boxes and class probabilities,
    and converting them back to their original scale.

    Args:
        predictions (tensor): The model predictions containing bounding boxes and class probabilities.
        image_shape (tuple): The original shape of the image.

    Returns:
        tuple: A tuple containing the bounding boxes and class probabilities.
    """
    bbox = predictions[:40]  # Extract bounding boxes
    class_probs = predictions[40:]  # Extract class probabilities

    # Reshape bounding boxes to original format
    bbox = tf.reshape(bbox, [-1, 4])
    height, width = image_shape[0], image_shape[1]

    # Convert bounding boxes back to original scale
    bbox = tf.stack([
        bbox[:, 0] * height,
        bbox[:, 1] * width,
        bbox[:, 2] * height,
        bbox[:, 3] * width
    ], axis=-1)



    return bbox, class_probs


if __name__ == '__main__':
    train_dataset = tfds.load('kitti', split='train')
    data = next(iter(train_dataset))
    image, bbox = preprocess_dataset(data)
    print(image.shape, bbox.shape)




### Aufbau Bounding Box: [ymin, xmin, ymax, xmax]