import tensorflow as tf

def preprocess_dataset(data):
    """
    Preprocess the input dataset by resizing images, reshaping bounding boxes,
    and converting class labels to one-hot encoding.

    In the KITTI dataset we have less than 10 bounding boxes and three object classes. So we preprocess the information
    from the image in the following tensor:
    [
        [bbox_1_y_min, bbox_1_x_min, bbox_1_y_max, bbox_1_x_max, class_1, class_2, class_3],
        [bbox_2_y_min, bbox_2_x_min, bbox_2_y_max, bbox_2_x_max, class_1, class_2, class_3],
        ...
        [bbox_10_y_min, bbox_10_x_min, bbox_10_y_max, bbox_10_x_max, class_1, class_2, class_3]
    ]
    The class labels will be encoded as one-hot vectors. For example, if the class label is 0, the one-hot encoding
    will be [1, 0, 0]. If the class label is 1, the one-hot encoding will be [0, 1, 0]. If the class label is 2, the
    one-hot encoding will be [0, 0, 1].

    Args:
        data (dict): A dictionary containing 'image', 'objects' with 'bbox' and 'type'.

    Returns:
        tuple: A tuple containing the preprocessed image and concatenated bounding boxes and labels.
    """
    # load image, bounding boxes, and class labels from the input data
    image = data['image']
    bbox = data['objects']['bbox']
    class_labels = data['objects']['type']

    image = tf.image.resize(image, (416, 416))  # Resize image to 416x416
    bbox = tf.reshape(bbox, [-1, 4])  # Ensure bbox shape is consistent

    # Add empty bounding boxes when image has less than 10 objects
    bbox = tf.pad(bbox, [[0, 10 - tf.shape(bbox)[0]], [0, 0]])

    # Create label tensor
    labels = tf.one_hot(class_labels, depth=3)  # Convert class labels to one-hot encoding
    labels = tf.pad(labels, [[0, 10 - tf.shape(labels)[0]], [0, 0]])  # Pad to shape (10, 3) if necessary

    # concat bbox and labels tensor
    return image, tf.concat([bbox, labels], axis=-1)