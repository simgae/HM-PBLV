import tensorflow as tf

def preprocess_dataset(data):
    # load image, bounding boxes, and class labels from the input data
    image = data['image']
    bbox = data['objects']['bbox']
    class_labels = data['objects']['type']

    # Resize bounding boxes
    if tf.size(bbox) == 0:
        bbox = tf.zeros([0, 4])
    else:
        bbox = tf.reshape(bbox, [-1, 4])

    # Denormalize bounding boxes
    image_shape = tf.shape(image)
    bbox = bbox * [image_shape[1], image_shape[0], image_shape[1], image_shape[0]]

    # Calculate the center of bounding boxes
    center_x = (bbox[:, 0] + bbox[:, 2]) / 2
    center_y = (bbox[:, 1] + bbox[:, 3]) / 2

    # Calculate the width and height of bounding boxes
    width = bbox[:, 2] - bbox[:, 0]
    height = bbox[:, 3] - bbox[:, 1]

    # Normalize bounding boxes
    bbox = tf.stack([
        center_x / tf.cast(image_shape[1], tf.float32),
        center_y / tf.cast(image_shape[0], tf.float32),
        width / tf.cast(image_shape[1], tf.float32),
        height / tf.cast(image_shape[0], tf.float32)
    ], axis=-1)

    # Resize image to 416x416
    image = tf.image.resize(image, (416, 416))

    # Add One-hot encoding for class labels
    class_labels = tf.one_hot(class_labels, depth=3)

    # Add tensor for objectiveness
    objectiveness = tf.ones((tf.shape(bbox)[0], 1))

    # Concatenate class labels, bounding boxes, and objectiveness
    bbox = tf.concat([bbox, objectiveness, class_labels], axis=-1)

    # Ground truth tensor Shape = (batch_size, 13, 13, 3, 5 + num_classes)
    # Tensor: (32, 13, 13, 3, bbox)
    batch_size = 1
    num_classes = 3
    grid_size_large = 13
    grid_size_medium = 26
    grid_size_small = 52

    y_true_large = tf.zeros((batch_size, grid_size_large, grid_size_large, 3, 5 + num_classes))
    y_true_medium = tf.zeros((batch_size, grid_size_medium, grid_size_medium, 3, 5 + num_classes))
    y_true_small = tf.zeros((batch_size, grid_size_small, grid_size_small, 3, 5 + num_classes))

    # Insert bounding boxes into the ground truth tensors
    for i in range(tf.shape(bbox)[0]):
        y_true_large = tf.tensor_scatter_nd_update(y_true_large, [[0, 6, 6, 0]], [bbox[i]])
        y_true_medium = tf.tensor_scatter_nd_update(y_true_medium, [[0, 13, 13, 0]], [bbox[i]])
        y_true_small = tf.tensor_scatter_nd_update(y_true_small, [[0, 26, 26, 0]], [bbox[i]])

    return image, (y_true_large, y_true_medium, y_true_small)