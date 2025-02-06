import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import os
from datetime import datetime

# T-Net for input transformation
def tnet(inputs, num_features):
    x = layers.Conv1D(64, 1, activation='relu')(inputs)
    x = layers.Conv1D(128, 1, activation='relu')(x)
    x = layers.Conv1D(256, 1, activation='relu')(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(
        num_features * num_features,
        kernel_initializer='zeros',
        bias_initializer=tf.keras.initializers.Constant(np.eye(num_features).flatten())  # Correct Identity Init
    )(x)
    transformation = layers.Reshape((num_features, num_features))(x)
    return transformation


# PointNet Model
def create_pointnet(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)

    # Input transformation
    t_input = tnet(inputs, 3)
    x = layers.Dot(axes=(2, 1))([inputs, t_input])

    # Shared MLP layers for feature extraction
    x = layers.Conv1D(64, 1, activation='relu')(x)
    x = layers.Conv1D(64, 1, activation='relu')(x)

    # Feature transformation
    t_feature = tnet(x, 64)
    x = layers.Dot(axes=(2, 1))([x, t_feature])

    x = layers.Conv1D(64, 1, activation='relu')(x)
    x = layers.Conv1D(128, 1, activation='relu')(x)
    x = layers.Conv1D(1024, 1, activation='relu')(x)

    # Global feature aggregation
    x = layers.GlobalMaxPooling1D()(x)

    # Fully connected layers
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(num_classes, activation='softmax')(x)

    return Model(inputs, x)


def load_velodyne_bin(file_path):
    points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
    return points  # (x, y, z, intensity)

def load_labels(file_path):
    labels = []
    with open(file_path, 'r') as f:
        for line in f:
            labels.append(line.strip().split())
    return labels

LABEL_MAP = {
    "Car": 0, "Pedestrian": 1, "Cyclist": 2,
    "Truck": 3, "Van": 4, "Tram": 5, "Misc": 6,
    "DontCare": -1  # Ignore category
}

def parse_labels(label_data):
    """Convert KITTI labels into a list of numerical class labels."""
    objects = []
    for label in label_data:
        class_name = label[0]
        if class_name in LABEL_MAP and LABEL_MAP[class_name] != -1:
            objects.append(LABEL_MAP[class_name])
    return objects  # Returns list of object classes

NUM_POINTS = 2048  # Fixed number of points per sample

def preprocess_data(velodyne_dir, label_dir, num_samples=100):
    X_data, Y_data = [], []

    for i in range(num_samples):
        file_id = f"{i:06d}"  # KITTI file naming format

        # Load point cloud and labels
        velodyne_path = os.path.join(velodyne_dir, f"{file_id}.bin")
        label_path = os.path.join(label_dir, f"{file_id}.txt")

        point_cloud = load_velodyne_bin(velodyne_path)[:, :3]  # Only x, y, z
        labels = parse_labels(load_labels(label_path))

        # Skip if no labeled objects
        if not labels:
            continue

        # Downsample or pad points to NUM_POINTS
        if point_cloud.shape[0] > NUM_POINTS:
            sampled_idx = np.random.choice(point_cloud.shape[0], NUM_POINTS, replace=False)
            point_cloud = point_cloud[sampled_idx]
        elif point_cloud.shape[0] < NUM_POINTS:
            # Pad points if less than NUM_POINTS
            pad_size = NUM_POINTS - point_cloud.shape[0]
            point_cloud = np.pad(point_cloud, ((0, pad_size), (0, 0)), mode='constant', constant_values=0)

        # Multi-class labels (one-hot encoding)
        one_hot_labels = np.zeros(7)
        for label in labels:
            one_hot_labels[label] = 1  # Multi-label support

        X_data.append(point_cloud)
        Y_data.append(one_hot_labels)

    return np.array(X_data), np.array(Y_data)

# Training Example TODO: Make generic Path
velodyne_dir = '/Users/leonbender/Desktop/dev/data/kitti/3d_object/training/velodyne'
label_dir = '/Users/leonbender/Desktop/dev/data/kitti/3d_object/training/label_2'

point_clouds, labels = preprocess_data(velodyne_dir, label_dir, num_samples=5000)

model = create_pointnet(input_shape=(NUM_POINTS, 3), num_classes=7)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(point_clouds, labels, epochs=50, batch_size=16)

# Save the trained model
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
model_save_path = f'../pointnet_model.keras'  # Save with timestamp to avoid overwriting
model.save(model_save_path)

print(f"Model saved to {model_save_path}")
