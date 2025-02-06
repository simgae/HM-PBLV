import tensorflow_datasets as tfds
import keras
from src.models.yolo_model import YoloV3, yolo_v3_loss
from src.utils import preprocess_dataset
from functools import partial

# Load KITTI splits
train_dataset = tfds.load('kitti', split='train')
val_dataset = tfds.load('kitti', split='validation')
test_dataset = tfds.load('kitti', split='test')

# Preprocess
train_dataset = train_dataset.map(preprocess_dataset).batch(32)
val_dataset = val_dataset.map(preprocess_dataset).batch(32)
test_dataset = test_dataset.map(preprocess_dataset).batch(32)

if __name__ == '__main__':
    yolo_model = YoloV3()

    loss_fn = partial(yolo_v3_loss, anchors=anchors, num_classes=3)

    # Train the model
    yolo_model.compile(optimizer=keras.optimizers.Adadelta(1e-3), loss=)

    # yolo_model._evaluate_image('test_image.jpg')