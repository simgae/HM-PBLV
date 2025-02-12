import tensorflow_datasets as tfds
import tensorflow as tf
import cv2
import torch
from pytorchyolo import detect, models

from src.utils import preprocess_dataset


class PreTrainedYoloV3Model:

    def __init__(self):
        self.model = None
        self.num_classes = 3
        self.image_shape = (416, 416, 3)

        # Load the KITTI dataset
        self.train_dataset = tfds.load('kitti', split='train')
        self.val_dataset = tfds.load('kitti', split='validation')
        self.test_dataset = tfds.load('kitti', split='test')

        # Preprocess the validation and test datasets
        self.train_dataset = self.train_dataset.map(preprocess_dataset).batch(32)
        self.val_dataset = self.val_dataset.map(preprocess_dataset).batch(32)
        self.test_dataset = self.test_dataset.map(preprocess_dataset).batch(32)


    @staticmethod
    def build_pre_trained_yolo_v3_model():
        """
        Build pre-trained yolo model from configuration files.
        :return: pre-trained yolo model
        """
        return models.load_model('./models/pre-trained-yolo-model/yolov3-tiny.cfg', './models/pre-trained-yolo-model/yolov3-tiny.weights')

    def train_model(self, epochs, fine_tuning=True):
        """
        Load the pre-trained YOLO v3 model and fine-tune it on the KITTI dataset.
        :param fine_tuning: Boolean indicating whether to fine-tune the model on the KITTI dataset.
        """
        self.model = self.build_pre_trained_yolo_v3_model()

        if fine_tuning:
            # TODO: fine tune the model on the KITTI dataset
            pass

        self._save_model()

    def evaluate_model(self):
        """
        Evaluates the pre-trained YOLO v3 model's performance on the test dataset.
        """
        # TODO: Evaluate image on test dataset
        pass

    def load_model(self, load_custom_model=False):
        """
        Loads a pre-trained YOLO v3 model from the configuration files.
        :param load_custom_model: Boolean indicating whether to load a custom model or the pre-trained-one.
        """
        if not load_custom_model:
            self.model = torch.load('./models/pre-trained-yolo-model/yolov3_ckpt_2.pth')
        else:
            self.model = torch.load('./pre_trained_yolo_v3_model.pth', weights_only=False)

    def evaluate_image(self, image_path):
        """
        Evaluate an image and save it with bounding boxes.
        :param image_path: path to image
        """
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        boxes = detect.detect_image(self.model, img, conf_thres=0.2, nms_thres=0.5)

        # format of boxes [[x1, y1, x2, y2, conf, cls], ...]
        # add bboxes to image
        for box in boxes:
            # add box as rectangle to image
            cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)

            # add label to image
            cv2.putText(img, f'{box[4]:.2f}', (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # save image as output_pre_trained_yolo_v3.jpg
        cv2.imwrite('output_pre_trained_yolo_v3.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    def _save_model(self):
        """
        Saves the pre-trained YOLO V3 model to a file.
        """
        torch.save(self.model, './pre_trained_yolo_v3_model.pth')

