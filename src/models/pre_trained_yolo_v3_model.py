import cv2
import torch
from pytorchyolo import detect, models

class PreTrainedYoloV3Model:

    def __init__(self):
        self.model = None

    @staticmethod
    def build_pre_trained_yolo_v3_model():
        """
        Build pre-trained yolo model from configuration files.
        :return: pre-trained yolo model
        """
        return models.load_model('./models/pre-trained-yolo-model/yolov3.cfg', './models/pre-trained-yolo-model/yolov3.weights')

    def train_model(self):
        """
        Load the pre-trained YOLO v3 model.
        """
        self.model = self.build_pre_trained_yolo_v3_model()

        self._save_model()

    def load_model(self, load_custom_model=False):
        """
        Loads a pre-trained YOLO v3 model from the configuration files.
        :param load_custom_model: Boolean indicating whether to load a custom model or the pre-trained-one.
        """
        if not load_custom_model:
            self.model = self.build_pre_trained_yolo_v3_model()
        else:
            self.model = torch.load('./pre_trained_yolo_v3_model.pth', weights_only=False)

    def evaluate_image(self, image_path):
        """
        Evaluate an image and save it with bounding boxes.
        :param image_path: path to image
        """
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        boxes = detect.detect_image(self.model, img, conf_thres=0.5, nms_thres=0.5)

        # format of boxes [[x1, y1, x2, y2, conf, cls], ...]
        # add bboxes to image
        for box in boxes:
            # add box as rectangle to image
            cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)

            # add label to image
            label = box[5]
            if 2.1 >= label >= -2.1:
                cv2.putText(img, 'Car ' + f'{box[4]:.2f}', (int(box[0] - 0.0002), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            else:
                cv2.putText(img, str(label) + f'{box[4]:.2f}', (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # save image as output_pre_trained_yolo_v3.jpg
        cv2.imwrite('output_pre_trained_yolo_v3.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    def _save_model(self):
        """
        Saves the pre-trained YOLO V3 model to a file.
        """
        torch.save(self.model, './pre_trained_yolo_v3_model.pth')

