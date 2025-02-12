from src.app import yolo_model
from src.models.pre_trained_yolo_v3_model import PreTrainedYoloV3Model
from src.models.yolo_v3_model import YoloV3Model

if __name__ == '__main__':
    yolo_model = PreTrainedYoloV3Model()
    yolo_model.load_model()