from src.models.pre_trained_yolo_v3_model import PreTrainedYoloV3Model
from src.models.yolo_v3_model import YoloV3Model

if __name__ == '__main__':
    yolo_v3_model = YoloV3Model()
    #yolo_v3_model.train_model(False)
    yolo_v3_model.load_model()
    # yolo_v3_model.evaluate_model()

    yolo_v3_model.evaluate_image('../data/kitti-test-plain/000024.png')
