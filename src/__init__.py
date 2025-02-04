from src.models.yolo_model import YoloModel

if __name__ == '__main__':
    yolo_model = YoloModel()
    yolo_model.train_model()
    yolo_model.evaluate_model()