from src.models.yolo_model import YoloModel

if __name__ == '__main__':
    yolo_model = YoloModel()
    yolo_model.train_model(epochs=2)
    yolo_model.evaluate_model()