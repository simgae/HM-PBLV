from src.models.yolo_v3_model import YoloV3Model

if __name__ == '__main__':
    yolo_v3_model = YoloV3Model()
    #yolo_v3_model.train_model(2)
    #yolo_v3_model.evaluate_model()

    yolo_v3_model._evaluate_image('../data/image.jpeg')
