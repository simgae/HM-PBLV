from .faster_rcnn_model import FasterRCNNModel

if __name__ == '__main__':
    # Train and evaluate Faster R-CNN model
    faster_rcnn_model = FasterRCNNModel()
    faster_rcnn_model.train_model(epochs=2)
    faster_rcnn_model.evaluate_model()


