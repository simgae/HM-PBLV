import os
import cv2
import numpy as np
import zipfile

# Import the YOLO model loader and detector from pytorchyolo
from pytorchyolo import detect, models
# For downloading files via torchvision
from torchvision.datasets.utils import download_url


# --- IoU Calculation ---
def iou(box1, box2):
    """
    Compute Intersection over Union (IoU) between two boxes.
    Each box is defined as [x1, y1, x2, y2].
    """
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    inter_area = (x_right - x_left) * (y_bottom - y_top)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter_area / float(area1 + area2 - inter_area)


# --- COCO Download Helper ---
def download_and_extract_coco(root_dir):
    """
    Download and extract the COCO 2017 validation images and annotations
    into root_dir if they do not already exist.
    Returns the images directory and the path to the instances_val2017.json file.
    """
    # Paths for images and annotations
    images_dir = os.path.join(root_dir, 'val2017')
    annotations_dir = os.path.join(root_dir, 'annotations')
    os.makedirs(root_dir, exist_ok=True)

    # Download validation images if not present.
    if not os.path.exists(images_dir):
        print("Downloading COCO 2017 validation images...")
        images_zip_path = os.path.join(root_dir, 'val2017.zip')
        download_url("http://images.cocodataset.org/zips/val2017.zip",
                     root=root_dir, filename="val2017.zip", md5=None)
        print("Extracting validation images...")
        with zipfile.ZipFile(images_zip_path, 'r') as zip_ref:
            zip_ref.extractall(root_dir)
        os.remove(images_zip_path)
    else:
        print("COCO 2017 validation images already exist.")

    # Download annotations if not present.
    annotation_file = os.path.join(annotations_dir, 'instances_val2017.json')
    if not os.path.exists(annotation_file):
        print("Downloading COCO 2017 annotations...")
        ann_zip_path = os.path.join(root_dir, 'annotations_trainval2017.zip')
        download_url("http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
                     root=root_dir, filename="annotations_trainval2017.zip", md5=None)
        print("Extracting annotations...")
        with zipfile.ZipFile(ann_zip_path, 'r') as zip_ref:
            zip_ref.extractall(root_dir)
        os.remove(ann_zip_path)
    else:
        print("COCO 2017 annotations already exist.")

    return images_dir, annotation_file


# --- Mapping between YOLO indices and COCO category IDs ---
# List of 80 class names in order (as in the standard coco.names file)
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]
# Mapping: YOLO index (0–79) -> COCO category id
YOLO2COCO = {
    0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10,
    10: 11, 11: 13, 12: 14, 13: 15, 14: 16, 15: 17, 16: 18, 17: 19, 18: 20, 19: 21,
    20: 22, 21: 23, 22: 24, 23: 25, 24: 27, 25: 28, 26: 31, 27: 32, 28: 33, 29: 34,
    30: 35, 31: 36, 32: 37, 33: 38, 34: 39, 35: 40, 36: 41, 37: 42, 38: 43, 39: 44,
    40: 46, 41: 47, 42: 48, 43: 49, 44: 50, 45: 51, 46: 52, 47: 53, 48: 54, 49: 55,
    50: 56, 51: 57, 52: 58, 53: 59, 54: 60, 55: 61, 56: 62, 57: 63, 58: 64, 59: 65,
    60: 67, 61: 70, 62: 72, 63: 73, 64: 74, 65: 75, 66: 76, 67: 77, 68: 78, 69: 79,
    70: 80, 71: 81, 72: 82, 73: 84, 74: 85, 75: 86, 76: 87, 77: 88, 78: 89, 79: 90
}
# Inverse mapping: COCO category id -> YOLO index
COCO2YOLO = {coco_id: yolo_idx for yolo_idx, coco_id in YOLO2COCO.items()}


# --- Evaluation on COCO ---
def evaluate_on_coco(model, images_dir, annotation_file, iou_threshold=0.5, conf_threshold=0.5, max_images=None):
    """
    Evaluate the YOLO model on the COCO 2017 validation set.
    Uses pycocotools to load ground truth annotations.

    Parameters:
      model         : your loaded YOLO model.
      images_dir    : directory containing the COCO val images.
      annotation_file: path to instances_val2017.json.
      iou_threshold : IoU threshold to consider a detection a true positive.
      conf_threshold: minimum confidence for predicted boxes.
      max_images    : if provided, limits evaluation to a subset of images.
    """
    from pycocotools.coco import COCO
    coco = COCO(annotation_file)
    image_ids = coco.getImgIds()
    if max_images is not None:
        image_ids = image_ids[:max_images]

    total_TP = 0
    total_FP = 0
    total_FN = 0

    for image_id in image_ids:
        # Load image info from COCO
        img_info = coco.loadImgs(image_id)[0]
        img_path = os.path.join(images_dir, img_info['file_name'])
        img = cv2.imread(img_path)
        if img is None:
            print(f"Could not load image {img_path}. Skipping...")
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width = img.shape[:2]

        # Run YOLO model on the image.
        pred_boxes = detect.detect_image(model, img_rgb)
        predictions = []
        if pred_boxes is not None and len(pred_boxes) > 0:
            for det in pred_boxes:
                # Each detection is [x1, y1, x2, y2, confidence, class]
                if det[4] < conf_threshold:
                    continue
                predictions.append({
                    'box': list(det[:4]),
                    'confidence': det[4],
                    'class': int(det[5])
                })

        # Get ground truth annotations for the image.
        ann_ids = coco.getAnnIds(imgIds=image_id)
        anns = coco.loadAnns(ann_ids)
        gt_boxes = []
        for ann in anns:
            # Optionally, skip crowd annotations.
            if ann.get('iscrowd', 0) == 1:
                continue
            # COCO format: [x, y, width, height] – convert to [x1, y1, x2, y2]
            bbox = ann['bbox']
            gt_box = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
            # Convert COCO category id to YOLO index.
            coco_cat_id = ann['category_id']
            if coco_cat_id not in COCO2YOLO:
                continue
            gt_class = COCO2YOLO[coco_cat_id]
            gt_boxes.append({'box': gt_box, 'class': gt_class})

        # Match predictions to ground truth boxes.
        # (Sort predictions so that higher-confidence detections are considered first.)
        predictions = sorted(predictions, key=lambda x: x['confidence'], reverse=True)
        matched_gt = set()
        TP = 0
        FP = 0
        for pred in predictions:
            best_iou = 0.0
            best_gt_idx = -1
            for idx, gt in enumerate(gt_boxes):
                if idx in matched_gt:
                    continue  # already matched
                if pred['class'] != gt['class']:
                    continue  # different class
                current_iou = iou(pred['box'], gt['box'])
                if current_iou > best_iou:
                    best_iou = current_iou
                    best_gt_idx = idx
            if best_iou >= iou_threshold:
                TP += 1
                matched_gt.add(best_gt_idx)
            else:
                FP += 1
        FN = len(gt_boxes) - len(matched_gt)
        total_TP += TP
        total_FP += FP
        total_FN += FN

        print(f"Image {img_info['file_name']} -- TP: {TP}, FP: {FP}, FN: {FN}")

    # Compute overall precision, recall, and F1 score.
    precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0
    recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print("\nCOCO Evaluation Results:")
    print(f"Total True Positives : {total_TP}")
    print(f"Total False Positives: {total_FP}")
    print(f"Total False Negatives: {total_FN}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1_score:.4f}")


# --- Main Script ---
if __name__ == '__main__':
    # Update these paths with the actual locations of your YOLO configuration and weights.
    config_path = "./models/pre-trained-yolo-model/yolov3.cfg"
    weights_path = "./models/pre-trained-yolo-model/yolov3.weights"

    print("Loading YOLO model...")
    model = models.load_model(config_path, weights_path)

    # Directory in which COCO data will be stored.
    coco_root = "coco"  # Change this if desired.
    images_dir, annotation_file = download_and_extract_coco(coco_root)

    # Optionally, you can limit the number of images evaluated (e.g., max_images=100)
    evaluate_on_coco(model, images_dir, annotation_file, iou_threshold=0.5, conf_threshold=0.1, max_images=None)
