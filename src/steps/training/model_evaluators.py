import json
import numpy as np
from typing import List, Dict, Tuple
from ultralytics import YOLO
import os


def load_test_dataset(test_dataset_path: str) -> List[Dict]:
    """
    Load test dataset annotations from a JSON file.
    """
    with open(test_dataset_path, 'r') as file:
        test_data = json.load(file)
    return test_data

def calculate_iou(predicted_bbox, ground_truth_bbox) -> float:
    """
    Calculate Intersection over Union (IoU) between predicted and ground truth bounding boxes.
    Both boxes are in the format [x_center, y_center, width, height].
    """
    # Convert [x_center, y_center, width, height] to [x1, y1, x2, y2] for both predicted and ground truth bboxes
    pred_x1 = predicted_bbox[0] - (predicted_bbox[2] / 2)
    pred_y1 = predicted_bbox[1] - (predicted_bbox[3] / 2)
    pred_x2 = predicted_bbox[0] + (predicted_bbox[2] / 2)
    pred_y2 = predicted_bbox[1] + (predicted_bbox[3] / 2)

    gt_x1 = ground_truth_bbox[0] - (ground_truth_bbox[2] / 2)
    gt_y1 = ground_truth_bbox[1] - (ground_truth_bbox[3] / 2)
    gt_x2 = ground_truth_bbox[0] + (ground_truth_bbox[2] / 2)
    gt_y2 = ground_truth_bbox[1] + (ground_truth_bbox[3] / 2)

    # Calculate the (x, y) coordinates of the intersection rectangle
    inter_x1 = max(pred_x1, gt_x1)
    inter_y1 = max(pred_y1, gt_y1)
    inter_x2 = min(pred_x2, gt_x2)
    inter_y2 = min(pred_y2, gt_y2)

    # Compute the area of intersection rectangle
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    # Compute the area of both the prediction and ground truth rectangles
    pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
    gt_area = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)

    # Compute the Intersection over Union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = inter_area / float(pred_area + gt_area - inter_area)

    return iou



def model_evaluator(model_path: str, data_config_path: str, label_map: dict[int, str]):
    # Load the model
    model = YOLO(model_path)

    # # Get all files in the test dataset path
    # files = os.listdir(test_dataset_path)

    # # Filter the files to include only PNG images
    # list_test_data = [file for file in files if file.endswith(".png")]

    # # Initialize counters for metrics calculation
    # total_predictions = 0
    # total_ground_truths = 0
    # true_positives = 0
    # false_positives = 0
    # false_negatives = 0
    # iou_scores = []
    
    # predictions_to_ground_truth_comparison_list = []
    # for item in list_test_data:
    #     image_path = item["image_path"]
    #     ground_truth_bboxes = item["bbox"]
    #     ground_truth_labels = item["label"]

        
        
    #     # Perform model inference
    #     results = model(image_path)
        
    #     # For each image, store predictions and ground truths in  a structured way
    #     predictions_to_ground_truth_comparison_list.append({"predicted_bboxes": result., "predicted_labels": result.,
    #                         "ground_truth_bboxes": item., "ground_truth_labels": item.})

    # # Mettre ça dans une fonction à part
    # iou_scores = []
    # for item in predictions_to_ground_truth_comparison_list:
    #     predicted_bboxes = item["predicted_bboxes"]
    #     ground_truth_bboxes = item["ground_truth_bboxes"]
        
    #     for i in range(len(predicted_bboxes)):
    #         predicted_bbox = predicted_bboxes[i]
    #         ground_truth_bbox = ground_truth_bboxes[i]
            
    #         iou = calculate_iou(predicted_bbox, ground_truth_bbox)
    #         iou_scores.append(iou)
    
  

    # Validate the model
    metrics = model.val(data=data_config_path, split="validation") 
    

    

    return metrics.box.maps


