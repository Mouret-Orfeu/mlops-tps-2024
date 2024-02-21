import os
from src.config import settings
import json
import numpy as np
from typing import List, Dict, Tuple
from ultralytics import YOLO
from zenml import step
import torch

@step
def model_evaluator(model_path: str, pipeline_config: dict, dataset_path: str):
    """
    Evaluate a YOLOV8 model on test split from dataset

    Args:
        model_path: path of the YOLOV8 model in PyTorch format
        pipeline_config: dict containing the hyperparameters of the model
        dataset_path: path of the dataset 
    Returns:
        mAP0.5-0.95 metric of each label
    """
    # Load the model
    model = YOLO(model_path)


    # Create the string variable of the path of yaml data file of YOLO formatted data
    data_config_path = os.path.join(dataset_path, settings.DATASET_YOLO_CONFIG_NAME)

    # Get the model hyperparameters from the pipeline_config
    img_size = pipeline_config["model"]["img_size"]
    batch_size = pipeline_config["model"]["batch_size"]
    device = pipeline_config["model"]["device"]
  
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())


    metrics = model.val(data=data_config_path, split="test",imgsz = img_size,batch=batch_size,device=device) 


    metrics.confusion_matrix.plot(normalize=True,save_dir='result')   

    return metrics.box.maps


