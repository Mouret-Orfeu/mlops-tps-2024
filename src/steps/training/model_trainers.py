import os
from src.config import settings
from ultralytics import YOLO
from pathlib import Path
from zenml import step
from train import train_model


@step
def model_trainer(pipeline_config: dict, dataset_path: str):
    data_config_path = os.path.join(dataset_path, settings.DATASET_YOLO_CONFIG_NAME)

    trained_model_path = train_model(pipeline_config, data_config_path)
    return trained_model_path

def model_predict(model_path: str, image_paths: list[str]):
    # Load a model
    model = YOLO(model_path)

    # Run batched inference on a list of images
    results = model(image_paths)  # return a list of Results objects

    # Visualize the results
    results.show()

@step
def test_path(path: str):
    print("\033[91m" + path + "\033[0m")
    exit(0)
    return