import os
from src.config import settings
from ultralytics import YOLO
from pathlib import Path
from zenml import step
import torch
import requests

def download_pre_trained_model(url: str, destination_folder: str, file_name: str):
    """
    Download a file from an specific url

    Args:
        url: url to request to download the file
        destination_folder: folder in which the file is downloaded
        file_name: file we want to download
    Returns:
        path of the downloaded file
    """
    # Créer le dossier de destination s'il n'existe pas
    os.makedirs(destination_folder, exist_ok=True)
    file_path = os.path.join(destination_folder, file_name)

    # Télécharger le fichier et l'écrire dans le dossier de destination
    response = requests.get(url)
    if response.status_code == 200:
        with open(file_path, 'wb') as file:
            file.write(response.content)
        print(f"Downloaded {file_name} to {destination_folder}")
    else:
        print(f"Failed to download {file_name}. Status code: {response.status_code}")
    
    return file_path

@step
def model_trainer(pipeline_config: dict, dataset_path: str):
    """
    Train a YOLOV8 model

    Args:
        pipeline_config: dict containing the hyperparameters of the model
        data_config_path: path of the dataset 
    Returns:
        path of the trained model
    """
    model_url = settings.YOLO_PRE_TRAINED_WEIGHTS_URL
    model_folder = settings.YOLO_PRE_TRAINED_WEIGHTS_PATH
    model_name = settings.YOLO_PRE_TRAINED_WEIGHTS_NAME

    # Dowload pre_trained model
    pre_trained_model_path = download_pre_trained_model(model_url, model_folder, model_name)

    # Get the model hyperparameters from the pipeline_config
    nb_epochs = pipeline_config["model"]["nb_epochs"]
    img_size = pipeline_config["model"]["img_size"]
    batch_size = pipeline_config["model"]["batch_size"]
    device = pipeline_config["model"]["device"]
    
    # Load a pretrained YOLO model (recommended for training)
    model = YOLO(pre_trained_model_path) # Mettre le vrai chemin ici

    # Create the string variable of the path of yaml data file of YOLO formatted data
    data_config_path = os.path.join(dataset_path, settings.DATASET_YOLO_CONFIG_NAME)
    
    # Convert path str to Path object for OS consistancy
    data_config_path = Path(data_config_path)
    pre_trained_model_path = Path(pre_trained_model_path)

    # Check if yaml data file exists
    if not os.path.exists(data_config_path):
        raise FileNotFoundError(f"Data config file not found: {data_config_path}")
    
    #print necessary to active cuda
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    
    model.train(data=data_config_path, epochs=nb_epochs, imgsz=img_size, batch=batch_size, device=device)

    trained_model_path = "ultralytics/yolov8s_trained.pt"
    model.save(trained_model_path)

    return trained_model_path



@step
def model_predict(model_path: str, images_path: list[str]):
    """
    Use a YOLOV8 model on several images to detect objets on

    Args:
        model_path: path of a YOLOV8 model in PyTorch format
        data_config_path: list of paths of image to try the model on 
    """

    # Load a model
    model = YOLO(model_path)

    for image_path in images_path:
        model.predict(image_path, show = True, conf=0.5)
