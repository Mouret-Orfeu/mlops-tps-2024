import os
import requests
from src.config import settings
from ultralytics import YOLO
from pathlib import Path
from zenml import step
from train import train_model

@step
# Fonction pour télécharger un fichier en utilisant requests
def download_pre_trained_model(url, destination_folder, file_name):
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

def get_pre_trained_weights_path():
    pass

@step
def model_trainer(
    data_config_path: str,
    pre_trained_model_path: str,
    fine_tuned_model_path: str,
    
):
    fine_tuned_model = train_model(data_config_path, pre_trained_model_path, fine_tuned_model_path)
    
    
    return fine_tuned_model

def model_predict(model_path: str, images_path: list[str]):
    # Load a model
    model = YOLO(model_path)

    # Run batched inference on a list of images
    results = model(images_path)  # return a list of Results objects

