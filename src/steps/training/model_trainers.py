import os
import requests
from src.config import settings
from ultralytics import YOLO
from pathlib import Path
# from yolov5 import train  # Assurez-vous que c'est le bon module import



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

def get_pre_trained_weights_path():
    pass

def model_trainer(
    img_size: int,
    batch_size: int,
    nb_epochs: int,
    data_config_path: str,
    pre_trained_model_path: str,
    fine_tuned_model_path: str,
    cache_images: bool,
    device: str = None  # Par exemple, 'cpu' ou '0' pour le premier GPU
):
    
    # Load a pretrained YOLO model (recommended for training)
    model = YOLO(pre_trained_model_path) # Mettre le vrai chemin ici

    # Convertir les chemins en objets Path
    data_config_path = Path(data_config_path)
    pre_trained_model_path = Path(pre_trained_model_path)

    # Assurez-vous que les chemins existent
    if not data_config_path.exists():
        raise FileNotFoundError(f"Data config file not found: {data_config_path}")
    if not pre_trained_model_path.exists():
        raise FileNotFoundError(f"Weights file not found: {pre_trained_model_path}")
    if not fine_tuned_model_path.exists():
        raise FileNotFoundError(f"Output folder not found: {fine_tuned_model_path}")
    
    results = model.train(data=data_config_path, epochs=nb_epochs, imgsz=img_size, batch_size=batch_size, device=device, cache_images=cache_images)
    results.save(fine_tuned_model_path)
