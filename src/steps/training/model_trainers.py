import os
from src.config import settings
from ultralytics import YOLO
from pathlib import Path
from zenml import step
import torch
import requests

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

def train_model(pipeline_config: dict, data_config_path: str):

    model_url = settings.YOLO_PRE_TRAINED_WEIGHTS_URL
    model_folder = settings.YOLO_PRE_TRAINED_WEIGHTS_PATH
    model_name = settings.YOLO_PRE_TRAINED_WEIGHTS_NAME

    # Dowload pre_trained model
    pre_trained_model_path = download_pre_trained_model(model_url, model_folder, model_name)

    # Récupérer les paramètres du fichier de configuration
    nb_epochs = pipeline_config["model"]["nb_epochs"]
    img_size = pipeline_config["model"]["img_size"]
    batch_size = pipeline_config["model"]["batch_size"]
    device = pipeline_config["model"]["device"]
    
    # Load a pretrained YOLO model (recommended for training)
    model = YOLO(pre_trained_model_path) # Mettre le vrai chemin ici

    # Convertir les chemins en objets Path
    data_config_path = Path(data_config_path)
    pre_trained_model_path = Path(pre_trained_model_path)

    # Assurez-vous que les chemins existent
    if not os.path.exists(data_config_path):
        raise FileNotFoundError(f"Data config file not found: {data_config_path}")
    
    print(torch.cuda.is_available())

    print(torch.cuda.device_count())
    
    model_metrics = model.train(data=data_config_path, epochs=nb_epochs, imgsz=img_size, batch=batch_size, device=device)

    print("entrainement terminé")
    trained_model_path = "ultralytics/yolov8s_trained.pt"

    if model_metrics is not None:
        
        # Save the model's state dictionary
        torch.save(model.state_dict(), trained_model_path)
        print(f"Model successfully saved at {trained_model_path}")
    else:
        print("Training did not complete successfully.")

    return trained_model_path



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