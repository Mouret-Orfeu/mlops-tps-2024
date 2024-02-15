import hydra
from omegaconf import DictConfig, OmegaConf
from ultralytics import YOLO
from pathlib import Path

@hydra.main(config_path="src/config/", config_name="config", version_base="1.2")
def train_model(cfg: DictConfig, data_config_path, pre_trained_model_path, fine_tuned_model_path):

    # Récupérer les paramètres du fichier de configuration
    nb_epochs = cfg.nb_epochs
    img_size = cfg.img_size
    batch_size = cfg.batch_size
    device = cfg.device

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
    
    # S'assurer qu'on entraine bien avec les données d'entrainement
    fine_tuned_model = model.train(data=data_config_path, split="train", epochs=nb_epochs, imgsz=img_size, batch_size=batch_size, device=device)
    
    # print(OmegaConf.to_yaml(cfg))
    
    return fine_tuned_model

  

# DEBUG et visualisation
if __name__ == "__main__":
    train_model()


