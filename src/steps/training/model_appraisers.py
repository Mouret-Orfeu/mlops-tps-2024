import hydra
from omegaconf import DictConfig, OmegaConf
from ultralytics import YOLO
from pathlib import Path
from zenml import step

@step
@hydra.main(config_path="src/config/", config_name="config", version_base="1.2")
def model_appraiser(cfg: DictConfig, mean_iou_score: int):

    threshold = cfg.Threshold_map50_95_for_all_labels
    
    if mean_iou_score > threshold:
        return True
    else:
        return False
