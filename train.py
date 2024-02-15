import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path="src/config/", config_name="config", version_base="1.2")
def train_model(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
  

if __name__ == "__main__":
    train_model()
