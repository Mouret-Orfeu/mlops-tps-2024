import hydra
from zenml import step
import numpy as np

@step
def model_appraiser(map_50_95_scores: np.ndarray, pipeline_config: dict):
    """
    Appraise the model based on the metrics and return a decision if the model can be deployed.

    Args:
        model_metrics: The metrics of the model.

    Returns:
        bool: True if the model can be deployed, False otherwise.
    """

    threshold = pipeline_config["evaluation"]["threshold"]
    for map_50_95_score in map_50_95_scores:
        if map_50_95_score<threshold:
            return False

    return True
