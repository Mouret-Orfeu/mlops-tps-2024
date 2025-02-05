from omegaconf import OmegaConf
from zenml import pipeline
from src.config import settings
from src.models.model_dataset import Dataset
from src.models.model_bucket_client import MinioClient
from src.config.settings import MLFLOW_END_TO_END_PIPELINE_NAME
from src.steps.data.datalake_initializers import (
    data_source_list_initializer,
    minio_client_initializer,
)
from src.steps.data.dataset_preparators import (
    dataset_creator,
    dataset_extractor,
    dataset_to_yolo_converter,
)
from src.steps.training.model_trainers import (
    model_trainer
)
from src.steps.training.model_evaluators import model_evaluator # Créer model_evaluator.py et coder fonction
from src.steps.training.model_appraisers import model_appraiser


@pipeline(name=MLFLOW_END_TO_END_PIPELINE_NAME)
def gitflow_experiment_pipeline(cfg: str) -> None:
    """
    Train and serve a new model if it performs better than the model
    currently served.

    Args:
        cfg: The Hydra configuration.
    """
    pipeline_config = OmegaConf.to_container(OmegaConf.create(cfg))

    minio_client: MinioClient = minio_client_initializer()
    data_source_list = data_source_list_initializer()
    bucket_name = "data-sources"

    distribution_weights = [0.6, 0.2, 0.2]
    
    # Prepare/create the dataset
    dataset = dataset_creator(
        data_source_list,
        1234,
        bucket_name,
        distribution_weights
    )

    # Extract the dataset to a folder
    extraction_path = dataset_extractor(
        dataset,
        minio_client,
        bucket_name
    )

    # If necessary, convert the dataset to a YOLO format
    dataset_to_yolo_converter(
        dataset,
        extraction_path
    )

    # Train the model
    trained_model_path = model_trainer(
        pipeline_config,
        extraction_path
    )

    # Evaluate the model
    test_metrics_result = model_evaluator(
        trained_model_path,
        pipeline_config,
        extraction_path)

    # Retrieve a decision if the model should be deployed
    can_model_be_deployed = model_appraiser(
        test_metrics_result,
        pipeline_config
    )

    if can_model_be_deployed:
        print("Model can be deployed")
    
    else:
        print("Model cannot be deployed")
