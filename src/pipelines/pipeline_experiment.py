from omegaconf import OmegaConf
from zenml import pipeline
from src.config import settings
from src.models.model_dataset import Dataset
from src.models.model_bucket_client import MinioClient
from src.config.settings import EXTRACTED_DATASETS_PATH


from src.config.settings import EXTRACTED_DATASETS_PATH, MLFLOW_EXPERIMENT_PIPELINE_NAME
# from src.steps.data.data_extractor import dataset_extractor
from src.steps.data.datalake_initializers import (
    data_source_list_initializer,
    minio_client_initializer,
)
from src.steps.data.dataset_preparators import (
    dataset_creator,
    datasource_extractor,
    dataset_to_yolo_converter,
)
from src.steps.training.model_appraisers import model_appraiser 
from src.steps.training.model_evaluators import model_evaluator # Créer model_evaluator.py et coder fonction
from src.steps.training.model_trainers import (
    get_pre_trained_weights_path,
    model_trainer,
    download_pre_trained_model
)


@pipeline(name=MLFLOW_EXPERIMENT_PIPELINE_NAME)
def gitflow_experiment_pipeline(cfg: str) -> None:
    """
    Experiment a local training and evaluate if the model can be deployed.

    Args:
        cfg: The Hydra configuration.
    """
    pipeline_config = OmegaConf.to_container(OmegaConf.create(cfg))

    minio_client: MinioClient = minio_client_initializer()
    data_source_list = data_source_list_initializer()

    bucket_name = "data-sources"

    distribution_weights = [0.6, 0.2, 0.2]

    # Extract the data sources to a folder
    dataset: Dataset = dataset_creator(data_source_list, 1234, bucket_name, distribution_weights)

    # model_url = settings.YOLO_PRE_TRAINED_WEIGHTS_URL
    # model_folder = settings.YOLO_PRE_TRAINED_WEIGHTS_PATH
    # model_name = settings.YOLO_PRE_TRAINED_WEIGHTS_NAME

    # # Dowload pre_trained model
    # download_pre_trained_model(model_url, model_folder, model_name)

    # Utilisez la méthode download pour télécharger les données de votre bucket
    # destination_root_path = "./downloaded_dataset"  # Remplacez par le chemin souhaité
    # minio_client.download_folder(bucket_name, data_source.name, EXTRACTED_DATASETS_PATH
    dataset.download(minio_client, EXTRACTED_DATASETS_PATH)

    # Vous pouvez ajouter des étapes supplémentaires ici si nécessaire
    # Par exemple, convertir les données au format YOLO si nécessaire
    dataset.to_yolo_format(EXTRACTED_DATASETS_PATH)

    # Extract the data sources to a folder
    datasource_extractor(data_source_list, minio_client, bucket_name)

    # model_url = settings.YOLO_PRE_TRAINED_WEIGHTS_URL
    # model_folder = settings.YOLO_PRE_TRAINED_WEIGHTS_PATH
    # model_name = settings.YOLO_PRE_TRAINED_WEIGHTS_NAME

    # # Dowload pre_trained model
    # download_pre_trained_model(model_url, model_folder, model_name)

    # explore_dataset()

    # Prepare/create the dataset
    # dataset = dataset_creator(
    #     ...
    # )

    # Extract the dataset to a folder
    # extraction_path = dataset_extractor(
    #     ...
    # )

    # If necessary, convert the dataset to a YOLO format
    # dataset_to_yolo_converter(
    #     ...
    # )

    # Train the model
    # trained_model_path = model_trainer(
    #     ...
    # )

    # Evaluate the model
    # test_metrics_result = model_evaluator(
    #     ...
    # )

    # Validation test for deployment readiness
    # test_metrics_result = model_appraiser(
    #     ...
    # )
