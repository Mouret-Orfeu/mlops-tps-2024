from zenml import step
# from src.models.model_dataset import (to_yolo_format)
from src.models.model_dataset import Dataset
# from src.clients.minio_client import MinioClient
from src.models.model_bucket_client import BucketClient, MinioClient
from src.models.model_data_source import DataSourceList
from src.materializers.materializer_dataset import DatasetMaterializer
import os
from src.config.settings import EXTRACTED_DATASETS_PATH

import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import os
from src.models.model_dataset import Dataset 

@step(output_materializers=DatasetMaterializer)
def dataset_creator(data_source_list: DataSourceList, seed: int, bucket_name: str, distribution_weights: list[float])-> Dataset:

    data_source = data_source_list.data_sources[0]

    # Assurez-vous d'avoir toutes les informations nécessaires pour instancier l'objet
    dataset = Dataset(
        bucket_name=bucket_name,
        uuid=data_source.name,  
        seed=seed,  # Utilisez une graine appropriée pour la reproductibilité
        # Vous pouvez spécifier un UUID ou le laisser générer automatiquement
        # La distribution_weights doit correspondre à la répartition souhaitée de vos données
        distribution_weights=distribution_weights,
        # Vous devez charger votre label_map à partir du fichier JSON
        label_map={
            0: "PLASTIC_BAG",
            1: "PLASTIC_BOTTLE",
            2: "OTHER_PLASTIC_WASTE",
            3: "NOT_PLASTIC_WASTE"
        }
    )

    return dataset

@step
def dataset_extractor(dataset: Dataset, minio_client: MinioClient, bucket_name: str):

    # No need to pass down the parent folder, it should be directly the dataset folder?
    # Double-check and fix this if necessary
    dataset.download(minio_client, EXTRACTED_DATASETS_PATH)

    # # Faut que destination_path soit ./destination_folder
    # destination_path = os.path.join(os.path.basename("./"), destination_folder)

    return os.path.join(EXTRACTED_DATASETS_PATH, dataset.uuid)

@step
def dataset_to_yolo_converter(dataset : Dataset, dataset_path: str):
    dataset.to_yolo_format(dataset_path)
