from zenml import step
from src.models.model_dataset import Dataset
from src.models.model_bucket_client import MinioClient
from src.models.model_data_source import DataSourceList
from src.materializers.materializer_dataset import DatasetMaterializer
import os
from src.config.settings import EXTRACTED_DATASETS_PATH

from src.models.model_dataset import Dataset 
import shutil
import json

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
    """
    Reorganizes and converts a dataset into YOLO format, including splitting images and annotations
    into train, test, and validation folders. Then, it rearranges the dataset structure for YOLO compatibility
    and converts annotations from JSON to YOLO's .txt format.

    Initial structure:
    datasets/plastic_in_river
    ├── images
    │   └── img1.png, img2.png, ...
    └── annotations
        └── img1.json, img2.json, ...

    Intermediate structure (after splitting):
    datasets/plastic_in_river
    ├── train
    │   ├── images
    │   │   └── img1.png, img2.png, ...
    │   └── labels
    │       └── img1.json, img2.json, ...
    ├── test
    │   ├── images
    │   │   └── img1.png, img2.png, ...
    │   └── labels
    │       └── img1.json, img2.json, ...
    └── validation
        ├── images
        │   └── img1.png, img2.png, ...
        └── labels
            └── img1.json, img2.json, ...

    Final structure (after converting to YOLO format):
    datasets/plastic_in_river
    ├── images
    │   ├── train
    │   │   └── img1.png, img2.png, ...
    │   ├── test
    │   │   └── img1.png, img2.png, ...
    │   └── validation
    │       └── img1.png, img2.png, ...
    ├── labels
    │   ├── train
    │   │   └── img1.txt, img2.txt, ... (YOLO format)
    │   ├── test
    │   │   └── img1.txt, img2.txt, ...
    │   └── validation
    │       └── img1.txt, img2.txt, ...
    └── dataset.yaml

    The process skips conversion if a 'dataset.yaml' file exists. It first moves files to the correct
    split directories based on distribution weights. Then, it rearranges files for YOLO compatibility and
    converts JSON annotations to YOLO format (.txt files), finalizing with a 'dataset.yaml' file creation
    for model training.

    Parameters:
    - dataset (Dataset): Dataset instance with distribution weights and methods for YOLO conversion.
    - dataset_path (str): Path to the dataset directory.
    """

    if os.path.exists(os.path.join(dataset_path, "dataset.yaml")):
        return

    # Paths to the original images and labels directories
    initial_images_path = os.path.join(dataset_path, "images")
    initial_annotations_path = os.path.join(dataset_path, "annotations")

    # Ensure split directories for images and labels exist
    for split_name in dataset.split_names:
        images_split_path = os.path.join(dataset_path, split_name, dataset.images_path)
        labels_split_path = os.path.join(dataset_path, split_name, dataset.annotations_path)
        os.makedirs(images_split_path, exist_ok=True)
        os.makedirs(labels_split_path, exist_ok=True)

    # Iterate over label files to distribute them and their corresponding images
    for label_file in os.listdir(initial_annotations_path):
        label_file_path = os.path.join(initial_annotations_path, label_file)
        with open(label_file_path, 'r') as file:
            label_data = json.load(file)
        
        # Extract image filename from the label data
        image_filename = os.path.basename(label_data["image_path"])
        image_file_path = os.path.join(initial_images_path, image_filename)

        # Determine the split for this pair
        split_name = dataset.get_next_split()

        # Move the label file
        new_label_path = os.path.join(dataset_path, split_name, dataset.annotations_path, label_file)
        new_image_path = os.path.join(dataset_path, split_name, dataset.images_path, image_filename)

        # Move the corresponding image file
        shutil.move(label_file_path, new_label_path)
        if os.path.exists(image_file_path): # Check if the image exists to avoid errors
            shutil.move(image_file_path, new_image_path)

    if not os.listdir(initial_images_path):
        os.rmdir(initial_images_path)
    if not os.listdir(initial_annotations_path):
        os.rmdir(initial_annotations_path)

    # After distribution, convert the dataset to YOLO format
    dataset.to_yolo_format(dataset_path)
