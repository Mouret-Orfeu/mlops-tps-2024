import json
import os
import random
import shutil
import threading

import ulid
import yaml
from PIL import Image

from src.config.settings import DATASET_YOLO_CONFIG_NAME
from src.models.model_bucket_client import BucketClient


class Dataset:
    def __init__(
        self,
        bucket_name,
        seed: int,
        uuid: str | None = None,
        annotations_path: str = "labels",
        images_path: str = "images",
        distribution_weights: list[float] | None = None,
        label_map: dict[int, str] | None = None,
    ):
        if distribution_weights is None:
            distribution_weights = [0.6, 0.2, 0.2]

        self.uuid = uuid or self.get_data_source_uuid()
        self.bucket_name = bucket_name
        self.annotations_path = annotations_path
        self.images_path = images_path
        self.distribution_weights = distribution_weights
        self.seed = seed
        self.random_instance = random.Random(seed)

        self.train_split_name = "train"
        self.test_split_name = "test"
        self.validation_split_name = "validation"

        self.split_names = [
            self.train_split_name,
            self.test_split_name,
            self.validation_split_name,
        ]

        self.label_map = label_map or {}

    def format_bucket_image_path(self, image_file_path: str, split_name: str) -> str:
        """
        Formats the bucket path for an image file based on the dataset's UUID and folder distribution.

        Args:
            image_file_path (str): The original path of the image file.
            split_name (str): The name of the split (train, test, or validation) the file will go into.

        Returns:
            str: A formatted bucket path for the image file.
        """
        image_filename = image_file_path.split("/")[-1]
        return f"{self.uuid}/{split_name}/{self.images_path}/{image_filename}"

    def format_bucket_annotation_path(
        self, annotation_file_path: str, split_name: str
    ) -> str:
        """
        Formats the bucket path for an annotation file based on the dataset's UUID and folder distribution.

        Args:
            annotation_file_path (str): The original path of the annotation file.
            split_name (str): The name of the split (train, test, or validation) the file will go into.

        Returns:
            str: A formatted bucket path for the annotation file.
        """
        annotation_filename = annotation_file_path.split("/")[-1]
        return f"{self.uuid}/{split_name}/{self.annotations_path}/{annotation_filename}"

    @staticmethod
    def get_data_source_uuid() -> str:
        """
        Generates a new ULID (Universally Unique Lexicographically Sortable Identifier) for the dataset.

        Returns:
            str: A ULID string.
        """
        return str(ulid.new())

    def get_next_split(self) -> str:
        """
        Randomly selects a folder ("train", "test", or "validation") based on the specified distribution weights.

        Returns:
            str: The name of the selected folder.
        """
        return self.random_instance.choices(
            self.split_names, self.distribution_weights
        )[0]

    def download(self, bucket_client: BucketClient, destination_root_path: str) -> None:
        bucket_client.download_folder(
            bucket_name=self.bucket_name,
            folder_name=self.uuid,
            destination_path=destination_root_path,
        )

    def to_yolo_format(self, dataset_path: str):
        """
        Converts a custom dataset to YOLO format.

        Args:
            dataset_path (str): The path where dataset has been downloaded.


        Before:
        datasets/plastic_in_river
        ├── train
        │   ├── images
        │   │   └── img1.png, img2.png, ...
        │   └── labels
        │       └── img1.json, img2.json, ... (JSON files)
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

        After:
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
        └── dataset.yaml (=DATASET_YOLO_CONFIG_NAME)
        """

        try:
            yolo_categories = [self.annotations_path, self.images_path]

            if os.path.isdir(
                os.path.join(dataset_path, yolo_categories[0])
            ) or os.path.isdir(os.path.join(dataset_path, yolo_categories[1])):
                return

            for category in yolo_categories:
                for split_name in self.split_names:
                    old_folder = os.path.join(dataset_path, split_name, category)
                    new_folder = os.path.join(dataset_path, category, split_name)

                    os.makedirs(new_folder, exist_ok=True)

                    for filename in os.listdir(old_folder):
                        old_file = os.path.join(old_folder, filename)
                        new_file = os.path.join(new_folder, filename)

                        shutil.move(old_file, new_file)

                    if not os.listdir(old_folder):
                        os.rmdir(old_folder)

            for split_name in self.split_names:
                split_path = os.path.join(dataset_path, split_name)
                if not os.listdir(split_path):
                    os.rmdir(split_path)

            self._create_yolo_yaml_file(dataset_path=dataset_path)
            self._convert_annotations_to_yolo_format(dataset_path=dataset_path)

        except Exception as e:
            # Handle any exception
            raise Exception("Error restructuring dataset") from e

    def _create_yolo_yaml_file(
        self, dataset_path: str, yaml_file_name: str = DATASET_YOLO_CONFIG_NAME
    ):
        data = {
            "path": self.uuid,
            "train": f"images/{self.train_split_name}/",
            "test": f"images/{self.test_split_name}/",
            "val": f"images/{self.validation_split_name}/",
            "names": self.label_map,
        }

        with open(os.path.join(dataset_path, yaml_file_name), "w") as file:
            yaml.dump(data, file, default_flow_style=False)

    def update_label_map(self, new_label_map: dict[int, str]) -> None:
        """
        Update the dataset's label_map by adding new key values.

        Args:
            new_label_map: The new label_map with potential new values.

        Raises:
            ValueError: When the new_label_map has identical keys as the dataset's labelmap but with different values.

        """
        for key, value in new_label_map.items():
            key = int(key)
            if key in self.label_map and self.label_map[key] != value:
                raise ValueError(
                    "Label maps are not compatible"
                    f"Current label_map: {self.label_map}"
                    f"is incompatible with the new label_map: {new_label_map}"
                )
            elif key not in self.label_map:
                self.label_map[key] = value

    def _get_yolo_data_from_json_data(self, json_data, img_width, img_height) -> list:
        """
        Converts JSON annotation data to YOLO format.

        Args:
            json_data (dict): JSON data containing labels and bounding boxes.
            img_width (int): Width of the corresponding image.
            img_height (int): Height of the corresponding image.

        Returns:
            list: A list of strings, each representing an object in YOLO annotation format.
        """
        yolo_format = []
        for i in range(len(json_data["label"])):
            label = json_data["label"][i]
            bbox = json_data["bbox"][i]
            x_center, y_center, width, height = bbox
            yolo_format.append(f"{label} {x_center} {y_center} {width} {height}")
        return yolo_format

    def _process_json_file(self, json_path, img_path) -> None:
        """
        Processes a single JSON file and its corresponding image to produce a YOLO format annotation.

        Args:
            json_path (str): The file path to the JSON file.
            img_path (str): The file path to the corresponding image file.

        Raises:
            Exception: If there is an error in processing the file.
        """
        try:
            with open(json_path) as file:
                json_data = json.load(file)

            with Image.open(img_path) as img:
                img_width, img_height = img.size

            yolo_annotations = self._get_yolo_data_from_json_data(
                json_data, img_width, img_height
            )

            txt_path = json_path.replace(".json", ".txt")
            with open(txt_path, "w") as file:
                file.write("\n".join(yolo_annotations))

            os.remove(json_path)
        except Exception as e:
            raise Exception(f"Error processing {json_path}") from e

    def _convert_annotations_to_yolo_format(self, dataset_path) -> None:
        """
        Converts JSON labels in a dataset to YOLO format.

        This function walks through a dataset directory, finds all JSON files,
        converts their labels to YOLO format, and writes them to `.txt` files.
        It uses multithreading to process multiple files simultaneously.

        Args:
            dataset_path (str): The root path of the dataset.

        Usage Example:
            to_yolo_format('path/to/your/dataset_name')
        """
        threads = []
        for root, _, files in os.walk(dataset_path):
            for file in files:
                if file.endswith(".json") and file!="label_map.json":
                    json_path = os.path.join(root, file)
                    img_path = json_path.replace("labels", "images").replace(
                        ".json", ".png"
                    )
                    thread = threading.Thread(
                        target=self._process_json_file, args=(json_path, img_path)
                    )
                    threads.append(thread)
                    thread.start()

        for thread in threads:
            thread.join()
