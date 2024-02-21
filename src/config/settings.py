from decouple import AutoConfig

config = AutoConfig(search_path="src/config")

MINIO_HOST: str = config("MINIO_HOST")
MINIO_PORT: str = config("MINIO_PORT")
MINIO_ENDPOINT: str = f"{MINIO_HOST}:{MINIO_PORT}"

MINIO_ROOT_USER: str = config("MINIO_ROOT_USER")
MINIO_ROOT_PASSWORD: str = config("MINIO_ROOT_PASSWORD")

MINIO_PENDING_ANNOTATIONS_BUCKET_NAME: str = config(
    "MINIO_PENDING_ANNOTATIONS_BUCKET_NAME"
)
MINIO_PENDING_REVIEWS_BUCKET_NAME: str = config("MINIO_PENDING_REVIEWS_BUCKET_NAME")
MINIO_DATA_SOURCES_BUCKET_NAME: str = config("MINIO_DATA_SOURCES_BUCKET_NAME")
MINIO_DATASETS_BUCKET_NAME: str = config("MINIO_DATASETS_BUCKET_NAME")

YOLO_PRE_TRAINED_WEIGHTS_PATH: str = "ultralytics"
EXTRACTED_DATASETS_PATH: str = "datasets"
DATASET_YOLO_CONFIG_NAME: str = "dataset.yaml"
YOLO_PRE_TRAINED_WEIGHTS_NAME: str = "yolov8s.pt"
YOLO_PRE_TRAINED_WEIGHTS_URL: str = (
    f"https://github.com/ultralytics/assets/releases/download/v8.1.0/{YOLO_PRE_TRAINED_WEIGHTS_NAME}"
)

MLFLOW_EXPERIMENT_PIPELINE_NAME: str = "local-experiment-pipeline"
MLFLOW_END_TO_END_PIPELINE_NAME: str = "production-end-to-end-pipeline"
