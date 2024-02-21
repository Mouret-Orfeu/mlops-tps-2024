from zenml import step
from zenml.logger import get_logger

from src.config.settings import (
    MINIO_DATA_SOURCES_BUCKET_NAME,
)
from src.models.model_bucket_client import BucketClient
from src.models.model_data_source import DataSource, DataSourceList
from src.services.service_data_uploader import DataUploaderService
from src.steps.data.datalake_initializers import validate_bucket_connection


def get_data_sources_bucket_name() -> str:
    """
    Retrieves the bucket name from the configuration.
    """
    return MINIO_DATA_SOURCES_BUCKET_NAME


def verify_data_source_path(data_source: DataSource) -> None:
    """
    Checks if the provided data source's path is valid.
    """
    logger = get_logger(__name__)

    try:
        data_source.verify_data_source_path()
    except Exception as e:
        logger.error(f"The data source's path is not valid: {type(e).__name__}: {e}")
        raise


def upload_data(
    data_uploader_service: DataUploaderService,
    bucket_name: str,
    data_source: DataSource,
) -> None:
    """
    Uploads data from the provided path to the bucket.
    """
    logger = get_logger(__name__)

    try:
        data_uploader_service.upload_data(
            bucket_name=bucket_name, data_source=data_source
        )
    except TypeError:
        logger.error(
            f"Couldn't upload the data source, type {type(data_source).__name__} is not"
            " supported."
        )
        raise


@step
def data_sources_uploader(
    bucket_client: BucketClient, data_source_list: DataSourceList
) -> None:
    """
    Flow for preparing data sources, which includes validating the data path, checking the bucket connection,
    configuring the bucket, and uploading the data.
    """
    data_uploader_service = DataUploaderService(bucket_client)
    validate_bucket_connection(bucket_client=bucket_client)

    for data_source in data_source_list.data_sources:
        verify_data_source_path(data_source=data_source)

        upload_data(
            data_uploader_service=data_uploader_service,
            bucket_name=get_data_sources_bucket_name(),
            data_source=data_source,
        )
