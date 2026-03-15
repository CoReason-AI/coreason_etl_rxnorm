# Copyright (c) 2026 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_etl_rxnorm

import shutil

import boto3
from pydantic import BaseModel, Field

from coreason_etl_rxnorm.config import FederatedRxNormConfigurationContract
from coreason_etl_rxnorm.network_manifold import SpatialExtractionManifest
from coreason_etl_rxnorm.utils.logger import logger


class EpistemicBronzeUploadManifest(BaseModel):
    """
    AGENT INSTRUCTION: This object represents the mandatory epistemic state
    of successfully uploaded RxNorm RRF files in the Bronze S3 zone.
    """

    uploaded_s3_uris: list[str] = Field(
        ...,
        description="The specific S3 URIs of the uploaded RRF files.",
    )


def execute_bronze_lake_upload_task(
    manifest: SpatialExtractionManifest,
    config: FederatedRxNormConfigurationContract,
) -> EpistemicBronzeUploadManifest:
    """
    AGENT INSTRUCTION: Executes the intent to upload the extracted RxNorm files
    to the designated Bronze S3 prefix using boto3. Once uploaded, the local
    extraction directory is explicitly purged to free disk space.

    Args:
        manifest: The SpatialExtractionManifest containing the local paths to extracted files.
        config: The configuration contract containing the S3 bronze bucket URI.

    Returns:
        An EpistemicBronzeUploadManifest containing the S3 URIs of the uploaded files.

    Raises:
        botocore.exceptions.ClientError: If the upload fails due to S3 or permissions errors.
    """
    logger.info("Executing BronzeLakeUploadTask to upload raw RxNorm files to Bronze layer.")

    # Remove 's3://' prefix if present for boto3 bucket parsing
    # But since Pydantic does not enforce it as AnyUrl, we strip if present
    # Better: explicitly handle "s3://bucket-name"
    bronze_uri = config.bronze_bucket
    bucket_name = bronze_uri.removeprefix("s3://")

    # In case of trailing slash, strip it
    bucket_name = bucket_name.strip("/")

    s3_client = boto3.client("s3")
    uploaded_uris: list[str] = []

    try:
        for file_path in manifest.extracted_files:
            # We enforce write_disposition="replace" implicitly by overwriting whatever is at the key.
            # Base prefix: /rxnorm/raw/
            s3_key = f"rxnorm/raw/{file_path.name}"

            logger.debug(f"Uploading {file_path} to s3://{bucket_name}/{s3_key}")

            s3_client.upload_file(str(file_path), bucket_name, s3_key)

            uploaded_uris.append(f"s3://{bucket_name}/{s3_key}")

    except boto3.exceptions.S3UploadFailedError as e:
        logger.exception("Failed to upload SpatialExtractionManifest to Bronze lake.")
        # botocore ClientError is inside boto3 exceptions often,
        # but to strictly match standard requirements, we can catch Exception and re-raise.
        raise e
    except Exception as e:  # pragma: no cover
        logger.exception("Failed to upload SpatialExtractionManifest to Bronze lake.")
        raise e

    # Purge local files after successful upload
    try:
        if manifest.extraction_path.exists():
            shutil.rmtree(str(manifest.extraction_path))
            logger.info("Successfully purged local extraction manifold.", path=str(manifest.extraction_path))
    except Exception as e:  # pragma: no cover
        logger.warning(f"Failed to purge local extraction manifold: {e}", path=str(manifest.extraction_path))

    logger.info("Successfully transmuted SpatialExtractionManifest into EpistemicBronzeUploadManifest.")
    return EpistemicBronzeUploadManifest(uploaded_s3_uris=uploaded_uris)
