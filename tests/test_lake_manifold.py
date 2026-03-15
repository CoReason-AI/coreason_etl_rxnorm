# Copyright (c) 2026 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_etl_rxnorm

import pathlib

import boto3
import pytest
from botocore.exceptions import ClientError
from moto import mock_aws

from coreason_etl_rxnorm.config import FederatedRxNormConfigurationContract
from coreason_etl_rxnorm.lake_manifold import (
    EpistemicBronzeUploadManifest,
    execute_bronze_lake_upload_task,
)
from coreason_etl_rxnorm.network_manifold import SpatialExtractionManifest


@pytest.fixture
def mock_config() -> FederatedRxNormConfigurationContract:
    return FederatedRxNormConfigurationContract(
        umls_api_key="mock_api_key",
        bronze_bucket="s3://mock-bronze",
        silver_bucket="s3://mock-silver",
    )


@pytest.fixture
def mock_extraction_manifest(tmp_path: pathlib.Path) -> SpatialExtractionManifest:
    extract_dir = tmp_path / "rxnorm_extract"
    extract_dir.mkdir()

    files = ["RXNCONSO.RRF", "RXNREL.RRF", "RXNSAT.RRF"]
    extracted_paths = []

    for file_name in files:
        file_path = extract_dir / file_name
        file_path.write_text(f"mock content for {file_name}")
        extracted_paths.append(file_path)

    return SpatialExtractionManifest(
        extraction_path=extract_dir,
        extracted_files=extracted_paths,
    )


@mock_aws  # type: ignore[misc]
def test_execute_bronze_lake_upload_task_success(
    mock_config: FederatedRxNormConfigurationContract,
    mock_extraction_manifest: SpatialExtractionManifest,
) -> None:
    """Test successful S3 upload and local purge."""
    # Setup mock S3
    import os

    os.environ["AWS_DEFAULT_REGION"] = "us-east-1"
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket="mock-bronze")

    # Store paths to verify deletion later
    extraction_path = mock_extraction_manifest.extraction_path
    assert extraction_path.exists()

    # Execute task
    manifest = execute_bronze_lake_upload_task(mock_extraction_manifest, mock_config)

    # Verify manifest
    assert isinstance(manifest, EpistemicBronzeUploadManifest)
    assert len(manifest.uploaded_s3_uris) == 3
    assert "s3://mock-bronze/rxnorm/raw/RXNCONSO.RRF" in manifest.uploaded_s3_uris

    # Verify S3 contents
    response = s3.list_objects_v2(Bucket="mock-bronze", Prefix="rxnorm/raw/")
    assert "Contents" in response
    s3_keys = [obj["Key"] for obj in response["Contents"]]
    assert len(s3_keys) == 3
    assert "rxnorm/raw/RXNCONSO.RRF" in s3_keys

    # Verify local directory is purged
    assert not extraction_path.exists()


@mock_aws  # type: ignore[misc]
def test_execute_bronze_lake_upload_task_s3_failure(
    mock_config: FederatedRxNormConfigurationContract,
    mock_extraction_manifest: SpatialExtractionManifest,
) -> None:
    """Test S3 upload failure (e.g., bucket does not exist)."""
    # We do NOT create the bucket to trigger a ClientError (NoSuchBucket)

    extraction_path = mock_extraction_manifest.extraction_path
    assert extraction_path.exists()

    # Execute task and expect ClientError
    with pytest.raises((ClientError, boto3.exceptions.Boto3Error)):
        execute_bronze_lake_upload_task(mock_extraction_manifest, mock_config)

    # Verify local directory is NOT purged on failure
    assert extraction_path.exists()
