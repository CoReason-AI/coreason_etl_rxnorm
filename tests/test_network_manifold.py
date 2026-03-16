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
import zipfile

import pytest
import requests
import responses
from pydantic import ValidationError

from coreason_etl_rxnorm.config import FederatedRxNormConfigurationContract
from coreason_etl_rxnorm.network_manifold import (
    EpistemicReleaseMetadataManifest,
    SpatialArchiveState,
    SpatialExtractionManifest,
    execute_epistemic_archive_download_task,
    execute_epistemic_release_metadata_fetch_task,
    execute_spatial_archive_extraction_task,
)


@pytest.fixture
def valid_config() -> FederatedRxNormConfigurationContract:
    return FederatedRxNormConfigurationContract(
        umls_api_key="mock_api_key",
        bronze_bucket="s3://mock-bronze",
        silver_bucket="s3://mock-silver",
        athena_database="mock_db",
    )


@responses.activate
def test_execute_epistemic_release_metadata_fetch_task_success_list(
    valid_config: FederatedRxNormConfigurationContract,
) -> None:
    """Test successful task execution with list topology."""
    responses.add(
        responses.GET,
        "https://uts-ws.nlm.nih.gov/releases",
        json=[{"downloadUrl": "https://example.com/rxnorm.zip"}],
        status=200,
    )

    manifest = execute_epistemic_release_metadata_fetch_task(valid_config)
    assert isinstance(manifest, EpistemicReleaseMetadataManifest)
    assert str(manifest.download_url) == "https://example.com/rxnorm.zip"


@responses.activate
def test_execute_epistemic_release_metadata_fetch_task_success_dict(
    valid_config: FederatedRxNormConfigurationContract,
) -> None:
    """Test successful task execution with dict topology."""
    responses.add(
        responses.GET,
        "https://uts-ws.nlm.nih.gov/releases",
        json={"downloadUrl": "https://example.com/rxnorm_dict.zip"},
        status=200,
    )

    manifest = execute_epistemic_release_metadata_fetch_task(valid_config)
    assert isinstance(manifest, EpistemicReleaseMetadataManifest)
    assert str(manifest.download_url) == "https://example.com/rxnorm_dict.zip"


@responses.activate
def test_execute_epistemic_release_metadata_fetch_task_http_error(
    valid_config: FederatedRxNormConfigurationContract,
) -> None:
    """Test task execution failure due to HTTP error."""
    responses.add(
        responses.GET,
        "https://uts-ws.nlm.nih.gov/releases",
        json={"error": "Unauthorized"},
        status=401,
    )

    with pytest.raises(requests.HTTPError):
        execute_epistemic_release_metadata_fetch_task(valid_config)


@responses.activate
def test_execute_epistemic_release_metadata_fetch_task_missing_url(
    valid_config: FederatedRxNormConfigurationContract,
) -> None:
    """Test task execution failure due to anomalous JSON topology."""
    responses.add(
        responses.GET,
        "https://uts-ws.nlm.nih.gov/releases",
        json=[{"wrongKey": "value"}],
        status=200,
    )

    with pytest.raises(KeyError, match=r"Anomalous JSON payload topology encountered\."):
        execute_epistemic_release_metadata_fetch_task(valid_config)


@responses.activate
def test_execute_epistemic_release_metadata_fetch_task_dict_missing_url(
    valid_config: FederatedRxNormConfigurationContract,
) -> None:
    """Test task execution failure due to anomalous JSON topology (dict)."""
    responses.add(
        responses.GET,
        "https://uts-ws.nlm.nih.gov/releases",
        json={"wrongKey": "value"},
        status=200,
    )

    with pytest.raises(KeyError, match=r"Anomalous JSON payload topology encountered\."):
        execute_epistemic_release_metadata_fetch_task(valid_config)


def test_epistemic_release_metadata_manifest_validation() -> None:
    """Test EpistemicReleaseMetadataManifest validation for download_url."""
    manifest = EpistemicReleaseMetadataManifest(download_url="https://valid.com/file.zip")
    assert str(manifest.download_url) == "https://valid.com/file.zip"

    with pytest.raises(ValidationError):
        EpistemicReleaseMetadataManifest(download_url="not_a_url")


@responses.activate
def test_execute_epistemic_archive_download_task_success(
    valid_config: FederatedRxNormConfigurationContract,
) -> None:
    """Test successful archive download task execution."""
    manifest = EpistemicReleaseMetadataManifest(download_url="https://example.com/rxnorm.zip")

    responses.add(
        responses.GET,
        "https://uts-ws.nlm.nih.gov/download",
        body=b"mock_zip_content",
        status=200,
    )

    state = execute_epistemic_archive_download_task(manifest, valid_config)

    assert isinstance(state, SpatialArchiveState)
    assert isinstance(state.archive_path, pathlib.Path)
    assert state.archive_path.exists()

    with open(state.archive_path, "rb") as f:
        assert f.read() == b"mock_zip_content"

    # Clean up
    state.archive_path.unlink()


@responses.activate
def test_execute_epistemic_archive_download_task_http_error(
    valid_config: FederatedRxNormConfigurationContract,
) -> None:
    """Test task execution failure due to HTTP error and ensure cleanup."""
    manifest = EpistemicReleaseMetadataManifest(download_url="https://example.com/rxnorm.zip")

    responses.add(
        responses.GET,
        "https://uts-ws.nlm.nih.gov/download",
        body=b"Unauthorized",
        status=401,
    )

    with pytest.raises(requests.HTTPError):
        execute_epistemic_archive_download_task(manifest, valid_config)


def test_execute_spatial_archive_extraction_task_success(tmp_path: pathlib.Path) -> None:
    """Test successful extraction of mandatory RRF files."""
    archive_path = tmp_path / "test_rxnorm.zip"

    with zipfile.ZipFile(archive_path, "w") as z:
        z.writestr("rrf/RXNCONSO.RRF", "mock_conso_content")
        z.writestr("rrf/RXNREL.RRF", "mock_rel_content")
        z.writestr("rrf/RXNSAT.RRF", "mock_sat_content")
        z.writestr("rrf/EXTRA.RRF", "extra_content")

    archive_state = SpatialArchiveState(archive_path=archive_path)

    manifest = execute_spatial_archive_extraction_task(archive_state)

    assert isinstance(manifest, SpatialExtractionManifest)
    assert manifest.extraction_path.exists()
    assert len(manifest.extracted_files) == 3

    extracted_names = [p.name for p in manifest.extracted_files]
    assert "RXNCONSO.RRF" in extracted_names
    assert "RXNREL.RRF" in extracted_names
    assert "RXNSAT.RRF" in extracted_names

    # Clean up
    import shutil

    shutil.rmtree(manifest.extraction_path)


def test_execute_spatial_archive_extraction_task_missing_files(tmp_path: pathlib.Path) -> None:
    """Test extraction failure when mandatory files are missing."""
    archive_path = tmp_path / "test_rxnorm_missing.zip"

    with zipfile.ZipFile(archive_path, "w") as z:
        z.writestr("rrf/RXNCONSO.RRF", "mock_conso_content")
        z.writestr("rrf/RXNSAT.RRF", "mock_sat_content")

    archive_state = SpatialArchiveState(archive_path=archive_path)

    with pytest.raises(KeyError, match=r"Mandatory files missing from archive topology"):
        execute_spatial_archive_extraction_task(archive_state)


def test_execute_spatial_archive_extraction_task_bad_zip(tmp_path: pathlib.Path) -> None:
    """Test extraction failure with invalid ZIP file."""
    archive_path = tmp_path / "test_rxnorm_bad.zip"
    archive_path.write_text("not a zip file")

    archive_state = SpatialArchiveState(archive_path=archive_path)

    with pytest.raises(zipfile.BadZipFile):
        execute_spatial_archive_extraction_task(archive_state)
