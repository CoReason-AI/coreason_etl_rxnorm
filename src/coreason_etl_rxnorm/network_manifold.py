# Copyright (c) 2026 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_etl_rxnorm

import contextlib
import pathlib
import tempfile

import requests
from pydantic import BaseModel, Field, HttpUrl

from coreason_etl_rxnorm.config import FederatedRxNormConfigurationContract
from coreason_etl_rxnorm.utils.logger import logger


class EpistemicReleaseMetadataManifest(BaseModel):
    """
    AGENT INSTRUCTION: This object represents the mandatory epistemic state
    of a fetched RxNorm release metadata payload.
    """

    download_url: HttpUrl = Field(
        ...,
        description="The resolved HTTP URL string for downloading the current RxNorm full monthly release archive.",
    )


def execute_epistemic_release_metadata_fetch_task(
    config: FederatedRxNormConfigurationContract,
) -> EpistemicReleaseMetadataManifest:
    """
    AGENT INSTRUCTION: Executes the intent to fetch the current monthly release metadata
    from the NLM UTS API and transmutes it into an EpistemicReleaseMetadataManifest.

    Args:
        config: The validated configuration contract containing the UMLS API key.

    Returns:
        An EpistemicReleaseMetadataManifest containing the target download URL.

    Raises:
        requests.HTTPError: If the upstream NLM API rejects the request or fails.
        KeyError: If the expected JSON topology is structurally anomalous.
    """
    logger.info("Executing EpistemicReleaseMetadataFetchTask against NLM UTS Release API.")

    url = "https://uts-ws.nlm.nih.gov/releases"
    params = {
        "releaseType": "rxnorm-full-monthly-release",
        "current": "true",
        "apiKey": config.umls_api_key,
    }

    response = requests.get(url, params=params, timeout=30.0)
    response.raise_for_status()

    payload = response.json()

    # Extract the download URL according to expected JSON structure
    try:
        # Assuming the API returns a list of release objects or a single object.
        # Based on typical UTS API for releases, it returns a list when queried,
        # or a single result if structured as such. Let's inspect the payload.
        # Assuming typical payload format where the url is under result/downloadUrl or similar.
        # NLM API returns: [{"downloadUrl": "...", "version": "...", ...}] for current=true
        if isinstance(payload, list) and len(payload) > 0:
            download_url = payload[0]["downloadUrl"]
        else:
            # Fallback if it's a dict and has 'downloadUrl' directly or nested
            download_url = payload.get("downloadUrl")
            if not download_url:
                raise KeyError("downloadUrl not found in payload topology.")
    except (KeyError, TypeError, IndexError) as e:
        logger.exception("Failed to extract downloadUrl from anomalous JSON payload topology.")
        raise KeyError("Anomalous JSON payload topology encountered.") from e

    logger.info("Successfully fetched EpistemicReleaseMetadataManifest.", url=str(download_url))

    return EpistemicReleaseMetadataManifest(download_url=download_url)


class SpatialArchiveState(BaseModel):
    """
    AGENT INSTRUCTION: This object represents the mandatory spatial state
    of a downloaded RxNorm ZIP archive resting on local disk.
    """

    archive_path: pathlib.Path = Field(
        ...,
        description="The local filesystem path to the completely downloaded RxNorm ZIP archive.",
    )


def execute_epistemic_archive_download_task(
    manifest: EpistemicReleaseMetadataManifest,
    config: FederatedRxNormConfigurationContract,
) -> SpatialArchiveState:
    """
    AGENT INSTRUCTION: Executes the intent to securely download the remote RxNorm archive
    specified in the metadata manifest to a local temporary file.

    Args:
        manifest: The EpistemicReleaseMetadataManifest containing the download URL.
        config: The FederatedRxNormConfigurationContract containing the UMLS API key.

    Returns:
        A SpatialArchiveState containing the local file path to the downloaded archive.

    Raises:
        requests.HTTPError: If the upstream NLM API rejects the request or fails.
    """
    logger.info("Executing EpistemicArchiveDownloadTask to stream the RxNorm archive.")

    # Using UTS Download API: https://uts-ws.nlm.nih.gov/download?url=<downloadUrl>&apiKey=<apiKey>
    url = "https://uts-ws.nlm.nih.gov/download"
    params = {
        "url": str(manifest.download_url),
        "apiKey": config.umls_api_key,
    }

    # Use a secure NamedTemporaryFile that persists after closing
    # suffix=".zip" makes it recognizable for extraction tasks
    # delete=False means it must be manually purged later
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as temp_file:
        temp_path = pathlib.Path(temp_file.name)

    logger.debug("Streaming archive to temporary manifold.", path=str(temp_path))

    try:
        with requests.get(url, params=params, stream=True, timeout=60.0) as response:
            response.raise_for_status()

            # Stream chunks securely to disk
            with open(temp_path, "wb") as f:
                f.writelines(response.iter_content(chunk_size=8192))
    except Exception as e:
        logger.exception("Failed to stream archive to SpatialArchiveState.")
        # Attempt to cleanup the partial file on failure
        if temp_path.exists():
            with contextlib.suppress(Exception):
                temp_path.unlink()
        raise e

    logger.info("Successfully transmuted stream into SpatialArchiveState.", path=str(temp_path))

    return SpatialArchiveState(archive_path=temp_path)
