# Copyright (c) 2026 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_etl_rxnorm

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
