# Copyright (c) 2026 CoReason, Inc.
# Source Code: https://github.com/CoReason-AI/coreason_etl_rxnorm

import sys

from coreason_etl_rxnorm.config import FederatedRxNormConfigurationContract
from coreason_etl_rxnorm.lake_manifold import execute_omop_to_postgres_pipeline
from coreason_etl_rxnorm.network_manifold import (
    execute_epistemic_archive_download_task,
    execute_epistemic_release_metadata_fetch_task,
    execute_spatial_archive_extraction_task,
)
from coreason_etl_rxnorm.utils.logger import logger


def execute_federated_pipeline_intent(
    config: FederatedRxNormConfigurationContract,
) -> None:
    logger.info("Initiating federated RxNorm ETL pipeline intent.")

    # 1. Network Manifold: Fetch, Download, Extract
    metadata_manifest = execute_epistemic_release_metadata_fetch_task(config)
    archive_state = execute_epistemic_archive_download_task(metadata_manifest, config)
    extraction_manifest = execute_spatial_archive_extraction_task(archive_state)

    # 2. Lake Manifold: Transform and Load to Postgres
    execute_omop_to_postgres_pipeline(extraction_manifest, config)

    logger.info("Successfully executed federated RxNorm ETL pipeline intent.")


def main() -> None:
    try:
        config = FederatedRxNormConfigurationContract()
        execute_federated_pipeline_intent(config)
    except Exception:
        logger.exception("Pipeline execution failed.")
        sys.exit(1)


if __name__ == "__main__":  # pragma: no cover
    main()
