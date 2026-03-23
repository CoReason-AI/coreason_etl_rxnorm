# Copyright (c) 2026 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_etl_rxnorm

import sys

from coreason_etl_rxnorm.config import FederatedRxNormConfigurationContract
from coreason_etl_rxnorm.lake_manifold import (
    EpistemicGoldRegistrationManifest,
    execute_bronze_lake_upload_task,
    execute_gold_athena_registration_task,
    execute_gold_postgres_load_task,
    execute_silver_conso_transmutation_task,
    execute_silver_rel_transmutation_task,
    execute_silver_sat_transmutation_task,
)
from coreason_etl_rxnorm.network_manifold import (
    execute_epistemic_archive_download_task,
    execute_epistemic_release_metadata_fetch_task,
    execute_spatial_archive_extraction_task,
)
from coreason_etl_rxnorm.utils.logger import logger


def execute_federated_pipeline_intent(
    config: FederatedRxNormConfigurationContract,
) -> EpistemicGoldRegistrationManifest:
    """
    AGENT INSTRUCTION: Executes the complete end-to-end RxNorm ETL pipeline topology.

    Args:
        config: The validated configuration contract for the ETL process.

    Returns:
        The final registration manifest confirming availability in the Gold lake.
    """
    logger.info("Initiating federated RxNorm ETL pipeline intent.")

    # 1. Network Manifold: Fetch, Download, Extract
    metadata_manifest = execute_epistemic_release_metadata_fetch_task(config)
    archive_state = execute_epistemic_archive_download_task(metadata_manifest, config)
    extraction_manifest = execute_spatial_archive_extraction_task(archive_state)

    # 2. Lake Manifold: Bronze Upload
    _bronze_manifest = execute_bronze_lake_upload_task(extraction_manifest, config)

    # 3. Lake Manifold: Silver Transmutations
    # Find the specific extracted RRF files
    conso_file = next(f for f in extraction_manifest.extracted_files if f.name == "RXNCONSO.RRF")
    rel_file = next(f for f in extraction_manifest.extracted_files if f.name == "RXNREL.RRF")
    sat_file = next(f for f in extraction_manifest.extracted_files if f.name == "RXNSAT.RRF")

    conso_manifest = execute_silver_conso_transmutation_task(conso_file, config)
    rel_manifest = execute_silver_rel_transmutation_task(rel_file, config)
    sat_manifest = execute_silver_sat_transmutation_task(sat_file, config)

    # 4. Lake Manifold: Gold Athena Registration
    gold_manifest = execute_gold_athena_registration_task(
        conso_manifest=conso_manifest,
        rel_manifest=rel_manifest,
        sat_manifest=sat_manifest,
        config=config,
    )

    # 5. Optional Lake Manifold: Gold PostgreSQL Load
    if config.pghost and config.pgport and config.pguser and config.pgpassword and config.pgdatabase:
        _postgres_manifest = execute_gold_postgres_load_task(
            conso_manifest=conso_manifest,
            rel_manifest=rel_manifest,
            sat_manifest=sat_manifest,
            config=config,
        )

    logger.info("Successfully executed federated RxNorm ETL pipeline intent.")

    return gold_manifest


def main() -> None:
    """
    AGENT INSTRUCTION: Primary entry point for executing the pipeline via CLI.
    """
    try:
        # Pydantic Settings natively parses CLI arguments when `cli_parse_args=True`
        # is set in the model configuration. This avoids manual argparse boilerplate.
        config = FederatedRxNormConfigurationContract()
        execute_federated_pipeline_intent(config)
    except Exception:
        logger.exception("Pipeline execution failed.")
        sys.exit(1)


if __name__ == "__main__":  # pragma: no cover
    main()
