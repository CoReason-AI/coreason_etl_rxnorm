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
from unittest.mock import MagicMock, patch

from coreason_etl_rxnorm.config import FederatedRxNormConfigurationContract
from coreason_etl_rxnorm.lake_manifold import (
    EpistemicBronzeUploadManifest,
    EpistemicGoldPostgresManifest,
    EpistemicGoldRegistrationManifest,
    EpistemicSilverConsoManifest,
    EpistemicSilverRelManifest,
    EpistemicSilverSatManifest,
)
from coreason_etl_rxnorm.main import execute_federated_pipeline_intent, main
from coreason_etl_rxnorm.network_manifold import (
    EpistemicReleaseMetadataManifest,
    SpatialArchiveState,
    SpatialExtractionManifest,
)


def test_execute_federated_pipeline_intent() -> None:
    config = FederatedRxNormConfigurationContract(
        umls_api_key="mock_key",
        bronze_bucket="s3://mock-bronze",
        silver_bucket="s3://mock-silver",
        athena_database="mock_db",
    )

    mock_metadata_manifest = EpistemicReleaseMetadataManifest(download_url="https://example.com/rxnorm.zip")
    mock_archive_state = SpatialArchiveState(archive_path=pathlib.Path("mock_dir/mock.zip"))
    mock_extraction_manifest = SpatialExtractionManifest(
        extraction_path=pathlib.Path("mock_dir/mock_extract"),
        extracted_files=[
            pathlib.Path("mock_dir/mock_extract/RXNCONSO.RRF"),
            pathlib.Path("mock_dir/mock_extract/RXNREL.RRF"),
            pathlib.Path("mock_dir/mock_extract/RXNSAT.RRF"),
        ],
    )
    mock_bronze_manifest = EpistemicBronzeUploadManifest(uploaded_s3_uris=["s3://mock-bronze/rxnorm.zip"])
    mock_conso_manifest = EpistemicSilverConsoManifest(uploaded_s3_uri="s3://mock-silver/conso.parquet")
    mock_rel_manifest = EpistemicSilverRelManifest(uploaded_s3_uri="s3://mock-silver/rel.parquet")
    mock_sat_manifest = EpistemicSilverSatManifest(uploaded_s3_uri="s3://mock-silver/sat.parquet")
    mock_gold_manifest = EpistemicGoldRegistrationManifest(registered_tables=["table1"])

    with (
        patch(
            "coreason_etl_rxnorm.main.execute_epistemic_release_metadata_fetch_task",
            return_value=mock_metadata_manifest,
        ) as mock_fetch,
        patch(
            "coreason_etl_rxnorm.main.execute_epistemic_archive_download_task", return_value=mock_archive_state
        ) as mock_download,
        patch(
            "coreason_etl_rxnorm.main.execute_spatial_archive_extraction_task", return_value=mock_extraction_manifest
        ) as mock_extract,
        patch(
            "coreason_etl_rxnorm.main.execute_bronze_lake_upload_task", return_value=mock_bronze_manifest
        ) as mock_bronze,
        patch(
            "coreason_etl_rxnorm.main.execute_silver_conso_transmutation_task", return_value=mock_conso_manifest
        ) as mock_conso,
        patch(
            "coreason_etl_rxnorm.main.execute_silver_rel_transmutation_task", return_value=mock_rel_manifest
        ) as mock_rel,
        patch(
            "coreason_etl_rxnorm.main.execute_silver_sat_transmutation_task", return_value=mock_sat_manifest
        ) as mock_sat,
        patch(
            "coreason_etl_rxnorm.main.execute_gold_athena_registration_task", return_value=mock_gold_manifest
        ) as mock_gold,
    ):
        result = execute_federated_pipeline_intent(config)

        assert result == mock_gold_manifest

        mock_fetch.assert_called_once_with(config)
        mock_download.assert_called_once_with(mock_metadata_manifest, config)
        mock_extract.assert_called_once_with(mock_archive_state)
        mock_bronze.assert_called_once_with(mock_extraction_manifest, config)

        # Verify specific file objects are correctly filtered and passed
        conso_call_args = mock_conso.call_args[0]
        assert conso_call_args[0].name == "RXNCONSO.RRF"
        assert conso_call_args[1] == config

        rel_call_args = mock_rel.call_args[0]
        assert rel_call_args[0].name == "RXNREL.RRF"
        assert rel_call_args[1] == config

        sat_call_args = mock_sat.call_args[0]
        assert sat_call_args[0].name == "RXNSAT.RRF"
        assert sat_call_args[1] == config

        mock_gold.assert_called_once_with(
            conso_manifest=mock_conso_manifest,
            rel_manifest=mock_rel_manifest,
            sat_manifest=mock_sat_manifest,
            config=config,
        )


def test_main_invokes_postgres_load() -> None:
    config = FederatedRxNormConfigurationContract(
        umls_api_key="mock_key",
        bronze_bucket="s3://mock-bronze",
        silver_bucket="s3://mock-silver",
        athena_database="mock_db",
        pghost="localhost",
        pgport=5432,
        pguser="user",
        pgpassword="password",
        pgdatabase="db",
    )

    mock_metadata_manifest = EpistemicReleaseMetadataManifest(download_url="https://example.com/rxnorm.zip")
    mock_archive_state = SpatialArchiveState(archive_path=pathlib.Path("mock_dir/mock.zip"))
    mock_extraction_manifest = SpatialExtractionManifest(
        extraction_path=pathlib.Path("mock_dir/mock_extract"),
        extracted_files=[
            pathlib.Path("mock_dir/mock_extract/RXNCONSO.RRF"),
            pathlib.Path("mock_dir/mock_extract/RXNREL.RRF"),
            pathlib.Path("mock_dir/mock_extract/RXNSAT.RRF"),
        ],
    )
    mock_bronze_manifest = EpistemicBronzeUploadManifest(uploaded_s3_uris=["s3://mock-bronze/rxnorm.zip"])
    mock_conso_manifest = EpistemicSilverConsoManifest(uploaded_s3_uri="s3://mock-silver/conso.parquet")
    mock_rel_manifest = EpistemicSilverRelManifest(uploaded_s3_uri="s3://mock-silver/rel.parquet")
    mock_sat_manifest = EpistemicSilverSatManifest(uploaded_s3_uri="s3://mock-silver/sat.parquet")
    mock_gold_manifest = EpistemicGoldRegistrationManifest(registered_tables=["table1"])
    mock_postgres_manifest = EpistemicGoldPostgresManifest(loaded_tables=["table1"])

    with (
        patch(
            "coreason_etl_rxnorm.main.execute_epistemic_release_metadata_fetch_task",
            return_value=mock_metadata_manifest,
        ),
        patch("coreason_etl_rxnorm.main.execute_epistemic_archive_download_task", return_value=mock_archive_state),
        patch(
            "coreason_etl_rxnorm.main.execute_spatial_archive_extraction_task", return_value=mock_extraction_manifest
        ),
        patch("coreason_etl_rxnorm.main.execute_bronze_lake_upload_task", return_value=mock_bronze_manifest),
        patch("coreason_etl_rxnorm.main.execute_silver_conso_transmutation_task", return_value=mock_conso_manifest),
        patch("coreason_etl_rxnorm.main.execute_silver_rel_transmutation_task", return_value=mock_rel_manifest),
        patch("coreason_etl_rxnorm.main.execute_silver_sat_transmutation_task", return_value=mock_sat_manifest),
        patch("coreason_etl_rxnorm.main.execute_gold_athena_registration_task", return_value=mock_gold_manifest),
        patch(
            "coreason_etl_rxnorm.main.execute_gold_postgres_load_task", return_value=mock_postgres_manifest
        ) as mock_postgres,
    ):
        result = execute_federated_pipeline_intent(config)

        assert result == mock_gold_manifest

        mock_postgres.assert_called_once_with(
            conso_manifest=mock_conso_manifest,
            rel_manifest=mock_rel_manifest,
            sat_manifest=mock_sat_manifest,
            config=config,
        )


def test_main_skips_postgres_load() -> None:
    config = FederatedRxNormConfigurationContract(
        umls_api_key="mock_key",
        bronze_bucket="s3://mock-bronze",
        silver_bucket="s3://mock-silver",
        athena_database="mock_db",
    )

    mock_metadata_manifest = EpistemicReleaseMetadataManifest(download_url="https://example.com/rxnorm.zip")
    mock_archive_state = SpatialArchiveState(archive_path=pathlib.Path("mock_dir/mock.zip"))
    mock_extraction_manifest = SpatialExtractionManifest(
        extraction_path=pathlib.Path("mock_dir/mock_extract"),
        extracted_files=[
            pathlib.Path("mock_dir/mock_extract/RXNCONSO.RRF"),
            pathlib.Path("mock_dir/mock_extract/RXNREL.RRF"),
            pathlib.Path("mock_dir/mock_extract/RXNSAT.RRF"),
        ],
    )
    mock_bronze_manifest = EpistemicBronzeUploadManifest(uploaded_s3_uris=["s3://mock-bronze/rxnorm.zip"])
    mock_conso_manifest = EpistemicSilverConsoManifest(uploaded_s3_uri="s3://mock-silver/conso.parquet")
    mock_rel_manifest = EpistemicSilverRelManifest(uploaded_s3_uri="s3://mock-silver/rel.parquet")
    mock_sat_manifest = EpistemicSilverSatManifest(uploaded_s3_uri="s3://mock-silver/sat.parquet")
    mock_gold_manifest = EpistemicGoldRegistrationManifest(registered_tables=["table1"])

    with (
        patch(
            "coreason_etl_rxnorm.main.execute_epistemic_release_metadata_fetch_task",
            return_value=mock_metadata_manifest,
        ),
        patch("coreason_etl_rxnorm.main.execute_epistemic_archive_download_task", return_value=mock_archive_state),
        patch(
            "coreason_etl_rxnorm.main.execute_spatial_archive_extraction_task", return_value=mock_extraction_manifest
        ),
        patch("coreason_etl_rxnorm.main.execute_bronze_lake_upload_task", return_value=mock_bronze_manifest),
        patch("coreason_etl_rxnorm.main.execute_silver_conso_transmutation_task", return_value=mock_conso_manifest),
        patch("coreason_etl_rxnorm.main.execute_silver_rel_transmutation_task", return_value=mock_rel_manifest),
        patch("coreason_etl_rxnorm.main.execute_silver_sat_transmutation_task", return_value=mock_sat_manifest),
        patch("coreason_etl_rxnorm.main.execute_gold_athena_registration_task", return_value=mock_gold_manifest),
        patch("coreason_etl_rxnorm.main.execute_gold_postgres_load_task") as mock_postgres,
    ):
        result = execute_federated_pipeline_intent(config)

        assert result == mock_gold_manifest

        mock_postgres.assert_not_called()


@patch(
    "sys.argv",
    [
        "main.py",
        "--umls-api-key",
        "cli_key",
        "--bronze-bucket",
        "s3://cli-bronze",
        "--silver-bucket",
        "s3://cli-silver",
        "--athena-database",
        "cli_db",
    ],
)
@patch("coreason_etl_rxnorm.main.execute_federated_pipeline_intent")
def test_cli_execution_success(mock_execute: MagicMock) -> None:
    main()
    mock_execute.assert_called_once()
    config_arg = mock_execute.call_args[0][0]
    assert config_arg.umls_api_key == "cli_key"
    assert config_arg.bronze_bucket == "s3://cli-bronze"
    assert config_arg.silver_bucket == "s3://cli-silver"
    assert config_arg.athena_database == "cli_db"


@patch("sys.argv", ["main.py"])
@patch("coreason_etl_rxnorm.main.execute_federated_pipeline_intent")
@patch.dict(
    "os.environ",
    {
        "UMLS_API_KEY": "env_key",
        "BRONZE_BUCKET": "s3://env-bronze",
        "SILVER_BUCKET": "s3://env-silver",
        "ATHENA_DATABASE": "env_db",
    },
)
def test_cli_execution_fallback_to_env(mock_execute: MagicMock) -> None:
    main()
    mock_execute.assert_called_once()
    config_arg = mock_execute.call_args[0][0]
    assert config_arg.umls_api_key == "env_key"
    assert config_arg.bronze_bucket == "s3://env-bronze"
    assert config_arg.silver_bucket == "s3://env-silver"
    assert config_arg.athena_database == "env_db"


@patch("sys.argv", ["main.py", "--umls-api-key", "cli_key"])
@patch.dict("os.environ", {}, clear=True)
def test_cli_execution_failure_missing_args() -> None:
    import pytest

    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 1


@patch("sys.argv", ["main.py", "--pgport", "invalid_port"])
@patch.dict("os.environ", {}, clear=True)
def test_cli_execution_invalid_port_type() -> None:
    import pytest

    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 2


@patch("sys.argv", ["main.py", "--unknown-arg", "value"])
@patch.dict("os.environ", {}, clear=True)
def test_cli_execution_unrecognized_arg() -> None:
    import pytest

    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 2


@patch(
    "sys.argv",
    [
        "main.py",
        "--umls-api-key",
        "cli_key",
        "--pghost",
        "cli_host",
    ],
)
@patch.dict(
    "os.environ",
    {
        "BRONZE_BUCKET": "s3://env-bronze",
        "SILVER_BUCKET": "s3://env-silver",
        "ATHENA_DATABASE": "env_db",
        "PGPORT": "5432",
    },
)
@patch("coreason_etl_rxnorm.main.execute_federated_pipeline_intent")
def test_cli_execution_mixed_args_and_env(mock_execute: MagicMock) -> None:
    main()
    mock_execute.assert_called_once()
    config_arg = mock_execute.call_args[0][0]

    # CLI args
    assert config_arg.umls_api_key == "cli_key"
    assert config_arg.pghost == "cli_host"

    # Env vars
    assert config_arg.bronze_bucket == "s3://env-bronze"
    assert config_arg.silver_bucket == "s3://env-silver"
    assert config_arg.athena_database == "env_db"
    assert config_arg.pgport == 5432


@patch(
    "sys.argv",
    [
        "main.py",
        "--umls-api-key",
        "cli_key",
        "--bronze-bucket",
        "s3://cli-bronze",
        "--silver-bucket",
        "s3://cli-silver",
        "--athena-database",
        "cli_db",
    ],
)
@patch("coreason_etl_rxnorm.main.execute_federated_pipeline_intent", side_effect=Exception("Pipeline failed"))
def test_cli_execution_pipeline_failure(_mock_execute: MagicMock) -> None:
    import pytest

    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 1


def test_main_module_execution() -> None:
    import contextlib
    import runpy
    import sys

    with (
        patch.object(sys, "argv", ["coreason_etl_rxnorm", "--help"]),
        patch("coreason_etl_rxnorm.main.main") as mock_main,
    ):
        with contextlib.suppress(SystemExit):
            runpy.run_module("coreason_etl_rxnorm", run_name="__main__")
        mock_main.assert_called_once()
