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
import tempfile
import typing

import boto3
import polars as pl
import pytest
from botocore.exceptions import ClientError
from moto import mock_aws
from pydantic import ValidationError

from coreason_etl_rxnorm.config import NAMESPACE_RXNORM, FederatedRxNormConfigurationContract
from coreason_etl_rxnorm.lake_manifold import (
    EpistemicBronzeUploadManifest,
    EpistemicGoldPostgresManifest,
    EpistemicGoldRegistrationManifest,
    EpistemicSilverConsoManifest,
    EpistemicSilverRelManifest,
    EpistemicSilverSatManifest,
    execute_bronze_lake_upload_task,
    execute_gold_athena_registration_task,
    execute_gold_postgres_load_task,
    execute_silver_conso_transmutation_task,
    execute_silver_rel_transmutation_task,
    execute_silver_sat_transmutation_task,
)
from coreason_etl_rxnorm.network_manifold import SpatialExtractionManifest


@pytest.fixture
def mock_config() -> FederatedRxNormConfigurationContract:
    return FederatedRxNormConfigurationContract(
        umls_api_key="mock_api_key",
        bronze_bucket="s3://mock-bronze",
        silver_bucket="s3://mock-silver",
        athena_database="mock_db",
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


@mock_aws
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


@mock_aws
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


@pytest.fixture
def mock_rxnconso_file(tmp_path: pathlib.Path) -> pathlib.Path:
    conso_file = tmp_path / "RXNCONSO.RRF"
    # Format: 18 pipe-delimited values + trailing pipe per line.
    # Columns: RXCUI, LAT, TS, LUI, STT, SUI, ISPREF, RXAUI, SAUI, SCUI, SDUI, SAB, TTY, CODE, STR, SRL, SUPPRESS, CVF
    lines = [
        "100|ENG||||||1001||||RXNORM|SCD|100|Concept One||N||",  # Valid
        "101|FRE||||||1002||||RXNORM|SCD|101|Concept Two French||N||",  # Non-English, should be filtered
        "102|ENG||||||1003||||RXNORM|SCD|102|Concept Three Obsolete||O||",  # Obsolete, should be filtered
        "103|ENG||||||1004||||RXNORM|SCD|103|Concept Four Suppressed||Y||",  # Suppressed, should be filtered
        "104|ENG||||||1005||||RXNORM|SCD|104|Concept Five Unquantified||E||",  # E is allowed
    ]
    conso_file.write_text("\n".join(lines))
    return conso_file


@mock_aws
def test_execute_silver_conso_transmutation_task_success(
    mock_config: FederatedRxNormConfigurationContract,
    mock_rxnconso_file: pathlib.Path,
) -> None:
    """Test successful transmutation and upload of RXNCONSO to Silver layer."""
    import os
    import uuid

    os.environ["AWS_DEFAULT_REGION"] = "us-east-1"
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket="mock-silver")

    manifest = execute_silver_conso_transmutation_task(mock_rxnconso_file, mock_config)

    assert isinstance(manifest, EpistemicSilverConsoManifest)
    assert manifest.uploaded_s3_uri == "s3://mock-silver/rxnorm/clean/dim_rxnorm_concept/dim_rxnorm_concept.parquet"

    # Download Parquet from S3 to verify contents
    s3.download_file(
        "mock-silver",
        "rxnorm/clean/dim_rxnorm_concept/dim_rxnorm_concept.parquet",
        str(mock_rxnconso_file.parent / "downloaded.parquet"),
    )

    df = pl.read_parquet(mock_rxnconso_file.parent / "downloaded.parquet")

    # Should only contain RXCUI 100 and 104
    assert len(df) == 2

    # Check typing and specific values
    assert df["rxcui_id"].dtype == pl.String
    assert df["rxcui_id"].to_list() == ["100", "104"]
    assert df["concept_name"].to_list() == ["Concept One", "Concept Five Unquantified"]

    # Check Dual ID (coreason_id) deterministic hashing
    expected_id_100 = str(uuid.uuid5(NAMESPACE_RXNORM, "100"))
    expected_id_104 = str(uuid.uuid5(NAMESPACE_RXNORM, "104"))
    assert df["coreason_id"].to_list() == [expected_id_100, expected_id_104]


@mock_aws
def test_execute_silver_conso_transmutation_task_s3_error(
    mock_config: FederatedRxNormConfigurationContract,
    mock_rxnconso_file: pathlib.Path,
) -> None:
    """Test failure of S3 upload for Silver layer and ensure temp cleanup."""
    import os

    os.environ["AWS_DEFAULT_REGION"] = "us-east-1"

    # Do NOT create the bucket mock-silver to cause ClientError
    with pytest.raises((ClientError, boto3.exceptions.Boto3Error)):
        execute_silver_conso_transmutation_task(mock_rxnconso_file, mock_config)


def test_epistemic_silver_rel_manifest_validation() -> None:
    """Test validation of EpistemicSilverRelManifest."""
    manifest = EpistemicSilverRelManifest(
        uploaded_s3_uri="s3://my-silver-bucket/rxnorm/clean/fact_rxnorm_relationship/fact_rxnorm_relationship.parquet"
    )
    assert (
        manifest.uploaded_s3_uri
        == "s3://my-silver-bucket/rxnorm/clean/fact_rxnorm_relationship/fact_rxnorm_relationship.parquet"
    )

    with pytest.raises(ValidationError):
        EpistemicSilverRelManifest()  # type: ignore[call-arg]


@mock_aws
def test_execute_silver_rel_transmutation_task_success(
    mock_config: FederatedRxNormConfigurationContract,
) -> None:
    """Test successful transmutation of RXNREL to Silver zone."""
    import os

    os.environ["AWS_DEFAULT_REGION"] = "us-east-1"
    s3_client = boto3.client("s3")
    bucket_name = mock_config.silver_bucket.removeprefix("s3://").strip("/")
    s3_client.create_bucket(Bucket=bucket_name)

    # Create dummy RXNREL.RRF
    with tempfile.NamedTemporaryFile(suffix=".RRF", delete=False, mode="w") as temp_file:
        # Columns:
        # Columns omitted for brevity

        # valid row (SAB=RXNORM)
        temp_file.write("1111|||RO|2222|||has_ingredient|||RXNORM||||N|||\n")
        # invalid row (SAB=SNOMEDCT_US)
        temp_file.write("3333|||RO|4444|||has_part|||SNOMEDCT_US||||N|||\n")

        temp_path = pathlib.Path(temp_file.name)

    try:
        manifest = execute_silver_rel_transmutation_task(temp_path, mock_config)
        assert (
            manifest.uploaded_s3_uri
            == f"s3://{bucket_name}/rxnorm/clean/fact_rxnorm_relationship/fact_rxnorm_relationship.parquet"
        )

        # Verify file exists in mock S3
        response = s3_client.head_object(
            Bucket=bucket_name,
            Key="rxnorm/clean/fact_rxnorm_relationship/fact_rxnorm_relationship.parquet",
        )
        assert response["ContentLength"] > 0
    finally:
        if temp_path.exists():
            temp_path.unlink()


@mock_aws
def test_execute_silver_rel_transmutation_task_s3_error(
    mock_config: FederatedRxNormConfigurationContract,
) -> None:
    """Test transmutation fails gracefully when S3 bucket is missing."""
    import os

    os.environ["AWS_DEFAULT_REGION"] = "us-east-1"

    with tempfile.NamedTemporaryFile(suffix=".RRF", delete=False, mode="w") as temp_file:
        temp_file.write("1111|||RO|2222|||has_ingredient|||RXNORM||||N|||\n")
        temp_path = pathlib.Path(temp_file.name)

    try:
        with pytest.raises((ClientError, boto3.exceptions.Boto3Error, Exception)):
            execute_silver_rel_transmutation_task(temp_path, mock_config)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def test_execute_silver_rel_transmutation_task_file_not_found(
    mock_config: FederatedRxNormConfigurationContract,
) -> None:
    """Test transmutation fails when RRF file does not exist."""
    fake_path = pathlib.Path("non_existent_rxnrel.rrf")

    with pytest.raises(FileNotFoundError):
        execute_silver_rel_transmutation_task(fake_path, mock_config)


def test_epistemic_silver_sat_manifest_validation() -> None:
    """Test validation of EpistemicSilverSatManifest."""
    manifest = EpistemicSilverSatManifest(
        uploaded_s3_uri="s3://my-silver-bucket/rxnorm/clean/bridge_rxnorm_ndc/bridge_rxnorm_ndc.parquet"
    )
    assert manifest.uploaded_s3_uri == "s3://my-silver-bucket/rxnorm/clean/bridge_rxnorm_ndc/bridge_rxnorm_ndc.parquet"

    with pytest.raises(ValidationError):
        EpistemicSilverSatManifest()  # type: ignore[call-arg]


@mock_aws
def test_execute_silver_sat_transmutation_task_success(
    mock_config: FederatedRxNormConfigurationContract,
) -> None:
    """Test successful transmutation of RXNSAT to Silver zone."""
    import os

    os.environ["AWS_DEFAULT_REGION"] = "us-east-1"
    s3_client = boto3.client("s3")
    bucket_name = mock_config.silver_bucket.removeprefix("s3://").strip("/")
    s3_client.create_bucket(Bucket=bucket_name)

    # Create dummy RXNSAT.RRF
    with tempfile.NamedTemporaryFile(suffix=".RRF", delete=False, mode="w") as temp_file:
        # valid row (ATN=NDC)
        temp_file.write("1111||||||||NDC|RXNORM|00000000000|||\n")
        # invalid row (ATN=SNOMEDCT_US or ATN=ATC)
        temp_file.write("3333||||||||ATC|SNOMEDCT_US|A01AA|||\n")

        temp_path = pathlib.Path(temp_file.name)

    try:
        manifest = execute_silver_sat_transmutation_task(temp_path, mock_config)
        assert (
            manifest.uploaded_s3_uri == f"s3://{bucket_name}/rxnorm/clean/bridge_rxnorm_ndc/bridge_rxnorm_ndc.parquet"
        )

        # Download Parquet from S3 to verify contents
        s3_client.download_file(
            bucket_name,
            "rxnorm/clean/bridge_rxnorm_ndc/bridge_rxnorm_ndc.parquet",
            str(temp_path.parent / "downloaded_sat.parquet"),
        )

        df = pl.read_parquet(temp_path.parent / "downloaded_sat.parquet")

        # Should only contain RXCUI 1111
        assert len(df) == 1

        # Check typing and specific values
        assert df["rxcui_id"].dtype == pl.String
        assert df["rxcui_id"].to_list() == ["1111"]
        assert df["ndc_code"].to_list() == ["00000000000"]

        import uuid

        expected_id_1111 = str(uuid.uuid5(NAMESPACE_RXNORM, "1111"))
        assert df["coreason_id"].to_list() == [expected_id_1111]

    finally:
        if temp_path.exists():
            temp_path.unlink()
        if (temp_path.parent / "downloaded_sat.parquet").exists():
            (temp_path.parent / "downloaded_sat.parquet").unlink()


@mock_aws
def test_execute_silver_sat_transmutation_task_s3_error(
    mock_config: FederatedRxNormConfigurationContract,
) -> None:
    """Test transmutation fails gracefully when S3 bucket is missing."""
    import os

    os.environ["AWS_DEFAULT_REGION"] = "us-east-1"

    with tempfile.NamedTemporaryFile(suffix=".RRF", delete=False, mode="w") as temp_file:
        temp_file.write("1111||||||||NDC|RXNORM|00000000000|||\n")
        temp_path = pathlib.Path(temp_file.name)

    try:
        with pytest.raises((ClientError, boto3.exceptions.Boto3Error, Exception)):
            execute_silver_sat_transmutation_task(temp_path, mock_config)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def test_execute_silver_sat_transmutation_task_file_not_found(
    mock_config: FederatedRxNormConfigurationContract,
) -> None:
    """Test transmutation fails when RRF file does not exist."""
    fake_path = pathlib.Path("non_existent_rxnsat.rrf")

    with pytest.raises(FileNotFoundError):
        execute_silver_sat_transmutation_task(fake_path, mock_config)


def test_epistemic_gold_registration_manifest_validation() -> None:
    """Test validation of EpistemicGoldRegistrationManifest."""
    manifest = EpistemicGoldRegistrationManifest(registered_tables=["table1", "table2"])
    assert manifest.registered_tables == ["table1", "table2"]

    with pytest.raises(ValidationError):
        EpistemicGoldRegistrationManifest()  # type: ignore[call-arg]


@mock_aws
def test_execute_gold_athena_registration_task_success(
    mock_config: FederatedRxNormConfigurationContract,
) -> None:
    """Test successful registration of Gold tables in Athena."""
    import os

    os.environ["AWS_DEFAULT_REGION"] = "us-east-1"
    s3_client = boto3.client("s3")
    glue_client = boto3.client("glue", region_name="us-east-1")

    bucket_name = mock_config.silver_bucket.removeprefix("s3://").strip("/")
    s3_client.create_bucket(Bucket=bucket_name)

    # Create the Glue database
    glue_client.create_database(DatabaseInput={"Name": mock_config.athena_database})

    # Prepare dummy parquet files so awswrangler can discover schema
    df = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})

    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as temp_file:
        temp_path_str = temp_file.name

    try:
        df.write_parquet(temp_path_str)

        # Upload to expected locations
        s3_client.upload_file(temp_path_str, bucket_name, "rxnorm/clean/dim_rxnorm_concept/dim_rxnorm_concept.parquet")
        s3_client.upload_file(
            temp_path_str, bucket_name, "rxnorm/clean/fact_rxnorm_relationship/fact_rxnorm_relationship.parquet"
        )
        s3_client.upload_file(temp_path_str, bucket_name, "rxnorm/clean/bridge_rxnorm_ndc/bridge_rxnorm_ndc.parquet")
    finally:
        temp_path = pathlib.Path(temp_path_str)
        if temp_path.exists():
            temp_path.unlink()

    conso_manifest = EpistemicSilverConsoManifest(
        uploaded_s3_uri=f"s3://{bucket_name}/rxnorm/clean/dim_rxnorm_concept/dim_rxnorm_concept.parquet"
    )
    rel_manifest = EpistemicSilverRelManifest(
        uploaded_s3_uri=f"s3://{bucket_name}/rxnorm/clean/fact_rxnorm_relationship/fact_rxnorm_relationship.parquet"
    )
    sat_manifest = EpistemicSilverSatManifest(
        uploaded_s3_uri=f"s3://{bucket_name}/rxnorm/clean/bridge_rxnorm_ndc/bridge_rxnorm_ndc.parquet"
    )

    manifest = execute_gold_athena_registration_task(
        conso_manifest,
        rel_manifest,
        sat_manifest,
        mock_config,
    )

    assert isinstance(manifest, EpistemicGoldRegistrationManifest)
    assert manifest.registered_tables == [
        "bridge_rxnorm_ndc",
        "dim_rxnorm_concept",
        "fact_rxnorm_relationship",
    ]

    # Verify tables actually exist in Glue
    res = glue_client.get_tables(DatabaseName=mock_config.athena_database)
    tables = [t["Name"] for t in res["TableList"]]
    assert "dim_rxnorm_concept" in tables
    assert "fact_rxnorm_relationship" in tables
    assert "bridge_rxnorm_ndc" in tables


def test_epistemic_gold_postgres_manifest_validation() -> None:
    """Test validation of EpistemicGoldPostgresManifest."""
    manifest = EpistemicGoldPostgresManifest(loaded_tables=["table1", "table2"])
    assert manifest.loaded_tables == ["table1", "table2"]

    with pytest.raises(ValidationError):
        EpistemicGoldPostgresManifest()  # type: ignore[call-arg]


@mock_aws
def test_execute_gold_postgres_load_task_success(
    mock_config: FederatedRxNormConfigurationContract,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test successful load of Gold tables into PostgreSQL."""
    import os

    os.environ["AWS_DEFAULT_REGION"] = "us-east-1"

    mock_config.pghost = "localhost"
    mock_config.pgport = 5432
    mock_config.pguser = "postgres"
    mock_config.pgpassword = "password"
    mock_config.pgdatabase = "coreason"

    # Mock dlt pipeline
    import dlt

    class MockPipeline:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def run(self, _data: object, **kwargs: object) -> None:
            to_sql_calls.append(kwargs)

    to_sql_calls: list[dict[str, object]] = []

    def mock_pipeline(*args: object, **kwargs: object) -> MockPipeline:
        return MockPipeline(*args, **kwargs)

    monkeypatch.setattr(dlt, "pipeline", mock_pipeline)

    # Need to execute the generator to cover stream_parquet_chunks
    original_run = MockPipeline.run

    def mocked_run(self: MockPipeline, data: object, **kwargs: object) -> None:
        original_run(self, data, **kwargs)
        # Execute generator
        import types

        if isinstance(data, types.GeneratorType):
            list(data)

    monkeypatch.setattr(MockPipeline, "run", mocked_run)

    import awswrangler as wr
    import pandas as pd

    def mock_read_parquet(*_args: object, **_kwargs: object) -> typing.Iterator[pd.DataFrame]:
        yield pd.DataFrame({"col1": [1]})

    monkeypatch.setattr(wr.s3, "read_parquet", mock_read_parquet)

    conso_manifest = EpistemicSilverConsoManifest(
        uploaded_s3_uri="s3://mock-silver/rxnorm/clean/dim_rxnorm_concept/dim_rxnorm_concept.parquet"
    )
    rel_manifest = EpistemicSilverRelManifest(
        uploaded_s3_uri="s3://mock-silver/rxnorm/clean/fact_rxnorm_relationship/fact_rxnorm_relationship.parquet"
    )
    sat_manifest = EpistemicSilverSatManifest(
        uploaded_s3_uri="s3://mock-silver/rxnorm/clean/bridge_rxnorm_ndc/bridge_rxnorm_ndc.parquet"
    )

    manifest = execute_gold_postgres_load_task(
        conso_manifest,
        rel_manifest,
        sat_manifest,
        mock_config,
    )

    assert isinstance(manifest, EpistemicGoldPostgresManifest)
    assert manifest.loaded_tables == [
        "bridge_rxnorm_ndc",
        "dim_rxnorm_concept",
        "fact_rxnorm_relationship",
    ]

    assert len(to_sql_calls) == 3

    # Verify expected parameters
    tables_called = [call.get("table_name") for call in to_sql_calls]
    assert "dim_rxnorm_concept" in tables_called
    assert "fact_rxnorm_relationship" in tables_called
    assert "bridge_rxnorm_ndc" in tables_called

    for call in to_sql_calls:
        assert call.get("write_disposition") == "merge"
        if call.get("table_name") == "fact_rxnorm_relationship":
            assert call.get("primary_key") == [
                "source_coreason_id",
                "target_coreason_id",
                "relationship_type",
            ]
        elif call.get("table_name") == "bridge_rxnorm_ndc":
            assert call.get("primary_key") == ["coreason_id", "ndc_code"]
        else:
            assert call.get("primary_key") == ["coreason_id"]


@mock_aws
def test_execute_gold_postgres_load_task_failure(
    mock_config: FederatedRxNormConfigurationContract,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test failure of Gold Postgres load task."""
    import os

    os.environ["AWS_DEFAULT_REGION"] = "us-east-1"

    mock_config.pghost = "localhost"
    mock_config.pgport = 5432
    mock_config.pguser = "postgres"
    mock_config.pgpassword = "password"
    mock_config.pgdatabase = "coreason"

    import dlt

    def mock_pipeline_fail(*_args: object, **_kwargs: object) -> None:
        raise Exception("Mocked connection failure")

    monkeypatch.setattr(dlt, "pipeline", mock_pipeline_fail)

    conso_manifest = EpistemicSilverConsoManifest(
        uploaded_s3_uri="s3://mock-silver/rxnorm/clean/dim_rxnorm_concept/dim_rxnorm_concept.parquet"
    )
    rel_manifest = EpistemicSilverRelManifest(
        uploaded_s3_uri="s3://mock-silver/rxnorm/clean/fact_rxnorm_relationship/fact_rxnorm_relationship.parquet"
    )
    sat_manifest = EpistemicSilverSatManifest(
        uploaded_s3_uri="s3://mock-silver/rxnorm/clean/bridge_rxnorm_ndc/bridge_rxnorm_ndc.parquet"
    )

    with pytest.raises(Exception, match=r"Mocked connection failure"):
        execute_gold_postgres_load_task(
            conso_manifest,
            rel_manifest,
            sat_manifest,
            mock_config,
        )


def test_execute_gold_postgres_load_task_missing_config(
    mock_config: FederatedRxNormConfigurationContract,
) -> None:
    """Test failure of Gold Postgres load task when config is missing."""
    conso_manifest = EpistemicSilverConsoManifest(
        uploaded_s3_uri="s3://mock-silver/rxnorm/clean/dim_rxnorm_concept/dim_rxnorm_concept.parquet"
    )
    rel_manifest = EpistemicSilverRelManifest(
        uploaded_s3_uri="s3://mock-silver/rxnorm/clean/fact_rxnorm_relationship/fact_rxnorm_relationship.parquet"
    )
    sat_manifest = EpistemicSilverSatManifest(
        uploaded_s3_uri="s3://mock-silver/rxnorm/clean/bridge_rxnorm_ndc/bridge_rxnorm_ndc.parquet"
    )

    with pytest.raises(ValueError, match=r"PostgreSQL configuration is incomplete\."):
        execute_gold_postgres_load_task(
            conso_manifest,
            rel_manifest,
            sat_manifest,
            mock_config,
        )


@mock_aws
def test_execute_gold_athena_registration_task_failure(
    mock_config: FederatedRxNormConfigurationContract,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test failure during registration of Gold tables."""
    import os

    os.environ["AWS_DEFAULT_REGION"] = "us-east-1"

    # Mock awswrangler to throw a standard exception to avoid pickling errors
    # with dynamically generated botocore exceptions during xdist multiprocessing
    import awswrangler as wr

    def mock_store(*_args: object, **_kwargs: object) -> None:
        raise Exception("Mocked failure")

    monkeypatch.setattr(wr.s3, "store_parquet_metadata", mock_store)

    conso_manifest = EpistemicSilverConsoManifest(
        uploaded_s3_uri="s3://mock-silver/rxnorm/clean/dim_rxnorm_concept/dim_rxnorm_concept.parquet"
    )
    rel_manifest = EpistemicSilverRelManifest(
        uploaded_s3_uri="s3://mock-silver/rxnorm/clean/fact_rxnorm_relationship/fact_rxnorm_relationship.parquet"
    )
    sat_manifest = EpistemicSilverSatManifest(
        uploaded_s3_uri="s3://mock-silver/rxnorm/clean/bridge_rxnorm_ndc/bridge_rxnorm_ndc.parquet"
    )

    with pytest.raises(Exception, match=r"Mocked failure"):
        execute_gold_athena_registration_task(
            conso_manifest,
            rel_manifest,
            sat_manifest,
            mock_config,
        )
