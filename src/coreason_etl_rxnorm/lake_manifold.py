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
import shutil
import tempfile
import uuid

import awswrangler as wr
import boto3
import polars as pl
from pydantic import BaseModel, Field

from coreason_etl_rxnorm.config import NAMESPACE_RXNORM, FederatedRxNormConfigurationContract
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


class EpistemicSilverConsoManifest(BaseModel):
    """
    AGENT INSTRUCTION: This object represents the mandatory epistemic state
    of successfully processed RXNCONSO Parquet files in the Silver S3 zone.
    """

    uploaded_s3_uri: str = Field(
        ...,
        description="The specific S3 URI of the uploaded Silver Parquet file.",
    )


def execute_silver_conso_transmutation_task(
    conso_file_path: pathlib.Path,
    config: FederatedRxNormConfigurationContract,
) -> EpistemicSilverConsoManifest:
    """
    AGENT INSTRUCTION: Executes the intent to transmute the raw RXNCONSO.RRF file
    into a typed, filtered, and identity-resolved Parquet file in the Silver zone.

    Args:
        conso_file_path: The local filesystem path to the extracted RXNCONSO.RRF file.
        config: The configuration contract containing the S3 silver bucket URI.

    Returns:
        An EpistemicSilverConsoManifest containing the S3 URI of the Silver Parquet file.

    Raises:
        botocore.exceptions.ClientError: If the upload fails due to S3 or permissions errors.
    """
    logger.info("Executing SilverConsoTransmutationTask for RXNCONSO.", file_path=str(conso_file_path))

    # Explicit schema definition for RXNCONSO.RRF based on NLM specs
    # A phantom column `_trailing_empty` is added to handle the trailing pipe
    conso_columns = [
        "RXCUI",
        "LAT",
        "TS",
        "LUI",
        "STT",
        "SUI",
        "ISPREF",
        "RXAUI",
        "SAUI",
        "SCUI",
        "SDUI",
        "SAB",
        "TTY",
        "CODE",
        "STR",
        "SRL",
        "SUPPRESS",
        "CVF",
        "_trailing_empty",
    ]

    lazy_df = pl.scan_csv(
        conso_file_path,
        separator="|",
        has_header=False,
        new_columns=conso_columns,
        truncate_ragged_lines=True,
    )

    # Transform the DataFrame
    transformed_df = (
        lazy_df.drop("_trailing_empty")
        .with_columns(
            pl.col("RXCUI").cast(pl.String, strict=True),
        )
        .filter((pl.col("LAT") == "ENG") & (~pl.col("SUPPRESS").is_in(["O", "Y"])) & (pl.col("SAB") == "RXNORM"))
        .with_columns(
            pl.col("RXCUI")
            .map_elements(
                lambda x: str(uuid.uuid5(NAMESPACE_RXNORM, str(x))),
                return_dtype=pl.String,
            )
            .alias("coreason_id"),
        )
        .rename(
            {
                "RXCUI": "rxcui_id",
                "STR": "concept_name",
                "TTY": "term_type",
                "SAB": "source_vocabulary",
            }
        )
        .select(["rxcui_id", "coreason_id", "concept_name", "term_type", "source_vocabulary"])
    )

    # Secure local temporary file for the Parquet output to manage memory
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as temp_file:
        temp_parquet_path = pathlib.Path(temp_file.name)

    logger.debug("Writing Transmuted DataFrame to temporary Silver Parquet manifold.", path=str(temp_parquet_path))

    try:
        # Collect streaming is required for multi-gigabyte files
        transformed_df.collect(engine="streaming").write_parquet(temp_parquet_path)

        silver_uri = config.silver_bucket
        bucket_name = silver_uri.removeprefix("s3://").strip("/")
        s3_key = "rxnorm/clean/dim_rxnorm_concept/dim_rxnorm_concept.parquet"

        s3_client = boto3.client("s3")
        logger.debug(f"Uploading Silver Parquet to s3://{bucket_name}/{s3_key}")

        s3_client.upload_file(str(temp_parquet_path), bucket_name, s3_key)
        uploaded_uri = f"s3://{bucket_name}/{s3_key}"

    except Exception as e:
        logger.exception("Failed to transmute and upload RXNCONSO to Silver lake.")
        raise e
    finally:
        # Immediately purge the local temporary parquet file
        if temp_parquet_path.exists():
            temp_parquet_path.unlink()
            logger.debug("Purged temporary Silver Parquet manifold.", path=str(temp_parquet_path))

    logger.info("Successfully transmuted RXNCONSO to EpistemicSilverConsoManifest.")
    return EpistemicSilverConsoManifest(uploaded_s3_uri=uploaded_uri)


class EpistemicSilverRelManifest(BaseModel):
    """
    AGENT INSTRUCTION: This object represents the mandatory epistemic state
    of successfully processed RXNREL Parquet files in the Silver S3 zone.
    """

    uploaded_s3_uri: str = Field(
        ...,
        description="The specific S3 URI of the uploaded Silver Parquet file.",
    )


def execute_silver_rel_transmutation_task(
    rel_file_path: pathlib.Path,
    config: FederatedRxNormConfigurationContract,
) -> EpistemicSilverRelManifest:
    """
    AGENT INSTRUCTION: Executes the intent to transmute the raw RXNREL.RRF file
    into a typed, filtered, and identity-resolved Parquet file in the Silver zone.

    Args:
        rel_file_path: The local filesystem path to the extracted RXNREL.RRF file.
        config: The configuration contract containing the S3 silver bucket URI.

    Returns:
        An EpistemicSilverRelManifest containing the S3 URI of the Silver Parquet file.

    Raises:
        botocore.exceptions.ClientError: If the upload fails due to S3 or permissions errors.
    """
    logger.info("Executing SilverRelTransmutationTask for RXNREL.", file_path=str(rel_file_path))

    # Explicit schema definition for RXNREL.RRF based on NLM specs
    # A phantom column `_trailing_empty` is added to handle the trailing pipe
    rel_columns = [
        "RXCUI1",
        "RXAUI1",
        "STYPE1",
        "REL",
        "RXCUI2",
        "RXAUI2",
        "STYPE2",
        "RELA",
        "RUI",
        "SRUI",
        "SAB",
        "SL",
        "DIR",
        "RG",
        "SUPPRESS",
        "CVF",
        "_trailing_empty",
    ]

    lazy_df = pl.scan_csv(
        rel_file_path,
        separator="|",
        has_header=False,
        new_columns=rel_columns,
        truncate_ragged_lines=True,
    )

    # Transform the DataFrame
    transformed_df = (
        lazy_df.drop("_trailing_empty")
        .with_columns(
            pl.col("RXCUI1").cast(pl.String, strict=True),
            pl.col("RXCUI2").cast(pl.String, strict=True),
        )
        .filter(pl.col("SAB") == "RXNORM")
        .with_columns(
            pl.col("RXCUI1")
            .map_elements(
                lambda x: str(uuid.uuid5(NAMESPACE_RXNORM, str(x))),
                return_dtype=pl.String,
            )
            .alias("source_coreason_id"),
            pl.col("RXCUI2")
            .map_elements(
                lambda x: str(uuid.uuid5(NAMESPACE_RXNORM, str(x))),
                return_dtype=pl.String,
            )
            .alias("target_coreason_id"),
        )
        .rename(
            {
                "RXCUI1": "source_rxcui_id",
                "RXCUI2": "target_rxcui_id",
                "RELA": "relationship_type",
            }
        )
        .select(
            [
                "source_rxcui_id",
                "source_coreason_id",
                "target_rxcui_id",
                "target_coreason_id",
                "relationship_type",
            ]
        )
    )

    # Secure local temporary file for the Parquet output to manage memory
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as temp_file:
        temp_parquet_path = pathlib.Path(temp_file.name)

    logger.debug("Writing Transmuted DataFrame to temporary Silver Parquet manifold.", path=str(temp_parquet_path))

    try:
        # Collect streaming is required for multi-gigabyte files
        transformed_df.collect(engine="streaming").write_parquet(temp_parquet_path)

        silver_uri = config.silver_bucket
        bucket_name = silver_uri.removeprefix("s3://").strip("/")
        s3_key = "rxnorm/clean/fact_rxnorm_relationship/fact_rxnorm_relationship.parquet"

        s3_client = boto3.client("s3")
        logger.debug(f"Uploading Silver Parquet to s3://{bucket_name}/{s3_key}")

        s3_client.upload_file(str(temp_parquet_path), bucket_name, s3_key)
        uploaded_uri = f"s3://{bucket_name}/{s3_key}"

    except Exception as e:
        logger.exception("Failed to transmute and upload RXNREL to Silver lake.")
        raise e
    finally:
        # Immediately purge the local temporary parquet file
        if temp_parquet_path.exists():
            temp_parquet_path.unlink()
            logger.debug("Purged temporary Silver Parquet manifold.", path=str(temp_parquet_path))

    logger.info("Successfully transmuted RXNREL to EpistemicSilverRelManifest.")
    return EpistemicSilverRelManifest(uploaded_s3_uri=uploaded_uri)


class EpistemicSilverSatManifest(BaseModel):
    """
    AGENT INSTRUCTION: This object represents the mandatory epistemic state
    of successfully processed RXNSAT Parquet files in the Silver S3 zone.
    """

    uploaded_s3_uri: str = Field(
        ...,
        description="The specific S3 URI of the uploaded Silver Parquet file.",
    )


def execute_silver_sat_transmutation_task(
    sat_file_path: pathlib.Path,
    config: FederatedRxNormConfigurationContract,
) -> EpistemicSilverSatManifest:
    """
    AGENT INSTRUCTION: Executes the intent to transmute the raw RXNSAT.RRF file
    into a typed, filtered, and identity-resolved Parquet file in the Silver zone.

    Args:
        sat_file_path: The local filesystem path to the extracted RXNSAT.RRF file.
        config: The configuration contract containing the S3 silver bucket URI.

    Returns:
        An EpistemicSilverSatManifest containing the S3 URI of the Silver Parquet file.

    Raises:
        botocore.exceptions.ClientError: If the upload fails due to S3 or permissions errors.
    """
    logger.info("Executing SilverSatTransmutationTask for RXNSAT.", file_path=str(sat_file_path))

    # Explicit schema definition for RXNSAT.RRF based on NLM specs
    # A phantom column `_trailing_empty` is added to handle the trailing pipe
    sat_columns = [
        "RXCUI",
        "LUI",
        "SUI",
        "RXAUI",
        "STYPE",
        "CODE",
        "ATUI",
        "SATUI",
        "ATN",
        "SAB",
        "ATV",
        "SUPPRESS",
        "CVF",
        "_trailing_empty",
    ]

    lazy_df = pl.scan_csv(
        sat_file_path,
        separator="|",
        has_header=False,
        new_columns=sat_columns,
        truncate_ragged_lines=True,
    )

    # Transform the DataFrame
    transformed_df = (
        lazy_df.drop("_trailing_empty")
        .with_columns(
            pl.col("RXCUI").cast(pl.String, strict=True),
        )
        .filter(pl.col("ATN") == "NDC")
        .with_columns(
            pl.col("RXCUI")
            .map_elements(
                lambda x: str(uuid.uuid5(NAMESPACE_RXNORM, str(x))),
                return_dtype=pl.String,
            )
            .alias("coreason_id"),
        )
        .rename(
            {
                "RXCUI": "rxcui_id",
                "ATV": "ndc_code",
            }
        )
        .select(
            [
                "rxcui_id",
                "coreason_id",
                "ndc_code",
            ]
        )
    )

    # Secure local temporary file for the Parquet output to manage memory
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as temp_file:
        temp_parquet_path = pathlib.Path(temp_file.name)

    logger.debug("Writing Transmuted DataFrame to temporary Silver Parquet manifold.", path=str(temp_parquet_path))

    try:
        # Collect streaming is required for multi-gigabyte files
        transformed_df.collect(engine="streaming").write_parquet(temp_parquet_path)

        silver_uri = config.silver_bucket
        bucket_name = silver_uri.removeprefix("s3://").strip("/")
        s3_key = "rxnorm/clean/bridge_rxnorm_ndc/bridge_rxnorm_ndc.parquet"

        s3_client = boto3.client("s3")
        logger.debug(f"Uploading Silver Parquet to s3://{bucket_name}/{s3_key}")

        s3_client.upload_file(str(temp_parquet_path), bucket_name, s3_key)
        uploaded_uri = f"s3://{bucket_name}/{s3_key}"

    except Exception as e:
        logger.exception("Failed to transmute and upload RXNSAT to Silver lake.")
        raise e
    finally:
        # Immediately purge the local temporary parquet file
        if temp_parquet_path.exists():
            temp_parquet_path.unlink()
            logger.debug("Purged temporary Silver Parquet manifold.", path=str(temp_parquet_path))

    logger.info("Successfully transmuted RXNSAT to EpistemicSilverSatManifest.")
    return EpistemicSilverSatManifest(uploaded_s3_uri=uploaded_uri)


class EpistemicGoldRegistrationManifest(BaseModel):
    """
    AGENT INSTRUCTION: This object represents the mandatory epistemic state
    of successfully registered Gold tables in the AWS Glue Data Catalog.
    """

    registered_tables: list[str] = Field(
        ...,
        description="The names of the tables successfully registered in Athena.",
    )


def execute_gold_athena_registration_task(
    conso_manifest: EpistemicSilverConsoManifest,
    rel_manifest: EpistemicSilverRelManifest,
    sat_manifest: EpistemicSilverSatManifest,
    config: FederatedRxNormConfigurationContract,
) -> EpistemicGoldRegistrationManifest:
    """
    AGENT INSTRUCTION: Executes the intent to register the Silver Parquet outputs
    as Athena tables in the AWS Glue Data Catalog using awswrangler.

    Args:
        conso_manifest: The EpistemicSilverConsoManifest containing the Silver Parquet URI.
        rel_manifest: The EpistemicSilverRelManifest containing the Silver Parquet URI.
        sat_manifest: The EpistemicSilverSatManifest containing the Silver Parquet URI.
        config: The configuration contract containing the Athena database name.

    Returns:
        An EpistemicGoldRegistrationManifest containing the names of registered tables.

    Raises:
        Exception: If the registration fails.
    """
    logger.info("Executing GoldAthenaRegistrationTask to register Silver artifacts.")

    database = config.athena_database
    registered_tables = []

    # Map the manifests to their expected table names and paths
    # Note: awswrangler needs the directory path for the dataset, not the explicit file
    datasets = [
        (
            "dim_rxnorm_concept",
            conso_manifest.uploaded_s3_uri.rsplit("/", 1)[0] + "/",
        ),
        (
            "fact_rxnorm_relationship",
            rel_manifest.uploaded_s3_uri.rsplit("/", 1)[0] + "/",
        ),
        (
            "bridge_rxnorm_ndc",
            sat_manifest.uploaded_s3_uri.rsplit("/", 1)[0] + "/",
        ),
    ]

    try:
        for table_name, path in datasets:
            logger.debug(f"Registering Gold table {table_name} at {path} in database {database}.")

            # We use awswrangler to store the parquet metadata in the Glue Catalog
            wr.s3.store_parquet_metadata(
                path=path,
                database=database,
                table=table_name,
                dataset=True,
            )
            registered_tables.append(table_name)
    except Exception as e:
        logger.exception("Failed to execute GoldAthenaRegistrationTask.")
        raise e

    logger.info("Successfully registered Gold tables into EpistemicGoldRegistrationManifest.")

    # Sort to ensure deterministic output
    registered_tables.sort()

    return EpistemicGoldRegistrationManifest(registered_tables=registered_tables)
