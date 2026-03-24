# Copyright (c) 2026 CoReason, Inc.
# Source Code: https://github.com/CoReason-AI/coreason_etl_rxnorm

import pathlib
import shutil
import urllib.parse
import uuid

import polars as pl

from coreason_etl_rxnorm.config import NAMESPACE_RXNORM, FederatedRxNormConfigurationContract
from coreason_etl_rxnorm.network_manifold import SpatialExtractionManifest
from coreason_etl_rxnorm.utils.logger import logger


def generate_uuid(omop_id: str) -> str:
    if omop_id is None:
        return None
    return str(uuid.uuid5(NAMESPACE_RXNORM, str(omop_id)))


def execute_omop_to_postgres_pipeline(
    manifest: SpatialExtractionManifest,
    config: FederatedRxNormConfigurationContract,
) -> None:
    """
    AGENT INSTRUCTION: Replaces the AWS Lake Manifold with direct Polars ADBC ingestion 
    for OMOP RxNorm datasets.
    """
    logger.info("Executing Lake Manifold Transmutations (OMOP to PostgreSQL).")

    # 1. Construct DB URI
    safe_password = urllib.parse.quote_plus(config.pgpassword) if config.pgpassword else ""
    db_uri = f"postgresql://{config.pguser}:{safe_password}@{config.pghost}:{config.pgport}/{config.pgdatabase}"

    ext_dir = manifest.extraction_path
    
    def load_table(lf: pl.LazyFrame, table_name: str):
        logger.info(f"Loading table {table_name}...")
        lf.collect().write_database(table_name=table_name, connection=db_uri, if_table_exists="replace", engine="adbc")

    try:
        # --- BRONZE LAYER ---
        logger.info("Building Bronze Layer...")
        for filename, table_name in [
            ("CONCEPT.csv", "rxnorm_bronze_concept"),
            ("CONCEPT_SYNONYM.csv", "rxnorm_bronze_description"),
            ("CONCEPT_RELATIONSHIP.csv", "rxnorm_bronze_relationship")
        ]:
            lf = pl.scan_csv(ext_dir / filename, separator="\t", infer_schema_length=0, quote_char=None)
            load_table(lf, table_name)

        # --- SILVER LAYER ---
        logger.info("Building Silver Layer...")
        
        # Concept
        concept_lf = pl.scan_csv(
            ext_dir / "CONCEPT.csv", separator="\t", quote_char=None, ignore_errors=True,
            schema_overrides={"concept_id": pl.String, "valid_start_date": pl.String, "invalid_reason": pl.String}
        )
        silver_concept = (
            concept_lf.filter(pl.col("invalid_reason").is_null())
            .with_columns(
                pl.col("concept_id").alias("source_id"),
                pl.col("concept_id").map_elements(generate_uuid, return_dtype=pl.String).alias("coreason_id"),
            )
        )
        load_table(silver_concept, "rxnorm_silver_concept")

        # Synonym
        synonym_lf = pl.scan_csv(
            ext_dir / "CONCEPT_SYNONYM.csv", separator="\t", quote_char=None, ignore_errors=True,
            schema_overrides={"concept_id": pl.String, "concept_synonym_name": pl.String}
        )
        silver_desc = (
            synonym_lf.with_columns(
                pl.col("concept_id").map_elements(generate_uuid, return_dtype=pl.String).alias("concept_coreason_id"),
                pl.col("concept_synonym_name").alias("term")
            )
        )
        load_table(silver_desc, "rxnorm_silver_description")

        # Relationship
        rel_lf = pl.scan_csv(
            ext_dir / "CONCEPT_RELATIONSHIP.csv", separator="\t", quote_char=None, ignore_errors=True,
            schema_overrides={"concept_id_1": pl.String, "concept_id_2": pl.String, "relationship_id": pl.String, "invalid_reason": pl.String}
        )
        silver_rel = (
            rel_lf.filter(pl.col("invalid_reason").is_null())
            .with_columns(
                pl.col("concept_id_1").map_elements(generate_uuid, return_dtype=pl.String).alias("source_coreason_id"),
                pl.col("concept_id_2").map_elements(generate_uuid, return_dtype=pl.String).alias("destination_coreason_id"),
                pl.col("relationship_id").map_elements(generate_uuid, return_dtype=pl.String).alias("type_coreason_id"),
            )
        )
        load_table(silver_rel, "rxnorm_silver_relationship")

        # --- GOLD LAYER ---
        logger.info("Building Gold Layer...")
        gold_concept = silver_concept.select([pl.col("coreason_id"), pl.col("concept_name").alias("name")])
        load_table(gold_concept, "rxnorm_gold_dim_concept")

        gold_synonym = (
            silver_desc.group_by("concept_coreason_id")
            .agg(pl.col("term").sort().alias("synonyms"))
            .rename({"concept_coreason_id": "coreason_id"})
        )
        load_table(gold_synonym, "rxnorm_gold_bridge_synonym")

        gold_rel = silver_rel.select([
            pl.col("source_coreason_id"), 
            pl.col("destination_coreason_id"), 
            pl.col("type_coreason_id").alias("relationship_type_coreason_id")
        ])
        load_table(gold_rel, "rxnorm_gold_fact_relationship")

    finally:
        if ext_dir.exists():
            shutil.rmtree(str(ext_dir))
