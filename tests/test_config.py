# Copyright (c) 2026 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_etl_rxnorm

import uuid

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from coreason_etl_rxnorm.config import NAMESPACE_RXNORM, FederatedRxNormConfigurationContract


def test_namespace_rxnorm_is_valid_uuid() -> None:
    """Verify that NAMESPACE_RXNORM is a valid UUID."""
    assert isinstance(NAMESPACE_RXNORM, uuid.UUID)


def test_deterministic_uuid5_generation() -> None:
    """Verify that UUID5 generation using NAMESPACE_RXNORM is deterministic."""
    test_string = "RX12345"
    uuid1 = uuid.uuid5(NAMESPACE_RXNORM, test_string)
    uuid2 = uuid.uuid5(NAMESPACE_RXNORM, test_string)
    assert uuid1 == uuid2

    # Also verify different strings give different UUIDs
    uuid3 = uuid.uuid5(NAMESPACE_RXNORM, "RX54321")
    assert uuid1 != uuid3


def test_valid_configuration() -> None:
    """Test valid configuration contract initialization."""
    config = FederatedRxNormConfigurationContract(
        umls_api_key="valid_key",
        bronze_bucket="s3://bronze-bucket",
        silver_bucket="s3://silver-bucket",
        athena_database="my_database",
    )
    assert config.umls_api_key == "valid_key"
    assert config.bronze_bucket == "s3://bronze-bucket"
    assert config.silver_bucket == "s3://silver-bucket"
    assert config.athena_database == "my_database"
    # Ensure optional fields are None by default
    assert config.pghost is None
    assert config.pgport is None
    assert config.pguser is None
    assert config.pgpassword is None
    assert config.pgdatabase is None


def test_configuration_with_postgres() -> None:
    """Test valid configuration contract initialization with PostgreSQL settings."""
    config = FederatedRxNormConfigurationContract(
        umls_api_key="valid_key",
        bronze_bucket="s3://bronze-bucket",
        silver_bucket="s3://silver-bucket",
        athena_database="my_database",
        pghost="localhost",
        pgport=5432,
        pguser="postgres",
        pgpassword="password",
        pgdatabase="coreason",
    )
    assert config.pghost == "localhost"
    assert config.pgport == 5432
    assert config.pguser == "postgres"
    assert config.pgpassword == "password"
    assert config.pgdatabase == "coreason"


def test_missing_fields() -> None:
    """Test missing required fields raise ValidationError."""
    with pytest.raises(ValidationError):
        FederatedRxNormConfigurationContract(
            umls_api_key="valid_key",
            bronze_bucket="s3://bronze-bucket",
            # silver_bucket missing
            athena_database="my_database",
        )


@given(api_key=st.text())
def test_hypothesis_umls_api_key(api_key: str) -> None:
    """Property-based test for UMLS API key (as arbitrary string)."""
    config = FederatedRxNormConfigurationContract(
        umls_api_key=api_key,
        bronze_bucket="s3://bronze-bucket",
        silver_bucket="s3://silver-bucket",
        athena_database="my_database",
    )
    assert config.umls_api_key == api_key
