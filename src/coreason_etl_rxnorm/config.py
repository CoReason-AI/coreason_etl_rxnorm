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

from pydantic import Field
from pydantic_settings import BaseSettings

# Determine UUID5 base for RxNorm to ensure deterministic identity.
NAMESPACE_RXNORM = uuid.UUID("f3b9c7b0-7b3b-4b3b-8b3b-0b3b3b3b3b3b")


class FederatedRxNormConfigurationContract(BaseSettings):
    """
    AGENT INSTRUCTION: This object represents the mandatory epistemic configuration state
    for the RxNorm ETL pipeline.

    It validates required fields for processing RxNorm.
    """

    umls_api_key: str = Field(
        ..., description="The API key required to authenticate with the UMLS Terminology Services."
    )
    bronze_bucket: str = Field(..., description="The S3 URI prefix for the Bronze data lake zone.")
    silver_bucket: str = Field(..., description="The S3 URI prefix for the Silver data lake zone.")
    athena_database: str = Field(..., description="The name of the AWS Glue Data Catalog database for Athena queries.")
