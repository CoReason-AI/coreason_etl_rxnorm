# Copyright (c) 2026 CoReason, Inc.
# Source Code: https://github.com/CoReason-AI/coreason_etl_rxnorm

import contextlib
import pathlib
import shutil
import tempfile
import zipfile

from pydantic import BaseModel, Field

from coreason_etl_rxnorm.config import FederatedRxNormConfigurationContract
from coreason_etl_rxnorm.utils.logger import logger


class EpistemicReleaseMetadataManifest(BaseModel):
    download_url: str = Field(..., description="Bypassed URL.")

def execute_epistemic_release_metadata_fetch_task(config: FederatedRxNormConfigurationContract) -> EpistemicReleaseMetadataManifest:
    logger.info("BYPASSING API FETCH: Using local rxnorm_data.zip for development.")
    return EpistemicReleaseMetadataManifest(download_url="local://rxnorm_data.zip")


class SpatialArchiveState(BaseModel):
    archive_path: pathlib.Path = Field(...)

def execute_epistemic_archive_download_task(manifest: EpistemicReleaseMetadataManifest, config: FederatedRxNormConfigurationContract) -> SpatialArchiveState:
    local_zip_path = pathlib.Path("rxnorm_data.zip")
    if not local_zip_path.exists():
        raise FileNotFoundError(f"Local archive not found at: {local_zip_path.absolute()}")

    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as temp_file:
        temp_path = pathlib.Path(temp_file.name)

    logger.debug(f"Copying local {local_zip_path} to temporary workspace {temp_path}")
    shutil.copy(local_zip_path, temp_path)
    return SpatialArchiveState(archive_path=temp_path)


class SpatialExtractionManifest(BaseModel):
    extraction_path: pathlib.Path = Field(...)
    extracted_files: list[pathlib.Path] = Field(...)

def execute_spatial_archive_extraction_task(archive_state: SpatialArchiveState) -> SpatialExtractionManifest:
    logger.info("Executing SpatialArchiveExtractionTask.")

    target_file_patterns = {"CONCEPT.csv", "CONCEPT_RELATIONSHIP.csv", "CONCEPT_SYNONYM.csv"}
    extracted_paths: list[pathlib.Path] = []
    temp_dir = pathlib.Path(tempfile.mkdtemp(prefix="rxnorm_extract_"))

    try:
        with zipfile.ZipFile(archive_state.archive_path, "r") as z:
            for file_info in z.infolist():
                if file_info.is_dir():
                    continue

                filename = pathlib.Path(file_info.filename).name
                if filename in target_file_patterns:
                    target_path = temp_dir / filename
                    with z.open(file_info) as source, open(target_path, "wb") as target:
                        shutil.copyfileobj(source, target)
                    extracted_paths.append(target_path)
                    logger.info(f"Extracted {filename} to {target_path}")

    finally:
        archive_state.archive_path.unlink(missing_ok=True)

    extracted_paths.sort()
    return SpatialExtractionManifest(extraction_path=temp_dir, extracted_files=extracted_paths)
