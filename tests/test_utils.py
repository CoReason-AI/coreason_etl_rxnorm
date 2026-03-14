# Copyright (c) 2026 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_etl_rxnorm

import importlib
from unittest.mock import patch

import coreason_etl_rxnorm.utils.logger


def test_logger_mkdir() -> None:
    with (
        patch("pathlib.Path.exists", return_value=False),
        patch("pathlib.Path.mkdir") as mock_mkdir,
        patch("loguru.logger.add"),
        patch("loguru.logger.remove"),
    ):
        importlib.reload(coreason_etl_rxnorm.utils.logger)
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
