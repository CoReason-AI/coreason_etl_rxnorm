import uuid

import polars as pl

NAMESPACE_RXNORM = uuid.UUID("f3b9c7b0-7b3b-4b3b-8b3b-0b3b3b3b3b3b")

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

with open("dummy_rxnrel.rrf", "w") as f:
    # row 1: valid rxnorm relation
    f.write("1111|||RO|2222|||has_ingredient|||RXNORM||||N|||\n")
    # row 2: other vocabulary relation
    f.write("3333|||RO|4444|||has_part|||SNOMEDCT_US||||N|||\n")

lazy_df = pl.scan_csv(
    "dummy_rxnrel.rrf",
    separator="|",
    has_header=False,
    new_columns=rel_columns,
    truncate_ragged_lines=True,
)

transformed_df = (
    lazy_df.drop("_trailing_empty")
    .filter(pl.col("SAB") == "RXNORM")
    .with_columns(
        pl.col("RXCUI1").cast(pl.String, strict=True),
        pl.col("RXCUI2").cast(pl.String, strict=True),
    )
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
    .select(["source_rxcui_id", "source_coreason_id", "target_rxcui_id", "target_coreason_id", "relationship_type"])
)

df = transformed_df.collect()
print(df)
