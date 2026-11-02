"""Build the OMOP warehouse: fresh schemas, optional Athena vocabulary, then each ETL."""

import duckdb

from clinical_ml import config
from clinical_ml.omop import schema, vocab


def init_vocab(con: duckdb.DuckDBPyConnection) -> dict:
    schema.create_schema(con, config.VOCAB, vocab=True)
    return vocab.load_athena(con)


def build_source(con: duckdb.DuckDBPyConnection, source: str) -> dict:
    import importlib

    etl = importlib.import_module(
        {"d130": "clinical_ml.etl.diabetes130", "mimic": "clinical_ml.etl.fhir_mimic"}[source]
    )
    con.execute(f"CREATE SCHEMA IF NOT EXISTS {config.RESULTS}")
    schema.create_schema(con, config.CDM_SCHEMAS[source], vocab=False)
    return etl.run(con)


def build_all(sources=("d130", "mimic")) -> dict:
    with schema.connect() as con:
        out = {"vocab": init_vocab(con)}
        for s in sources:
            out[s] = build_source(con, s)
    return out
