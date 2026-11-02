"""Create OMOP CDM v5.4 schemas from the official OHDSI DuckDB DDL.

The DDL files in ./ddl are vendored unmodified from OHDSI/CommonDataModel (tag v5.5.0,
inst/ddl/5.4/duckdb, Apache-2.0).
"""

import re
from pathlib import Path

import duckdb

from clinical_ml import config

DDL_DIR = Path(__file__).parent / "ddl"
VOCAB_TABLES = {
    "concept", "vocabulary", "domain", "concept_class", "concept_relationship", "relationship",
    "concept_synonym", "concept_ancestor", "source_to_concept_map", "drug_strength",
}


def _statements(filename: str) -> list[tuple[str, str]]:
    """Return (table_name, statement) pairs from an OHDSI DDL file."""
    sql = (DDL_DIR / filename).read_text()
    out = []
    for stmt in sql.split(";"):
        m = re.search(r"@cdmDatabaseSchema\.(\w+)", stmt)
        if m:
            out.append((m.group(1).lower(), stmt.strip()))
    return out


def create_schema(con: duckdb.DuckDBPyConnection, schema: str, vocab: bool) -> None:
    """(Re)create one schema holding either the clinical tables or the vocabulary tables."""
    con.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    con.execute(f"CREATE SCHEMA {schema}")
    for fname in ("OMOPCDM_duckdb_5.4_ddl.sql", "OMOPCDM_duckdb_5.4_primary_keys.sql"):
        for table, stmt in _statements(fname):
            if (table in VOCAB_TABLES) == vocab:
                con.execute(stmt.replace("@cdmDatabaseSchema", schema))


def connect(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    config.WAREHOUSE.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(config.WAREHOUSE), read_only=read_only)


def table_names(con: duckdb.DuckDBPyConnection, schema: str) -> list[str]:
    rows = con.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = ? ORDER BY 1",
        [schema],
    ).fetchall()
    return [r[0] for r in rows]
