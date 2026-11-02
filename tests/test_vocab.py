import pytest

from clinical_ml import config
from clinical_ml.omop import vocab


@pytest.mark.parametrize("raw, expected", [
    ("4280", "428.0"),     # MIMIC: dots stripped
    ("V462", "V46.2"),
    ("E8790", "E879.0"),
    ("428", "428"),        # 3-digit category
    ("8", "008"),          # Diabetes-130: leading zeros stripped
    ("41.9", "041.9"),
    ("250.83", "250.83"),
    ("V57", "V57"),
    ("E909", "E909"),
])
def test_icd9cm(raw, expected):
    assert vocab.icd9cm(raw) == expected


def test_icd10_and_proc():
    assert vocab.icd10cm("I5033") == "I50.33"
    assert vocab.icd10cm("I10") == "I10"
    assert vocab.icd9proc("5491") == "54.91"


def test_code_map_with_fake_athena(con):
    v = config.VOCAB
    con.execute(f"""
        INSERT INTO {v}.concept VALUES
          (1, 'Heart failure (ICD9)', 'Condition', 'ICD9CM', '4-dig billing code', NULL,
           '428.0', DATE '1970-01-01', DATE '2099-12-31', NULL),
          (2, 'Heart failure', 'Condition', 'SNOMED', 'Clinical Finding', 'S',
           '84114007', DATE '1970-01-01', DATE '2099-12-31', NULL),
          (3, 'insulin', 'Drug', 'RxNorm', 'Ingredient', 'S',
           '5856', DATE '1970-01-01', DATE '2099-12-31', NULL);
        INSERT INTO {v}.concept_relationship VALUES
          (1, 2, 'Maps to', DATE '1970-01-01', DATE '2099-12-31', NULL);
        CREATE TEMP TABLE staged AS SELECT * FROM (VALUES
          ('ICD9CM', '428.0'), ('ICD9CM', '999.9'), ('RxNorm-ingredient-name', 'Insulin'))
          t(source_vocabulary_id, source_code);
    """)
    name = vocab.build_code_map(con, "staged")
    got = dict(con.execute(
        f"SELECT source_code, concept_id FROM {name}").fetchall())
    assert got == {"428.0": 2, "999.9": 0, "Insulin": 3}
