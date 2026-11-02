import hashlib

import pytest

from clinical_ml import config, sources


def test_fhir_demo_verifies_existing_files_without_network(tmp_path, monkeypatch):
    root = tmp_path / "mimic_iv_fhir_demo"
    (root / "fhir").mkdir(parents=True)
    payload = b"{}\n"
    (root / "fhir" / "MimicPatient.ndjson.gz").write_bytes(payload)
    (root / "SHA256SUMS.txt").write_text(
        f"{hashlib.sha256(payload).hexdigest()} fhir/MimicPatient.ndjson.gz\n")
    monkeypatch.setattr(config, "FHIR_DEMO_DIR", root)

    def no_network(*_):
        raise AssertionError("tried to download a file that is already present")

    monkeypatch.setattr(sources, "_fetch", no_network)
    assert sources.download_fhir_demo()["verified_files"] == 1


def test_fhir_demo_rejects_tampered_file(tmp_path, monkeypatch):
    root = tmp_path / "mimic_iv_fhir_demo"
    (root / "fhir").mkdir(parents=True)
    (root / "fhir" / "MimicPatient.ndjson.gz").write_bytes(b"tampered")
    (root / "SHA256SUMS.txt").write_text(f"{'0' * 64} fhir/MimicPatient.ndjson.gz\n")
    monkeypatch.setattr(config, "FHIR_DEMO_DIR", root)
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        sources.download_fhir_demo()
