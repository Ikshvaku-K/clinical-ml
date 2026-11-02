"""Download the public source datasets and verify their checksums.

Diabetes 130-US Hospitals (UCI #296, CC BY 4.0). UCI publishes no checksum, so the
SHA-256 of the zip as first downloaded on 2026-11-02 is pinned below.
MIMIC-IV Clinical Database Demo on FHIR v2.1.0 (PhysioNet, ODbL). Verified against
PhysioNet's own SHA256SUMS.txt.
"""

import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path

from clinical_ml import config

UCI_URL = (
    "https://archive.ics.uci.edu/static/public/296/"
    "diabetes+130-us+hospitals+for+years+1999-2008.zip"
)
UCI_ZIP_SHA256 = "f82ac129da2ddd2299391ff6fbae3a6a58b3edcf59ac9d7bd480c00fe453112a"

PHYSIONET_BASE = "https://physionet.org/files/mimic-iv-fhir-demo/2.1.0/"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fetch(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "clinical-ml/0.1"})
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    tmp.rename(dest)


def _safe_extract(zip_path: Path, out_dir: Path) -> list[str]:
    """Extract only the CSVs, flattening paths so nothing can escape out_dir."""
    names = []
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            name = Path(info.filename).name
            if info.is_dir() or not name.endswith(".csv"):
                continue
            (out_dir / name).write_bytes(z.read(info))
            names.append(name)
    return sorted(names)


def download_diabetes130() -> dict:
    out = config.DIABETES_DIR
    out.mkdir(parents=True, exist_ok=True)
    zip_path = out / "diabetes130.zip"
    if not zip_path.exists():
        _fetch(UCI_URL, zip_path)
    digest = sha256(zip_path)
    if UCI_ZIP_SHA256 and digest != UCI_ZIP_SHA256:
        raise RuntimeError(f"Diabetes-130 checksum mismatch: {digest}")
    files = _safe_extract(zip_path, out)
    return {"url": UCI_URL, "zip_sha256": digest, "files": {f: sha256(out / f) for f in files}}


def download_fhir_demo() -> dict:
    out = config.FHIR_DEMO_DIR
    out.mkdir(parents=True, exist_ok=True)
    sums_path = out / "SHA256SUMS.txt"
    if not sums_path.exists():
        _fetch(PHYSIONET_BASE + "SHA256SUMS.txt", sums_path)
    expected = dict(
        (name, digest)
        for digest, name in (line.split() for line in sums_path.read_text().splitlines() if line)
    )
    verified = {}
    for name, digest in sorted(expected.items()):
        dest = out / name
        if not dest.exists():
            _fetch(PHYSIONET_BASE + name, dest)
        actual = sha256(dest)
        if actual != digest:
            raise RuntimeError(f"{name}: checksum mismatch ({actual} != {digest})")
        verified[name] = digest
    return {"url": PHYSIONET_BASE, "verified_files": len(verified), "sha256": verified}


def download_all() -> dict:
    manifest = {
        "diabetes130": download_diabetes130(),
        "mimic_iv_fhir_demo": download_fhir_demo(),
    }
    (config.RAW / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
