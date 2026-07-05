"""Dataset download / cache / unpack / verification utilities.

Called by load_dataset() in data/registry.py. Design goal: turn the scattered wget / Dataverse / Zenodo links
in each paper's notebook into a declarative download—only write the URL and filename in DatasetSpec,
and converge the actual download / cache / unpack / verification logic here.

Cache directory priority:
    1. Environment variable NEURALRNN_CACHE
    2. ~/.cache/neuralrnn/datasets

Note: the repository runtime may disable network. This file is a runnable template: it works directly online;
when offline, it returns from cache if present, otherwise raises a clear error asking the user to place the file manually.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path

from .registry import DatasetSpec


def cache_root() -> Path:
    root = os.environ.get("NEURALRNN_CACHE")
    base = Path(root) if root else Path.home() / ".cache" / "neuralrnn" / "datasets"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dst: Path) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"[neuralrnn] Downloading {url}\n          -> {dst}")
    try:
        with urllib.request.urlopen(url) as resp, open(dst, "wb") as f:  # noqa: S310
            shutil.copyfileobj(resp, f)
    except Exception as e:  # Network disabled / invalid link
        raise RuntimeError(
            f"Download failed: {url}\n"
            f"If you are offline, please place the file manually at: {dst}\nOriginal error: {e}"
        ) from e


def _unpack(archive: Path, kind: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if kind == "zip":
        with zipfile.ZipFile(archive) as z:
            z.extractall(out_dir)
    elif kind == "tar":
        with tarfile.open(archive) as t:
            t.extractall(out_dir)  # noqa: S202
    else:
        raise ValueError(f"Unknown archive type: {kind}")


def ensure_files(spec: DatasetSpec) -> dict[str, str]:
    """Ensure the files required by spec are ready and return {logical_name: local absolute path}.

    Conventions (matching registry.load_dataset):
      - When spec.files gives {logical_name: filename}, return a local-path dict with the same keys;
        load_dataset will convert them to `<logical_name>_path=...` keyword arguments passed to loader.
      - Otherwise return {"file": single-file path}.
    """
    assert spec.url is not None, "A dataset that needs downloading must provide spec.url"
    # One subdirectory per dataset, named from the URL/filename for idempotence
    key = spec.filename or Path(spec.url).name or "dataset"
    ds_dir = cache_root() / Path(key).stem
    ds_dir.mkdir(parents=True, exist_ok=True)

    download_name = spec.filename or Path(spec.url).name or "download.bin"
    archive_path = ds_dir / download_name
    _download(spec.url, archive_path)

    if spec.sha256:
        got = _sha256(archive_path)
        if got != spec.sha256:
            raise RuntimeError(f"{archive_path} checksum failed: expected {spec.sha256}, got {got}")

    if spec.unpack:
        _unpack(archive_path, spec.unpack, ds_dir)

    # Map logical name -> local path
    if spec.files:
        out: dict[str, str] = {}
        for logical, fname in spec.files.items():
            # File may be in the unpack root or a subdirectory: recursively find the first match
            cand = ds_dir / fname
            if not cand.exists():
                matches = list(ds_dir.rglob(fname))
                if not matches:
                    raise FileNotFoundError(f"{fname} not found after extraction (in {ds_dir})")
                cand = matches[0]
            out[logical] = str(cand.resolve())
        return out

    return {"file": str(archive_path.resolve())}
