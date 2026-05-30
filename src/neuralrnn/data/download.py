"""数据集下载 / 缓存 / 解压 / 校验工具。

被 data/registry.py 的 load_dataset() 调用。设计目标：把每篇论文 notebook 里
散落的 wget / Dataverse / Zenodo 链接，统一成"声明式下载"——只在 DatasetSpec
里写 URL 与文件名，真正的下载/缓存/解压/校验逻辑全部收敛到这里。

缓存目录优先级：
    1. 环境变量 NEURALRNN_CACHE
    2. ~/.cache/neuralrnn/datasets

注意：本仓库运行环境可能禁用网络。本文件是"可运行模板"：在有网环境直接可用；
无网时若缓存已存在则直接命中，否则抛出清晰的错误提示用户手动放置文件。
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
    print(f"[neuralrnn] 下载 {url}\n          -> {dst}")
    try:
        with urllib.request.urlopen(url) as resp, open(dst, "wb") as f:  # noqa: S310
            shutil.copyfileobj(resp, f)
    except Exception as e:  # 网络禁用 / 链接失效
        raise RuntimeError(
            f"下载失败：{url}\n"
            f"若处于无网环境，请手动把文件放到：{dst}\n原始错误：{e}"
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
        raise ValueError(f"未知解压类型: {kind}")


def ensure_files(spec: DatasetSpec) -> dict[str, str]:
    """确保 spec 所需文件已就绪，返回 {逻辑名: 本地绝对路径}。

    约定（与 registry.load_dataset 对接）：
      - spec.files 给定 {逻辑名: 文件名} 时，返回同名 key 的本地路径字典；
        load_dataset 会把它们转成 `<逻辑名>_path=...` 关键字传给 loader。
      - 否则返回 {"file": 单文件路径}。
    """
    assert spec.url is not None, "需要下载的数据集必须提供 spec.url"
    # 每个数据集一个子目录，名取自 URL/文件名，保证幂等
    key = spec.filename or Path(spec.url).name or "dataset"
    ds_dir = cache_root() / Path(key).stem
    ds_dir.mkdir(parents=True, exist_ok=True)

    download_name = spec.filename or Path(spec.url).name or "download.bin"
    archive_path = ds_dir / download_name
    _download(spec.url, archive_path)

    if spec.sha256:
        got = _sha256(archive_path)
        if got != spec.sha256:
            raise RuntimeError(f"{archive_path} 校验失败：期望 {spec.sha256}，实得 {got}")

    if spec.unpack:
        _unpack(archive_path, spec.unpack, ds_dir)

    # 解析逻辑名 -> 本地路径
    if spec.files:
        out: dict[str, str] = {}
        for logical, fname in spec.files.items():
            # 文件可能在解压根目录或其子目录里：递归找第一个匹配名
            cand = ds_dir / fname
            if not cand.exists():
                matches = list(ds_dir.rglob(fname))
                if not matches:
                    raise FileNotFoundError(f"解压后未找到 {fname}（于 {ds_dir}）")
                cand = matches[0]
            out[logical] = str(cand.resolve())
        return out

    return {"file": str(archive_path.resolve())}
