#!/usr/bin/env python3
"""Download and extract the official Radar Ghost Dataset v1.1.

The dataset is deliberately not committed to this repository.  This utility
downloads the hand-labelled ``original.zip`` archive directly from the fixed
Zenodo v1.1 record, supports resuming an interrupted transfer, verifies the
published size and MD5 checksum, and rejects unsafe ZIP paths before
extraction.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import sys
import time
from urllib.request import Request, urlopen
import zipfile


ZENODO_RECORD = 6676246
ARCHIVE_NAME = "original.zip"
ARCHIVE_URL = (
    "https://zenodo.org/api/records/6676246/files/original.zip/content"
)
ARCHIVE_SIZE = 5_818_814_597
ARCHIVE_MD5 = "3873152766839286469b4b7e63ceba12"
DOWNLOAD_CHUNK_BYTES = 8 * 1024 * 1024
DISK_MARGIN_BYTES = 1024 * 1024 * 1024


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="data/radar_ghost_v1_1",
        help="dataset root (default: data/radar_ghost_v1_1)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="discard an invalid/complete archive and extract again",
    )
    parser.add_argument(
        "--delete-archive",
        action="store_true",
        help="delete original.zip after successful verified extraction",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="HTTP connection/read timeout in seconds",
    )
    return parser.parse_args()


def _human_bytes(value):
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(size) < 1024.0 or unit == "TiB":
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TiB"


def _md5(path):
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while True:
            block = handle.read(DOWNLOAD_CHUNK_BYTES)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _check_free_space(path, required_bytes, purpose):
    free = shutil.disk_usage(path).free
    if free < required_bytes + DISK_MARGIN_BYTES:
        raise OSError(
            f"Not enough free space to {purpose}: need at least "
            f"{_human_bytes(required_bytes + DISK_MARGIN_BYTES)}, have "
            f"{_human_bytes(free)}"
        )


def _verified_archive(path):
    if not path.is_file() or path.stat().st_size != ARCHIVE_SIZE:
        return False
    print(f"Verifying MD5 for {path} ...")
    return _md5(path) == ARCHIVE_MD5


def _download(archive_path, timeout_s, force=False):
    partial_path = archive_path.with_name(f"{archive_path.name}.part")
    if archive_path.exists():
        if _verified_archive(archive_path):
            print(f"Using verified archive: {archive_path}")
            return
        if not force:
            raise RuntimeError(
                f"Existing archive is invalid: {archive_path}. "
                "Remove it or rerun with --force."
            )
        archive_path.unlink()
    if force and partial_path.exists():
        partial_path.unlink()

    downloaded = partial_path.stat().st_size if partial_path.exists() else 0
    if downloaded > ARCHIVE_SIZE:
        raise RuntimeError(
            f"Partial archive is larger than expected: {partial_path}. "
            "Rerun with --force."
        )
    if downloaded == ARCHIVE_SIZE:
        print(f"Verifying completed partial archive: {partial_path}")
        checksum = _md5(partial_path)
        if checksum != ARCHIVE_MD5:
            raise RuntimeError(
                f"Checksum mismatch for {partial_path}: expected "
                f"{ARCHIVE_MD5}, got {checksum}. Rerun with --force."
            )
        os.replace(partial_path, archive_path)
        return
    _check_free_space(
        archive_path.parent,
        ARCHIVE_SIZE - downloaded,
        "download the archive",
    )

    headers = {"User-Agent": "carla-radar-research/1.0"}
    if downloaded:
        headers["Range"] = f"bytes={downloaded}-"
        print(f"Resuming at {_human_bytes(downloaded)}")
    request = Request(ARCHIVE_URL, headers=headers)
    with urlopen(request, timeout=timeout_s) as response:
        status = response.getcode()
        if downloaded and status != 206:
            print("Server did not accept the range request; restarting download")
            partial_path.unlink()
            downloaded = 0
            _check_free_space(
                archive_path.parent,
                ARCHIVE_SIZE,
                "restart the archive download",
            )
            mode = "wb"
        else:
            mode = "ab" if downloaded else "wb"
        session_start_bytes = downloaded
        started = time.monotonic()
        last_report = started
        with partial_path.open(mode) as handle:
            while True:
                block = response.read(DOWNLOAD_CHUNK_BYTES)
                if not block:
                    break
                handle.write(block)
                downloaded += len(block)
                now = time.monotonic()
                if now - last_report >= 1.0:
                    elapsed = max(now - started, 1.0e-6)
                    rate = (downloaded - session_start_bytes) / elapsed
                    percent = 100.0 * downloaded / ARCHIVE_SIZE
                    sys.stdout.write(
                        f"\r{percent:6.2f}%  {_human_bytes(downloaded)} / "
                        f"{_human_bytes(ARCHIVE_SIZE)}  {_human_bytes(rate)}/s"
                    )
                    sys.stdout.flush()
                    last_report = now
    print()
    if partial_path.stat().st_size != ARCHIVE_SIZE:
        raise RuntimeError(
            f"Incomplete download: expected {ARCHIVE_SIZE} bytes, got "
            f"{partial_path.stat().st_size}. Rerun the same command to resume."
        )
    print("Verifying downloaded archive ...")
    checksum = _md5(partial_path)
    if checksum != ARCHIVE_MD5:
        raise RuntimeError(
            f"Checksum mismatch for {partial_path}: expected {ARCHIVE_MD5}, "
            f"got {checksum}. Rerun with --force."
        )
    os.replace(partial_path, archive_path)


def _archive_layout(archive):
    files = [
        PurePosixPath(info.filename)
        for info in archive.infolist()
        if not info.is_dir()
    ]
    if not files:
        raise RuntimeError("Official archive contains no files")
    first_parts = {path.parts[0].lower() for path in files if path.parts}
    if "original" in first_parts:
        return "contains_original"
    if {"train", "val", "test"}.issubset(first_parts):
        return "contains_splits"
    raise RuntimeError(
        "Unexpected archive layout; expected original/{train,val,test} or "
        "top-level train/val/test"
    )


def _safe_target(root, member_name):
    relative = PurePosixPath(member_name)
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(f"Unsafe path in ZIP archive: {member_name!r}")
    target = root.joinpath(*relative.parts)
    root_text = str(root.resolve())
    target_text = str(target.resolve())
    if os.path.commonpath((root_text, target_text)) != root_text:
        raise RuntimeError(f"Unsafe path in ZIP archive: {member_name!r}")
    return target


def _extract(archive_path, output_root, force=False):
    marker = output_root / ".original_v1_1_extracted.json"
    prepared_root = output_root / "original"
    expected_splits = tuple(
        prepared_root / name for name in ("train", "val", "test")
    )
    if (
        marker.is_file()
        and all(path.is_dir() for path in expected_splits)
        and not force
    ):
        with marker.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        if metadata.get("archive_md5") == ARCHIVE_MD5:
            print(f"Dataset is already extracted: {prepared_root}")
            return prepared_root

    with zipfile.ZipFile(archive_path, "r") as archive:
        layout = _archive_layout(archive)
        extraction_root = (
            output_root if layout == "contains_original" else prepared_root
        )
        members = archive.infolist()
        uncompressed = sum(info.file_size for info in members)
        _check_free_space(output_root, uncompressed, "extract the archive")
        print(
            f"Extracting {len(members)} entries "
            f"({_human_bytes(uncompressed)}) to {extraction_root}"
        )
        for index, info in enumerate(members, start=1):
            mode = info.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise RuntimeError(
                    f"Symbolic link is not allowed: {info.filename!r}"
                )
            target = _safe_target(extraction_root, info.filename)
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info, "r") as source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination, DOWNLOAD_CHUNK_BYTES)
            if index % 10 == 0 or index == len(members):
                print(f"  extracted {index}/{len(members)}", flush=True)

    if not all(path.is_dir() for path in expected_splits):
        raise RuntimeError(
            f"Extraction completed but train/val/test were not found under "
            f"{prepared_root}"
        )
    metadata = {
        "archive": ARCHIVE_NAME,
        "archive_md5": ARCHIVE_MD5,
        "archive_size": ARCHIVE_SIZE,
        "doi": f"10.5281/zenodo.{ZENODO_RECORD}",
        "input_directory": str(prepared_root.resolve()),
        "version": "1.1",
    }
    with marker.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return prepared_root


def main():
    args = parse_args()
    if args.timeout <= 0.0:
        raise ValueError("--timeout must be positive")
    output_root = Path(args.output).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    archive_path = output_root / ARCHIVE_NAME
    print("Radar Ghost Dataset v1.1 (CC BY-NC-SA 4.0)")
    print(f"DOI: https://doi.org/10.5281/zenodo.{ZENODO_RECORD}")
    print(f"Archive size: {_human_bytes(ARCHIVE_SIZE)}")
    _download(archive_path, args.timeout, force=args.force)
    prepared_root = _extract(
        archive_path,
        output_root,
        force=args.force,
    )
    if args.delete_archive:
        archive_path.unlink()
        print(f"Deleted verified archive: {archive_path}")
    print(f"Ready. Use this as --input: {prepared_root}")


if __name__ == "__main__":
    main()
