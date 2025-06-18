import os
import time
import zipfile
from pathlib import Path

import pytest


def _acquire_lock(lock_path: Path, timeout: float = 60.0, check_interval: float = 0.1):
    start = time.time()
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
            os.close(fd)
            return
        except FileExistsError:
            if time.time() - start > timeout:
                raise TimeoutError(f"Timeout waiting for lock {lock_path}")
            time.sleep(check_interval)


def _release_lock(lock_path: Path):
    try:
        lock_path.unlink()
    except FileNotFoundError:
        pass


def _extract_zip_to_dir(zip_path: Path, extract_dir: Path):
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        for member in zf.infolist():
            # Skip macOS metadata
            if member.filename.startswith("__MACOSX/"):
                continue
            parts = Path(member.filename).parts
            # Skip top-level directory entries
            if len(parts) <= 1:
                continue
            relative_path = Path(*parts[1:])
            target_path = extract_dir / relative_path
            if member.is_dir():
                target_path.mkdir(parents=True, exist_ok=True)
            else:
                target_path.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as src, open(target_path, "wb") as dst:
                    dst.write(src.read())


@pytest.fixture(scope="session", autouse=True)
def ibsi_i_data_dir():
    zip_path = Path(__file__).parent / 'data' / 'IBSI_I.zip'
    extract_dir = Path(__file__).parent / 'data' / 'IBSI_I'
    extraction_flag = extract_dir.joinpath('.extraction_finished.flag')
    lock_file = extract_dir.with_suffix('.lock')

    _acquire_lock(lock_file)
    try:
        if not extraction_flag.exists():
            if not extract_dir.exists():
                _extract_zip_to_dir(zip_path, extract_dir)
            extraction_flag.touch()
    finally:
        _release_lock(lock_file)

    while not extraction_flag.exists():
        time.sleep(1)
    return extract_dir


@pytest.fixture(scope="session", autouse=True)
def ibsi_ii_data_dir():
    zip_path = Path(__file__).parent / 'data' / 'IBSI_II.zip'
    extract_dir = Path(__file__).parent / 'data' / 'IBSI_II'
    extraction_flag = extract_dir.joinpath('.extraction_finished.flag')
    lock_file = extract_dir.with_suffix('.lock')

    _acquire_lock(lock_file)
    try:
        if not extraction_flag.exists():
            if not extract_dir.exists():
                _extract_zip_to_dir(zip_path, extract_dir)
            extraction_flag.touch()
    finally:
        _release_lock(lock_file)

    while not extraction_flag.exists():
        time.sleep(1)
    return extract_dir
