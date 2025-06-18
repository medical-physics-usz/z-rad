import os
import time
import zipfile
from pathlib import Path

import pytest


def _acquire_lock(lock_path: Path, timeout: float = 60.0, check_interval: float = 0.1):
    """
    Acquire an exclusive file-based lock by creating a lock file.

    Args:
        lock_path (Path): The path to the lock file.
        timeout (float): Maximum time in seconds to wait for the lock.
        check_interval (float): Time in seconds between lock attempts.

    Raises:
        TimeoutError: If the lock could not be acquired within the timeout.
    """
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
    """
    Release a previously acquired lock by deleting the lock file.

    Args:
        lock_path (Path): The path to the lock file to remove.
    """
    try:
        lock_path.unlink()
    except FileNotFoundError:
        pass


def _extract_zip_to_dir(zip_path: Path, extract_dir: Path):
    """
    Extract all files from a ZIP archive into a target directory.

    Skips macOS metadata entries and top-level directory placeholders.

    Args:
        zip_path (Path): Path to the ZIP archive.
        extract_dir (Path): Directory where files will be extracted.
    """
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


def _prepare_data_dir(zip_path: Path, extract_dir: Path):
    """
    Ensure a ZIP archive is extracted exactly once across processes.

    Uses a lock file to serialize extraction, and a flag file to mark completion.
    Waits for extraction to finish if another process is performing it.

    Args:
        zip_path (Path): Path to the ZIP archive to extract.
        extract_dir (Path): Target directory for extraction.

    Returns:
        Path: The directory where the data has been extracted.
    """
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
def ibsi_i_data_dir():
    """
    Pytest fixture that provides the extracted IBSI_I data directory.

    Ensures the IBSI_I.zip archive is unpacked once per test session.
    """
    zip_path = Path(__file__).parent / 'data' / 'IBSI_I.zip'
    extract_dir = Path(__file__).parent / 'data' / 'IBSI_I'
    return _prepare_data_dir(zip_path, extract_dir)


@pytest.fixture(scope="session", autouse=True)
def ibsi_ii_data_dir():
    """
    Pytest fixture that provides the extracted IBSI_II data directory.

    Ensures the IBSI_II.zip archive is unpacked once per test session.
    """
    zip_path = Path(__file__).parent / 'data' / 'IBSI_II.zip'
    extract_dir = Path(__file__).parent / 'data' / 'IBSI_II'
    return _prepare_data_dir(zip_path, extract_dir)
