import hashlib
import os
import time
import zipfile
import zlib
from pathlib import Path

import pytest


def _acquire_file_lock(lock_path: Path, timeout: float = 60.0, check_interval: float = 0.1):
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
            # The lock existed when os.open() ran. It may be released before
            # this process retries, which is a normal lock handoff.
            pass
        except PermissionError:
            # On Windows, opening an existing lock file with O_EXCL may raise
            # PermissionError rather than FileExistsError. Disambiguate that
            # case from an unrelated permissions problem.
            if not lock_path.exists():
                raise
        if time.time() - start > timeout:
            raise TimeoutError(f"Timeout waiting for lock {lock_path}")
        time.sleep(check_interval)


def _release_file_lock(lock_path: Path):
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
            # ZIP entry names are standardized with forward slashes, but
            # normalize legacy archives created on Windows as well.
            parts = Path(member.filename.replace("\\", "/")).parts
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
    Ensure extracted files match the current ZIP archive across processes.

    Uses a lock to serialize integrity checks and extraction. The completion
    marker stores the archive fingerprint; member CRCs detect missing or
    modified extracted files even when the marker is present.

    Args:
        zip_path (Path): Path to the ZIP archive to extract.
        extract_dir (Path): Target directory for extraction.

    Returns:
        Path: The directory where the data has been extracted.
    """
    extraction_flag = extract_dir / '.extraction_finished.flag'
    fingerprint = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    lock_file = extract_dir.with_suffix('.lock')
    _acquire_file_lock(lock_file)
    try:
        complete = extraction_flag.exists() and extraction_flag.read_text() == fingerprint
        with zipfile.ZipFile(zip_path) as archive:
            for member in archive.infolist():
                parts = Path(member.filename.replace('\\', '/')).parts
                if member.filename.startswith('__MACOSX/') or member.is_dir() or len(parts) <= 1:
                    continue
                target = extract_dir / Path(*parts[1:])
                if not target.is_file() or zlib.crc32(target.read_bytes()) != member.CRC:
                    complete = False
                    break
        if not complete:
            extraction_flag.unlink(missing_ok=True)
            _extract_zip_to_dir(zip_path, extract_dir)
            extraction_flag.write_text(fingerprint)
    finally:
        _release_file_lock(lock_file)
    return extract_dir


@pytest.fixture(scope="session")
def ibsi_i_data_dir():
    """
    Pytest fixture that provides the extracted IBSI_I data directory.

    Ensures the IBSI_I.zip archive is unpacked once per test session.
    """
    zip_path = Path(__file__).parent / 'data' / 'IBSI_I.zip'
    extract_dir = Path(__file__).parent / 'data' / 'IBSI_I'
    return _prepare_data_dir(zip_path, extract_dir)


@pytest.fixture(scope="session")
def ibsi_ii_data_dir():
    """
    Pytest fixture that provides the extracted IBSI_II data directory.

    Ensures the IBSI_II.zip archive is unpacked once per test session.
    """
    zip_path = Path(__file__).parent / 'data' / 'IBSI_II.zip'
    extract_dir = Path(__file__).parent / 'data' / 'IBSI_II'
    return _prepare_data_dir(zip_path, extract_dir)


@pytest.fixture(scope="session")
def ibsi_suv_data_dir():
    """Provide the extracted official IBSI-SUV v3.0.1 DRO directory."""
    zip_path = Path(__file__).parent / "data" / "IBSI_SUV.zip"
    extract_dir = Path(__file__).parent / "data" / "IBSI_SUV"
    return _prepare_data_dir(zip_path, extract_dir)
