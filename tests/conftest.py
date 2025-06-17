import time
import zipfile
from pathlib import Path

import pytest


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
    if not extract_dir.exists():
        _extract_zip_to_dir(zip_path, extract_dir)
        extraction_flag.touch()
    else:
        while not extraction_flag.exists():
            time.sleep(1)
    return extract_dir


@pytest.fixture(scope="session", autouse=True)
def ibsi_ii_data_dir():
    zip_path = Path(__file__).parent / 'data' / 'IBSI_II.zip'
    extract_dir = Path(__file__).parent / 'data' / 'IBSI_II'
    extraction_flag = extract_dir.joinpath('.extraction_finished.flag')
    if not extract_dir.exists():
        _extract_zip_to_dir(zip_path, extract_dir)
        extraction_flag.touch()
    else:
        while not extraction_flag.exists():
            time.sleep(1)
    return extract_dir
