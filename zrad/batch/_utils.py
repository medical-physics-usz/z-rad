import contextlib
from pathlib import Path
from typing import Callable, Sequence

import joblib

from ..exceptions import InvalidInputParametersError


def normalize_optional_text(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def require_text(value, message: str) -> str:
    text = normalize_optional_text(value)
    if not text:
        raise InvalidInputParametersError(message)
    return text


def normalize_names(values: Sequence[str] | str | None) -> list[str] | None:
    if values is None:
        return None
    if isinstance(values, str):
        values = values.split(',')
    normalized = [str(value).strip() for value in values if str(value).strip()]
    return normalized or None


def resolve_patient_folders(input_directory: Path, patient_folders, start_folder, stop_folder) -> list[str]:
    if start_folder and stop_folder:
        try:
            start = int(start_folder)
            stop = int(stop_folder)
        except ValueError:
            raise InvalidInputParametersError("start_folder and stop_folder must be integers.")
        return [
            path.name
            for path in sorted(input_directory.iterdir())
            if path.is_dir() and path.name.isdigit() and start <= int(path.name) <= stop
        ]

    if patient_folders:
        return list(patient_folders)

    if start_folder or stop_folder:
        raise InvalidInputParametersError("start_folder and stop_folder must be provided together.")

    return [
        path.name
        for path in sorted(input_directory.iterdir())
        if path.is_dir() and not path.name.startswith('.')
    ]


def find_nifti_file(directory: Path, name: str | None) -> Path | None:
    if name is None:
        return None
    base_path = directory / name
    for candidate in [base_path.with_suffix(base_path.suffix + '.gz'), base_path.with_suffix('.nii'), base_path]:
        if candidate.exists():
            return candidate
    nii_gz_path = directory / f'{name}.nii.gz'
    if nii_gz_path.exists():
        return nii_gz_path
    return None


@contextlib.contextmanager
def joblib_progress(progress_callback: Callable[[int], None] | None = None):
    class ProgressBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            if progress_callback:
                progress_callback(self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = ProgressBatchCompletionCallback
    try:
        yield
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
