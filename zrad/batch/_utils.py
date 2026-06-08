import contextlib
from pathlib import Path
from typing import Callable, Sequence

import joblib

from ..exceptions import InvalidInputParametersError

VALID_MODALITIES = ['CT', 'MRI', 'PET', 'MG', 'RTDOSE']


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


def normalize_common_batch_options(batch) -> None:
    batch.input_directory = Path(batch.input_directory)
    batch.output_directory = Path(batch.output_directory)
    batch.input_data_type = str(batch.input_data_type).strip().lower()
    batch.modality = str(batch.modality).strip().upper()
    batch.parallel_backend = str(batch.parallel_backend).strip().lower()
    batch.patient_folders = normalize_names(batch.patient_folders)
    batch.start_folder = normalize_optional_text(batch.start_folder)
    batch.stop_folder = normalize_optional_text(batch.stop_folder)

    if batch.input_data_type not in ['dicom', 'nifti']:
        raise InvalidInputParametersError("input_data_type must be 'dicom' or 'nifti'.")
    if batch.modality not in VALID_MODALITIES:
        raise InvalidInputParametersError("modality must be one of CT, MRI, PET, MG, or RTDOSE.")
    if not batch.input_directory.exists():
        raise InvalidInputParametersError(f"Input directory '{batch.input_directory}' does not exist.")
    if not batch.input_directory.is_dir():
        raise InvalidInputParametersError(f"Input directory '{batch.input_directory}' is not a directory.")

    try:
        batch.number_of_threads = int(batch.number_of_threads)
    except (TypeError, ValueError):
        raise InvalidInputParametersError("number_of_threads must be a positive integer.")
    if batch.number_of_threads < 1:
        raise InvalidInputParametersError("number_of_threads must be a positive integer.")

    if batch.parallel_backend not in ['processes', 'threads']:
        raise InvalidInputParametersError("parallel_backend must be 'processes' or 'threads'.")


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

    return [path.name for path in sorted(input_directory.iterdir()) if path.is_dir() and not path.name.startswith('.')]


def find_nifti_file(directory: Path, name: str | None) -> Path | None:
    if name is None:
        return None
    base_path = directory / name
    if str(name).endswith(('.nii.gz', '.nii')):
        candidates = [base_path]
    else:
        candidates = [directory / f'{name}.nii.gz', directory / f'{name}.nii', base_path]

    for candidate in candidates:
        if candidate.exists():
            return candidate
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
