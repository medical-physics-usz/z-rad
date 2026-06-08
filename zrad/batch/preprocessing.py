import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from joblib import Parallel, delayed

from ..exceptions import DataStructureError, InvalidInputParametersError
from ..image import Image
from ..io import get_all_structure_names, get_dicom_files
from ..preprocessing import ImageResampler, MaskResampler
from ._utils import (
    find_nifti_file,
    joblib_progress,
    normalize_common_batch_options,
    normalize_names,
    normalize_optional_text,
    require_text,
    resolve_patient_folders,
)
from .results import BatchResult

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingCaseResult:
    case_name: str
    status: str
    image_output_path: Path | None = None
    mask_output_paths: dict[str, Path] = field(default_factory=dict)
    mask_union_output_path: Path | None = None
    processed_structures: list[str] = field(default_factory=list)
    skipped_structures: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass
class BatchPreprocessor:
    """Run preprocessing over many case folders and write NIfTI outputs.

    ``validate()`` normalizes public attributes in place. After validation,
    directories are ``Path`` objects, ``input_data_type`` is lower-case, modality
    is upper-case, and comma-separated folders/structures are stored as lists.
    """

    input_directory: str | Path
    output_directory: str | Path
    input_data_type: str
    modality: str
    number_of_threads: int = 1
    patient_folders: Sequence[str] | None = None
    start_folder: str | int | None = None
    stop_folder: str | int | None = None
    structures: Sequence[str] | None = None
    use_all_structures: bool = False
    nifti_image_name: str | None = None
    just_save_as_nifti: bool = False
    resample_resolution: float | None = None
    resample_dimension: str | None = None
    image_interpolation_method: str | None = None
    mask_interpolation_method: str | None = None
    mask_interpolation_threshold: float = 0.5
    mask_union: bool = False
    parallel_backend: str = 'processes'

    def validate(self) -> None:
        normalize_common_batch_options(self)

        if self.input_data_type == 'nifti':
            self.nifti_image_name = normalize_optional_text(self.nifti_image_name)
            if not self.nifti_image_name:
                raise InvalidInputParametersError("nifti_image_name is required for NIfTI preprocessing.")
            if self.use_all_structures:
                raise InvalidInputParametersError("use_all_structures is only supported for DICOM preprocessing.")

        self.structures = normalize_names(self.structures)

        if not self.just_save_as_nifti:
            if self.resample_resolution is None:
                raise InvalidInputParametersError("resample_resolution is required when resampling is enabled.")
            try:
                self.resample_resolution = float(self.resample_resolution)
            except (TypeError, ValueError):
                raise InvalidInputParametersError("resample_resolution must be a positive number.")
            if self.resample_resolution <= 0:
                raise InvalidInputParametersError("resample_resolution must be a positive number.")

            self.resample_dimension = str(self.resample_dimension).strip().upper()
            if self.resample_dimension not in ['2D', '3D']:
                raise InvalidInputParametersError("resample_dimension must be '2D' or '3D'.")
            self.image_interpolation_method = require_text(
                self.image_interpolation_method,
                "image_interpolation_method is required when resampling is enabled.",
            )
            self.mask_interpolation_method = require_text(
                self.mask_interpolation_method,
                "mask_interpolation_method is required when resampling is enabled.",
            )
            try:
                self.mask_interpolation_threshold = float(self.mask_interpolation_threshold)
            except (TypeError, ValueError):
                raise InvalidInputParametersError("mask_interpolation_threshold must be a number.")

    def plan(self) -> list[str]:
        self.validate()
        return self._resolve_patient_folders()

    def run(self, progress_callback: Callable[[int], None] | None = None) -> BatchResult:
        self.validate()
        self.output_directory.mkdir(parents=True, exist_ok=True)
        patient_folders = self._resolve_patient_folders()

        if self.number_of_threads == 1:
            case_results = []
            for patient_folder in patient_folders:
                case_results.append(self._process_case(patient_folder))
                if progress_callback:
                    progress_callback(1)
        else:
            with joblib_progress(progress_callback):
                case_results = Parallel(n_jobs=self.number_of_threads, prefer=self.parallel_backend)(
                    delayed(self._process_case)(patient_folder) for patient_folder in patient_folders
                )

        return BatchResult(workflow='preprocessing', case_results=case_results)

    def _resolve_patient_folders(self) -> list[str]:
        return resolve_patient_folders(
            self.input_directory,
            self.patient_folders,
            self.start_folder,
            self.stop_folder,
        )

    def _process_case(self, case_name: str) -> PreprocessingCaseResult:
        case_dir = self.input_directory / case_name
        case_output_dir = self.output_directory / case_name
        result = PreprocessingCaseResult(case_name=case_name, status='processed')
        logger.info("Processing patient's %s image.", case_name)

        try:
            image = self._load_image(case_dir)
        except (DataStructureError, FileNotFoundError, ValueError) as exc:
            return PreprocessingCaseResult(case_name=case_name, status='skipped', error=str(exc))
        except Exception as exc:
            logger.exception("Patient %s failed while loading image.", case_name)
            return PreprocessingCaseResult(case_name=case_name, status='failed', error=str(exc))

        try:
            image_new = image.copy() if self.just_save_as_nifti else self._resample_image(image)
            image_output_path = case_output_dir / 'image.nii.gz'
            image_new.save_as_nifti(image_output_path)
            result.image_output_path = image_output_path

            structure_names, rtstruct_path = self._resolve_structures(case_dir)
            if structure_names:
                self._process_masks(case_dir, result, image, structure_names, rtstruct_path)
            return result
        except Exception as exc:
            logger.exception("Patient %s failed during preprocessing.", case_name)
            result.status = 'failed'
            result.error = str(exc)
            return result

    def _load_image(self, case_dir: Path) -> Image:
        if self.input_data_type == 'dicom':
            return Image.from_dicom(case_dir, modality=self.modality)

        image_path = find_nifti_file(case_dir, self.nifti_image_name)
        if image_path is None:
            raise FileNotFoundError(case_dir / self.nifti_image_name)
        return Image.from_nifti(image_path)

    def _resolve_structures(self, case_dir: Path) -> tuple[list[str], str | None]:
        if self.input_data_type == 'nifti':
            return list(self.structures or []), None

        rtstructs = get_dicom_files(case_dir, modality='RTSTRUCT')
        rtstruct_path = rtstructs[0]['file_path'] if rtstructs else None
        if self.use_all_structures and rtstruct_path:
            return get_all_structure_names(rtstruct_path), rtstruct_path
        return list(self.structures or []), rtstruct_path

    def _process_masks(
        self,
        case_dir: Path,
        result: PreprocessingCaseResult,
        image: Image,
        structure_names: Sequence[str],
        rtstruct_path: str | None,
    ) -> None:
        mask_union = None
        case_output_dir = self.output_directory / result.case_name

        for structure_name in structure_names:
            try:
                mask = self._load_mask(case_dir, structure_name, image, rtstruct_path)
            except Exception as exc:
                result.skipped_structures.append(structure_name)
                logger.warning("Skipping patient's %s ROI %s: %s", result.case_name, structure_name, exc)
                continue

            if mask is None or mask.array is None:
                result.skipped_structures.append(structure_name)
                continue

            logger.info("Processing patient's %s ROI: %s.", result.case_name, structure_name)
            mask_new = mask.copy() if self.just_save_as_nifti else self._resample_mask(mask)
            mask_output_path = case_output_dir / f'{structure_name}.nii.gz'
            mask_new.save_as_nifti(mask_output_path)

            result.mask_output_paths[structure_name] = mask_output_path
            result.processed_structures.append(structure_name)

            if self.mask_union:
                if mask_union:
                    mask_union.array = np.bitwise_or(
                        _binary_mask_array(mask_union.array),
                        _binary_mask_array(mask_new.array),
                    ).astype(np.int16)
                else:
                    mask_union = mask_new.copy()
                    mask_union.array = _binary_mask_array(mask_union.array)

        if mask_union:
            mask_union_output_path = case_output_dir / 'mask_union.nii.gz'
            mask_union.save_as_nifti(mask_union_output_path)
            result.mask_union_output_path = mask_union_output_path

    def _load_mask(
        self,
        case_dir: Path,
        structure_name: str,
        image: Image,
        rtstruct_path: str | None,
    ) -> Image | None:
        if self.input_data_type == 'dicom':
            if not rtstruct_path:
                return None
            return Image.from_dicom_mask(rtstruct_path=rtstruct_path, structure_name=structure_name, reference=image)

        mask_path = find_nifti_file(case_dir, structure_name)
        if mask_path is None:
            return None
        return Image.from_nifti_mask(mask_path, reference=image)

    def _resample_image(self, image: Image) -> Image:
        return ImageResampler(
            resolution=self._target_resolution(image),
            method=self.image_interpolation_method,
            intensity_rounding='nearest_integer' if self.modality == 'CT' else None,
        ).apply(image)

    def _resample_mask(self, mask: Image) -> Image:
        return MaskResampler(
            resolution=self._target_resolution(mask),
            method=self.mask_interpolation_method,
            partial_volume_threshold=self.mask_interpolation_threshold,
        ).apply(mask)

    def _target_resolution(self, image: Image) -> tuple[float, float, float]:
        if self.resample_dimension == '3D':
            return (self.resample_resolution, self.resample_resolution, self.resample_resolution)
        if self.resample_dimension == '2D':
            return (self.resample_resolution, self.resample_resolution, image.spacing[2])
        raise ValueError(f"Resample dimension '{self.resample_dimension}' is not supported.")


def _binary_mask_array(array):
    return np.where(array > 0, 1, 0).astype(np.int16)
