import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from joblib import Parallel, delayed

from ..exceptions import DataStructureError, InvalidInputParametersError
from ..image import Image
from ..io import get_all_structure_names, get_dicom_files
from ..preprocessing import IntensityMaskBuilder, Resegmenter, RoiData, TextureDiscretizer
from ..radiomics import Radiomics
from ._utils import (
    find_nifti_file,
    joblib_progress,
    normalize_names,
    normalize_optional_text,
    require_text,
    resolve_patient_folders,
)
from .results import BatchResult

logger = logging.getLogger(__name__)


@dataclass
class RadiomicsCaseResult:
    case_name: str
    status: str
    processed_structures: list[str] = field(default_factory=list)
    skipped_structures: list[str] = field(default_factory=list)
    feature_count: int = 0
    error: str | None = None


@dataclass
class BatchRadiomicsExtractor:
    """Extract radiomics features for many case folders and write one CSV."""

    input_directory: str | Path
    output_directory: str | Path
    input_data_type: str
    modality: str
    aggregation_dimension: str
    aggregation_method: str
    discretization_method: str
    number_of_threads: int = 1
    patient_folders: Sequence[str] | None = None
    start_folder: str | int | None = None
    stop_folder: str | int | None = None
    structures: Sequence[str] | None = None
    use_all_structures: bool = False
    nifti_image_name: str | None = None
    nifti_filtered_image_name: str | None = None
    slice_weighting: bool = False
    slice_median: bool = False
    number_of_bins: int | str | None = None
    bin_size: float | str | None = None
    intensity_range: Sequence[float] | None = None
    outlier_range: float | str | None = None
    output_filename: str = 'radiomics.csv'
    parallel_backend: str = 'processes'

    def validate(self) -> None:
        self.input_directory = Path(self.input_directory)
        self.output_directory = Path(self.output_directory)
        self.input_data_type = str(self.input_data_type).strip().lower()
        self.modality = str(self.modality).strip().upper()
        self.parallel_backend = str(self.parallel_backend).strip().lower()
        self.aggregation_dimension = require_text(
            self.aggregation_dimension,
            "aggregation_dimension is required.",
        ).upper()
        self.aggregation_method = require_text(
            self.aggregation_method,
            "aggregation_method is required.",
        ).upper()
        self.discretization_method = require_text(
            self.discretization_method,
            "discretization_method is required.",
        )
        self.output_filename = require_text(self.output_filename, "output_filename is required.")

        if self.input_data_type not in ['dicom', 'nifti']:
            raise InvalidInputParametersError("input_data_type must be 'dicom' or 'nifti'.")
        if self.modality not in ['CT', 'MRI', 'PET', 'MG', 'RTDOSE']:
            raise InvalidInputParametersError("modality must be one of CT, MRI, PET, MG, or RTDOSE.")
        if self.aggregation_dimension not in ['2D', '2.5D', '3D']:
            raise InvalidInputParametersError("aggregation_dimension must be '2D', '2.5D', or '3D'.")
        if self.aggregation_method not in ['MERG', 'AVER', 'SLICE_MERG', 'DIR_MERG']:
            raise InvalidInputParametersError(
                "aggregation_method must be one of MERG, AVER, SLICE_MERG, or DIR_MERG."
            )
        if not self.input_directory.exists():
            raise InvalidInputParametersError(f"Input directory '{self.input_directory}' does not exist.")
        if not self.input_directory.is_dir():
            raise InvalidInputParametersError(f"Input directory '{self.input_directory}' is not a directory.")

        try:
            self.number_of_threads = int(self.number_of_threads)
        except (TypeError, ValueError):
            raise InvalidInputParametersError("number_of_threads must be a positive integer.")
        if self.number_of_threads < 1:
            raise InvalidInputParametersError("number_of_threads must be a positive integer.")
        if self.parallel_backend not in ['processes', 'threads']:
            raise InvalidInputParametersError("parallel_backend must be 'processes' or 'threads'.")

        self.structures = normalize_names(self.structures)
        self.patient_folders = normalize_names(self.patient_folders)
        self.start_folder = normalize_optional_text(self.start_folder)
        self.stop_folder = normalize_optional_text(self.stop_folder)
        self.nifti_filtered_image_name = normalize_optional_text(self.nifti_filtered_image_name)

        if self.input_data_type == 'nifti':
            self.nifti_image_name = normalize_optional_text(self.nifti_image_name)
            if not self.nifti_image_name:
                raise InvalidInputParametersError("nifti_image_name is required for NIfTI radiomics.")
            if self.use_all_structures:
                raise InvalidInputParametersError("use_all_structures is only supported for DICOM radiomics.")
            if not self.structures:
                raise InvalidInputParametersError("structures are required for NIfTI radiomics.")
        elif not self.use_all_structures and not self.structures:
            raise InvalidInputParametersError("DICOM radiomics requires structures or use_all_structures=True.")

        self.slice_weighting = _as_bool(self.slice_weighting)
        self.slice_median = _as_bool(self.slice_median)
        if self.slice_weighting and self.slice_median:
            raise InvalidInputParametersError("slice_weighting and slice_median cannot both be enabled.")

        self.intensity_range = _normalize_intensity_range(self.intensity_range)
        self._validate_discretization()
        self.outlier_range = _normalize_positive_float(self.outlier_range, "outlier_range must be positive.")

    def plan(self) -> list[str]:
        self.validate()
        return self._resolve_patient_folders()

    def run(self, progress_callback: Callable[[int], None] | None = None) -> BatchResult:
        self.validate()
        self.output_directory.mkdir(parents=True, exist_ok=True)
        patient_folders = self._resolve_patient_folders()

        if self.number_of_threads == 1:
            processed = []
            for patient_folder in patient_folders:
                processed.append(self._process_case(patient_folder))
                if progress_callback:
                    progress_callback(1)
        else:
            with joblib_progress(progress_callback):
                processed = Parallel(n_jobs=self.number_of_threads, prefer=self.parallel_backend)(
                    delayed(self._process_case)(patient_folder) for patient_folder in patient_folders
                )

        case_results = [case_result for case_result, _features in processed]
        feature_rows = [row for _case_result, features in processed for row in features]
        _write_radiomics_csv(self.output_directory / self.output_filename, feature_rows)
        return BatchResult(workflow='radiomics', case_results=case_results)

    def _resolve_patient_folders(self) -> list[str]:
        return resolve_patient_folders(
            self.input_directory,
            self.patient_folders,
            self.start_folder,
            self.stop_folder,
        )

    def _process_case(self, case_name: str) -> tuple[RadiomicsCaseResult, list[dict]]:
        case_dir = self.input_directory / case_name
        result = RadiomicsCaseResult(case_name=case_name, status='processed')
        feature_rows = []
        logger.info("Processing patient: %s.", case_name)

        try:
            image, filtered_image = self._load_images(case_dir)
        except (DataStructureError, FileNotFoundError, ValueError) as exc:
            return RadiomicsCaseResult(case_name=case_name, status='skipped', error=str(exc)), []
        except Exception as exc:
            logger.exception("Patient %s failed while loading image.", case_name)
            return RadiomicsCaseResult(case_name=case_name, status='failed', error=str(exc)), []

        try:
            structure_names, rtstruct_path = self._resolve_structures(case_dir)
        except Exception as exc:
            logger.exception("Patient %s failed while resolving structures.", case_name)
            return RadiomicsCaseResult(case_name=case_name, status='failed', error=str(exc)), []

        for structure_name in structure_names:
            try:
                mask = self._load_mask(case_dir, structure_name, image, rtstruct_path)
                if mask is None or mask.array is None or not np.any(mask.array):
                    result.skipped_structures.append(structure_name)
                    continue

                logger.info("Processing patient: %s with ROI: %s.", case_name, structure_name)
                features = self._extract_structure_features(image, filtered_image, mask)
            except (DataStructureError, ValueError) as exc:
                logger.warning("Patient %s with mask %s skipped: %s", case_name, structure_name, exc)
                result.skipped_structures.append(structure_name)
                continue
            except Exception:
                logger.exception("Patient %s failed for mask %s.", case_name, structure_name)
                result.skipped_structures.append(structure_name)
                continue

            result.feature_count += len(features)
            features['pat_id'] = case_name
            features['mask_id'] = structure_name
            feature_rows.append(features)
            result.processed_structures.append(structure_name)

        if result.processed_structures:
            return result, feature_rows

        result.status = 'skipped'
        if result.skipped_structures:
            result.error = "No structures were successfully processed for radiomics extraction."
        else:
            result.error = "No structures were available for radiomics extraction."
        return result, feature_rows

    def _load_images(self, case_dir: Path) -> tuple[Image, Image | None]:
        image = self._load_image(case_dir, self.nifti_image_name)
        filtered_image = None
        if self.input_data_type == 'nifti' and self.nifti_filtered_image_name:
            filtered_image = self._load_image(case_dir, self.nifti_filtered_image_name)
        return image, filtered_image

    def _load_image(self, case_dir: Path, nifti_name: str | None = None) -> Image:
        if self.input_data_type == 'dicom':
            return Image.from_dicom(case_dir, modality=self.modality)

        image_path = find_nifti_file(case_dir, nifti_name)
        if image_path is None:
            raise FileNotFoundError(case_dir / str(nifti_name))
        return Image.from_nifti(image_path)

    def _resolve_structures(self, case_dir: Path) -> tuple[list[str], str | None]:
        if self.input_data_type == 'nifti':
            return list(self.structures or []), None

        rtstructs = get_dicom_files(case_dir, modality='RTSTRUCT')
        rtstruct_path = rtstructs[0]['file_path'] if rtstructs else None
        if self.use_all_structures:
            if not rtstruct_path:
                return [], None
            return get_all_structure_names(rtstruct_path), rtstruct_path
        return list(self.structures or []), rtstruct_path

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

    def _extract_structure_features(self, image: Image, filtered_image: Image | None, mask: Image) -> dict:
        roi_data = IntensityMaskBuilder().apply(
            RoiData(
                image=image,
                filtered_image=filtered_image,
                morphological_mask=mask,
            )
        )
        roi_data = Resegmenter(
            intensity_range=self.intensity_range,
            outlier_range=self.outlier_range,
        ).apply(roi_data)
        roi_data = TextureDiscretizer(
            number_of_bins=self.number_of_bins,
            bin_size=self.bin_size,
        ).apply(roi_data)
        return Radiomics(
            aggr_dim=self.aggregation_dimension,
            aggr_method=self.aggregation_method,
            slice_weighting=self.slice_weighting,
            slice_median=self.slice_median,
        ).extract_features(
            roi_data=roi_data,
            include_metadata=True,
        )

    def _validate_discretization(self) -> None:
        if self.discretization_method == 'Number of Bins':
            self.number_of_bins = _require_positive_int(self.number_of_bins, "number_of_bins is required.")
            self.bin_size = None
        elif self.discretization_method == 'Bin Size':
            self.bin_size = _require_positive_float(self.bin_size, "bin_size is required.")
            if self.intensity_range is None:
                raise InvalidInputParametersError("Bin Size discretization requires intensity_range.")
            self.number_of_bins = None
        else:
            raise InvalidInputParametersError("discretization_method must be 'Number of Bins' or 'Bin Size'.")


def _write_radiomics_csv(file_path: Path, features: list[dict]) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    if not features:
        file_path.touch()
        return

    fieldnames = []
    for key in ("pat_id", "mask_id", "bounding_box_min", "no_voxels", "no_bins"):
        if any(key in row for row in features):
            fieldnames.append(key)

    for row in features:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with open(file_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(features)


def _require_positive_int(value, message: str) -> int:
    if value is None or str(value).strip() == '':
        raise InvalidInputParametersError(message)
    try:
        result = int(value)
    except (TypeError, ValueError):
        raise InvalidInputParametersError(message)
    if result <= 0:
        raise InvalidInputParametersError(message)
    return result


def _require_positive_float(value, message: str) -> float:
    result = _normalize_positive_float(value, message)
    if result is None:
        raise InvalidInputParametersError(message)
    return result


def _normalize_positive_float(value, message: str) -> float | None:
    if value is None or str(value).strip() == '':
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise InvalidInputParametersError(message)
    if not np.isfinite(result) or result <= 0:
        raise InvalidInputParametersError(message)
    return result


def _normalize_intensity_range(values: Sequence[float] | None) -> tuple[float, float] | None:
    if values is None:
        return None
    if isinstance(values, str):
        values = [value.strip() for value in values.split(',')]
    if len(values) != 2:
        raise InvalidInputParametersError("intensity_range must contain exactly two values.")
    try:
        lower, upper = (float(value) for value in values)
    except (TypeError, ValueError):
        raise InvalidInputParametersError("intensity_range must contain numeric values.")
    if not np.isfinite(lower) or np.isnan(upper) or lower > upper:
        raise InvalidInputParametersError("intensity_range must have a finite lower bound and lower <= upper.")
    return lower, upper


def _as_bool(value) -> bool:
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ['true', 'yes', '1', 'enable', 'enabled']:
            return True
        if text in ['false', 'no', '0', 'disable', 'disabled', '']:
            return False
    return bool(value)
