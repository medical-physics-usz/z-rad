import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from joblib import Parallel, delayed

from ..exceptions import DataStructureError, InvalidInputParametersError
from ..filtering import create_filter
from ..image import Image
from ._utils import (
    find_nifti_file,
    joblib_progress,
    normalize_common_batch_options,
    normalize_optional_text,
    require_text,
    resolve_patient_folders,
)
from .results import BatchResult

logger = logging.getLogger(__name__)


@dataclass
class FilteringCaseResult:
    """Per-case result returned by ``BatchFilter``.

    Attributes
    ----------
    case_name : str
        Name of the case folder.
    status : {"processed", "skipped", "failed"}
        Filtering status for the case.
    output_path : pathlib.Path or None, optional
        Path to the written filtered image when filtering succeeds.
    error : str or None, optional
        Case-level error message when image loading or filtering fails.
    """

    case_name: str
    status: str
    output_path: Path | None = None
    error: str | None = None


@dataclass
class BatchFilter:
    """Run one image filter over many case folders and write NIfTI outputs.

    ``BatchFilter`` is the batch counterpart to the lower-level filtering
    classes. It discovers case folders, loads one image per case, applies the
    configured filter, and writes one filtered NIfTI image per processed case.
    The API is save-to-disk only.

    Parameters
    ----------
    input_directory : str or pathlib.Path
        Directory containing one subfolder per case.
    output_directory : str or pathlib.Path
        Directory where filtered case folders are written.
    input_data_type : {"dicom", "nifti"}
        Input format. Values are normalized to lower-case during validation.
    modality : {"CT", "MRI", "PET", "MG", "RTDOSE"}
        Image modality used by the image reader.
    filter_type : {"Mean", "Laplacian of Gaussian", "Laws Kernels", "Gabor", "Wavelets"}
        Filter family to apply.
    filter_dimension : {"2D", "3D"}
        Apply the filter slice-wise in 2D or volumetrically in 3D.
    padding_type : str
        Boundary handling mode passed to the selected filter.
    number_of_threads : int, optional
        Number of cases to process in parallel. The default is ``1``.
    patient_folders : sequence of str or str, optional
        Explicit case folders to process. Comma-separated strings are accepted.
    start_folder, stop_folder : str or int, optional
        Inclusive numeric folder range. Both values must be provided together.
    nifti_image_name : str, optional
        Image file name or stem used for NIfTI input.
    filter-specific settings : optional
        Numeric and text settings required by the selected ``filter_type``:
        ``mean_support``, ``log_sigma``, ``log_cutoff``,
        ``laws_response_map``, ``laws_pooling``, ``laws_distance``,
        ``wavelet_response_map``, ``wavelet_type``,
        ``wavelet_decomposition_level``, ``gabor_res_mm``,
        ``gabor_sigma_mm``, ``gabor_lambda_mm``, ``gabor_gamma``, and
        ``gabor_theta``.
    filter-specific enable settings : bool or str, optional
        Enable/disable settings for Laws, Wavelets, and Gabor filters.
        GUI-style ``"Enable"`` and ``"Disable"`` values are accepted.
    parallel_backend : {"processes", "threads"}, optional
        Joblib backend preference used when ``number_of_threads`` is greater
        than one. The default is ``"processes"``.

    Notes
    -----
    ``validate()`` normalizes public attributes in place. After validation,
    directories are ``Path`` objects, ``input_data_type`` is lower-case,
    modality is upper-case, and numeric filter settings are converted to
    numeric Python values.
    """

    input_directory: str | Path
    output_directory: str | Path
    input_data_type: str
    modality: str
    filter_type: str
    filter_dimension: str
    padding_type: str
    number_of_threads: int = 1
    patient_folders: Sequence[str] | None = None
    start_folder: str | int | None = None
    stop_folder: str | int | None = None
    nifti_image_name: str | None = None
    mean_support: int | str | None = None
    log_sigma: float | str | None = None
    log_cutoff: float | str | None = None
    laws_response_map: str | None = None
    laws_rotation_invariance: bool | str = False
    laws_pooling: str | None = None
    laws_energy_map: bool | str = False
    laws_distance: int | str | None = None
    wavelet_response_map: str | None = None
    wavelet_type: str | None = None
    wavelet_decomposition_level: int | str | None = None
    wavelet_rotation_invariance: bool | str = False
    gabor_res_mm: float | str | None = None
    gabor_sigma_mm: float | str | None = None
    gabor_lambda_mm: float | str | None = None
    gabor_gamma: float | str | None = None
    gabor_theta: float | str | None = None
    gabor_rotation_invariance: bool | str = False
    gabor_orthogonal_planes: bool | str = False
    parallel_backend: str = 'processes'

    def validate(self) -> None:
        """Validate and normalize filtering configuration.

        Raises
        ------
        InvalidInputParametersError
            If the input directory, data type, modality, folder selection,
            threading, backend, filter type, or filter-specific settings are
            invalid.
        """
        normalize_common_batch_options(self)
        self.filter_type = require_text(self.filter_type, "filter_type is required.")
        self.filter_dimension = require_text(self.filter_dimension, "filter_dimension is required.").upper()
        self.padding_type = require_text(self.padding_type, "padding_type is required.")

        if self.filter_dimension not in ['2D', '3D']:
            raise InvalidInputParametersError("filter_dimension must be '2D' or '3D'.")

        if self.input_data_type == 'nifti':
            self.nifti_image_name = normalize_optional_text(self.nifti_image_name)
            if not self.nifti_image_name:
                raise InvalidInputParametersError("nifti_image_name is required for NIfTI filtering.")

        self._validate_filter_parameters()

    def plan(self) -> list[str]:
        """Return the case folders selected for filtering.

        Returns
        -------
        folders : list of str
            Deterministically ordered case folder names selected by
            ``patient_folders`` or the numeric ``start_folder`` /
            ``stop_folder`` range. If neither option is set, all non-hidden
            subfolders are returned.

        Raises
        ------
        InvalidInputParametersError
            If validation fails before folder selection.
        """
        self.validate()
        return self._resolve_patient_folders()

    def run(self, progress_callback: Callable[[int], None] | None = None) -> BatchResult:
        """Run filtering and write filtered NIfTI images.

        Parameters
        ----------
        progress_callback : callable, optional
            Function called as ``progress_callback(step_count)`` after cases
            complete. ``step_count`` may be greater than one during parallel
            execution.

        Returns
        -------
        result : BatchResult
            Aggregate result with one ``FilteringCaseResult`` per selected
            case.

        Notes
        -----
        Case-level failures are recorded in the returned result and do not stop
        the batch.
        """
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

        return BatchResult(workflow='filtering', case_results=case_results)

    def get_output_filename(self) -> str:
        """Return the GUI-compatible output file name for the configured filter.

        Returns
        -------
        filename : str
            File name ending in ``.nii.gz``.

        Raises
        ------
        InvalidInputParametersError
            If ``filter_type`` is not supported.
        """
        values = self._filename_values()
        filter_formats = {
            'Mean': "{filter_type}_{filter_dimension}_{filter_mean_support}support_{filter_padding_type}",
            'Laplacian of Gaussian': "{filter_type}_{filter_dimension}_{filter_log_sigma}sigma_"
            "{filter_log_cutoff}cutoff_{filter_padding_type}",
            'Laws Kernels': "{filter_type}_{filter_dimension}_{filter_laws_response_map}_"
            "{filter_laws_rot_inv}_{filter_laws_pooling}_"
            "{filter_laws_energy_map}_{filter_laws_distance}_{filter_padding_type}",
            'Gabor': "Gabor_{filter_dimension}_"
            "{filter_gabor_res_mm}resmm_"
            "{filter_gabor_sigma_mm}sigmm_"
            "{filter_gabor_lambda_mm}lambmm_"
            "g{filter_gabor_gamma}_"
            "t{filter_gabor_theta}_"
            "{filter_gabor_rotinv}_"
            "{filter_gabor_ortho}_"
            "{filter_padding_type}",
        }

        if self.filter_type == 'Wavelets':
            filename = (
                "{filter_wavelet_type}_{filter_dimension}_"
                "{filter_wavelet_resp_map}_"
                "{filter_wavelet_decomp_lvl}_"
                "{filter_wavelet_rot_inv}_"
                "{filter_padding_type}"
            ).format(**values)
        elif self.filter_type in filter_formats:
            filename = filter_formats[self.filter_type].format(**values)
        else:
            raise InvalidInputParametersError(f"Unknown filter type: {self.filter_type}")
        return f"{filename}.nii.gz"

    def _resolve_patient_folders(self) -> list[str]:
        return resolve_patient_folders(
            self.input_directory,
            self.patient_folders,
            self.start_folder,
            self.stop_folder,
        )

    def _process_case(self, case_name: str) -> FilteringCaseResult:
        case_dir = self.input_directory / case_name
        logger.info("Filtering patient's %s image.", case_name)

        try:
            image = self._load_image(case_dir)
        except (DataStructureError, FileNotFoundError, ValueError) as exc:
            return FilteringCaseResult(case_name=case_name, status='skipped', error=str(exc))
        except Exception as exc:
            logger.exception("Patient %s failed while loading image.", case_name)
            return FilteringCaseResult(case_name=case_name, status='failed', error=str(exc))

        try:
            filtering = self._create_filter()
            image_new = filtering.apply(image)
            output_path = self.output_directory / case_name / self.get_output_filename()
            image_new.save_as_nifti(output_path)
            return FilteringCaseResult(case_name=case_name, status='processed', output_path=output_path)
        except Exception as exc:
            logger.exception("Patient %s failed during filtering.", case_name)
            return FilteringCaseResult(case_name=case_name, status='failed', error=str(exc))

    def _load_image(self, case_dir: Path) -> Image:
        if self.input_data_type == 'dicom':
            return Image.from_dicom(case_dir, modality=self.modality)

        image_path = find_nifti_file(case_dir, self.nifti_image_name)
        if image_path is None:
            raise FileNotFoundError(case_dir / self.nifti_image_name)
        return Image.from_nifti(image_path)

    def _create_filter(self):
        if self.filter_type == 'Mean':
            return create_filter(
                filtering_method='Mean',
                padding_type=self.padding_type,
                support=self.mean_support,
                dimensionality=self.filter_dimension,
            )
        if self.filter_type == 'Laplacian of Gaussian':
            return create_filter(
                filtering_method='Laplacian of Gaussian',
                padding_type=self.padding_type,
                sigma_mm=self.log_sigma,
                cutoff=self.log_cutoff,
                dimensionality=self.filter_dimension,
            )
        if self.filter_type == 'Laws Kernels':
            return create_filter(
                filtering_method='Laws Kernels',
                response_map=self.laws_response_map,
                padding_type=self.padding_type,
                dimensionality=self.filter_dimension,
                rotation_invariance=_as_bool(self.laws_rotation_invariance),
                pooling=self.laws_pooling,
                energy_map=_as_bool(self.laws_energy_map),
                distance=self.laws_distance,
            )
        if self.filter_type == 'Gabor':
            return create_filter(
                filtering_method='Gabor',
                padding_type=self.padding_type,
                res_mm=self.gabor_res_mm,
                sigma_mm=self.gabor_sigma_mm,
                lambda_mm=self.gabor_lambda_mm,
                gamma=self.gabor_gamma,
                theta=self.gabor_theta,
                rotation_invariance=_as_bool(self.gabor_rotation_invariance),
                orthogonal_planes=_as_bool(self.gabor_orthogonal_planes),
            )
        if self.filter_type == 'Wavelets':
            return create_filter(
                filtering_method='Wavelets',
                dimensionality=self.filter_dimension,
                padding_type=self.padding_type,
                wavelet_type=self.wavelet_type,
                response_map=self.wavelet_response_map,
                decomposition_level=self.wavelet_decomposition_level,
                rotation_invariance=_as_bool(self.wavelet_rotation_invariance),
            )
        raise InvalidInputParametersError(f"Filter_type {self.filter_type} not supported.")

    def _validate_filter_parameters(self) -> None:
        if self.filter_type == 'Mean':
            self.mean_support = _require_positive_int(self.mean_support, "mean_support is required.")
        elif self.filter_type == 'Laplacian of Gaussian':
            self.log_sigma = _require_float(self.log_sigma, "log_sigma is required.")
            self.log_cutoff = _require_float(self.log_cutoff, "log_cutoff is required.")
        elif self.filter_type == 'Laws Kernels':
            self.laws_response_map = require_text(self.laws_response_map, "laws_response_map is required.")
            self.laws_pooling = require_text(self.laws_pooling, "laws_pooling is required.")
            self.laws_distance = _require_positive_int(self.laws_distance, "laws_distance is required.")
            self.laws_rotation_invariance = _normalize_enable_disable(self.laws_rotation_invariance)
            self.laws_energy_map = _normalize_enable_disable(self.laws_energy_map)
        elif self.filter_type == 'Gabor':
            self.gabor_res_mm = _require_float(self.gabor_res_mm, "gabor_res_mm is required.")
            self.gabor_sigma_mm = _require_float(self.gabor_sigma_mm, "gabor_sigma_mm is required.")
            self.gabor_lambda_mm = _require_float(self.gabor_lambda_mm, "gabor_lambda_mm is required.")
            self.gabor_gamma = _require_float(self.gabor_gamma, "gabor_gamma is required.")
            self.gabor_theta = _require_float(self.gabor_theta, "gabor_theta is required.")
            self.gabor_rotation_invariance = _normalize_enable_disable(self.gabor_rotation_invariance)
            self.gabor_orthogonal_planes = _normalize_enable_disable(self.gabor_orthogonal_planes)
        elif self.filter_type == 'Wavelets':
            self.wavelet_type = require_text(self.wavelet_type, "wavelet_type is required.")
            self.wavelet_response_map = require_text(self.wavelet_response_map, "wavelet_response_map is required.")
            self.wavelet_decomposition_level = _require_positive_int(
                self.wavelet_decomposition_level,
                "wavelet_decomposition_level is required.",
            )
            self.wavelet_rotation_invariance = _normalize_enable_disable(self.wavelet_rotation_invariance)
        else:
            raise InvalidInputParametersError(f"Filter_type {self.filter_type} not supported.")

    def _filename_values(self) -> dict:
        return {
            'filter_type': self.filter_type,
            'filter_dimension': self.filter_dimension,
            'filter_padding_type': self.padding_type,
            'filter_mean_support': self.mean_support,
            'filter_log_sigma': self.log_sigma,
            'filter_log_cutoff': self.log_cutoff,
            'filter_laws_response_map': self.laws_response_map,
            'filter_laws_rot_inv': _enable_disable_text(self.laws_rotation_invariance),
            'filter_laws_pooling': self.laws_pooling,
            'filter_laws_energy_map': _enable_disable_text(self.laws_energy_map),
            'filter_laws_distance': self.laws_distance,
            'filter_wavelet_type': self.wavelet_type,
            'filter_wavelet_resp_map': self.wavelet_response_map,
            'filter_wavelet_decomp_lvl': self.wavelet_decomposition_level,
            'filter_wavelet_rot_inv': _enable_disable_text(self.wavelet_rotation_invariance),
            'filter_gabor_res_mm': self.gabor_res_mm,
            'filter_gabor_sigma_mm': self.gabor_sigma_mm,
            'filter_gabor_lambda_mm': self.gabor_lambda_mm,
            'filter_gabor_gamma': self.gabor_gamma,
            'filter_gabor_theta': self.gabor_theta,
            'filter_gabor_rotinv': _enable_disable_text(self.gabor_rotation_invariance),
            'filter_gabor_ortho': _enable_disable_text(self.gabor_orthogonal_planes),
        }


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


def _require_float(value, message: str) -> float:
    if value is None or str(value).strip() == '':
        raise InvalidInputParametersError(message)
    try:
        return float(value)
    except (TypeError, ValueError):
        raise InvalidInputParametersError(message)


def _normalize_enable_disable(value) -> str:
    if isinstance(value, str):
        text = value.strip()
        if text in ['Enable', 'Disable']:
            return text
        if text.lower() in ['true', 'yes', '1']:
            return 'Enable'
        if text.lower() in ['false', 'no', '0']:
            return 'Disable'
    return 'Enable' if bool(value) else 'Disable'


def _as_bool(value) -> bool:
    return _normalize_enable_disable(value) == 'Enable'


def _enable_disable_text(value) -> str:
    return _normalize_enable_disable(value)
