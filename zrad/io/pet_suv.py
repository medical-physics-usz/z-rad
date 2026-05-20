import re
import warnings
from datetime import datetime

import numpy as np
import pydicom
import SimpleITK as sitk

from ..exceptions import DataStructureError, DataStructureWarning


def is_fdg(name):
    fdg_pattern = re.compile(
        r"(fdg|fluorodeoxy|fludeoxy|2[-\s]?\[?18f\]?[-\s]?fluoro)",
        re.IGNORECASE,
    )
    return bool(fdg_pattern.search(name))


def parse_time(time_str):
    """Parse a DICOM time string into a datetime object."""
    if isinstance(time_str, bytes):
        time_str = time_str.decode("utf-8").strip()

    for fmt in ("%H%M%S.%f", "%H%M%S", "%Y%m%d%H%M%S.%f", "%Y%m%d%H%M%S"):
        try:
            return datetime.strptime(time_str, fmt)
        except ValueError:
            continue
        except TypeError:
            continue
    raise ValueError(f"Time data '{time_str}' does not match expected formats")


def calc_elapsed_time(ds, decay_constant, acquisition_time, injection_time):
    frame_reference_time = float(ds.FrameReferenceTime) / 1000
    decay_during_frame = decay_constant * ds.ActualFrameDuration / 1000
    avg_count_rate_time = (1 / decay_constant) * np.log(decay_during_frame / (1 - np.exp(-decay_during_frame)))

    return (acquisition_time - injection_time).total_seconds() + avg_count_rate_time - frame_reference_time


def get_patient_height_cm(ds):
    """Return patient height in cm."""
    if hasattr(ds, "PatientSize") and ds.PatientSize not in [None, ""]:
        height = float(ds.PatientSize)
    elif (0x0010, 0x1020) in ds and ds[(0x0010, 0x1020)].value not in [None, ""]:
        height = float(ds[(0x0010, 0x1020)].value)
    else:
        raise DataStructureError("Patient height tag (0010,1020) is missing.")

    if height <= 0:
        raise DataStructureError("Patient height must be > 0.")

    return height * 100.0 if height <= 3 else height


def calculate_bsa_du_bois(height_cm, weight_kg):
    """Calculate Du Bois body surface area in m^2."""
    if height_cm <= 0 or weight_kg <= 0:
        raise DataStructureError("Height and weight must be > 0 to compute BSA.")
    return 0.007184 * (height_cm**0.725) * (weight_kg**0.425)


def get_patient_sex(ds):
    """Return normalized patient sex: 'M', 'F', or 'O'."""
    sex = getattr(ds, "PatientSex", None)
    if sex is None and (0x0010, 0x0040) in ds:
        sex = ds[(0x0010, 0x0040)].value

    if sex is None:
        raise DataStructureError("Patient sex tag (0010,0040) is missing.")

    sex = str(sex).strip().upper()
    if sex == "":
        raise DataStructureError("Patient sex tag (0010,0040) is empty.")
    if sex not in {"M", "F", "O"}:
        raise DataStructureError(f"Unsupported PatientSex '{sex}'. Expected one of 'M', 'F', or 'O'.")
    return sex


def calculate_lbm_morgan(height_cm, weight_kg, sex):
    """Calculate lean body mass using the Morgan/Sugawara-style formula."""
    if height_cm <= 0 or weight_kg <= 0:
        raise DataStructureError("Height and weight must be > 0 to compute LBM.")

    male_lbm = 1.10 * weight_kg - 120.0 * ((weight_kg / height_cm) ** 2)
    female_lbm = 1.07 * weight_kg - 148.0 * ((weight_kg / height_cm) ** 2)

    if sex == "M":
        lbm = male_lbm
    elif sex == "F":
        lbm = female_lbm
    elif sex == "O":
        lbm = 0.5 * (male_lbm + female_lbm)
    else:
        raise DataStructureError(f"Unsupported sex '{sex}' for LBM calculation.")

    if lbm <= 0:
        raise DataStructureError(f"Computed Morgan LBM is non-positive: {lbm}.")
    return lbm


def calculate_lbm_james128(height_cm, weight_kg, sex):
    """Calculate lean body mass using the James/Morgan-128 formula."""
    if height_cm <= 0 or weight_kg <= 0:
        raise DataStructureError("Height and weight must be > 0 to compute LBM.")

    male_lbm = 1.10 * weight_kg - 128.0 * ((weight_kg / height_cm) ** 2)
    female_lbm = 1.07 * weight_kg - 148.0 * ((weight_kg / height_cm) ** 2)

    if sex == "M":
        lbm = male_lbm
    elif sex == "F":
        lbm = female_lbm
    elif sex == "O":
        lbm = 0.5 * (male_lbm + female_lbm)
    else:
        raise DataStructureError(f"Unsupported sex '{sex}' for LBM calculation.")

    if lbm <= 0:
        raise DataStructureError(f"Computed James128 LBM is non-positive: {lbm}.")
    return lbm


def calculate_lbm_janmahasatian(height_cm, weight_kg, sex):
    """Calculate lean body mass using the Janmahasatian formula."""
    if height_cm <= 0 or weight_kg <= 0:
        raise DataStructureError("Height and weight must be > 0 to compute LBM.")

    height_m = height_cm * 1e-2
    bmi = weight_kg / (height_m**2)

    male_lbm = 9270.0 * weight_kg / (6680.0 + 216.0 * bmi)
    female_lbm = 9270.0 * weight_kg / (8780.0 + 244.0 * bmi)

    if sex == "M":
        lbm = male_lbm
    elif sex == "F":
        lbm = female_lbm
    elif sex == "O":
        lbm = 0.5 * (male_lbm + female_lbm)
    else:
        raise DataStructureError(f"Unsupported sex '{sex}' for LBM calculation.")

    if lbm <= 0:
        raise DataStructureError(f"Computed Janmahasatian LBM is non-positive: {lbm}.")
    return lbm


def calculate_ibw(height_cm, sex):
    """Calculate ideal body weight."""
    if height_cm <= 0:
        raise DataStructureError("Height must be > 0 to compute IBW.")

    male_ibw = 48.0 + 1.06 * (height_cm - 152.0)
    female_ibw = 45.5 + 0.91 * (height_cm - 152.0)

    if sex == "M":
        ibw = male_ibw
    elif sex == "F":
        ibw = female_ibw
    elif sex == "O":
        ibw = 0.5 * (male_ibw + female_ibw)
    else:
        raise DataStructureError(f"Unsupported sex '{sex}' for IBW calculation.")

    if ibw <= 0:
        raise DataStructureError(f"Computed IBW is non-positive: {ibw}.")
    return ibw


def get_gml_normalization_info(ds):
    """Parse GML SUV normalization metadata and compute the normalization factor."""
    suv_type_elem = ds.get((0x0054, 0x1006), None)
    suv_type = (
        "BW" if suv_type_elem is None or suv_type_elem.value in [None, ""] else str(suv_type_elem.value).strip().upper()
    )

    try:
        patient_weight = float(ds.PatientWeight)
    except Exception:
        raise DataStructureError("Patient weight tag is missing or invalid for GML normalization.")

    if patient_weight <= 0:
        raise DataStructureError("Patient weight must be > 0 for GML normalization.")

    if suv_type == "BW":
        return suv_type, patient_weight

    height_cm = get_patient_height_cm(ds)
    sex = get_patient_sex(ds)

    if suv_type == "LBM":
        factor = calculate_lbm_morgan(height_cm, patient_weight, sex)
    elif suv_type == "LBMJAMES128":
        factor = calculate_lbm_james128(height_cm, patient_weight, sex)
    elif suv_type == "LBMJANMA":
        factor = calculate_lbm_janmahasatian(height_cm, patient_weight, sex)
    elif suv_type == "IBW":
        factor = calculate_ibw(height_cm, sex)
    else:
        raise DataStructureError(
            f"GML with SUV Type '{suv_type}' is not supported. "
            f"Supported types are BW, LBM, LBMJAMES128, LBMJANMA, and IBW."
        )

    return suv_type, factor


def validate_pet_dicom_tags(dicom_files):
    for dcm_file in dicom_files:
        ds = dcm_file["ds"]
        image_id = dcm_file["file_path"]

        try:
            pat_weight = ds[(0x0010, 0x1030)].value
            if float(pat_weight) < 1:
                warning_msg = f"For patient's {image_id} image, patient's weight tag (0071, 1022) contains weight < 1kg. Patient is excluded from the analysis."
                warnings.warn(warning_msg, DataStructureWarning)

        except (KeyError, TypeError):
            warning_msg = f"For patient's {image_id} image, patient's weight tag (0071, 1022) is not present. Patient is excluded from the analysis."
            warnings.warn(warning_msg, DataStructureWarning)
        if pat_weight <= 0:
            raise DataStructureError("Patient weight must be > 0.")
        if "DECY" not in ds[(0x0028, 0x0051)].value or "ATTN" not in ds[(0x0028, 0x0051)].value:
            warning_msg = f"For patient's {image_id} image, in DICOM tag (0028, 0051) either no 'DECY' (decay correction) or 'ATTN' (attenuation correction). Patient is excluded from the analysis."
            warnings.warn(warning_msg, DataStructureWarning)

        if ds.Units == "BQML":
            try:
                injection_time = parse_time(ds.RadiopharmaceuticalInformationSequence[0].RadiopharmaceuticalStartTime)
            except AttributeError:
                injection_time = parse_time(
                    ds.RadiopharmaceuticalInformationSequence[0].RadiopharmaceuticalStartDateTime
                )
            half_life = float(ds.RadiopharmaceuticalInformationSequence[0].RadionuclideHalfLife)
            decay_constant = np.log(2) / half_life

            if ds.DecayCorrection == "START":
                if "PHILIPS" in ds.Manufacturer.upper():
                    acquisition_time = parse_time(ds.AcquisitionTime).replace(
                        year=injection_time.year,
                        month=injection_time.month,
                        day=injection_time.day,
                    )

                    elapsed_time = calc_elapsed_time(ds, decay_constant, acquisition_time, injection_time)
                elif (
                    "SIEMENS" in ds.Manufacturer.upper()
                    or "CPS" in ds.Manufacturer.upper()
                    or "CTI" in ds.Manufacturer.upper()
                ):
                    try:
                        elapsed_time = (
                            parse_time(ds[(0x0071, 0x1022)].value).replace(
                                year=injection_time.year,
                                month=injection_time.month,
                                day=injection_time.day,
                            )
                            - injection_time
                        ).total_seconds()

                    except (KeyError, TypeError):
                        acquisition_time = parse_time(ds.AcquisitionTime).replace(
                            year=injection_time.year,
                            month=injection_time.month,
                            day=injection_time.day,
                        )

                        elapsed_time = calc_elapsed_time(ds, decay_constant, acquisition_time, injection_time)
                elif "GE" in ds.Manufacturer.upper():
                    try:
                        elapsed_time = (
                            parse_time(ds[(0x0009, 0x100D)].value).replace(
                                year=injection_time.year,
                                month=injection_time.month,
                                day=injection_time.day,
                            )
                            - injection_time
                        ).total_seconds()
                    except (KeyError, TypeError):
                        acquisition_time = parse_time(ds.AcquisitionTime).replace(
                            year=injection_time.year,
                            month=injection_time.month,
                            day=injection_time.day,
                        )
                        frame_reference_time = float(ds.FrameReferenceTime) / 1000

                        elapsed_time = (acquisition_time - injection_time).total_seconds() - frame_reference_time
                else:
                    acquisition_time = parse_time(ds.AcquisitionTime).replace(
                        year=injection_time.year,
                        month=injection_time.month,
                        day=injection_time.day,
                    )

                    elapsed_time = calc_elapsed_time(ds, decay_constant, acquisition_time, injection_time)

                    warning_msg = f"For patient's {image_id} image, an unknown PET scaner manufacturer is present {ds.Manufacturer}. Siemens/Philips strategy is applied."
                    warnings.warn(warning_msg, DataStructureWarning)

            elif ds.DecayCorrection == "ADMIN":
                elapsed_time = 0
            elif ds.DecayCorrection == "NONE":
                elapsed_time = None
            else:
                warning_msg = f"For patient's {image_id} image, An unsupported Decay Correction {ds.DecayCorrection} is present. Only supported are 'NONE', 'START' and 'ADMIN'. Patient is excluded from the analysis."
                raise DataStructureError(warning_msg)
            if elapsed_time is not None and elapsed_time < 0:
                error_msg = f"For patient's {image_id} image, patient is excluded from the analysis due to the negative time difference in the decay factor."
                raise DataStructureError(error_msg)
            elif (
                elapsed_time is not None
                and elapsed_time > 0
                and abs(elapsed_time) < 1800
                and ds.DecayCorrection != "ADMIN"
            ):
                warning_msg = f"Only {abs(elapsed_time) / 60} minutes after the injection."
                warnings.warn(warning_msg, DataStructureWarning)
        elif ds.Units == "CNTS" and "PHILIPS" in ds.Manufacturer.upper():
            if not (
                ((0x7053, 0x1009) in ds and ds[(0x7053, 0x1009)].value != 0)
                or ((0x7053, 0x1000) in ds and ds[(0x7053, 0x1000)].value != 0)
            ):
                error_msg = f"For patient's {image_id} image, patient is excluded, Philips scale factors not present (PET units CNTS)"
                raise DataStructureError(error_msg)
        elif ds.Units == "GML":
            try:
                _suv_type, _factor = get_gml_normalization_info(ds)
            except Exception as e:
                error_msg = f"For patient's {image_id} image, patient is excluded, GML normalization is invalid: {e}"
                raise DataStructureError(error_msg)

        elif ds.Units == "CM2ML":
            suv_type = ds.get((0x0054, 0x1006), None)
            if suv_type is not None and suv_type.value != "BSA":
                error_msg = f"For patient's {image_id} image, patient is excluded, SUV Type is not BSA (CM2ML units)"
                raise DataStructureError(error_msg)

            try:
                patient_weight = float(ds.PatientWeight)
                height_cm = get_patient_height_cm(ds)
                _ = calculate_bsa_du_bois(height_cm, patient_weight)
            except Exception as e:
                error_msg = (
                    f"For patient's {image_id} image, CM2ML requires valid patient "
                    f"height and weight for Du Bois BSA calculation: {e}"
                )
                raise DataStructureError(error_msg)

        else:
            error_msg = f"For patient's {image_id} image, patient is excluded, only supported PET Units are BQML for Philips, Siemens and GE or CNTS for Philips"
            raise DataStructureError(error_msg)


def apply_suv_correction(dicom_files, suv_image):
    def process_single_slice(dicom_file_path):
        ds = pydicom.dcmread(dicom_file_path)

        def get_injection_time(ds):
            rph = ds.RadiopharmaceuticalInformationSequence[0]
            try:
                return parse_time(rph.RadiopharmaceuticalStartTime)
            except AttributeError:
                return parse_time(rph.RadiopharmaceuticalStartDateTime)

        def get_tracer_name(rph_item):
            value = getattr(rph_item, "Radiopharmaceutical", None)
            if value is None and (0x0018, 0x0031) in rph_item:
                value = rph_item[(0x0018, 0x0031)].value
            return str(value) if value is not None else None

        def get_datetime_on_injection_day(time_value, injection_time):
            return parse_time(time_value).replace(
                year=injection_time.year,
                month=injection_time.month,
                day=injection_time.day,
            )

        def compute_elapsed_time_for_start_decay_correction(ds, injection_time, decay_constant):
            manufacturer = ds.Manufacturer.upper()

            acquisition_time = get_datetime_on_injection_day(ds.AcquisitionTime, injection_time)
            series_time = get_datetime_on_injection_day(ds.SeriesTime, injection_time)

            if "PHILIPS" in manufacturer:
                if acquisition_time == series_time:
                    return (acquisition_time - injection_time).total_seconds()
                return calc_elapsed_time(ds, decay_constant, acquisition_time, injection_time)

            if "SIEMENS" in manufacturer or "CPS" in manufacturer or "CTI" in manufacturer:
                try:
                    private_time = get_datetime_on_injection_day(ds[(0x0071, 0x1022)].value, injection_time)
                    return (private_time - injection_time).total_seconds()
                except (KeyError, TypeError):
                    if acquisition_time == series_time:
                        return (acquisition_time - injection_time).total_seconds()
                    return calc_elapsed_time(ds, decay_constant, acquisition_time, injection_time)

            if "GE" in manufacturer:
                try:
                    private_time = get_datetime_on_injection_day(ds[(0x0009, 0x100D)].value, injection_time)
                    return (private_time - injection_time).total_seconds()
                except (KeyError, TypeError):
                    if acquisition_time == series_time:
                        return (acquisition_time - injection_time).total_seconds()
                    frame_reference_time = float(ds.FrameReferenceTime) / 1000.0
                    return (acquisition_time - injection_time).total_seconds() - frame_reference_time

            if acquisition_time == series_time:
                return (acquisition_time - injection_time).total_seconds()
            return calc_elapsed_time(ds, decay_constant, acquisition_time, injection_time)

        def process_gml(pixel_array_units, ds):
            suv_type, factor = get_gml_normalization_info(ds)
            patient_weight = float(ds.PatientWeight)

            if suv_type == "BW":
                return pixel_array_units

            return pixel_array_units * (patient_weight / factor)

        def process_cm2ml(pixel_array_units, ds):
            suv_type = ds.get((0x0054, 0x1006), None)
            if suv_type is not None and suv_type.value != "BSA":
                raise DataStructureError(f"CM2ML with {suv_type.value} SUV normalization is not supported!")

            patient_weight = float(ds.PatientWeight)
            height_cm = get_patient_height_cm(ds)
            bsa_m2 = calculate_bsa_du_bois(height_cm, patient_weight)

            return pixel_array_units * (patient_weight / (bsa_m2 * 10.0))

        def process_bqml(activity_concentration, ds):
            rph = ds.RadiopharmaceuticalInformationSequence[0]
            injection_time = get_injection_time(ds)
            patient_weight = float(ds.PatientWeight)
            injected_dose = float(rph.RadionuclideTotalDose)

            tracer_name = get_tracer_name(rph)
            if tracer_name is not None and is_fdg(tracer_name) and injected_dose < 10000:
                injected_dose *= 1000000
                warnings.warn(
                    f"Injected dose is {injected_dose} Bq, it is too low for FDG, assumed to be in MBq",
                    DataStructureWarning,
                )

            if injected_dose <= 0:
                raise DataStructureError("The injected PET tracer dose is zero.")

            half_life = float(rph.RadionuclideHalfLife)
            decay_constant = np.log(2) / half_life
            decay_correction = ds.DecayCorrection

            if decay_correction == "START":
                elapsed_time = compute_elapsed_time_for_start_decay_correction(ds, injection_time, decay_constant)
                decay_factor = np.exp(-(np.log(2) * elapsed_time) / half_life)
                decay_corrected_dose = injected_dose * decay_factor
                return activity_concentration / (decay_corrected_dose / (patient_weight * 1000))

            if decay_correction == "ADMIN":
                return activity_concentration / (injected_dose / (patient_weight * 1000))

            if decay_correction == "NONE":
                acquisition_time = get_datetime_on_injection_day(ds.AcquisitionTime, injection_time)

                decay_during_frame = decay_constant * float(ds.ActualFrameDuration) / 1000.0
                avg_count_rate_time = (1 / decay_constant) * np.log(
                    decay_during_frame / (1 - np.exp(-decay_during_frame))
                )
                decay_corrected_activity_concentration = activity_concentration * np.exp(
                    decay_constant * ((acquisition_time - injection_time).total_seconds() + avg_count_rate_time)
                )

                return decay_corrected_activity_concentration / (injected_dose / (patient_weight * 1000))

            raise DataStructureError(f"Decay correction {decay_correction} is not supported!")

        def process_cnts(pixel_array_units, ds):
            manufacturer = ds.Manufacturer.upper()
            if "PHILIPS" not in manufacturer:
                raise DataStructureError(f"Vendor {ds.Manufacturer} is not supported with CNTS units!")

            if (0x7053, 0x1009) in ds and ds[(0x7053, 0x1009)].value != 0:
                activity_concentration_bqml = pixel_array_units * ds[(0x7053, 0x1009)].value
                return process_bqml(activity_concentration_bqml, ds)

            if ds.DecayCorrection != "NONE" and (0x7053, 0x1000) in ds and ds[(0x7053, 0x1000)].value != 0:
                return pixel_array_units * ds[(0x7053, 0x1000)].value

            raise DataStructureError("Philips-specific scaling factors not present!")

        units = ds.Units
        pixel_array_units = (ds.pixel_array * ds.RescaleSlope) + ds.RescaleIntercept

        if units == "GML":
            suv = process_gml(pixel_array_units, ds)
        elif units == "CM2ML":
            suv = process_cm2ml(pixel_array_units, ds)
        elif units == "BQML":
            suv = process_bqml(pixel_array_units, ds)
        elif units == "CNTS":
            suv = process_cnts(pixel_array_units, ds)
        else:
            raise DataStructureError(f"Units {units} are not supported!")

        return suv.T

    intensity_array = np.zeros(suv_image.GetSize())

    dicom_files = sorted(dicom_files, key=lambda f: float(f["ds"].ImagePositionPatient[2]))
    for z_slice_id, dicom_file in enumerate(dicom_files):
        intensity_array[:, :, z_slice_id] = process_single_slice(dicom_file["file_path"])

    intensity_image = sitk.GetImageFromArray(intensity_array.T)
    intensity_image.SetOrigin(suv_image.GetOrigin())
    intensity_image.SetSpacing(np.array(suv_image.GetSpacing()))
    intensity_image.SetDirection(np.array(suv_image.GetDirection()))
    return intensity_image
