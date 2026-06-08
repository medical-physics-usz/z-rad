import numpy as np
import pydicom
import pytest
import SimpleITK as sitk
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

import zrad.io.dicom as dicom
from zrad.exceptions import DataStructureWarning


def _make_sitk_image(size=(5, 5, 3)):
    width, height, depth = size
    image = sitk.GetImageFromArray(np.zeros((depth, height, width), dtype=np.int16))
    image.SetOrigin((0.0, 0.0, 0.0))
    image.SetSpacing((1.0, 1.0, 1.0))
    image.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    return image


def _contour(x, y, z, contour_type="CLOSED_PLANAR"):
    return {
        "type": contour_type,
        "points": {
            "x": x,
            "y": y,
            "z": z,
        },
    }


def _square_contour(z=1.0, contour_type="CLOSED_PLANAR"):
    return _contour(
        x=[1.0, 3.0, 3.0, 1.0],
        y=[1.0, 1.0, 3.0, 3.0],
        z=[z, z, z, z],
        contour_type=contour_type,
    )


def _write_rtstruct(path, roi_name, contours):
    file_meta = FileMetaDataset()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.MediaStorageSOPClassUID = generate_uid()
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.ImplementationClassUID = generate_uid()

    ds = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.Modality = "RTSTRUCT"
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID

    structure = Dataset()
    structure.ROINumber = 1
    structure.ROIName = roi_name
    ds.StructureSetROISequence = Sequence([structure])

    roi_contour = Dataset()
    roi_contour.ReferencedROINumber = 1
    roi_contour.ContourSequence = Sequence()
    for contour in contours:
        contour_ds = Dataset()
        contour_ds.ContourGeometricType = contour["type"]
        points = contour["points"]
        contour_ds.NumberOfContourPoints = len(points["x"])
        contour_ds.ContourData = [value for point in zip(points["x"], points["y"], points["z"]) for value in point]
        roi_contour.ContourSequence.append(contour_ds)
    ds.ROIContourSequence = Sequence([roi_contour])

    pydicom.dcmwrite(path, ds, write_like_original=False)


@pytest.mark.unit
def test_rtstruct_mask_rasterizes_contour_inside_target_image():
    image = _make_sitk_image()

    mask, skipped, supported = dicom._generate_rtstruct_mask_array([_square_contour()], image)

    assert supported == 1
    assert skipped == 0
    assert mask.shape == (3, 5, 5)
    assert mask[1].any()
    assert not mask[0].any()
    assert not mask[2].any()


@pytest.mark.unit
def test_rtstruct_mask_clips_in_plane_contour_to_target_image():
    image = _make_sitk_image()
    contour = _contour(
        x=[-2.0, 2.0, 2.0, -2.0],
        y=[-2.0, -2.0, 2.0, 2.0],
        z=[1.0, 1.0, 1.0, 1.0],
    )

    mask, skipped, supported = dicom._generate_rtstruct_mask_array([contour], image)

    assert supported == 1
    assert skipped == 0
    assert mask[1].any()


@pytest.mark.unit
def test_rtstruct_mask_skips_contour_outside_target_z_without_index_error():
    image = _make_sitk_image()

    mask, skipped, supported = dicom._generate_rtstruct_mask_array([_square_contour(z=10.0)], image)

    assert supported == 1
    assert skipped == 1
    assert not mask.any()


@pytest.mark.unit
def test_rtstruct_xor_contour_is_applied_once_per_contour():
    image = _make_sitk_image()

    mask, skipped, supported = dicom._generate_rtstruct_mask_array(
        [_square_contour(contour_type="CLOSED_PLANAR_XOR")],
        image,
    )

    assert supported == 1
    assert skipped == 0
    assert mask[1].any()


@pytest.mark.unit
def test_extract_dicom_mask_warns_when_some_contours_are_outside_target_fov(tmp_path):
    rtstruct_path = tmp_path / "rtstruct.dcm"
    _write_rtstruct(rtstruct_path, "GTV", [_square_contour(), _square_contour(z=10.0)])
    image = _make_sitk_image()

    with pytest.warns(DataStructureWarning, match="Skipped 1 RTSTRUCT contour"):
        mask = dicom.extract_dicom_mask(rtstruct_path, "GTV", image)

    assert mask.array is not None
    assert mask.array.any()


@pytest.mark.unit
def test_extract_dicom_mask_returns_empty_image_when_roi_has_no_target_fov_overlap(tmp_path):
    rtstruct_path = tmp_path / "rtstruct.dcm"
    _write_rtstruct(rtstruct_path, "GTV", [_square_contour(z=10.0)])
    image = _make_sitk_image()

    with pytest.warns(DataStructureWarning, match="has no overlap"):
        mask = dicom.extract_dicom_mask(rtstruct_path, "GTV", image)

    assert mask.array is None
