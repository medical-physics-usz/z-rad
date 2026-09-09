from types import SimpleNamespace

import numpy as np
import pydicom
import pytest
import SimpleITK as sitk
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

import zrad.io.dicom as dicom
from zrad.exceptions import DataStructureError, DataStructureWarning


def _make_sitk_image(size=(5, 5, 3)):
    width, height, depth = size
    image = sitk.GetImageFromArray(np.zeros((depth, height, width), dtype=np.int16))
    image.SetOrigin((0.0, 0.0, 0.0))
    image.SetSpacing((1.0, 1.0, 1.0))
    image.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    return image


@pytest.mark.unit
def test_get_dicom_files_keeps_enhanced_pet_out_of_classic_geometry_sorting(monkeypatch, tmp_path):
    enhanced_pet = Dataset()
    enhanced_pet.Modality = "PT"
    enhanced_pet.SOPClassUID = "1.2.840.10008.5.1.4.1.1.130"

    class FakeSeriesReader:
        def GetGDCMSeriesIDs(self, directory):
            return ["enhanced-pet"]

        def GetGDCMSeriesFileNames(self, directory, series_id):
            return [str(tmp_path / "enhanced-pet.dcm")]

    monkeypatch.setattr(dicom.sitk, "ImageSeriesReader", FakeSeriesReader)
    monkeypatch.setattr(dicom.pydicom, "dcmread", lambda *args, **kwargs: enhanced_pet)

    def fail_if_sorted(_files):
        raise AssertionError("Enhanced PET must not use classic slice geometry")

    monkeypatch.setattr(dicom, "sort_by_geometric_position", fail_if_sorted)
    monkeypatch.setattr(dicom, "_sort_enhanced_instances", lambda files: files)

    result = dicom.get_dicom_files(str(tmp_path), "PET")

    assert result == [{"file_path": str(tmp_path / "enhanced-pet.dcm"), "ds": enhanced_pet}]


@pytest.mark.unit
@pytest.mark.parametrize(
    "modality, orientation",
    [
        ("CT", (1, 0, 0, 0, 1, 0)),
        ("MRI", (1, 0, 0, 0, 1, 0)),
        ("PET", (1, 0, 0, 0, 1, 0)),
        ("MG", (1, 0, 0, 0, 1, 0)),
        ("US", (1, 0, 0, 0, 1, 0)),
        ("CT", (0.6, 0.8, 0, 0, 0, 1)),
        ("MRI", (0.6, 0.8, 0, 0, 0, 1)),
        ("PET", (0.6, 0.8, 0, 0, 0, 1)),
    ],
    ids=["CT", "MRI", "PET", "MG", "US", "CT-oblique", "MRI-oblique", "PET-oblique"],
)
def test_process_dicom_series_maps_row_column_spacing_to_physical_axes(monkeypatch, modality, orientation):
    row_spacing, column_spacing, slice_spacing = 0.8, 0.3, 2.5
    x_direction = np.array(orientation[:3])
    y_direction = np.array(orientation[3:])
    normal = np.cross(x_direction, y_direction)
    origin = np.array([10.0, 20.0, 30.0]) if modality in ("CT", "MRI", "PET") else np.zeros(3)
    depth = 1 if modality == "MG" else 2
    pixels = np.arange(depth * 3 * 4, dtype=np.int16).reshape(depth, 3, 4) - 1000
    reader_image = sitk.GetImageFromArray(pixels)
    reader_image.SetOrigin(origin)
    dicom_files = []
    for index in range(2 if modality in ("CT", "MRI", "PET") else 1):
        ds = Dataset()
        ds.Modality = dicom.modality_mapping(modality)
        if modality == "MG":
            ds.ImagerPixelSpacing = [row_spacing, column_spacing]
            ds.BodyPartThickness = slice_spacing
        else:
            ds.PixelSpacing = [row_spacing, column_spacing]
            if modality == "US":
                ds.SliceThickness = slice_spacing
            else:
                ds.ImageOrientationPatient = list(orientation)
                ds.ImagePositionPatient = (origin + index * slice_spacing * normal).tolist()
        dicom_files.append({"file_path": f"slice-{index}.dcm", "ds": ds})

    class FakeSeriesReader:
        def SetFileNames(self, names):
            assert names == [item["file_path"] for item in dicom_files]

        def Execute(self):
            return reader_image

    monkeypatch.setattr(dicom.sitk, "ImageSeriesReader", FakeSeriesReader)
    monkeypatch.setattr(dicom.sitk, "ReadImage", lambda _path: reader_image)

    image = dicom.process_dicom_series(dicom_files, modality)

    assert image.GetSpacing() == pytest.approx((column_spacing, row_spacing, slice_spacing))
    # Check actual physical displacements, including when image axes are rotated.
    assert image.TransformIndexToPhysicalPoint((1, 0, 0)) == pytest.approx(origin + column_spacing * x_direction)
    assert image.TransformIndexToPhysicalPoint((0, 1, 0)) == pytest.approx(origin + row_spacing * y_direction)
    assert image.TransformIndexToPhysicalPoint((0, 0, 1)) == pytest.approx(origin + slice_spacing * normal)
    np.testing.assert_array_equal(sitk.GetArrayFromImage(image), pixels)


@pytest.mark.unit
def test_read_dicom_dose_preserves_reader_spacing_during_scaling(monkeypatch):
    pixels = np.arange(24, dtype=np.uint16).reshape(2, 3, 4)
    reader_image = sitk.GetImageFromArray(pixels)
    reader_image.SetSpacing((0.3, 0.8, 2.5))
    reader_image.SetOrigin((10.0, 20.0, 30.0))
    reader_image.SetDirection((0, -1, 0, 1, 0, 0, 0, 0, 1))
    ds = Dataset()
    ds.DoseUnits = "GY"
    ds.DoseType = "PHYSICAL"
    ds.DoseGridScaling = 0.01
    ds.PixelSpacing = [0.8, 0.3]
    monkeypatch.setattr(dicom.pydicom, "dcmread", lambda _path: ds)
    monkeypatch.setattr(dicom.sitk, "ReadImage", lambda _path: reader_image)

    image = dicom.read_dicom_dose("dose.dcm")

    assert image.GetSpacing() == pytest.approx((0.3, 0.8, 2.5))
    assert image.GetOrigin() == reader_image.GetOrigin()
    assert image.GetDirection() == reader_image.GetDirection()
    np.testing.assert_allclose(sitk.GetArrayFromImage(image), pixels * 0.01, rtol=1e-12, atol=0)


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


@pytest.mark.unit
def test_dicom_seg_selects_segment_by_label_and_places_frames(monkeypatch):
    segment = SimpleNamespace(SegmentNumber=2, SegmentLabel="Tumor lesions")
    other_segment = SimpleNamespace(SegmentNumber=1, SegmentLabel="Background")
    group = SimpleNamespace(
        SegmentIdentificationSequence=[SimpleNamespace(ReferencedSegmentNumber=2)],
        PlanePositionSequence=[SimpleNamespace(ImagePositionPatient=[0.0, 0.0, 1.0])],
    )
    seg = SimpleNamespace(
        Modality="SEG",
        SegmentationType="BINARY",
        SegmentSequence=[other_segment, segment],
        PerFrameFunctionalGroupsSequence=[group],
        pixel_array=np.array([[[0, 1, 0, 0, 0]] * 5], dtype=np.uint8),
    )
    monkeypatch.setattr(pydicom, "dcmread", lambda *_args, **_kwargs: seg)

    mask = dicom.read_dicom_mask("seg.dcm", "Tumor lesions", _make_sitk_image())

    assert mask.array.shape == (3, 5, 5)
    assert mask.array[1, 0, 1] == 1
    assert np.count_nonzero(mask.array[0]) == 0


@pytest.mark.unit
def test_dicom_seg_accepts_image_position_directly_in_per_frame_group(monkeypatch):
    group = SimpleNamespace(
        SegmentIdentificationSequence=[SimpleNamespace(ReferencedSegmentNumber=1)],
        ImagePositionPatient=[0.0, 0.0, 2.0],
    )
    seg = SimpleNamespace(
        Modality="SEG",
        SegmentationType="BINARY",
        SegmentSequence=[SimpleNamespace(SegmentNumber=1, SegmentLabel="Tumor")],
        PerFrameFunctionalGroupsSequence=[group],
        pixel_array=np.array([[[1, 0, 0, 0, 0]] * 5], dtype=np.uint8),
    )
    monkeypatch.setattr(pydicom, "dcmread", lambda *_args, **_kwargs: seg)

    mask = dicom.read_dicom_mask("seg.dcm", "Tumor", _make_sitk_image())

    assert mask.array.shape == (3, 5, 5)
    assert mask.array[2, 0, 0] == 1
    assert np.count_nonzero(mask.array[:2]) == 0


@pytest.mark.unit
@pytest.mark.parametrize("segmentation_type", ["FRACTIONAL", "LABELMAP", None])
def test_dicom_seg_rejects_non_binary_segmentation_types(monkeypatch, segmentation_type):
    seg = SimpleNamespace(
        Modality="SEG",
        SegmentSequence=[SimpleNamespace(SegmentNumber=1, SegmentLabel="Tumor")],
        SegmentationType=segmentation_type,
    )
    monkeypatch.setattr(pydicom, "dcmread", lambda *_args, **_kwargs: seg)

    with pytest.raises(DataStructureError, match="Only BINARY segmentations are supported"):
        dicom.read_dicom_mask("seg.dcm", "Tumor", _make_sitk_image())


@pytest.mark.unit
def test_get_all_structure_names_supports_dicom_seg(monkeypatch):
    seg = SimpleNamespace(
        Modality="SEG",
        SegmentSequence=[
            SimpleNamespace(SegmentNumber=1, SegmentLabel="Tumor lesions"),
            SimpleNamespace(SegmentNumber=2, SegmentLabel="Liver"),
        ],
    )
    monkeypatch.setattr(pydicom, "dcmread", lambda *_args, **_kwargs: seg)

    assert dicom.get_all_structure_names("seg.dcm") == ["Tumor lesions", "Liver"]


@pytest.mark.unit
@pytest.mark.parametrize(
    'stored,slope,intercept,expected',
    [
        ([0, 1, 2, 3], 1, -10, [-10, -9, -8, -7]),
        ([0, 1, 2, 3], 2, -3, [-3, -1, 1, 3]),
        ([0, 1, 2, 3], 1, 0, [0, 1, 2, 3]),
        ([0, 1, 2, 3], 2, 5, [5, 7, 9, 11]),
        ([0, 1, 2, 3], 0.5, -0.25, [-0.25, 0.25, 0.75, 1.25]),
        ([0, 1, 2, 3], 0.5, 0.25, [0.25, 0.75, 1.25, 1.75]),
        ([-2, -1, 0, 1], 2, 5, [1, 3, 5, 7]),
    ],
    ids=['negative', 'mixed', 'identity', 'positive', 'fractional-mixed', 'fractional-positive', 'signed-storage'],
)
def test_ct_dicom_rescaling_preserves_declared_values(tmp_path, stored, slope, intercept, expected):
    """Use real encoded DICOMs and explicit expected values, not a mocked reader."""
    from zrad.image import Image

    study_uid, series_uid, frame_uid = generate_uid(), generate_uid(), generate_uid()
    signed = min(stored) < 0
    for index in range(2):
        meta = FileMetaDataset()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        meta.MediaStorageSOPClassUID = pydicom.uid.CTImageStorage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        # Reverse filenames to require geometry-based ordering.
        path = tmp_path / f'{1 - index}.dcm'
        ds = FileDataset(str(path), {}, file_meta=meta, preamble=b'\0' * 128)
        ds.SOPClassUID = meta.MediaStorageSOPClassUID
        ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        ds.StudyInstanceUID, ds.SeriesInstanceUID, ds.FrameOfReferenceUID = study_uid, series_uid, frame_uid
        ds.PatientName, ds.PatientID = 'Rescale^Test', 'rescale-test'
        ds.Modality = 'CT'
        ds.ImageType = ['DERIVED', 'SECONDARY']
        ds.InstanceNumber = index + 1
        ds.ImagePositionPatient = [10, 20, 30 + 2 * index]
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.PixelSpacing = [0.8, 0.3]
        ds.SliceThickness = 2
        ds.Rows, ds.Columns = 2, 2
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = 'MONOCHROME2'
        ds.BitsAllocated = ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = int(signed)
        ds.RescaleSlope, ds.RescaleIntercept, ds.RescaleType = slope, intercept, 'HU'
        pixels = np.array(stored, dtype='<i2' if signed else '<u2')
        if index:
            pixels = pixels[::-1]
        ds.PixelData = pixels.tobytes()
        ds.save_as(path, enforce_file_format=True)

    image = Image.from_dicom(tmp_path, modality='CT')
    expected_volume = np.array([expected, expected[::-1]]).reshape(2, 2, 2)
    np.testing.assert_array_equal(image.array, expected_volume)
    assert image.spacing == pytest.approx((0.3, 0.8, 2))
    assert image.origin == pytest.approx((10, 20, 30))
