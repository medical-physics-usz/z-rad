# Z-Rad

[![Tests](https://github.com/medical-physics-usz/z-rad/actions/workflows/test.yml/badge.svg?branch=master)](https://github.com/medical-physics-usz/z-rad/actions/workflows/test.yml)
[![Ruff](https://github.com/medical-physics-usz/z-rad/actions/workflows/python-lint.yml/badge.svg?branch=master)](https://github.com/medical-physics-usz/z-rad/actions/workflows/python-lint.yml)
[![Docs](https://github.com/medical-physics-usz/z-rad/actions/workflows/docs.yml/badge.svg?branch=master)](https://github.com/medical-physics-usz/z-rad/actions/workflows/docs.yml)
[![PyPI](https://img.shields.io/pypi/v/z-rad.svg)](https://pypi.org/project/z-rad/)
[![Python](https://img.shields.io/pypi/pyversions/z-rad.svg)](https://pypi.org/project/z-rad/)
[![License](https://img.shields.io/pypi/l/z-rad.svg)](https://github.com/medical-physics-usz/z-rad/blob/master/LICENSE)

<p align="center">
  <img src="https://raw.githubusercontent.com/medical-physics-usz/z-rad/master/docs/logos/ZRadLogo.jpg" width="320" alt="Z-Rad — Zurich Radiomics" />
</p>

<p align="center">
  <strong>Extract quantitative features from medical images with a desktop interface or Python API.</strong><br />
  CT, PET, MR, MG, US, and RTDOSE · DICOM and NIfTI · Windows, macOS, and Linux<br />
  Developed by the Department of Radiation Oncology at University Hospital Zurich
</p>

<p align="center">
  <a href="#ibsi-implementation-and-validation"><strong>Full IBSI I preprocessing and feature coverage · All IBSI II filters</strong></a>
</p>

<p align="center">
  <a href="https://github.com/medical-physics-usz/z-rad/releases">Download</a> ·
  <a href="https://medical-physics-usz.github.io/z-rad/">Documentation</a> ·
  <a href="#python-quickstart">Python quickstart</a> ·
  <a href="https://medical-physics-usz.github.io/z-rad/examples/">Examples</a> ·
  <a href="#ibsi-implementation-and-validation">Validation</a>
</p>

## From images to a feature table

<p align="center">
  <img src="docs/images/readme_workflow.png" width="700" alt="Workflow from images and ROI masks through preprocessing, optional filtering, feature extraction, and export to a feature dictionary or CSV." />
</p>

Use the same processing stages interactively through the desktop application or automate them with Python. See the [workflow guide](https://medical-physics-usz.github.io/z-rad/user/gui_workflows.html) for data preparation and processing choices.

## See what you can do

| Capability | What you can do |
| --- | --- |
| Inspect images | View images and ROI masks together in the [desktop viewer](https://medical-physics-usz.github.io/z-rad/user/visualization.html). |
| Prepare images | Convert DICOM to NIfTI, resample images and masks, resegment intensities, and configure discretization. |
| Filter images | Apply standardized spatial and wavelet filters before feature extraction. |
| Extract features | Calculate shape, intensity, and texture features with 2D, 2.5D, and 3D aggregation options. |
| Process cohorts | Use the GUI or Python batch APIs to preprocess, filter, and extract features across case folders. |
| Use the results | Collect feature dictionaries in Python or export batch radiomics results to a CSV file. |

<p align="center">
  <img src="https://raw.githubusercontent.com/medical-physics-usz/z-rad/master/docs/images/Filtering_Res.png" width="380" alt="Four CT views comparing an unfiltered image with mean, Laplacian-of-Gaussian, and wavelet filtering" />
</p>

*IBSI II CT phantom: (A) unfiltered, (B) mean, (C) Laplacian-of-Gaussian, and (D) Daubechies 3 wavelet filtering. See [filter settings and the worked example](https://medical-physics-usz.github.io/z-rad/examples/gui_filtering.html).*

## Supported images and masks

| Data format | Data type | Supported types and notes |
| --- | --- | --- |
| DICOM | Image | CT, MR, PET (PT), mammography (MG), ultrasound (US), and RTDOSE. |
| DICOM | Mask | RTSTRUCT contours and BINARY DICOM SEG objects; select ROIs by structure name or segment label. |
| NIfTI | Image | Scalar image volumes with spatial geometry. |
| NIfTI | Mask | One binary ROI mask per file, paired with its reference image. |

Use scalar image volumes with masks on the same physical voxel grid. DICOM SEG support excludes fractional and label-map segmentations. Ultrasound input must be a single DICOM file with `PixelSpacing` and `SliceThickness` metadata.

See the [data-format and folder-layout guide](https://medical-physics-usz.github.io/z-rad/user/data_structure.html) for organizing cases, naming masks, and selecting DICOM structures, and the [Python image reference](https://medical-physics-usz.github.io/z-rad/reference/image.html) for in-memory inputs.

## Full IBSI implementation coverage

Z-Rad supports **all IBSI I preprocessing operations and radiomic features, and all IBSI II filters** defined by the Image Biomarker Standardisation Initiative (IBSI). Implementation coverage describes available operations; benchmark agreement applies to the tested configurations and features described in the [coverage and limitations](docs/ibsi/index.rst).

| Standard | Implementation coverage | Explore |
| --- | --- | --- |
| IBSI I · Preprocessing | All operations, including image and mask interpolation, resegmentation, and intensity discretization | [Preprocessing guide](https://medical-physics-usz.github.io/z-rad/user/preprocessing.html) |
| IBSI I · Features | All feature families, including morphology, local intensity, intensity statistics, histograms, intensity-volume histograms, and texture | [Feature guide](https://medical-physics-usz.github.io/z-rad/user/radiomics.html) |
| IBSI II · Filters | All filters, including mean, LoG, Laws, Gabor, separable wavelets, Simoncelli, and Riesz transforms | [Filtering guide](https://medical-physics-usz.github.io/z-rad/user/filtering.html) |


## Get started

### Desktop application

1. Open [Z-Rad releases](https://github.com/medical-physics-usz/z-rad/releases) and choose a release.
2. Download the asset for your platform:

   | Platform | Release asset | Launch |
   | --- | --- | --- |
   | Windows | `z-rad-<release-tag>-windows.exe` | Run the executable. |
   | Apple Silicon macOS | `z-rad-<release-tag>-macos-arm64.zip` | Extract the archive and open `Z-Rad.app`. |

3. Follow the [GUI quickstart](https://medical-physics-usz.github.io/z-rad/user/gui_quickstart.html) to select your input data, configure processing, and run your first analysis.

The macOS app is currently unsigned and unnotarized, so Gatekeeper may show a warning. For Linux, Intel Macs, or the current source version, follow [Run from a repository checkout](https://medical-physics-usz.github.io/z-rad/user/installation.html#run-from-a-repository-checkout) to install and launch the GUI.

### Python quickstart

**Requires Python 3.11 or newer.** Install the published package:

```sh
python -m pip install z-rad
```

The example below requires your own NIfTI image and binary ROI mask on the same physical voxel grid. Replace the two paths with your files. It extracts intensity statistics without resampling or filtering:

```python
from zrad.image import Image
from zrad.preprocessing import IntensityMaskBuilder, RoiData
from zrad.radiomics import Radiomics

image = Image.from_nifti("path/to/image.nii.gz")
mask = Image.from_nifti_mask("path/to/mask.nii.gz", reference=image)
roi = IntensityMaskBuilder().apply(RoiData(image=image, morphological_mask=mask))
features = Radiomics().extract_features(roi_data=roi, families=["intensity_statistics"])
print(features["stat_mean"])
```

The result is a dictionary of feature names and values. To try a complete example with supplied data and an expected result, follow the [bundled phantom example](https://medical-physics-usz.github.io/z-rad/user/api_quickstart.html#run-the-bundled-phantom-example). Texture and intensity-volume histogram features require additional preparation; see the [full Python workflow](https://medical-physics-usz.github.io/z-rad/user/api_quickstart.html#recommended-workflow) and follow the documentation for your installed release.

## IBSI implementation and validation

Z-Rad implements **all IBSI I preprocessing operations and radiomic features, and all IBSI II filters** defined by the Image Biomarker Standardisation Initiative (IBSI). Automated benchmarks compare results against published reference data:

- **IBSI I:** digital-phantom features and CT configurations A–E, including preprocessing diagnostics.
- **IBSI II:** digital-phantom filter response maps and features from filtered CT images.
- **IBSI-SUV:** SUV conversion checks using valid and intentionally invalid digital reference objects.

Implementation coverage describes available operations; benchmark agreement applies to the tested configurations and features. See [IBSI coverage and limitations](https://medical-physics-usz.github.io/z-rad/ibsi/) for the scope, comparison rules, and unavailable references, and [execution reports and reproduction](https://medical-physics-usz.github.io/z-rad/developer/testing.html#ibsi-benchmark-reports) to inspect results for a particular revision.

## Contribute and get in touch

Found a bug or have a feature request? [Open an issue](https://github.com/medical-physics-usz/z-rad/issues). To contribute code or documentation, start with the [contributing guide](https://medical-physics-usz.github.io/z-rad/developer/contributing.html).

For questions or research collaborations, contact [zrad@usz.ch](mailto:zrad@usz.ch). Z-Rad is released under the [MIT License](LICENSE).
