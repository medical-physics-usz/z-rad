API quickstart
==============

Extract your first features with the bundled CT phantom, then adapt the
workflow to your own images.

Run the bundled phantom example
-------------------------------

This example uses the repository's supplied CT phantom and binary ROI mask,
so you can verify a result before working with your own images. Clone the
repository and create a virtual environment:

.. code-block:: bash

   git clone https://github.com/medical-physics-usz/z-rad.git
   cd z-rad
   python -m venv .venv

Activate it with ``source .venv/bin/activate`` on macOS/Linux or
``.venv\Scripts\Activate.ps1`` in Windows PowerShell, then install:

.. code-block:: bash

   python -m pip install -e .

Run the following from the repository root. It extracts intensity statistics
without resampling or filtering:

.. code-block:: python

   from pathlib import Path
   from tempfile import TemporaryDirectory
   from zipfile import ZipFile

   from zrad.image import Image
   from zrad.preprocessing import IntensityMaskBuilder, RoiData
   from zrad.radiomics import Radiomics

   with TemporaryDirectory() as folder:
       with ZipFile("tests/data/ibsi_ct_radiomics_phantom.zip") as archive:
           for name in ("image/phantom.nii.gz", "mask/mask.nii.gz"):
               archive.extract(f"ibsi_ct_radiomics_phantom/nifti/{name}", folder)

       data = Path(folder) / "ibsi_ct_radiomics_phantom/nifti"
       image = Image.from_nifti(data / "image/phantom.nii.gz")
       mask = Image.from_nifti_mask(data / "mask/mask.nii.gz", reference=image)
       roi = IntensityMaskBuilder().apply(RoiData(image=image, morphological_mask=mask))
       features = Radiomics().extract_features(roi_data=roi, families=["intensity_statistics"])

       print(f"Mean intensity: {features['stat_mean']:.2f} HU")

Expected output:

.. code-block:: text

   Mean intensity: -46.88 HU

The result is a dictionary of feature names and values; this example prints
the mean CT intensity within the ROI. See the `bundled dataset attribution
and license terms <https://github.com/medical-physics-usz/z-rad/blob/master/tests/data/README.md>`_
before reusing or redistributing the phantom data. For texture and
intensity-volume histogram features, continue with :doc:`api_workflows`.


Next steps
----------

* :doc:`api_workflows`: resample images and masks and build an extraction pipeline.
* :doc:`api_filtering`: apply filters to individual images.
* :doc:`api_batch`: process case folders and save results to disk.
* :doc:`../reference/index`: look up classes and parameters.
