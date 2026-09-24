Compare image filters
=====================

Apply three different 3D filters to the same CT image: a 5-voxel Mean filter,
a 1.5 mm Laplacian-of-Gaussian (LoG) filter, and a first-level Daubechies 3
wavelet ``LLH`` response. Save each response and compare its mean and standard
deviation. Mean smoothing retains the CT intensity scale; LoG and wavelet
values are transformed responses rather than raw HU.

.. code-block:: python

   import numpy as np

   from zrad.filtering import LoG, Mean, Wavelets3D
   from zrad.image import Image

   image = Image.from_nifti("path/to/phantom.nii.gz")

   # Use the same input and boundary handling for each filter.
   filters = {
       "mean": Mean(padding_type="reflect", support=5, dimensionality="3D"),
       "log": LoG(padding_type="reflect", sigma_mm=1.5, cutoff=4.0, dimensionality="3D"),
       "wavelet_llh": Wavelets3D(
           wavelet_type="db3", padding_type="reflect",
           response_map="LLH", decomposition_level=1,
       ),
   }

   for name, image_filter in filters.items():
       response = image_filter.apply(image)
       response.save_as_nifti(f"output/{name}.nii.gz")
       # These summary statistics cover the full image, including outside the ROI.
       print(f"{name}: mean={np.mean(response.array):.3f}, std={np.std(response.array):.3f}")

The files ``mean.nii.gz``, ``log.nii.gz``, and ``wavelet_llh.nii.gz`` can be
opened in an image viewer to inspect spatial differences. See
:doc:`../user/api_filtering` for other filters and parameters.
