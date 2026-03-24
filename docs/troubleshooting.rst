Troubleshooting
===============

Documentation Build Fails
-------------------------

If ``make -C docs html`` fails with ``sphinx-build: command not found``, install
the docs dependencies first:

.. code-block:: bash

   pip install -r docs/requirements.txt

Module Import Errors During Docs Build
--------------------------------------

The documentation imports project modules through Sphinx autodoc. Build the docs
from the repository root after installing the package dependencies so imports
such as SimpleITK, NumPy, and PyQt5 are available.

Feature Extraction Errors
-------------------------

Common causes include:

* image and mask arrays not sharing the same geometry
* empty or too-small masks
* unsupported interpolation or aggregation settings
* invalid discretization settings

GUI Issues
----------

If the desktop application starts but does not behave as expected:

* inspect the generated log files
* verify the input modality and spacing assumptions
* confirm that paths to image and mask data are correct

Packaging And Version Mismatch
------------------------------

The package version is defined in ``zrad.__version__`` and should match the
documentation version shown by Sphinx. If they diverge, rebuild the docs after
updating the package metadata.
