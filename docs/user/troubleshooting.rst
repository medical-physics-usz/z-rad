:orphan:

Troubleshooting
===============

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

The package version is defined in ``zrad.__version__``. Build metadata and the
documentation version shown by Sphinx derive from that value. If documentation
still shows an older version, rebuild the docs after updating ``zrad.__version__``.
