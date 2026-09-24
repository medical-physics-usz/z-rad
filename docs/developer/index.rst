Development
===========

New contributors can start with :doc:`contributing` and
:doc:`development_environment`. For an existing checkout, use :doc:`testing`
and :doc:`code_quality` to check a change, :doc:`building_docs` to update the
manual, or :doc:`benchmarking` to investigate performance. Release maintainers
can go directly to :doc:`release_process`.

.. toctree::
   :maxdepth: 1

   contributing
   development_environment
   testing
   ibsi_validation
   code_quality
   building_docs
   benchmarking
   memory_profiling
   benchmark_methodology
   ci
   release_process

Find your way around the repository
-----------------------------------

.. list-table:: Source layout
   :header-rows: 1
   :widths: 35 65

   * - Location
     - Purpose
   * - ``zrad/image.py`` and ``zrad/io/``
     - Image representation and DICOM/NIfTI input and output.
   * - ``zrad/preprocessing/``, ``zrad/filtering/``, ``zrad/radiomics/``
     - Processing steps, filters, and feature extraction.
   * - ``zrad/batch/``
     - Workflows that process case folders and write results.
   * - ``main.py``, ``zrad/gui/``, and ``zrad/visualization/``
     - Desktop application, controls, and image viewer.
   * - ``tests/``
     - Unit and integration tests, reference data, and performance benchmarks.
   * - ``docs/``
     - Narrative guides, API reference pages, and Sphinx configuration.
   * - ``.github/workflows/``
     - Automated checks, documentation deployment, and release jobs.
