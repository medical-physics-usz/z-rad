Release process
===============

This guide is for maintainers preparing a release. Contributors can follow
:doc:`contributing` without performing these steps.

Prepare the release commit
--------------------------

1. Review the changes to include and check their test, lint, and validation
   results. Build the documentation locally using :doc:`building_docs`.
2. Set ``__version__`` in ``zrad/__init__.py`` to the intended release version,
   following the conventions below. Package metadata and documentation read
   this value; the release tag does not change it automatically.
3. Commit the version and release-related documentation changes. Confirm the
   intended commit contains the code and version you want to publish.

Version conventions
-------------------

Z-Rad uses date-based, PEP 440-compatible versions:

.. list-table:: Version examples
   :header-rows: 1
   :widths: 25 30 45

   * - Type
     - Format
     - Example
   * - Stable
     - ``YY.M.PATCH``
     - ``26.5.0`` for the first May 2026 release;
       ``26.5.1`` for a subsequent patch.
   * - Development
     - ``YY.M.PATCH.devN``
     - ``26.6.0.dev0`` for development toward the June 2026 release.

Keep the patch component, including ``.0`` for the first release of a month.
Use ``.dev0`` for the normal in-repository development version. Write the month
without leading zeroes: ``26.5.0``, not ``26.05.0``.

Publish the github release
--------------------------

Create or select the release tag on the intended commit and prepare the release
notes. Check that the tag and notes identify the package version in that commit.
Publish the GitHub release when it is ready for distribution.

Publication triggers both workflows below; they do not wait for one another.
Their trigger is ``release: published``, with no stable-release-only condition.
Do not assume marking a release as a prerelease suppresses publication jobs.

Package publication
~~~~~~~~~~~~~~~~~~~

``.github/workflows/publish.yml`` uses Python 3.12, installs ``build`` and
``twine``, runs ``python -m build``, and uploads ``dist/*`` to PyPI with Twine.
It reads the repository secrets ``PYPI_USERNAME`` and ``PYPI_PASSWORD`` for
upload credentials.

Desktop application builds
~~~~~~~~~~~~~~~~~~~~~~~~~~

``.github/workflows/release-executables.yml`` uses
``generate_executable.py`` and PyInstaller to build and attach:

* ``z-rad-<release-tag>-windows.exe`` from ``windows-latest``
* ``z-rad-<release-tag>-macos-arm64.zip`` from ``macos-latest``

The macOS ZIP contains an Apple Silicon ``Z-Rad.app`` bundle. Intel macOS
binaries are not produced. The app is unsigned and unnotarized, so Gatekeeper
warnings are expected. Adding signing and notarization requires Apple Developer
credentials and a change to the release workflow.

Verify the published artifacts
------------------------------

Check both workflow runs after publication. Verify that PyPI shows the intended
package version and that the GitHub release contains both desktop assets.
A successful package upload does not establish that the executable builds passed.

As a release check, install the published package in a fresh environment and
confirm its version and imports. Download the desktop assets and check that they
launch on their target platforms. If a job fails, inspect its logs and the
artifacts already published before deciding how to recover; publication can
succeed for one artifact while another fails.
