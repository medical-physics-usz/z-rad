Release Process
===============

Maintainer Scope
----------------

This page is for maintainers. Contributors do not need to perform release
steps before opening a pull request.

Current Publishing Workflow
---------------------------

The current PyPI publishing workflow is defined in
``.github/workflows/publish.yml``. When a GitHub release is published, GitHub
Actions:

* checks out the repository
* sets up Python 3.12
* installs ``build`` and ``twine``
* builds distribution artifacts with ``python -m build``
* uploads ``dist/*`` to PyPI with Twine

Version Identifiers
-------------------

Z-Rad uses date-based release identifiers. The package version is defined in
``zrad/__init__.py`` as ``zrad.__version__`` and is read by the build metadata
through ``pyproject.toml``.

Use PEP 440-compatible version identifiers directly in ``__version__``.

Stable releases should use the format::

   YY.M.PATCH

For example, the first stable release in May 2026 is::

   __version__ = "26.5.0"

The patch segment should be kept even for the first release of a month. This
leaves room for additional stable patch releases in the same month, such as
``26.5.1`` or ``26.5.2``.

Development versions should use the format::

   YY.M.PATCH.devN

For example, development toward the June 2026 release would start as::

   __version__ = "26.6.0.dev0"

Use ``.dev0`` for the normal in-repository development version.

Do not use leading zeroes in the month component. For example, use ``26.5.0``
rather than ``26.05.0``. Python packaging normalizes leading zeroes away, so
storing the normalized form avoids ambiguity.

Executable generation
-----------------------

Executable-generation policies will be documented here once those processes are defined in the
repository workflow.
