Installation
============

Requirements
------------

* Python 3.11 or newer for Python-based workflows.
* Windows, macOS, or Linux.
* Enough local storage for imaging data, masks, output files, and logs.

Run The Software
----------------

You can run Z-Rad either as a packaged desktop application or directly from a
repository checkout.

Run The Release Executable
^^^^^^^^^^^^^^^^^^^^^^^^^^

The simplest way to run the GUI is to start the executable attached to each
release.

Use the platform-specific release asset:

* On Windows, download and run ``z-rad-<release-tag>-windows.exe``.
* On Apple Silicon macOS, download ``z-rad-<release-tag>-macos-arm64.zip``,
  extract it, and start ``Z-Rad.app``.

The macOS release app is currently unsigned and unnotarized. macOS may show a
Gatekeeper warning when opening it. Intel macOS release binaries are not
provided; Intel Mac users should run Z-Rad from a Python environment.

Run From A Repository Checkout
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Download or clone the repository, create a Python environment, install Z-Rad
from the source checkout, and start the GUI from the repository root.

Create and activate a virtual environment:

.. code-block:: bash

   python -m venv .venv

Then install Z-Rad from the source checkout:

.. code-block:: bash

   pip install -e .

Launch the application:

.. code-block:: bash

   python main.py

Install As A Package
--------------------

If you want to use Z-Rad as an importable Python package in your own scripts or
pipelines, install it with one of the following methods.

Install From PyPI
^^^^^^^^^^^^^^^^^

Install the published package from PyPI:

.. code-block:: bash

   pip install z-rad

Install From Source
^^^^^^^^^^^^^^^^^^^

Clone or download the repository, move into the project root, and install the
package from source:

.. code-block:: bash

   pip install .
