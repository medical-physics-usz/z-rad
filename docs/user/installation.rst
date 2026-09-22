Installation
============

Choose the desktop application for GUI workflows, or install the Python package
for scripts and pipelines. Running from source also gives you access to the GUI
and the bundled examples.

Desktop application
-------------------

Download the asset for your platform from the
`releases page <https://github.com/medical-physics-usz/z-rad/releases>`_:

* Windows: download and run ``z-rad-<release-tag>-windows.exe``.
* Apple Silicon macOS: download ``z-rad-<release-tag>-macos-arm64.zip``,
  extract it, and open ``Z-Rad.app``.

The macOS app is unsigned and unnotarized, so macOS may show a Gatekeeper
warning. Intel Mac and Linux users can run the GUI from source as described
below. The packaged application does not require a separate Python installation.

Continue with :doc:`gui_quickstart` once the application opens.

Python environment
------------------

Python workflows require Python 3.11 or newer on Windows, macOS, or Linux.
Create a virtual environment in your working directory:

.. code-block:: bash

   python -m venv .venv

Activate it on macOS or Linux:

.. code-block:: bash

   source .venv/bin/activate

Or in Windows PowerShell:

.. code-block:: powershell

   .venv\Scripts\Activate.ps1

Install the package
-------------------

For use in your own scripts, install the published package in the activated
environment:

.. code-block:: bash

   python -m pip install z-rad

See :doc:`api_quickstart` for a first extraction using the repository's bundled
phantom. That example also requires a repository checkout to access the data.

Run from source
---------------

Clone the repository, or download and extract its source archive:

.. code-block:: bash

   git clone https://github.com/medical-physics-usz/z-rad.git
   cd z-rad

Create and activate a Python environment as described above, then install from
the repository root:

.. code-block:: bash

   python -m pip install -e .

The editable installation uses this checkout for both imports and the GUI. To
start the application, run:

.. code-block:: bash

   python main.py

If you only need a non-editable package installation from source, use
``python -m pip install .`` instead.
