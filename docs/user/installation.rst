Installation
============

Requirements
------------

* Python 3.10 or newer for Python-based workflows.
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

On Windows, use the ``z-rad.exe`` asset distributed with the release package.

Run From A Repository Checkout
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Download or clone the repository, create a Python environment, install the
project requirements, and start the GUI from the repository root.

Create and activate a virtual environment:

.. code-block:: bash

   python -m venv .venv

Then install the runtime requirements:

.. code-block:: bash

   pip install -r requirements.txt

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
