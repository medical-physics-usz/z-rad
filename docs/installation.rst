Installation
============

Requirements
------------

* Python 3.10 or newer for the Python-based workflow.
* Windows, macOS, or Linux.
* Enough local storage for imaging data, masks, output files, and logs.

Install From PyPI
-----------------

Install the published package:

.. code-block:: bash

   pip install z-rad

Install From A Local Checkout
-----------------------------

Clone the repository, move into the project root, and install the dependencies:

.. code-block:: bash

   pip install -r requirements.txt

You can then launch the desktop application with:

.. code-block:: bash

   python main.py

Windows Executable
------------------

If you prefer the packaged GUI on Windows, use the ``z-rad.exe`` asset attached
to each release.

Documentation Build Environment
-------------------------------

The documentation uses Sphinx with the Furo theme. To build it locally:

.. code-block:: bash

   pip install -r docs/requirements.txt
   make -C docs html

The generated site is written to ``docs/_build/html``.
