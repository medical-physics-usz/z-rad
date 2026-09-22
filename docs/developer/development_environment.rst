Development environment
=======================

Z-Rad requires Python 3.11 or newer. See :doc:`ci` for the versions tested in
GitHub Actions. Use a separate environment for each checkout so imports and
commands use the code you intend to change.

Get the source
--------------

Clone the repository, or use your existing checkout:

.. code-block:: bash

   git clone https://github.com/medical-physics-usz/z-rad.git
   cd z-rad

Run the remaining commands from the repository root. The examples use
``python``; use ``python3`` if that is how your system names the supported
interpreter when creating the environment.

Create and activate an environment
----------------------------------

.. code-block:: bash

   python -m venv .venv

On macOS or Linux:

.. code-block:: bash

   source .venv/bin/activate

On Windows PowerShell:

.. code-block:: powershell

   .venv\Scripts\Activate.ps1

Install development dependencies
--------------------------------

In the activated environment, install the editable package and contributor tools:

.. code-block:: bash

   python -m pip install --upgrade pip
   python -m pip install -e ".[dev]"
   python -m pip check

The editable installation imports Z-Rad from this checkout. ``pip check`` checks
installed dependency requirements. If imports resolve to unexpected code, check
the active interpreter and package location:

.. code-block:: bash

   python -c "import sys, zrad; print(sys.executable); print(zrad.__file__)"

Continue with :doc:`testing` or launch the GUI with ``python main.py``.

Optional dependencies
---------------------

Use ``dev`` for routine contribution work. Smaller extras are available when
you only need part of the toolchain:

* ``docs``: Sphinx and its theme dependencies.
* ``test``: pytest, coverage, parallel test execution, and benchmarking tools.
* ``lint``: Ruff formatting and linting.
* ``dev``: the docs, test, and lint tools together.
* ``profiling``: Memray on Linux and macOS; install it separately when following
  :doc:`memory_profiling`. It is not included in ``dev``.
