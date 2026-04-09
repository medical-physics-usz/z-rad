Building The Docs
=================

Requirements
------------

Install the documentation toolchain into the same Python environment you use to
run the project:

.. code-block:: bash

   python3 -m pip install -r docs/requirements.txt

Build Commands
--------------

Build the HTML site with either of these commands from the repository root:

.. code-block:: bash

   python3 -m sphinx -M html docs docs/_build

or:

.. code-block:: bash

   make -C docs html

Output
------

The generated documentation is written to ``docs/_build/html``.

Fixing Build Problems
---------------------

Common causes of docs failures include:

* Sphinx installed in a different Python environment than the one used to build
* missing runtime dependencies required by autodoc imports
* stale references in ``toctree`` blocks
