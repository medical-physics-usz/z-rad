===============
IBSI Compliance
===============

Overview
--------

Z-Rad is developed around IBSI-oriented radiomics workflows and includes test
data and regression tests derived from IBSI reference material.

Repository Assets
-----------------

The repository includes:

* IBSI test data under ``tests/data``
* reference feature values for IBSI I and IBSI II
* automated tests in ``tests/test_ibsi_1.py`` and ``tests/test_ibsi_2.py``

Reference Precision
-------------------

For IBSI I configuration A, the intensity-histogram quartile coefficient of
dispersion (``ih_qcod``) is compared with the published value ``0.0455`` after
rounding the computed value to four decimal places. The observed quartiles
are 21 and 23, giving ``(23 - 21) / (23 + 21) = 1/22``. The reference table's
zero tolerance is interpreted as exact agreement at its published precision,
not equality between the unrounded result and the rounded reference value.
This feature must be present for the comparison to pass. Other comparisons
retain their existing reference-value and tolerance checks.

Licensing
---------

The bundled IBSI datasets use multiple open licenses depending on the specific
component. See ``tests/data/README.md`` for the exact attribution and license
terms of each dataset subset.

Why This Matters
----------------

The IBSI tests provide a reproducibility baseline for:

* preprocessing choices
* filter definitions
* discretization behavior
* radiomics feature calculations

When changing feature code or preprocessing behavior, these tests should remain
part of the release and CI validation workflow.
