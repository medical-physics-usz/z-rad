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

For IBSI I configuration A, all participants submitted ``0.0455`` for the
intensity-histogram quartile coefficient of dispersion (``ih_qcod``). This
agreement at the reported precision accounts for the zero tolerance in the
reference table.

Z-Rad's observed quartiles are 21 and 23, giving
``(23 - 21) / (23 + 21) = 1/22 = 0.0454545...``, which rounds to ``0.0455``.
The validation therefore requires exact agreement with the reference after
rounding the computed value to four decimal places. This checks agreement
at the published precision; it does not require the unrounded value to equal
the rounded reference. The feature must be present for the comparison to
pass. Other comparisons retain their existing reference-value and tolerance
checks.

IBSI II Reference Availability
------------------------------

For IBSI II configuration 8.B (3D Simoncelli filtering, decomposition level 1),
Table 7.16 of the `IBSI II reference manual
<https://doi.org/10.48550/arXiv.2006.05470>`_ explicitly reports consensus as
``none`` for the quartile coefficient of dispersion (``stat_qcod``). IBSI
therefore provides neither a reference value nor a tolerance for this
feature/filter combination. The blank fields in the bundled reference CSV
match the `official IBSI reference data
<https://github.com/theibsi/ibsi_2_reference_data/blob/main/reference_feature_values/reference_values.csv>`_.

The 8.B test excludes only ``stat_qcod`` from the reference comparison and
continues to check the other 17 reference features. This is an intentional
exception due to the absence of IBSI consensus, not missing repository data
or evidence of a calculation defect. A passing 8.B comparison does not
establish IBSI agreement for ``stat_qcod``.

For IBSI II phase II configurations 10.A, 10.B, 11.A, and 11.B, IBSI provides
no reference values or tolerances for any of the 18 features. These entire
configurations are absent from the official reference-feature CSV and its
bundled copy, which cover configurations 1.A through 9.B. Consequently, no
IBSI phase II feature agreement can be established for 10.A, 10.B, 11.A, or
11.B using the published reference data.

This limitation concerns phase II feature values. Phase I response-map
comparisons are separate tests and do not establish phase II feature agreement
for these configurations.

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
