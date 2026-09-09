# IBSI 1 reference values

This directory contains reference values and tolerances for the IBSI 1 digital
phantom and CT radiomics phantom benchmark configurations A-E. The CSV files
were adapted from the official IBSI 1 submission table and the corresponding
tables in version 11 of the IBSI reference manual.

## Sources

- IBSI 1 project page and submission table:
  https://theibsi.github.io/ibsi1/
- Alex Zwanenburg, Stefan Leger, Martin Vallières, and Steffen Löck (for the
  Image Biomarker Standardisation Initiative), *Image biomarker
  standardisation initiative*, arXiv:1612.07003v11:
  https://arxiv.org/abs/1612.07003v11
- Related journal publication:
  https://doi.org/10.1148/radiol.2020191145

## Adaptation

The source tables were converted to CSV. The `consensus`, `your result`,
`difference`, and `check` columns were removed, leaving the following fields:

| Column | Meaning |
| --- | --- |
| `dataset` | Digital phantom or CT benchmark configuration |
| `family` | IBSI feature family and, where applicable, aggregation method |
| `feature` | Human-readable IBSI feature name |
| `reference value` | Published consensus reference value |
| `tolerance` | Published acceptance tolerance |
| `tag` | Z-Rad identifier used to match computed and reference features |

The `tag` column is Z-Rad-specific metadata. Reference values and tolerances
have not been recalculated. Empty reference values and tolerances are retained
for the five features for which IBSI 1 did not establish a reference:

- volume density, oriented minimum bounding box
- area density, oriented minimum bounding box
- volume density, minimum volume enclosing ellipsoid
- area density, minimum volume enclosing ellipsoid
- area under the IVH curve

## License

The source material and these adapted reference tables are distributed under
the Creative Commons Attribution 4.0 International license (CC BY 4.0). See
[LICENSE.md](LICENSE.md) for the license notice and attribution requirements.

## Provenance and integrity

The spreadsheet source is the [IBSI 1 submission table](https://ibsi.radiomics.hevs.ch/assets/IBSI-1-submission-table.xlsx).
The CC BY 4.0 attribution above refers to the reference manual; it does not
assert a separately verified license for the Excel workbook as a whole.
The manual version is pinned to arXiv:1612.07003v11. No upstream Git revision
has been established for the spreadsheet.

On 2026-09-09, all six CSVs were compared with the existing Z-Rad reference
CSVs: every retained field was identical, including numerical text and blanks.
This comparison establishes preservation during migration; it is not an
independent transcription audit against every table in the manual.
The files contain 411 rows each for A and B, 275 each for C, D, and E, and
487 for the digital phantom (excluding headers).

`SHA256SUMS` records every distributed file except itself, using paths relative
to this directory. Verify with `shasum -a 256 -c SHA256SUMS` from here.
