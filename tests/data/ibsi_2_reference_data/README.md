# IBSI II reference data

This directory contains published consensus response maps for IBSI II phase I
and reference feature values for phase II.

## Source and license

Source: [theibsi/ibsi_2_reference_data](https://github.com/theibsi/ibsi_2_reference_data/tree/5404579fb3e0d17e8db421f0e82d64ce2432ec03).

[//]: # (All 33 response maps, the reference CSV, and `LICENSE` were verified unchanged)

[//]: # (against upstream Git blob hashes at revision)

[//]: # (`5404579fb3e0d17e8db421f0e82d64ce2432ec03` on 2026-09-09.)

[//]: # (This is a verified source match, not the original download date.)

The source repository distributes these data under CC0 1.0 Universal; see
[LICENSE](LICENSE). The benchmark configurations and definitions are described
in the [IBSI II reference manual](https://doi.org/10.48550/arXiv.2006.05470).

## Contents and comparison rules

- `ibsi_2_response_maps.zip`: 33 NIfTI response maps, plus a standalone README,
  the CC0 license, and a SHA-256 manifest. The archive has one top-level
  `ibsi_2_response_maps/` directory containing `reference_response_maps/`.
  Every voxel must agree
  within 1% of the intensity range of the reference map.
- `reference_feature_values/reference_values.csv`: a UTF-8, semicolon-delimited
  table with `filter_id`, `feature`, `feature_tag`, `consensus_value`, and
  `tolerance` columns. It contains 324 rows for configurations 1.A through 9.B.
  Feature comparisons use the published reference values and tolerances.

The value and tolerance for 8.B `stat_qcod` are blank because IBSI did not
establish consensus.
Phase II configurations 10.A, 10.B, 11.A, and 11.B have no published feature
references in this table. These limits are separate from phase I response-map
availability; all 33 supplied maps are retained.

## Local changes and integrity

[//]: # (This README replaces the upstream overview with the local inventory, provenance,)

[//]: # (and reference-availability notes. No response-map bytes, reference CSV bytes,)

[//]: # (or license text were changed. These data also match the legacy Z-Rad copies.)

`SHA256SUMS` records every distributed file except itself, using paths relative
to this directory. Verify with `shasum -a 256 -c SHA256SUMS` from here.
The feature CSV, this README, and `LICENSE` remain ordinary files outside the
archive. Tests extract the response-map archive into the disposable
`tests/data/.cache/ibsi_2_response_maps/` directory.
