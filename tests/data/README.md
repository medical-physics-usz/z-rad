# IBSI test data

These assets come from the Image Biomarker Standardisation Initiative (IBSI).
Their licenses apply separately from the MIT license for Z-Rad code.

## Canonical datasets

| Dataset | Asset | License |
| --- | --- | --- |
| Shared IBSI I and II CT radiomics phantom | [ibsi_ct_radiomics_phantom](ibsi_ct_radiomics_phantom.zip) | [CC BY-NC 3.0](https://creativecommons.org/licenses/by-nc/3.0/) |
| IBSI I digital phantom | [ibsi_1_digital_phantom](ibsi_1_digital_phantom.zip) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| IBSI II digital phantoms | [ibsi_2_digital_phantom](ibsi_2_digital_phantom.zip) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| IBSI I reference values | [ibsi_1_reference_data](ibsi_1_reference_data/README.md) | [CC BY 4.0](ibsi_1_reference_data/LICENSE.md) |
| IBSI II response maps and feature values | [ibsi_2_reference_data](ibsi_2_reference_data/README.md) | [CC0 1.0](ibsi_2_reference_data/LICENSE) |

Each image archive contains its source, local changes, and file-layout README,
license notice, and `SHA256SUMS` integrity manifest. The readable reference-data
directories also include their own notices and manifests. The shared CT phantom
retains its attribution and noncommercial restriction. It is used with different
benchmark configurations in IBSI I and II.

The IBSI I digital phantom is stored at `nifti/image/phantom.nii.gz` and
`nifti/mask/mask.nii.gz` within its archive directory. Benchmark settings use no
interpolation or resegmentation, six-bin discretization preserving the original
grey levels, and all six texture aggregation modes.

## Packaging and extraction

The canonical image archives are:

- [ibsi_ct_radiomics_phantom.zip](ibsi_ct_radiomics_phantom.zip)
- [ibsi_1_digital_phantom.zip](ibsi_1_digital_phantom.zip)
- [ibsi_2_digital_phantom.zip](ibsi_2_digital_phantom.zip)
- [ibsi_2_response_maps.zip](ibsi_2_reference_data/ibsi_2_response_maps.zip)

Each archive includes a README, the relevant license, and a `SHA256SUMS`
manifest. It contains one top-level directory named after the archive.

[//]: # (Entries)

[//]: # (are sorted, use fixed timestamps and permissions, and exclude packaging metadata.)

[//]: # (`IMAGE_ARCHIVES.sha256` records archive checksums; verify from this directory)

[//]: # (with `shasum -a 256 -c IMAGE_ARCHIVES.sha256`.)

[//]: # (Reference CSVs remain ordinary files in their respective reference directories.)

[//]: # ()
[//]: # (Tests now use separate fixtures for the shared CT phantom, each digital-phantom)

[//]: # (collection, and the response maps. The four archives extract into `tests/data/.cache/`,)

[//]: # (with archive fingerprinting, member integrity checks, and parallel-extraction)

[//]: # (locking. Reference loaders read the two canonical reference directories directly.)

[//]: # (Execution reports fingerprint these archives and reference CSVs.)

[//]: # ()
[//]: # (Superseded archives, top-level reference CSVs, and unpacked source copies have)

[//]: # (been removed. Extracted image files are disposable test caches and are not)

[//]: # (repository source assets. To inspect an archive independently, extract it into)

[//]: # (a temporary directory and read its included README and license. IBSI-SUV)

[//]: # (continues to extract into the ignored `tests/data/IBSI_SUV/` directory.)

## IBSI-SUV

`IBSI_SUV.zip` contains the IBSI-SUV v3.0.1 digital reference objects from
[oncoray/suv_computation](https://github.com/oncoray/suv_computation). These data use [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).


## References

- [IBSI official site](https://theibsi.github.io/)
- [CC BY-NC 3.0](https://creativecommons.org/licenses/by-nc/3.0/)
- [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/)
