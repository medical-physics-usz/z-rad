# IBSI Test Data

This directory contains a collection of test data files from the **Image Biomarker Standardisation Initiative (IBSI)**. Different subsets of these data are governed by different open licenses. Below is an overview of each data component and its associated license terms.

---

## Data Components & Licenses

1. **IBSI 1 CT Radiomics Phantom**  
   - **License:** Creative Commons Attribution-NonCommercial 3.0 Unported (CC BY-NC 3.0)

2. **IBSI 1 Digital Phantom**  
   - **Source:** https://github.com/theibsi/data_sets/tree/master/ibsi_1_digital_phantom
   - **Files:** `ibsi_1_digital_phantom/image.nii.gz`, `mask.nii.gz`, and `LICENSE.md`
   - **Benchmark settings:** no interpolation or resegmentation; six-bin discretization preserves the original grey levels; all six texture aggregation modes.
   - **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)

3. **IBSI 2 CT Radiomics Phantom**  
   - **License:** Creative Commons Attribution-NonCommercial 3.0 Unported (CC BY-NC 3.0)

4. **IBSI 2 Digital Phantom**  
   - **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)

5. **IBSI 2 Response Maps**  
   - **License:** CC0 1.0 Universal (Public Domain Dedication)

6. **IBSI 2 Reference Feature Values**  
   - **License:** CC0 1.0 Universal (Public Domain Dedication)

7. **IBSI-SUV v3.0.1 Digital Reference Objects**
   - **Source:** https://github.com/oncoray/suv_computation
   - **Authors:** Michael Vácha, Alex Zwanenburg, and the Image Biomarker Standardisation Initiative
   - **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)


---
    
## Contact & References

- **IBSI Official Site:**  
  https://theibsi.github.io

- **Creative Commons Licenses:**  
  - CC BY-NC 3.0: https://creativecommons.org/licenses/by-nc/3.0/  
  - CC BY 4.0: https://creativecommons.org/licenses/by/4.0/  
  - CC0 1.0: https://creativecommons.org/publicdomain/zero/1.0/

If you have any questions regarding these datasets or their licensing, please refer to the License files in this directory or contact the repository maintainers.

### Digital phantom provenance

Downloaded unchanged from the official `theibsi/data_sets` repository on 2026-09-08.
SHA-256 checksums of the bundled NIfTI files:

- `image.nii.gz`: `83773ac2a288aa93cf819a98eaf301c18d35d5876182ec9c18733184e2b7a83a`
- `mask.nii.gz`: `3032d340944b577b83bc559ea7ae9a6034b84633a840eff39570ec17aa1d6281`
