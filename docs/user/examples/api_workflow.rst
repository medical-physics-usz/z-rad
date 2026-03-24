API Workflow Example
====================

This example outlines the recommended Python workflow for batch processing.

1. Load the image and mask into ``zrad.image.Image`` objects with aligned
   geometry.
2. Resample them with ``Preprocessing`` so image and mask share the intended
   voxel spacing.
3. Apply ``Filtering`` if the experiment requires a filtered representation.
4. Run ``Radiomics.extract_features()`` and collect ``features_`` for storage in
   a table or downstream analysis pipeline.
5. Keep the exact preprocessing, filtering, and discretization settings next to
   the extracted features so the run remains reproducible.

For parameter details, combine this example with the dedicated user-guide pages
and the API reference.
