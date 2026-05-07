from dataclasses import dataclass

import numpy as np

from ..exceptions import DataStructureError
from ..image import Image


@dataclass
class RoiMasks:
    """Morphological and intensity masks for one ROI processing stage."""

    morphological_mask: Image
    intensity_mask: Image


class RoiMaskValidator:
    """Validate ROI mask geometry for 2D, 2.5D, or 3D feature extraction."""

    def __init__(self, aggregation_dimension=None):
        self.aggregation_dimension = aggregation_dimension

    def get_params(self):
        """Return validation parameters mapped to their configured values."""
        return {
            'aggregation_dimension': self.aggregation_dimension,
        }

    def apply(self, mask):
        """Validate a mask and return a validated mask copy."""
        validated_mask = mask.copy()
        if self.aggregation_dimension is None:
            return validated_mask

        masked_array = validated_mask.array
        min_box_size = 3
        min_voxel_number_3d = 27
        min_voxel_number_2d = 9

        if self.aggregation_dimension == '3D':
            valid_coords = np.where(masked_array != 0)
            if len(valid_coords[0]) == 0:
                raise DataStructureError("No valid voxels in 3D array.")

            zmin, zmax = valid_coords[0].min(), valid_coords[0].max() + 1
            ymin, ymax = valid_coords[1].min(), valid_coords[1].max() + 1
            xmin, xmax = valid_coords[2].min(), valid_coords[2].max() + 1

            bbox_shape = (zmax - zmin, ymax - ymin, xmax - xmin)
            no_valid_voxels = len(valid_coords[0])
            if min(bbox_shape) < min_box_size:
                raise DataStructureError(f"3D bounding box dimension < {min_box_size}.")
            if no_valid_voxels < min_voxel_number_3d:
                raise DataStructureError(f"Valid voxel count < {min_voxel_number_3d}.")
        else:
            for z_idx in range(masked_array.shape[0]):
                slice_arr = masked_array[z_idx, :, :]
                if not np.any(slice_arr):
                    continue

                valid_coords = np.where(slice_arr != 0)
                no_valid_voxels = len(valid_coords[0])
                if no_valid_voxels == 0:
                    continue

                ymin, ymax = valid_coords[0].min(), valid_coords[0].max() + 1
                xmin, xmax = valid_coords[1].min(), valid_coords[1].max() + 1
                height = ymax - ymin
                width = xmax - xmin

                if min(height, width) < min_box_size or no_valid_voxels < min_voxel_number_2d:
                    slice_arr[:, :] = 0

            if not np.any(masked_array):
                raise DataStructureError(
                    "Not a single slice meets the minimum 2D/2.5D requirements. "
                    "Consider finer resampling or check the data."
                )
            validated_mask.array = masked_array

        return validated_mask
