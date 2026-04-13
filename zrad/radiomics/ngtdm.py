import numpy as np
from scipy.ndimage import convolve
from scipy.ndimage import distance_transform_cdt, label, generate_binary_structure, minimum
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import ConvexHull
from scipy.special import legendre
from scipy.stats import iqr
from skimage import measure

from ..exceptions import DataStructureError
from .base import BaseFeatureGroup

class NGTDM:
    """Neighbouring gray tone difference matrix features.

    Parameters
    ----------
    image : np.ndarray
        Discretized ROI image with integer gray levels and ``NaN`` values
        outside the ROI.
    slice_weight : bool, default=False
        If ``True``, weight slice-wise 2D features by the ROI voxel count of
        each slice.
    slice_median : bool, default=False
        If ``True`` and ``slice_weight`` is ``False``, aggregate 2D features by
        the median across slices instead of the mean.

    Notes
    -----
    NGTDM features summarize how much each gray level differs from the local
    neighborhood average and expose the IBSI coarseness, contrast, busyness,
    complexity, and strength metrics.
    """
    def __init__(self, image, slice_weight=False, slice_median=False):

        self.image = image  # Import image as (x, y, z) array
        self.lvl = int(np.nanmax(self.image) + 1)
        self.tot_no_of_roi_voxels = np.sum(~np.isnan(image))
        self.slice_weight = slice_weight
        self.slice_median = slice_median

        x_indices, y_indices, z_indices = np.where(~np.isnan(self.image))
        self.range_x = np.unique(x_indices)
        self.range_y = np.unique(y_indices)
        self.range_z = np.unique(z_indices)

        self.ngtd_2d_matrices = []
        self.ngtd_3d_matrix = None
        self.slice_no_of_roi_voxels = []

        self.coarseness = 0
        self.contrast = 0
        self.busyness = 0
        self.complexity = 0
        self.strength = 0

        self.coarsness_list = []
        self.contrast_list = []
        self.busyness_list = []
        self.complexity_list = []
        self.strength_list = []

    def calc_ngtd_3d_matrix(self):
        img = self.image
        # boolean mask of valid voxels
        valid = ~np.isnan(img)
        # fill NaNs with zero for the sum convolution
        img_filled = np.where(valid, img, 0.0)

        # 3×3×3 kernel of ones, with center zeroed
        kernel = np.ones((3, 3, 3), dtype=np.int8)
        kernel[1, 1, 1] = 0

        # sum of neighbor intensities
        neighbor_sum = convolve(img_filled, kernel, mode='constant', cval=0.0)
        # count of valid neighbors
        neighbor_count = convolve(valid.astype(np.int8), kernel, mode='constant', cval=0)

        # prepare output
        ngtdm = np.zeros((self.lvl, 2), dtype=np.float64)

        for lvl in range(self.lvl):
            # voxels at this grey‐level
            mask_lvl = (img == lvl)
            # require at least one valid neighbor
            mask_good = mask_lvl & (neighbor_count > 0)

            n_i = np.count_nonzero(mask_good)
            if n_i > 0:
                # mean neighbor value at each voxel
                mean_nb = neighbor_sum[mask_good] / neighbor_count[mask_good]
                # accumulate |i - μ_k|
                s_i = np.sum(np.abs(lvl - mean_nb))
            else:
                s_i = 0.0

            ngtdm[lvl, 0] = n_i
            ngtdm[lvl, 1] = s_i

        self.ngtd_3d_matrix = ngtdm

    def calc_ngtd_2d_matrices(self):
        # 3×3 kernel of ones with center zeroed
        kernel2d = np.ones((3,3), dtype=np.int8)
        kernel2d[1,1] = 0

        slice_matrices = []
        slice_voxel_counts = []

        for z in self.range_z:
            sl = self.image[:, :, z]
            valid = ~np.isnan(sl)
            n_vox = valid.sum()
            if n_vox == 0:
                continue

            # record how many ROI voxels in this slice
            slice_voxel_counts.append(int(n_vox))

            # replace NaNs with zero so they don't contribute to the sum
            filled = np.where(valid, sl, 0.0)

            # convolve to get per‑pixel neighbor sums and counts
            neighbor_sum   = convolve(filled,       kernel2d, mode='constant', cval=0.0)
            neighbor_count = convolve(valid.astype(np.int8),
                                     kernel2d, mode='constant', cval=0)

            # build the N_g × 2 matrix
            ngtdm_slice = np.zeros((self.lvl, 2), dtype=np.float64)
            for lvl in range(self.lvl):
                # voxels at this grey level with ≥1 valid neighbour
                mask = (sl == lvl) & (neighbor_count > 0)
                n_i = mask.sum()
                if n_i > 0:
                    mean_nb = neighbor_sum[mask] / neighbor_count[mask]
                    s_i = np.abs(lvl - mean_nb).sum()
                else:
                    s_i = 0.0

                ngtdm_slice[lvl, 0] = n_i
                ngtdm_slice[lvl, 1] = s_i

            slice_matrices.append(ngtdm_slice)

        # store results back into object
        self.slice_no_of_roi_voxels = slice_voxel_counts
        self.ngtd_2d_matrices       = np.array(slice_matrices)
    def calc_coarseness(self, matrix):
        num = np.sum(matrix[:, 0])
        denum = 0
        for i in range(matrix.shape[0]):
            denum += matrix[i, 0] * matrix[i, 1]
        if denum == 0:
            return 1_000_000  # IBSI 1 QCDE
        else:
            return num / denum

    def calc_contrast(self, matrix):
        n = np.sum(matrix[:, 0])
        if n == 0:
            raise DataStructureError(
                f" Denominator is zero in calc_contrast.")
        n_g = np.sum(matrix[:, 0] != 0)
        s_1 = 0
        s_2 = 0
        for i in range(matrix.shape[0]):
            s_2 += matrix[i, 1]
            for j in range(matrix.shape[0]):
                s_1 += (matrix[i, 0] * matrix[j, 0] * (i - j) ** 2) / (n ** 2)
        num = (s_1 * s_2)
        denum = (n_g * (n_g - 1) * np.sum(matrix[:, 0]))
        if denum == 0:
            return 0
        else:
            return num / denum

    def calc_busyness(self, matrix):
        n = np.sum(matrix[:, 0])
        if n == 0:
            raise DataStructureError(
                f" Denominator is zero in calc_busyness.")
        num = 0
        denum = 0
        for i in range(matrix.shape[0]):
            num += (matrix[i, 0] * matrix[i, 1]) / n
            for j in range(matrix.shape[0]):
                if matrix[i, 0] != 0 and matrix[j, 0] != 0:
                    denum += abs(i * matrix[i, 0] - j * matrix[j, 0]) / n
        if denum == 0:
            return 0
        else:
            return num / denum

    def calc_complexity(self, matrix):
        n = np.sum(matrix[:, 0])
        if n == 0:
            return 0

        sum_compl = 0.0
        # build the double‐sum
        for i in range(matrix.shape[0]):
            p_i, s_i = matrix[i, 0], matrix[i, 1]
            if p_i == 0:
                continue
            for j in range(matrix.shape[0]):
                p_j, s_j = matrix[j, 0], matrix[j, 1]
                if p_j == 0:
                    continue

                # per-IBSI numerator and denominator
                num = (p_i * s_i + p_j * s_j) * abs(i - j) / n
                den = (p_i + p_j) / n
                sum_compl += num / den

        # normalize by N_{v,c} = sum_i p_i
        N_vc = n
        if N_vc == 0:
            return 0
        return sum_compl / N_vc

    def calc_strength(self, matrix):
        n = np.sum(matrix[:, 0])
        if n == 0:
            raise DataStructureError(
                f" Denominator is zero in calc_strength.")
        num = 0
        denum = 0
        for i in range(matrix.shape[0]):
            denum += matrix[i, 1]
            for j in range(matrix.shape[0]):
                if matrix[i, 0] != 0 and matrix[j, 0] != 0:
                    num += ((matrix[i, 0] + matrix[j, 0]) * (i - j) ** 2) / n
        if denum == 0:
            return 0
        else:
            return num / denum

    def calc_2d_ngtdm_features(self):

        number_of_slices = self.ngtd_2d_matrices.shape[0]
        weights = []
        for i in range(number_of_slices):
            ngtdm_slice = self.ngtd_2d_matrices[i]
            weight = 1
            if self.slice_weight:
                if self.tot_no_of_roi_voxels == 0:
                    raise DataStructureError(
                        f" Denominator is zero in calc_2d_ngtdm_features.")
                weight = self.slice_no_of_roi_voxels[i] / self.tot_no_of_roi_voxels
            weights.append(weight)

            self.coarsness_list.append(self.calc_coarseness(ngtdm_slice))
            self.contrast_list.append(self.calc_contrast(ngtdm_slice))
            self.busyness_list.append(self.calc_busyness(ngtdm_slice))
            self.complexity_list.append(self.calc_complexity(ngtdm_slice))
            self.strength_list.append(self.calc_strength(ngtdm_slice))

        if self.slice_median and not self.slice_weight:
            self.coarseness = np.median(self.coarsness_list)
            self.contrast = np.median(self.contrast_list)
            self.busyness = np.median(self.busyness_list)
            self.complexity = np.median(self.complexity_list)
            self.strength = np.median(self.strength_list)

        elif not self.slice_median:
            self.coarseness = np.average(self.coarsness_list, weights=weights)
            self.contrast = np.average(self.contrast_list, weights=weights)
            self.busyness = np.average(self.busyness_list, weights=weights)
            self.complexity = np.average(self.complexity_list, weights=weights)
            self.strength = np.average(self.strength_list, weights=weights)
        else:
            print('Weighted median not supported. Aborted!')
            return

    def calc_2_5d_ngtdm_features(self):

        ngtdm_merged = np.sum(self.ngtd_2d_matrices, axis=0)

        self.coarseness = self.calc_coarseness(ngtdm_merged)
        self.contrast = self.calc_contrast(ngtdm_merged)
        self.busyness = self.calc_busyness(ngtdm_merged)
        self.complexity = self.calc_complexity(ngtdm_merged)
        self.strength = self.calc_strength(ngtdm_merged)

    def calc_3d_ngtdm_features(self):

        ngtdm = self.ngtd_3d_matrix

        self.coarseness = self.calc_coarseness(ngtdm)
        self.contrast = self.calc_contrast(ngtdm)
        self.busyness = self.calc_busyness(ngtdm)
        self.complexity = self.calc_complexity(ngtdm)
        self.strength = self.calc_strength(ngtdm)

from .texture_aggregation import format_texture_feature_names


NGTDM_FEATURE_NAMES = (
    'ngt_coarseness',
    'ngt_contrast',
    'ngt_busyness',
    'ngt_complexity',
    'ngt_strength',
)


class NGTDMFeatureGroup(BaseFeatureGroup):
    family = 'ngtdm'
    requirements = frozenset({'analysis_masks', 'discretized_intensity_image'})

    def supports(self, context):
        return True

    def output_names(self, context):
        return format_texture_feature_names(NGTDM_FEATURE_NAMES, context.aggr_dim)

    def feature_aliases(self, context):
        output_names = self.output_names(context)
        aliases = {name: name for name in output_names}
        aliases.update(dict(zip(NGTDM_FEATURE_NAMES, output_names)))
        return aliases

    def calculate(self, context, prepared_data):
        ngtdm = NGTDM(
            image=prepared_data.require_discretized_intensity_image().array.T,
            slice_weight=context.slice_weighting,
            slice_median=context.slice_median,
        )
        if context.aggr_dim == '3D':
            ngtdm.calc_ngtd_3d_matrix()
            ngtdm.calc_3d_ngtdm_features()
        elif context.aggr_dim == '2.5D':
            ngtdm.calc_ngtd_2d_matrices()
            ngtdm.calc_2_5d_ngtdm_features()
        else:
            ngtdm.calc_ngtd_2d_matrices()
            ngtdm.calc_2d_ngtdm_features()

        values = [
            ngtdm.coarseness,
            ngtdm.contrast,
            ngtdm.busyness,
            ngtdm.complexity,
            ngtdm.strength,
        ]
        return dict(zip(self.output_names(context), values))
