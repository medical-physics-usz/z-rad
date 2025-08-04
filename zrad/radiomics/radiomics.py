import sys

import numpy as np

from .radiomics_definitions import MorphologicalFeatures, LocalIntensityFeatures, IntensityBasedStatFeatures, GLCM, \
    GLRLM_GLSZM_GLDZM_NGLDM, NGTDM, IntensityVolumeHistogramFeatures
from ..exceptions import DataStructureError
from ..image import Image
from ..toolbox_logic import handle_uncaught_exception

sys.excepthook = handle_uncaught_exception


class Radiomics:

    def __init__(self,
                 aggr_dim='3D', aggr_method='AVER',
                 intensity_range=None, outlier_range=None,
                 number_of_bins=None, bin_size=None,
                 calc_ivh_features=False,
                 ivh_number_of_bins=None, ivh_bin_size=None,
                 calc_morph_moran_i_and_geary_c_features = False,
                 slice_weighting=False, slice_median=False):
        self.patient_morphological_mask = None
        self.patient_intensity_mask = None

        if slice_weighting and slice_median:
            raise ValueError('Only one slice median averaging is not supported with weighting strategy.')

        else:
            self.slice_weighting = slice_weighting
            self.slice_median = slice_median

        self.calc_intensity_mask = False
        if intensity_range is not None:
            self.calc_intensity_mask = True
            self.intensity_range = intensity_range
            self.discret_min_val = intensity_range[0]
            self.discret_max_val = intensity_range[1]

        self.calc_outlier_mask = False
        if str(outlier_range).strip().replace('.', '').isdigit():
            self.calc_outlier_mask = True
            self.outlier_range = outlier_range

        self.calc_discr_bin_number = False
        self.calc_discr_bin_size = False
        self.calc_morph_moran_i_and_geary_c_features = calc_morph_moran_i_and_geary_c_features
        if number_of_bins is not None:
            self.calc_discr_bin_number = True
            self.bin_number = number_of_bins

        if bin_size is not None:
            self.calc_discr_bin_size = True
            self.bin_size = bin_size

        self.calc_ivh_features = calc_ivh_features
        self.calc_discr_ivh_bin_number = False
        self.calc_discr_ivh_bin_size = False

        if ivh_number_of_bins is not None:
            self.calc_discr_ivh_bin_number = True
            self.ivh_bin_number = ivh_number_of_bins

        if ivh_bin_size is not None:
            self.calc_discr_ivh_bin_size = True
            self.ivh_bin_size = ivh_bin_size

        if aggr_dim in ['2D', '2.5D', '3D']:
            self.aggr_dim = aggr_dim
        else:
            raise ValueError(f"Wrong aggregation dim {aggr_dim}. Available '2D', '2.5D', and '3D'.")

        if aggr_method in ['MERG', 'AVER', 'SLICE_MERG', 'DIR_MERG']:
            self.aggr_method = aggr_method
        else:
            raise ValueError(f"Wrong aggregation dim {aggr_method}. "
                             "Available 'MERG', 'AVER', 'SLICE_MERG', and 'DIR_MERG'.")

        self.patient_folder = None
        self.patient_number = None

    def extract_features(self, image, mask, filtered_image=None):
        slice_2d = True if image.shape[2] == 1 else False
        
        columns = [
            'morph_volume', 'morph_vol_approx', 'morph_area_mesh', 'morph_av', 'morph_comp_1', 'morph_comp_2',
            'morph_sph_dispr', 'morph_sphericity', 'morph_asphericity', 'morph_com', 'morph_diam', 'morph_pca_maj_axis',
            'morph_pca_min_axis', 'morph_pca_least_axis', 'morph_pca_elongation', 'morph_pca_flatness',
            'morph_vol_dens_aabb', 'morph_area_dens_aabb', 'morph_vol_dens_aee', 'morph_area_dens_aee',
            'morph_vol_dens_conv_hull', 'morph_area_dens_conv_hull', 'morph_integ_int',
            'loc_peak_loc', 'loc_peak_glob',
            'stat_mean', 'stat_var', 'stat_skew', 'stat_kurt', 'stat_median', 'stat_min', 'stat_p10',
            'stat_p90', 'stat_max', 'stat_iqr', 'stat_range', 'stat_mad', 'stat_rmad', 'stat_medad', 'stat_cov',
            'stat_qcod', 'stat_energy', 'stat_rms',
            'ih_mean', 'ih_var', 'ih_skew', 'ih_kurt', 'ih_median', 'ih_min', 'ih_p10', 'ih_p90', 'ih_max', 'ih_mode',
            'ih_iqr', 'ih_range', 'ih_mad', 'ih_rmad', 'ih_medad', 'ih_cov', 'ih_qcod', 'ih_entropy', 'ih_uniformity',
            'ih_max_grad', 'ih_max_grad_g', 'ih_min_grad', 'ih_min_grad_g',
            'cm_joint_max', 'cm_joint_avg', 'cm_joint_var', 'cm_joint_entr', 'cm_diff_avg', 'cm_diff_var',
            'cm_diff_entr', 'cm_sum_avg', 'cm_sum_var', 'cm_sum_entr', 'cm_energy', 'cm_contrast', 'cm_dissimilarity',
            'cm_inv_diff', 'cm_inv_diff_norm', 'cm_inv_diff_mom', 'cm_inv_diff_mom_norm', 'cm_inv_var', 'cm_corr',
            'cm_auto_corr', 'cm_clust_tend', 'cm_clust_shade', 'cm_clust_prom', 'cm_info_corr1', 'cm_info_corr2',
            'rlm_sre', 'rlm_lre', 'rlm_lgre', 'rlm_hgre', 'rlm_srlge', 'rlm_srhge', 'rlm_lrlge', 'rlm_lrhge',
            'rlm_glnu', 'rlm_glnu_norm', 'rlm_rlnu', 'rlm_rlnu_norm', 'rlm_r_perc', 'rlm_gl_var', 'rlm_rl_var',
            'rlm_rl_entr',
            'szm_sze', 'szm_lze', 'szm_lgze', 'szm_hgze', 'szm_szlge', 'szm_szhge', 'szm_lzlge', 'szm_lzhge',
            'szm_glnu', 'szm_glnu_norm', 'szm_zsnu', 'szm_zsnu_norm', 'szm_z_perc', 'szm_gl_var', 'szm_zs_var',
            'szm_zs_entr',
            'dzm_sde', 'dzm_lde', 'dzm_lgze', 'dzm_hgze', 'dzm_sdlge', 'dzm_sdhge', 'dzm_ldlge', 'dzm_ldhge',
            'dzm_glnu', 'dzm_glnu_norm', 'dzm_zdnu', 'dzm_zdnu_norm', 'dzm_z_perc', 'dzm_gl_var', 'dzm_zd_var',
            'dzm_zd_entr',
            'ngt_coarseness', 'ngt_contrast', 'ngt_busyness', 'ngt_complexity', 'ngt_strength',
            'ngl_lde', 'ngl_hde', 'ngl_lgce', 'ngl_hgce', 'ngl_ldlge', 'ngl_ldhge', 'ngl_hdlge', 'ngl_hdhge',
            'ngl_glnu', 'ngl_glnu_norm', 'ngl_dcnu', 'ngl_dcnu_norm', 'ngl_dc_perc', 'ngl_gl_var', 'ngl_dc_var',
            'ngl_dc_entr', 'ngl_dc_energy']

        if slice_2d:
            self.columns = columns[23:]
        elif not slice_2d:
            self.columns = columns


        self.pat_binned_masked_image = {}
        self.patient_morf_features_list = []
        self.morph_moran_i_and_geary_c_features = {}
        self.patient_local_intensity_features_list = []
        self.intensity_features_list = []
        self.ivh_features = {}
        self.discr_intensity_features_list = []
        self.glcm_features_list = []
        self.glrlm_features_list = []
        self.glszm_features_list = []
        self.gldzm_features_list = []
        self.ngtdm_features_list = []
        self.ngldm_features_list = []

        self.orig_patient_image = image
        if filtered_image:
            self.patient_image = filtered_image
        else:
            self.patient_image = image

        # Extract non-discretized features
        if slice_2d:
            mask_validated = mask
        else:
            mask_validated = self._validate_mask(mask, '3D')
        self.patient_morphological_mask = mask_validated.copy()
        self.patient_morphological_mask.array = self.patient_morphological_mask.array.astype(np.int8)
        self.patient_intensity_mask = mask_validated.copy()
        self.patient_intensity_mask.array = np.where(self.patient_intensity_mask.array > 0, self.patient_image.array, np.nan)
        self._outlier_removal_and_intensity_truncation()
        self._calc_mask_intensity_features()
        if not slice_2d:
            self._calc_mask_morphological_features()
        if self.calc_morph_moran_i_and_geary_c_features:
            self._calc_morph_moran_i_and_geary_c_features()


        # Extract discretized features
        if self.aggr_dim != '3D':
            if slice_2d:
                mask_validated = mask
            else:
                mask_validated = self._validate_mask(mask, self.aggr_dim)
            self.patient_morphological_mask = mask_validated.copy()
            self.patient_morphological_mask.array = self.patient_morphological_mask.array.astype(np.int8)
            self.patient_intensity_mask = mask_validated.copy()
            self.patient_intensity_mask.array = np.where(self.patient_intensity_mask.array > 0, self.patient_image.array, np.nan)
            self._outlier_removal_and_intensity_truncation()
        if self.calc_ivh_features:
            self._calc_ivh_features()
        self._calc_discretized_intensity_features()
        self._calc_texture_features()

        # compile features
        all_features_list = [self.patient_local_intensity_features_list,
                             self.intensity_features_list, self.discr_intensity_features_list,
                             self.glcm_features_list,
                             self.glrlm_features_list, self.glszm_features_list,
                             self.gldzm_features_list, self.ngtdm_features_list, self.ngldm_features_list]
        if not slice_2d:
            all_features_list = [self.patient_morf_features_list] + all_features_list
        if self.calc_morph_moran_i_and_geary_c_features:
            columns += list(self.morph_moran_i_and_geary_c_features.keys())
            all_features_list += [[list(self.morph_moran_i_and_geary_c_features.values())]]
        all_features_list_flat = [item for sublist in all_features_list for item in sublist[0]]
        self.new_columns = []
        el_aggr_dim = '2_5D' if self.aggr_dim == '2.5D' else self.aggr_dim

        aggr_method_map = {'AVER': 'avg', 'DIR_MERG': 'avg', 'SLICE_MERG': 'comb', 'MERG': 'comb'}

        for el in self.columns:
            if el.startswith(('cm', 'rlm')):
                el_aggr_method = aggr_method_map.get(self.aggr_method, '')
                self.new_columns.append(f'{el}_{el_aggr_dim}_{el_aggr_method}')
            elif el.startswith(('szm', 'dzm', 'ngt', 'ngl')):
                self.new_columns.append(f'{el}_{el_aggr_dim}')
            else:
                self.new_columns.append(el)

        self.features_ = dict(zip(self.new_columns, all_features_list_flat))
        self.features_ = self.features_ | self.ivh_features | self.morph_moran_i_and_geary_c_features

    def _validate_mask(self, mask, aggr_dim):
        """
        Validates the intensity mask for a patient by checking bounding box dimensions
        and number of valid voxels. Skips or discards slices that do not meet criteria.
        """
        # Calculate the bounding box around the intensity mask and determine its shape.
        masked_array = mask.array

        # Define the minimum size and voxel count requirements for validation.
        min_box_size = 3
        min_voxel_number_3d = 27
        min_voxel_number_2d = 9

        # Check the bounding box size and the number of voxels based on the aggregation dimension.
        if aggr_dim == '3D':
            # Find all nonzero voxel coordinates
            valid_coords = np.where(masked_array != 0)
            if len(valid_coords[0]) == 0:
                raise DataStructureError("No valid voxels in 3D array.")

            # Compute bounding box from min to max across each dimension
            zmin, zmax = valid_coords[0].min(), valid_coords[0].max() + 1
            ymin, ymax = valid_coords[1].min(), valid_coords[1].max() + 1
            xmin, xmax = valid_coords[2].min(), valid_coords[2].max() + 1

            bbox_shape = (zmax - zmin, ymax - ymin, xmax - xmin)
            no_valid_voxels = len(valid_coords[0])

            # Validation checks for 3D
            if min(bbox_shape) < min_box_size:
                raise DataStructureError(f"3D bounding box dimension < {min_box_size}.")
            if no_valid_voxels < min_voxel_number_3d:
                raise DataStructureError(f"Valid voxel count < {min_voxel_number_3d}.")

        else:
            # 2D or 2.5D case: check each slice individually
            n_slices = masked_array.shape[0]
            for z in range(n_slices):
                slice_arr = masked_array[z, :, :]
                # If the slice is all zero, skip it
                if not np.any(slice_arr):
                    continue

                # Find all nonzero coordinates in this slice
                valid_coords = np.where(slice_arr != 0)
                no_valid_voxels = len(valid_coords[0])
                if no_valid_voxels == 0:
                    # It's effectively empty
                    continue

                ymin, ymax = valid_coords[0].min(), valid_coords[0].max() + 1
                xmin, xmax = valid_coords[1].min(), valid_coords[1].max() + 1
                height = ymax - ymin
                width = xmax - xmin

                # Validation checks for each slice
                if min(height, width) < min_box_size or no_valid_voxels < min_voxel_number_2d:
                    # Mark this entire slice as invalid by setting it to 0
                    slice_arr[:, :] = 0

            # Check if all slices are now invalid (all zeros)
            if not np.any(masked_array):
                raise DataStructureError(
                    "Not a single slice meets the minimum 2D/2.5D requirements. "
                    "Consider finer resampling or check the data."
                )

            # Assign the possibly updated mask back (already in-place, but for clarity)
            mask.array = masked_array

        return mask

    def _outlier_removal_and_intensity_truncation(self):
        if self.calc_intensity_mask:
            intensity_range_mask = np.where((self.orig_patient_image.array <= self.intensity_range[1])
                                            & (self.orig_patient_image.array >= self.intensity_range[0]),
                                            1, 0)
            self.patient_intensity_mask = Image(array=np.where((intensity_range_mask > 0)
                                                               & (~np.isnan(self.patient_intensity_mask.array)),
                                                               self.patient_intensity_mask.array, np.nan),
                                                origin=self.patient_intensity_mask.origin,
                                                spacing=self.patient_intensity_mask.spacing,
                                                direction=self.patient_intensity_mask.direction,
                                                shape=self.patient_intensity_mask.shape)
        if self.calc_outlier_mask:
            flattened_image = np.where(self.patient_morphological_mask.array > 0,
                                       self.orig_patient_image.array, np.nan).ravel()
            valid_values = flattened_image[~np.isnan(flattened_image)]
            mean = np.mean(valid_values)
            std = np.std(valid_values)
            outlier_mask = np.where((self.orig_patient_image.array <= mean + self.outlier_range * std)
                                    & (self.orig_patient_image.array >= mean - self.outlier_range * std)
                                    & (~np.isnan(self.patient_intensity_mask.array)),
                                    1, 0)

            self.patient_intensity_mask = Image(array=np.where((outlier_mask > 0)
                                                               & (~np.isnan(self.patient_intensity_mask.array)),
                                                               self.patient_intensity_mask.array, np.nan),
                                                origin=self.patient_intensity_mask.origin,
                                                spacing=self.patient_intensity_mask.spacing,
                                                direction=self.patient_intensity_mask.direction,
                                                shape=self.patient_intensity_mask.shape)

    def _bin_size_discr(self, image, min_val, bin_size):

        return Image(array=np.floor((image.array - min_val) / bin_size) + 1,
                     origin=image.origin,
                     spacing=image.spacing,
                     direction=image.direction,
                     shape=image.shape)

    def _bin_number_discr(self, image, bin_number):

        return Image(array=np.where(image.array != np.nanmax(image.array),
                                    np.floor(bin_number * (image.array - np.nanmin(image.array))
                                             / (np.nanmax(image.array) - np.nanmin(image.array))) + 1, bin_number),
                     origin=image.origin,
                     spacing=image.spacing,
                     direction=image.direction,
                     shape=image.shape)

    def _ivh_bin_size_discr(self, image, min_val, bin_size):

        return Image(array=min_val + ((np.floor((image.array - min_val) / bin_size) + 1) - 0.5) * bin_size,
                     origin=image.origin,
                     spacing=image.spacing,
                     direction=image.direction,
                     shape=image.shape)

    def _ivh_bin_number_discr(self, image, bin_number):

        return Image(array=np.where(image.array != np.nanmax(image.array),
                                    np.floor(bin_number * (image.array - np.nanmin(image.array))
                                             / (np.nanmax(image.array) - np.nanmin(image.array))) + 1, bin_number),
                     origin=image.origin,
                     spacing=image.spacing,
                     direction=image.direction,
                     shape=image.shape)

    def _calc_mask_intensity_features(self):

        local_intensity_features = LocalIntensityFeatures(self.patient_image.array,
                                                          self.patient_intensity_mask.array,
                                                          (self.patient_image.spacing[::-1]))

        local_intensity_features.calc_local_intensity_peak()
        local_intensity_features.calc_global_intensity_peak()
        self.local_intensity_features = [local_intensity_features.local_intensity_peak,
                                         local_intensity_features.global_intensity_peak]
        self.patient_local_intensity_features_list.append(self.local_intensity_features)

        intensity_features = IntensityBasedStatFeatures()
        intensity_features.calc_mean_intensity(self.patient_intensity_mask.array)
        intensity_features.calc_intensity_variance(self.patient_intensity_mask.array)
        intensity_features.calc_intensity_skewness(self.patient_intensity_mask.array)
        intensity_features.calc_intensity_kurtosis(self.patient_intensity_mask.array)
        intensity_features.calc_median_intensity(self.patient_intensity_mask.array)
        intensity_features.calc_min_intensity(self.patient_intensity_mask.array)
        intensity_features.calc_intensity_10th_percentile(self.patient_intensity_mask.array)
        intensity_features.calc_intensity_90th_percentile(self.patient_intensity_mask.array)
        intensity_features.calc_max_intensity(self.patient_intensity_mask.array)
        intensity_features.calc_intensity_iqr(self.patient_intensity_mask.array)
        intensity_features.calc_intensity_range(self.patient_intensity_mask.array)
        intensity_features.calc_intensity_based_mean_abs_deviation(self.patient_intensity_mask.array)
        intensity_features.calc_intensity_based_robust_mean_abs_deviation(self.patient_intensity_mask.array)
        intensity_features.calc_intensity_based_median_abs_deviation(self.patient_intensity_mask.array)
        intensity_features.calc_intensity_based_variation_coef(self.patient_intensity_mask.array)
        intensity_features.calc_intensity_based_quartile_coef_dispersion(self.patient_intensity_mask.array)
        intensity_features.calc_intensity_based_energy(self.patient_intensity_mask.array)
        intensity_features.calc_root_mean_square_intensity(self.patient_intensity_mask.array)

        self.intensity_based_features = [intensity_features.mean_intensity,
                                         intensity_features.intensity_variance,
                                         intensity_features.intensity_skewness,
                                         intensity_features.intensity_kurtosis,
                                         intensity_features.median_intensity,
                                         intensity_features.min_intensity,
                                         intensity_features.intensity_10th_percentile,
                                         intensity_features.intensity_90th_percentile,
                                         intensity_features.max_intensity,
                                         intensity_features.intensity_iqr,
                                         intensity_features.intensity_range,
                                         intensity_features.intensity_based_mean_abs_deviation,
                                         intensity_features.intensity_based_robust_mean_abs_deviation,
                                         intensity_features.intensity_based_median_abs_deviation,
                                         intensity_features.intensity_based_variation_coef,
                                         intensity_features.intensity_based_quartile_coef_dispersion,
                                         intensity_features.intensity_based_energy,
                                         intensity_features.root_mean_square_intensity]

        self.intensity_features_list.append(self.intensity_based_features)

    def _calc_ivh_features(self):
        self.ihv_patient_intensity_mask = self.patient_intensity_mask.copy()

        if self.calc_discr_ivh_bin_size:

            if self.calc_intensity_mask:

                self.ihv_patient_intensity_mask = Image(array=self.discret_min_val + (self._bin_size_discr(self.ihv_patient_intensity_mask,
                                                                           self.discret_min_val,
                                                                           self.ivh_bin_size).array - 0.5) * self.ivh_bin_size,
                                                        origin=self.patient_intensity_mask.origin,
                                                        spacing=self.patient_intensity_mask.spacing,
                                                        direction=self.patient_intensity_mask.direction,
                                                        shape=self.patient_intensity_mask.shape)
                ivh_features = IntensityVolumeHistogramFeatures(self.ihv_patient_intensity_mask.array,
                                                                self.discret_min_val + 0.5 * self.ivh_bin_size,
                                                                self.discret_max_val - 0.5 * self.ivh_bin_size,
                                                                self.ivh_bin_size)
            else:
                self.ihv_patient_intensity_mask = Image(
                    array=np.nanmin(self.ihv_patient_intensity_mask.array) + (self._bin_size_discr(self.ihv_patient_intensity_mask,
                                                                       np.nanmin(self.ihv_patient_intensity_mask.array),
                                                                       self.ivh_bin_size).array - 0.5) * self.ivh_bin_size,
                    origin=self.patient_intensity_mask.origin,
                    spacing=self.patient_intensity_mask.spacing,
                    direction=self.patient_intensity_mask.direction,
                    shape=self.patient_intensity_mask.shape)
                ivh_features = IntensityVolumeHistogramFeatures(self.ihv_patient_intensity_mask.array, np.nanmin(
                    self.ihv_patient_intensity_mask.array) + 0.5 * self.ivh_bin_size,
                                                                np.nanmax(
                                                                    self.ihv_patient_intensity_mask.array) - 0.5 * self.ivh_bin_size,
                                                                self.ivh_bin_size)
        if self.calc_discr_ivh_bin_number:
            self.ihv_patient_intensity_mask = self._bin_number_discr(self.ihv_patient_intensity_mask, self.ivh_bin_number)
            ivh_features = IntensityVolumeHistogramFeatures(self.ihv_patient_intensity_mask.array,
                                                            np.nanmin(self.ihv_patient_intensity_mask.array),
                                                            np.nanmax(self.ihv_patient_intensity_mask.array))
        if not self.calc_discr_ivh_bin_size and not self.calc_discr_ivh_bin_number:
            ivh_features = IntensityVolumeHistogramFeatures(self.ihv_patient_intensity_mask.array,
                                                            np.nanmin(self.ihv_patient_intensity_mask.array),
                                                            np.nanmax(self.ihv_patient_intensity_mask.array))

        self.ivh_features = {'ivh_v10': ivh_features.calc_volume_at_intensity_fraction(10),
                             'ivh_v90': ivh_features.calc_volume_at_intensity_fraction(90),
                             'ivh_i10': ivh_features.calc_intensity_at_volume_fraction(10),
                             'ivh_i90': ivh_features.calc_intensity_at_volume_fraction(90),
                             'ivh_diff_v10_v90': ivh_features.calc_volume_fraction_diff_intensity_fractions(),
                             'ivh_diff_i10_i90': ivh_features.calc_intensity_fraction_diff_volume_fractions()}

    def _calc_discretized_intensity_features(self):
        if self.calc_discr_bin_size:
            if self.calc_intensity_mask:
                self.patient_intensity_mask = self._bin_size_discr(self.patient_intensity_mask,
                                                                       self.discret_min_val,
                                                                       self.bin_size)
            else:
                self.patient_intensity_mask = self._bin_size_discr(self.patient_intensity_mask,
                                                                   np.nanmin(self.patient_intensity_mask.array),
                                                                   self.bin_size)
        if self.calc_discr_bin_number:
            self.patient_intensity_mask = self._bin_number_discr(self.patient_intensity_mask, self.bin_number)

        discr_intensity_features = IntensityBasedStatFeatures()
        discr_intensity_features.calc_mean_intensity(self.patient_intensity_mask.array)
        discr_intensity_features.calc_intensity_variance(self.patient_intensity_mask.array)
        discr_intensity_features.calc_intensity_skewness(self.patient_intensity_mask.array)
        discr_intensity_features.calc_intensity_kurtosis(self.patient_intensity_mask.array)
        discr_intensity_features.calc_median_intensity(self.patient_intensity_mask.array)
        discr_intensity_features.calc_min_intensity(self.patient_intensity_mask.array)
        discr_intensity_features.calc_intensity_10th_percentile(self.patient_intensity_mask.array)
        discr_intensity_features.calc_intensity_90th_percentile(self.patient_intensity_mask.array)
        discr_intensity_features.calc_max_intensity(self.patient_intensity_mask.array)
        discr_intensity_features.calc_discretised_intensity_mode(self.patient_intensity_mask.array)
        discr_intensity_features.calc_intensity_iqr(self.patient_intensity_mask.array)
        discr_intensity_features.calc_intensity_range(self.patient_intensity_mask.array)
        discr_intensity_features.calc_intensity_based_mean_abs_deviation(self.patient_intensity_mask.array)
        discr_intensity_features.calc_intensity_based_robust_mean_abs_deviation(self.patient_intensity_mask.array)
        discr_intensity_features.calc_intensity_based_median_abs_deviation(self.patient_intensity_mask.array)
        discr_intensity_features.calc_intensity_based_variation_coef(self.patient_intensity_mask.array)
        discr_intensity_features.calc_intensity_based_quartile_coef_dispersion(self.patient_intensity_mask.array)
        discr_intensity_features.calc_discretised_intensity_entropy(self.patient_intensity_mask.array)
        discr_intensity_features.calc_discretised_intensity_uniformity(self.patient_intensity_mask.array)
        discr_intensity_features.calc_max_hist_gradient(self.patient_intensity_mask.array)  # 3.4.20
        discr_intensity_features.calc_max_hist_gradient_intensity(self.patient_intensity_mask.array)  # 3.4.21
        discr_intensity_features.calc_min_hist_gradient(self.patient_intensity_mask.array)  # 3.4.22
        discr_intensity_features.calc_min_hist_gradient_intensity(self.patient_intensity_mask.array)  # 3.4.23

        self.discr_intensity_based_features = [discr_intensity_features.mean_intensity,
                                               discr_intensity_features.intensity_variance,
                                               discr_intensity_features.intensity_skewness,
                                               discr_intensity_features.intensity_kurtosis,
                                               discr_intensity_features.median_intensity,
                                               discr_intensity_features.min_intensity,
                                               discr_intensity_features.intensity_10th_percentile,
                                               discr_intensity_features.intensity_90th_percentile,
                                               discr_intensity_features.max_intensity,
                                               discr_intensity_features.intensity_hist_mode,
                                               discr_intensity_features.intensity_iqr,
                                               discr_intensity_features.intensity_range,
                                               discr_intensity_features.intensity_based_mean_abs_deviation,
                                               discr_intensity_features.intensity_based_robust_mean_abs_deviation,
                                               discr_intensity_features.intensity_based_median_abs_deviation,
                                               discr_intensity_features.intensity_based_variation_coef,
                                               discr_intensity_features.intensity_based_quartile_coef_dispersion,
                                               discr_intensity_features.discret_intensity_entropy,
                                               discr_intensity_features.discret_intensity_uniformity,
                                               discr_intensity_features.max_hist_gradient,
                                               discr_intensity_features.max_hist_gradient_intensity,
                                               discr_intensity_features.min_hist_gradient,
                                               discr_intensity_features.min_hist_gradient_intensity]

        self.discr_intensity_features_list.append(self.discr_intensity_based_features)

    def _calc_texture_features(self):
        glcm = GLCM(image=self.patient_intensity_mask.array.T, slice_weight=self.slice_weighting,
                    slice_median=self.slice_median)
        if self.aggr_dim == '3D':
            glcm.calc_glc_3d_matrix()
            if self.aggr_method == 'AVER':
                glcm.calc_3d_averaged_glcm_features()
            elif self.aggr_method == 'MERG':
                glcm.calc_3d_merged_glcm_features()

        elif self.aggr_dim == '2.5D' or self.aggr_dim == '2D':
            glcm.calc_glc_2d_matrices()
            if self.aggr_method == 'DIR_MERG':
                glcm.calc_2_5d_direction_merged_glcm_features()
            elif self.aggr_method == 'MERG':
                glcm.calc_2_5d_merged_glcm_features()
            elif self.aggr_method == 'AVER':
                glcm.calc_2d_averaged_glcm_features()
            elif self.aggr_method == 'SLICE_MERG':
                glcm.calc_2d_slice_merged_glcm_features()

        self.glcm_features = [glcm.joint_max,
                              glcm.joint_average,
                              glcm.joint_var,
                              glcm.joint_entropy,
                              glcm.dif_average,
                              glcm.dif_var,
                              glcm.dif_entropy,
                              glcm.sum_average,
                              glcm.sum_var,
                              glcm.sum_entropy,
                              glcm.ang_second_moment,
                              glcm.contrast,
                              glcm.dissimilarity,
                              glcm.inv_diff,
                              glcm.norm_inv_diff,
                              glcm.inv_diff_moment,
                              glcm.norm_inv_diff_moment,
                              glcm.inv_variance,
                              glcm.cor,
                              glcm.autocor,
                              glcm.cluster_tendency,
                              glcm.cluster_shade,
                              glcm.cluster_prominence,
                              glcm.inf_cor_1,
                              glcm.inf_cor_2]
        self.glcm_features_list.append(self.glcm_features)
        glrlm = GLRLM_GLSZM_GLDZM_NGLDM(image=self.patient_intensity_mask.array.T, slice_weight=self.slice_weighting,
                                        slice_median=self.slice_median)
        if self.aggr_dim == '3D':

            glrlm.calc_glrl_3d_matrix()
            if self.aggr_method == 'AVER':
                glrlm.calc_3d_averaged_glrlm_features()
            elif self.aggr_method == 'MERG':
                glrlm.calc_3d_merged_glrlm_features()

        elif self.aggr_dim == '2.5D' or self.aggr_dim == '2D':
            glrlm.calc_glrl_2d_matrices()

            if self.aggr_method == 'DIR_MERG':
                glrlm.calc_2_5d_direction_merged_glrlm_features()
            elif self.aggr_method == 'MERG':
                glrlm.calc_2_5d_merged_glrlm_features()
            elif self.aggr_method == 'AVER':
                glrlm.calc_2d_averaged_glrlm_features()
            elif self.aggr_method == 'SLICE_MERG':
                glrlm.calc_2d_slice_merged_glrlm_features()

        self.glrlm_features = [glrlm.short_runs_emphasis,
                               glrlm.long_runs_emphasis,
                               glrlm.low_grey_level_run_emphasis,
                               glrlm.high_gr_lvl_emphasis,
                               glrlm.short_low_gr_lvl_emphasis,
                               glrlm.short_high_gr_lvl_emphasis,
                               glrlm.long_low_gr_lvl_emphasis,
                               glrlm.long_high_gr_lvl_emphasis,
                               glrlm.non_uniformity,
                               glrlm.norm_non_uniformity,
                               glrlm.length_non_uniformity,
                               glrlm.norm_length_non_uniformity,
                               glrlm.percentage,
                               glrlm.gr_lvl_var,
                               glrlm.length_var,
                               glrlm.entropy]
        self.glrlm_features_list.append(self.glrlm_features)

        glszm_gldzm = GLRLM_GLSZM_GLDZM_NGLDM(image=self.patient_intensity_mask.array.T,
                                              slice_weight=self.slice_weighting, slice_median=self.slice_median)
        if self.aggr_dim == '3D':
            glszm_gldzm.calc_glsz_gldz_3d_matrices(self.patient_morphological_mask.array.T)
            glszm_gldzm.calc_3d_glszm_features()

            self.glszm_features = [glszm_gldzm.short_runs_emphasis,
                                   glszm_gldzm.long_runs_emphasis,
                                   glszm_gldzm.low_grey_level_run_emphasis,
                                   glszm_gldzm.high_gr_lvl_emphasis,
                                   glszm_gldzm.short_low_gr_lvl_emphasis,
                                   glszm_gldzm.short_high_gr_lvl_emphasis,
                                   glszm_gldzm.long_low_gr_lvl_emphasis,
                                   glszm_gldzm.long_high_gr_lvl_emphasis,
                                   glszm_gldzm.non_uniformity,
                                   glszm_gldzm.norm_non_uniformity,
                                   glszm_gldzm.length_non_uniformity,
                                   glszm_gldzm.norm_length_non_uniformity,
                                   glszm_gldzm.percentage,
                                   glszm_gldzm.gr_lvl_var,
                                   glszm_gldzm.length_var,
                                   glszm_gldzm.entropy]
            self.glszm_features_list.append(self.glszm_features)

            glszm_gldzm.reset_fields()
            glszm_gldzm.calc_3d_gldzm_features()

            self.gldzm_features = [glszm_gldzm.short_runs_emphasis,
                                   glszm_gldzm.long_runs_emphasis,
                                   glszm_gldzm.low_grey_level_run_emphasis,
                                   glszm_gldzm.high_gr_lvl_emphasis,
                                   glszm_gldzm.short_low_gr_lvl_emphasis,
                                   glszm_gldzm.short_high_gr_lvl_emphasis,
                                   glszm_gldzm.long_low_gr_lvl_emphasis,
                                   glszm_gldzm.long_high_gr_lvl_emphasis,
                                   glszm_gldzm.non_uniformity,
                                   glszm_gldzm.norm_non_uniformity,
                                   glszm_gldzm.length_non_uniformity,
                                   glszm_gldzm.norm_length_non_uniformity,
                                   glszm_gldzm.percentage,
                                   glszm_gldzm.gr_lvl_var,
                                   glszm_gldzm.length_var,
                                   glszm_gldzm.entropy]
            self.gldzm_features_list.append(self.gldzm_features)

        else:
            glszm_gldzm.calc_glsz_gldz_2d_matrices(self.patient_morphological_mask.array.T)
            if self.aggr_dim == '2.5D':
                glszm_gldzm.calc_2_5d_glszm_features()

                self.glszm_features = [glszm_gldzm.short_runs_emphasis,
                                       glszm_gldzm.long_runs_emphasis,
                                       glszm_gldzm.low_grey_level_run_emphasis,
                                       glszm_gldzm.high_gr_lvl_emphasis,
                                       glszm_gldzm.short_low_gr_lvl_emphasis,
                                       glszm_gldzm.short_high_gr_lvl_emphasis,
                                       glszm_gldzm.long_low_gr_lvl_emphasis,
                                       glszm_gldzm.long_high_gr_lvl_emphasis,
                                       glszm_gldzm.non_uniformity,
                                       glszm_gldzm.norm_non_uniformity,
                                       glszm_gldzm.length_non_uniformity,
                                       glszm_gldzm.norm_length_non_uniformity,
                                       glszm_gldzm.percentage,
                                       glszm_gldzm.gr_lvl_var,
                                       glszm_gldzm.length_var,
                                       glszm_gldzm.entropy]
                self.glszm_features_list.append(self.glszm_features)

                glszm_gldzm.reset_fields()
                glszm_gldzm.calc_2_5d_gldzm_features()

                self.gldzm_features = [glszm_gldzm.short_runs_emphasis,
                                       glszm_gldzm.long_runs_emphasis,
                                       glszm_gldzm.low_grey_level_run_emphasis,
                                       glszm_gldzm.high_gr_lvl_emphasis,
                                       glszm_gldzm.short_low_gr_lvl_emphasis,
                                       glszm_gldzm.short_high_gr_lvl_emphasis,
                                       glszm_gldzm.long_low_gr_lvl_emphasis,
                                       glszm_gldzm.long_high_gr_lvl_emphasis,
                                       glszm_gldzm.non_uniformity,
                                       glszm_gldzm.norm_non_uniformity,
                                       glszm_gldzm.length_non_uniformity,
                                       glszm_gldzm.norm_length_non_uniformity,
                                       glszm_gldzm.percentage,
                                       glszm_gldzm.gr_lvl_var,
                                       glszm_gldzm.length_var,
                                       glszm_gldzm.entropy]
                self.gldzm_features_list.append(self.gldzm_features)
            else:
                glszm_gldzm.calc_2d_glszm_features()

                self.glszm_features = [glszm_gldzm.short_runs_emphasis,
                                       glszm_gldzm.long_runs_emphasis,
                                       glszm_gldzm.low_grey_level_run_emphasis,
                                       glszm_gldzm.high_gr_lvl_emphasis,
                                       glszm_gldzm.short_low_gr_lvl_emphasis,
                                       glszm_gldzm.short_high_gr_lvl_emphasis,
                                       glszm_gldzm.long_low_gr_lvl_emphasis,
                                       glszm_gldzm.long_high_gr_lvl_emphasis,
                                       glszm_gldzm.non_uniformity,
                                       glszm_gldzm.norm_non_uniformity,
                                       glszm_gldzm.length_non_uniformity,
                                       glszm_gldzm.norm_length_non_uniformity,
                                       glszm_gldzm.percentage,
                                       glszm_gldzm.gr_lvl_var,
                                       glszm_gldzm.length_var,
                                       glszm_gldzm.entropy]
                self.glszm_features_list.append(self.glszm_features)

                glszm_gldzm.reset_fields()
                glszm_gldzm.calc_2d_gldzm_features()

                self.gldzm_features = [glszm_gldzm.short_runs_emphasis,
                                       glszm_gldzm.long_runs_emphasis,
                                       glszm_gldzm.low_grey_level_run_emphasis,
                                       glszm_gldzm.high_gr_lvl_emphasis,
                                       glszm_gldzm.short_low_gr_lvl_emphasis,
                                       glszm_gldzm.short_high_gr_lvl_emphasis,
                                       glszm_gldzm.long_low_gr_lvl_emphasis,
                                       glszm_gldzm.long_high_gr_lvl_emphasis,
                                       glszm_gldzm.non_uniformity,
                                       glszm_gldzm.norm_non_uniformity,
                                       glszm_gldzm.length_non_uniformity,
                                       glszm_gldzm.norm_length_non_uniformity,
                                       glszm_gldzm.percentage,
                                       glszm_gldzm.gr_lvl_var,
                                       glszm_gldzm.length_var,
                                       glszm_gldzm.entropy]
                self.gldzm_features_list.append(self.gldzm_features)

        ngtdm = NGTDM(image=self.patient_intensity_mask.array.T, slice_weight=self.slice_weighting,
                      slice_median=self.slice_median)
        if self.aggr_dim == '3D':
            ngtdm.calc_ngtd_3d_matrix()
            ngtdm.calc_3d_ngtdm_features()
        elif self.aggr_dim == '2.5D':
            ngtdm.calc_ngtd_2d_matrices()
            ngtdm.calc_2_5d_ngtdm_features()
        elif self.aggr_dim == '2D':
            ngtdm.calc_ngtd_2d_matrices()
            ngtdm.calc_2d_ngtdm_features()

        self.ngtdm_features = [ngtdm.coarseness,
                               ngtdm.contrast,
                               ngtdm.busyness,
                               ngtdm.complexity,
                               ngtdm.strength]
        self.ngtdm_features_list.append(self.ngtdm_features)

        ngldm = GLRLM_GLSZM_GLDZM_NGLDM(image=self.patient_intensity_mask.array.T, slice_weight=self.slice_weighting,
                                        slice_median=self.slice_median)
        if self.aggr_dim == '3D':
            ngldm.calc_ngld_3d_matrix()
            ngldm.calc_3d_ngldm_features()
        elif self.aggr_dim == '2.5D':
            ngldm.calc_ngld_2d_matrices()
            ngldm.calc_2_5d_ngldm_features()
        elif self.aggr_dim == '2D':
            ngldm.calc_ngld_2d_matrices()
            ngldm.calc_2d_ngldm_features()

        self.ngldm_features = [ngldm.short_runs_emphasis,
                               ngldm.long_runs_emphasis,
                               ngldm.low_grey_level_run_emphasis,
                               ngldm.high_gr_lvl_emphasis,
                               ngldm.short_low_gr_lvl_emphasis,
                               ngldm.short_high_gr_lvl_emphasis,
                               ngldm.long_low_gr_lvl_emphasis,
                               ngldm.long_high_gr_lvl_emphasis,
                               ngldm.non_uniformity,
                               ngldm.norm_non_uniformity,
                               ngldm.length_non_uniformity,
                               ngldm.norm_length_non_uniformity,
                               ngldm.percentage,
                               ngldm.gr_lvl_var,
                               ngldm.length_var,
                               ngldm.entropy,
                               ngldm.energy]
        self.ngldm_features_list.append(self.ngldm_features)

    def _calc_mask_morphological_features(self):
        morf_features = MorphologicalFeatures(self.patient_morphological_mask.array,
                                              (self.patient_morphological_mask.spacing[::-1]))
        morf_features.calc_mesh()
        morf_features.calc_vol_and_area_mesh()
        morf_features.calc_vol_count()
        morf_features.calc_surf_to_vol_ratio()
        morf_features.calc_compactness_1()
        morf_features.calc_compactness_2()
        morf_features.calc_spherical_disproportion()
        morf_features.calc_sphericity()
        morf_features.calc_asphericity()
        morf_features.calc_centre_of_shift(self.patient_intensity_mask.array)
        morf_features.calc_convex_hull()
        morf_features.calc_max_diameter()
        morf_features.calc_pca()
        morf_features.calc_major_minor_least_axes_len()
        morf_features.calc_elongation()
        morf_features.calc_flatness()
        morf_features.calc_vol_and_area_densities_aabb()
        morf_features.calc_vol_density_aee()
        morf_features.calc_area_density_aee()
        morf_features.calc_vol_density_ch()
        morf_features.calc_area_density_ch()
        morf_features.calc_integrated_intensity(self.patient_intensity_mask.array)

        self.mort_features = [morf_features.vol_mesh,
                              morf_features.vol_count,
                              morf_features.area_mesh,
                              morf_features.surf_to_vol_ratio,
                              morf_features.compactness_1,
                              morf_features.compactness_2,
                              morf_features.spherical_disproportion,
                              morf_features.sphericity,
                              morf_features.asphericity,
                              morf_features.centre_of_shift,
                              morf_features.max_diameter,
                              morf_features.major_axis_len,
                              morf_features.minor_axis_len,
                              morf_features.least_axis_len,
                              morf_features.elongation,
                              morf_features.flatness,
                              morf_features.vol_density_aabb,
                              morf_features.area_density_aabb,
                              morf_features.vol_density_aee,
                              morf_features.area_density_aee,
                              morf_features.vol_density_ch,
                              morf_features.area_density_ch,
                              morf_features.integrated_intensity,
                              ]

        self.patient_morf_features_list.append(self.mort_features)

    def _calc_morph_moran_i_and_geary_c_features(self):
        morf_features = MorphologicalFeatures(self.patient_morphological_mask.array,
                                              (self.patient_morphological_mask.spacing[::-1]))
        morf_features.calc_moran_i(self.patient_intensity_mask.array)
        morf_features.calc_geary_c(self.patient_intensity_mask.array)
        self.morph_moran_i_and_geary_c_features = {'morph_moran_i':  morf_features.moran_i,
         'morph_geary_c': morf_features.geary_c}
