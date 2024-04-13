import os
from multiprocessing import Pool, cpu_count

import SimpleITK as sitk
import numpy as np
import pandas as pd
import pydicom

from .radiomics_defenitions import MorphologicalFeatures, LocalIntensityFeatures, IntensityBasedStatFeatures, \
    IntensityVolumeHistogramFeatures, GLCM, GLRLM_GLSZM_GLDZM_NGLDM, NGTDM
from .toolbox_logic import Image, extract_dicom, extract_nifti_image, extract_nifti_mask, \
    list_folders_in_defined_range, check_dicom_spacing


class Radiomics:

    def __init__(self, load_dir, save_dir,
                 input_data_type, input_imaging_mod,
                 aggr_dim, aggr_method,
                 intensity_range=None, outlier_range=None,
                 number_of_bins=None, bin_size=None,
                 slice_weighting=False, slice_median=False,
                 start_folder=None, stop_folder=None, list_of_patient_folders=None,
                 structure_set=None, nifti_image=None,
                 number_of_threads=1):

        if os.path.exists(load_dir):
            self.load_dir = load_dir
        else:
            raise ValueError(f"Load directory '{load_dir}' does not exist.")

        if os.path.exists(save_dir):
            self.save_dir = save_dir
        else:
            os.makedirs(save_dir)
            self.save_dir = save_dir

        if (start_folder is not None and stop_folder is not None
                and isinstance(start_folder, int) and isinstance(stop_folder, int)):
            self.list_of_patient_folders = list_folders_in_defined_range(start_folder, stop_folder, self.load_dir)
        elif list_of_patient_folders is not None and list_of_patient_folders not in [[], ['']]:
            self.list_of_patient_folders = list_of_patient_folders
        elif list_of_patient_folders is None and start_folder is None and stop_folder is None:
            self.list_of_patient_folders = os.listdir(load_dir)
        else:
            raise ValueError('Incorrectly selected patient folders.')

        if input_data_type in ['DICOM', 'NIFTI']:
            self.input_data_type = input_data_type
        else:
            raise ValueError(f"Wrong input data type '{input_data_type}', available types: 'DICOM', 'NIFTI'.")

        if self.input_data_type == 'DICOM':
            list_to_del = set()
            for pat_index, pat_path in enumerate(self.list_of_patient_folders):
                pat_folder_path = os.path.join(load_dir, pat_path)
                if check_dicom_spacing(os.path.join(pat_folder_path)):
                    list_to_del.add(pat_index)
            for index_to_del in list_to_del:
                print(f'Patient {index_to_del} is excluded from the analysis due to the inconsistent z-spacing.')
                del self.list_of_patient_folders[index_to_del]

        if input_imaging_mod in ['CT', 'PT', 'MR']:
            self.input_imaging_mod = input_imaging_mod
        else:
            raise ValueError(f"Wrong input imaging type '{input_imaging_mod}', available types: 'CT', 'PT', 'MR'.")

        if self.input_data_type == 'NIFTI':
            if nifti_image is not None:
                image_exists = True
                for folder in self.list_of_patient_folders:
                    if not os.path.exists(os.path.join(load_dir, str(folder), nifti_image)):
                        image_exists = False
                        if not image_exists:
                            print(f"The NIFTI image file does not exist "
                                  f"'{os.path.join(load_dir, str(folder), nifti_image)}'")
                if image_exists:
                    self.nifti_image = nifti_image
            else:
                raise ValueError('Select the NIFTI image file.')

        if isinstance(number_of_threads, (int, float)) and 0 < number_of_threads <= cpu_count():
            self.number_of_threads = number_of_threads
        else:
            raise ValueError(f'Number of threads is not an integer or selected nubmer '
                             f'is greater than maximum number of available CPU. (Max available {cpu_count()} units)')

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

        self.calc_outlier_mask = False
        if str(outlier_range).strip().replace('.', '').isdigit():
            self.calc_outlier_mask = True
            self.outlier_range = outlier_range

        self.calc_discr_bin_number = False
        self.calc_discr_bin_size = False

        if number_of_bins is not None:
            self.calc_discr_bin_number = True
            self.bin_number = number_of_bins

        if bin_size is not None:
            self.calc_discr_bin_size = True
            self.bin_size = bin_size

        if aggr_dim in ['2D', '3D']:
            self.aggr_dim = aggr_dim
        else:
            raise ValueError(f"Wrong aggregation dim {aggr_dim}. Available '2D' and '3D'.")

        if aggr_method in ['MERG', 'AVER', 'SLICE_MERG', 'DIR_MERG']:
            self.aggr_method = aggr_method
        else:
            raise ValueError(f"Wrong aggregation dim {aggr_method}. "
                             "Available 'MERG', 'AVER', 'SLICE_MERG', and 'DIR_MERG'.")

        if structure_set is not None:
            self.structure_set = structure_set
        else:
            self.structure_set = ['']

        self.patient_folder = None
        self.patient_number = None

        self.columns = [
            'PAT::ROI',
            'vol_mesh', 'vol_count', 'area_mesh', 'surf_to_vol_ratio', 'compactness_1', 'compactness_2',
            'spherical_disproportion', 'sphericity', 'asphericity', 'centre_of_shift', 'max_diameter', 'major_axis_len',
            'minor_axis_len', 'least_axis_len', 'elongation', 'flatness', 'vol_density_aabb', 'area_density_aabb',
            'vol_density_aee', 'area_density_aee', 'vol_density_ch', 'area_density_ch', 'integrated_intensity',
            'local_intensity_peak', 'mean_intensity', 'intensity_variance', 'intensity_skewness', 'intensity_kurtosis',
            'median_intensity', 'min_intensity', 'intensity_10th_percentile', 'intensity_90th_percentile',
            'max_intensity', 'intensity_iqr', 'intensity_range', 'intensity_based_mean_abs_deviation',
            'intensity_based_robust_mean_abs_deviation', 'intensity_based_median_abs_deviation',
            'intensity_based_variation_coef', 'intensity_based_quartile_coef_dispersion', 'intensity_based_energy',
            'root_mean_square_intensity',
            'discr_mean_intensity', 'discr_intensity_varaince', 'discr_intensity_skewness', 'discr_intensity_kurtosis',
            'discr_median_intensity', 'discr_min_intensity', 'discr_intensity_10th_percentile',
            'discr_intensity_90th_percentile', 'discr_max_intensity', 'discr_intensity_hist_mode',
            'discr_intensity_iqr', 'discr_intensity_range', 'discr_intensity_based_mean_abs_deviation',
            'discr_intensity_based_robust_mean_abs_deviation', 'discr_intensity_based_median_abs_deviation',
            'discr_intensity_based_variation_coef', 'discr_intensity_based_quartile_coef_dispersion',
            'discr_intensity_entropy', 'discr_intensity_uniformity', 'discr_max_hist_gradient',
            'discr_max_hist_gradient_intensity', 'discr_min_hist_gradient', 'discr_min_hist_gradient_intensity',
            'vol_10%_intensity_frac', 'vol_90%_intensity_frac', 'intensity_10%_vol_frac', 'intensity_90%_vol_frac',
            'volume_frac_diff_between_intensity_frac', 'intensity_frac_diff_between_vol_frac',
            'glcm_joint_max', 'glcm_joint_average', 'glcm_joint_var', 'glcm_joint_entropy', 'glcm_dif_average',
            'glcm_dif_var', 'glcm_dif_entropy', 'glcm_sum_average', 'glcm_sum_var', 'glcm_sum_entropy',
            'glcm_ang_second_moment', 'glcm_contrast', 'glcm_dissimilarity', 'glcm_inv_diff', 'glcm_norm_inv_diff',
            'glcm_inv_diff_moment', 'glcm_norm_inv_diff_moment', 'glcm_inv_variance', 'glcm_cor', 'glcm_autocor',
            'glcm_cluster_tendency', 'glcm_cluster_shade', 'glcm_cluster_prominence', 'glcm_inf_cor_1',
            'glcm_inf_cor_2',
            'glrlm_short_runs_emphasis', 'glrlm_long_runs_emphasis', 'glrlm_low_grey_level_run_emphasis',
            'glrlm_high_gr_lvl_emphasis', 'glrlm_short_run_low_gr_lvl_emphasis', 'glrlm_short_run_high_gr_lvl_emphasis',
            'glrlm_long_run_low_gr_lvl_emphasis', 'glrlm_long_run_high_gr_lvl_emphasis', 'glrlm_non_uniformity',
            'glrlm_norm_non_uniformity', 'glrlm_run_length_non_uniformity', 'glrlm_norm_run_length_non_uniformity',
            'glrlm_percentage', 'glrlm_gr_lvl_var', 'glrlm_run_length_var', 'glrlm_run_entropy',
            'glszm_small_zone_emphasis', 'glszm_large_zone_emphasis', 'glszm_low_grey_level_zone_emphasis',
            'glszm_high_gr_lvl_zone_emphasis', 'glszm_small_zone_low_gr_lvl_emphasis',
            'glszm_small_zone_high_gr_lvl_emphasis', 'glszm_large_zone_low_gr_lvl_emphasis',
            'glszm_large_zone_high_gr_lvl_emphasis', 'glszm_non_uniformity', 'glszm_norm_non_uniformity',
            'glszm_zone_non_uniformity', 'glszm_norm_zone_non_uniformity', 'glszm_percentage', 'glszm_gr_lvl_var',
            'glszm_zone_size_var', 'glszm_entropy',
            'gldzm_small_dist_emphasis', 'gldzm_large_dist_emphasis', 'gldzm_low_grey_level_zone_emphasis',
            'gldzm_high_gr_lvl_zone_emphasis', 'gldzm_small_dist_low_gr_lvl_emphasis',
            'gldzm_small_dist_high_gr_lvl_emphasis', 'gldzm_large_dist_low_gr_lvl_emphasis',
            'gldzm_large_dist_high_gr_lvl_emphasis', 'gldzm_non_uniformity', 'gldzm_norm_non_uniformity',
            'gldzm_zone_dist_non_uniformity', 'gldzm_norm_zone_dist_non_uniformity', 'gldzm_percentage',
            'gldzm_gr_lvl_var', 'gldzm_zone_dist_var', 'gldzm_entropy',
            'ngtdm_coarseness', 'ngtdm_contrast', 'ngtdm_busyness', 'ngtdm_complexity', 'ngtdm_strength',
            'ngldm_low_depend_emphasis', 'ngldm_high_depend_emphasis', 'ngldm_low_gr_lvl_emphasis',
            'ngldm_high_gr_lvl_emphasis', 'ngldm_low_depend_low_gr_lvl_emphasis',
            'ngldm_low_depend_high_gr_lvl_emphasis', 'ngldm_high_depend_low_gr_lvl_emphasis',
            'ngldm_high_depend_high_gr_lvl_emphasis', 'ngldm_non_uniformity', 'ngldm_norm_non_uniformity',
            'ngldm_depend_count_non_uniformity', 'ngldm_norm_depend_count_non_uniformity', 'ngldm_percentage',
            'ngldm_gr_lvl_var', 'ngldm_depend_count_var', 'ngldm_entropy', 'ngldm_energy'
        ]

    def extract_radiomics(self):

        with Pool(self.number_of_threads) as pool:
            pool.map(self._load_patient, sorted(self.list_of_patient_folders))

        print('Completed!')

    def _load_patient(self, patient_number):
        self.patient_number = str(patient_number)
        self.patient_folder = os.path.join(self.load_dir, self.patient_number)
        self.pat_binned_masked_image = {}
        self.patient_mort_features_list = []
        self.patient_local_intensity_features_list = []
        self.intensity_features_list = []
        self.intensity_volume_features_list = []
        self.discr_intensity_features_list = []
        self.glcm_features_list = []
        self.glrlm_features_list = []
        self.glszm_features_list = []
        self.gldzm_features_list = []
        self.ngtdm_features_list = []
        self.ngldm_features_list = []

        if self.input_data_type == 'NIFTI':
            self._process_nifti_files()
        else:
            self._process_dicom_files()

        all_features_list = [self.patient_mort_features_list, self.patient_local_intensity_features_list,
                             self.intensity_features_list, self.discr_intensity_features_list,
                             self.intensity_volume_features_list,
                             self.glcm_features_list,
                             self.glrlm_features_list, self.glszm_features_list,
                             self.gldzm_features_list, self.ngtdm_features_list, self.ngldm_features_list]
        radiomics_features_df = pd.DataFrame(columns=self.columns)
        for mask_index in range(len(self.patient_mort_features_list)):
            save_list = []
            for feature_list in all_features_list:
                save_list += feature_list[mask_index]
            radiomics_features_df.loc[radiomics_features_df.shape[0]] = dict(zip(self.columns, save_list))
        radiomics_features_df.to_excel(
            os.path.join(self.save_dir, f'patient_{self.patient_number}_radiomics.xlsx'), index=False)

    def _process_dicom_files(self):
        self.patient_image = extract_dicom(dicom_dir=self.patient_folder, rtstract=False,
                                           modality=self.input_imaging_mod)

        if self.structure_set != ['']:
            for dicom_file in os.listdir(self.patient_folder):
                dcm_data = pydicom.dcmread(os.path.join(self.patient_folder, dicom_file))
                if dcm_data.Modality == 'RTSTRUCT':
                    masks = extract_dicom(dicom_dir=self.patient_folder, rtstract=True,
                                          rtstruct_file=os.path.join(self.patient_folder, dicom_file),
                                          selected_structures=self.structure_set,
                                          modality=self.input_imaging_mod)

                    for instance_key in masks.keys():

                        print(f'Patient {self.patient_number}. Mask {instance_key}.')
                        self.patient_morphological_mask = Image(masks[instance_key].array.astype(np.int8),
                                                                origin=self.patient_image.origin,
                                                                spacing=self.patient_image.spacing,
                                                                direction=self.patient_image.direction,
                                                                shape=self.patient_image.shape)

                        self.patient_intensity_mask = Image(array=np.where(self.patient_morphological_mask.array > 0,
                                                                           self.patient_image.array, np.nan),
                                                            origin=self.patient_image.origin,
                                                            spacing=self.patient_image.spacing,
                                                            direction=self.patient_image.direction,
                                                            shape=self.patient_image.shape)

                        self._calc_mask_intensity_features()
                        self._calc_mask_morphological_features(instance_key)
                        self._calc_discretized_intensity_features()
                        self._calc_texture_features()

    def _process_nifti_files(self):
        def extract_nifti(key, mask_file_name=None):
            reader = sitk.ImageFileReader()
            reader.SetImageIO("NiftiImageIO")

            if key == 'IMAGE':
                return extract_nifti_image(self, reader)

            elif key.startswith('MASK'):
                return extract_nifti_mask(self, reader, mask_file_name, self.patient_image)

        self.patient_image = extract_nifti('IMAGE')
        for nifti_mask in self.structure_set:
            instance_key = 'MASK_' + nifti_mask
            file_found, mask = extract_nifti(instance_key, nifti_mask)
            if file_found:
                print(f'Patient {self.patient_number}. Mask {instance_key[5:]}.')

                self.patient_morphological_mask = Image(array=mask.array.astype(np.int8),
                                                        origin=self.patient_image.origin,
                                                        spacing=self.patient_image.spacing,
                                                        direction=self.patient_image.direction,
                                                        shape=self.patient_image.shape)

                self.patient_intensity_mask = Image(array=np.where(self.patient_morphological_mask.array > 0,
                                                    self.patient_image.array, np.nan),
                                                    origin=self.patient_image.origin,
                                                    spacing=self.patient_image.spacing,
                                                    direction=self.patient_image.direction,
                                                    shape=self.patient_image.shape)

                self._calc_mask_intensity_features()
                self._calc_mask_morphological_features(instance_key)
                self._calc_discretized_intensity_features()
                self._calc_texture_features()

    def _calc_mask_intensity_features(self):
        if self.calc_intensity_mask:
            array = np.where((self.patient_intensity_mask.array <= self.intensity_range[1])
                             & (self.patient_intensity_mask.array >= self.intensity_range[0])
                             & (self.patient_morphological_mask.array > 0),
                             self.patient_image.array, np.nan)
            self.patient_intensity_mask = Image(array=array,
                                                origin=self.patient_intensity_mask.origin,
                                                spacing=self.patient_intensity_mask.spacing,
                                                direction=self.patient_intensity_mask.direction,
                                                shape=self.patient_intensity_mask.shape)
        if self.calc_outlier_mask:
            flattened_image = np.where(self.patient_morphological_mask.array > 0,
                                       self.patient_image.array, np.nan).ravel()
            valid_values = flattened_image[~np.isnan(flattened_image)]
            mean = np.mean(valid_values)
            std = np.std(valid_values)
            array = np.where((self.patient_intensity_mask.array <= mean + self.outlier_range * std)
                             & (self.patient_intensity_mask.array >= mean - self.outlier_range * std)
                             & (self.patient_morphological_mask.array > 0),
                             self.patient_image.array, np.nan)

            self.patient_intensity_mask = Image(array=array,
                                                origin=self.patient_intensity_mask.origin,
                                                spacing=self.patient_intensity_mask.spacing,
                                                direction=self.patient_intensity_mask.direction,
                                                shape=self.patient_intensity_mask.shape)

        local_intensity_features = LocalIntensityFeatures(self.patient_image.array,
                                                          self.patient_intensity_mask.array,
                                                          (self.patient_image.spacing[::-1]))

        local_intensity_features.calc_local_intensity_peak()
        self.local_intensity_features = [local_intensity_features.local_intensity_peak]
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

        intensity_vol_hist_features = IntensityVolumeHistogramFeatures(self.patient_intensity_mask.array.T)
        self.intensity_volume_features_list.append([
            intensity_vol_hist_features.calc_volume_at_intensity_fraction(10),
            intensity_vol_hist_features.calc_volume_at_intensity_fraction(90),
            intensity_vol_hist_features.calc_intensity_at_volume_fraction(10),
            intensity_vol_hist_features.calc_intensity_at_volume_fraction(90),
            intensity_vol_hist_features.calc_volume_fraction_diff_intensity_fractions(),
            intensity_vol_hist_features.calc_intensity_fraction_diff_volume_fractions()
            ])

    def _calc_discretized_intensity_features(self):
        if self.calc_discr_bin_size:
            if self.calc_intensity_mask:
                self.patient_intensity_mask = Image(array=np.floor(
                    (self.patient_intensity_mask.array - self.discret_min_val) / self.bin_size) + 1,
                                                         origin=self.patient_image.origin,
                                                         spacing=self.patient_image.spacing,
                                                         direction=self.patient_image.direction,
                                                         shape=self.patient_image.shape
                                                         )
            else:
                self.patient_intensity_mask = Image(array=np.floor((self.patient_intensity_mask.array - np.nanmin(
                    self.patient_intensity_mask.array)) / self.bin_size) + 1,
                                                         origin=self.patient_image.origin,
                                                         spacing=self.patient_image.spacing,
                                                         direction=self.patient_image.direction,
                                                         shape=self.patient_image.shape
                                                         )
        if self.calc_discr_bin_number:
            self.patient_intensity_mask = Image(
                array=np.where(self.patient_intensity_mask.array != np.nanmax(self.patient_intensity_mask.array),
                               np.floor(self.bin_number * (self.patient_intensity_mask.array - np.nanmin(
                                   self.patient_intensity_mask.array))
                                        / (np.nanmax(self.patient_intensity_mask.array)
                                           - np.nanmin(self.patient_intensity_mask.array))) + 1, self.bin_number),
                origin=self.patient_image.origin,
                spacing=self.patient_image.spacing,
                direction=self.patient_image.direction,
                shape=self.patient_image.shape
                )

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

    def _calc_mask_morphological_features(self, key):
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

        self.mort_features = [self.patient_number + '::' + key[5:],
                              morf_features.vol_mesh,
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
                              morf_features.integrated_intensity
                              ]

        self.patient_mort_features_list.append(self.mort_features)
