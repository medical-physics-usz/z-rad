import numpy as np
from scipy.ndimage import distance_transform_cdt
from scipy.spatial import ConvexHull
from scipy.special import legendre
from scipy.stats import iqr, skew, kurtosis
from skimage import measure
from sklearn.decomposition import PCA


class MorphologicalFeatures:
    def __init__(self,  # image,
                 mask, spacing):

        self.spacing = spacing
        # self.array_image = image
        self.array_mask = mask
        self.unit_vol = self.spacing[0] * self.spacing[1] * self.spacing[2]

        # ---------mech-----------
        self.mesh_verts = None
        self.mesh_faces = None
        self.mesh_normals = None
        self.mesh_values = None
        self.vol_mesh = None  # 3.1.1
        self.vol_count = None  # 3.1.2
        self.area_mesh = None  # 3.1.3
        self.surf_to_vol_ratio = None  # 3.1.4
        self.compactness_1 = None  # 3.1.5
        self.compactness_2 = None  # 3.1.6
        self.spherical_disproportion = None  # 3.1.7
        self.sphericity = None  # 3.1.8
        self.asphericity = None  # 3.1.9
        # ------------------------------
        self.centre_of_shift = None  # 3.1.10
        # -------------------------------
        self.conv_hull = None
        self.max_diameter = None  # 3.1.11

        # ------------PCA based---------
        self.pca_eigenvalues = None
        self.major_axis_len = None  # 3.1.12
        self.minor_axis_len = None  # 3.1.13
        self.least_axis_len = None  # 3.1.14
        self.elongation = None  # 3.1.15
        self.flatness = None  # 3.1.16

        # ----axis-aligned bounding box----
        self.vol_density_aabb = None  # 3.1.17
        self.area_density_aabb = None  # 3.1.18
        # -------------------------------------
        # 3.1.19 and 3.1.20 no cross-center validation
        # ------------AEE-----------------------
        self.vol_density_aee = None  # 3.1.21
        self.area_density_aee = None  # 3.1.22
        # -----------------------------------
        # 3.1.23 and 3.1.24 no cross-center validation
        # --------convex hull based-------------
        self.vol_density_ch = None  # 3.1.25
        self.area_density_ch = None  # 3.1.26
        # --------------------------------------
        self.integrated_intensity = None  # 3.1.27

    def calc_mesh(self):
        self.mesh_verts, self.mesh_faces, self.mesh_normals, self.mesh_values = measure.marching_cubes(self.array_mask,
                                                                                                       level=0.5)
        self.mesh_verts = self.mesh_verts * self.spacing

    def calc_vol_and_area_mesh(self):

        def volume(a, b, c):
            return np.dot(a, np.cross(b, c)) / 6

        def area(a, b, c):
            return np.linalg.norm(np.cross(b - a, c - a)) / 2

        self.vol_mesh = 0
        self.area_mesh = 0

        for face in self.mesh_faces:
            a, b, c = self.mesh_verts[face]
            self.vol_mesh += volume(a, b, c)
            self.area_mesh += area(a, b, c)

        self.vol_mesh = abs(self.vol_mesh)

    def calc_vol_count(self):
        self.vol_count = np.sum(self.array_mask) * self.unit_vol

    def calc_surf_to_vol_ratio(self):
        self.surf_to_vol_ratio = self.area_mesh / self.vol_mesh

    def calc_compactness_1(self):
        self.compactness_1 = self.vol_mesh / (np.pi ** (1 / 2) * self.area_mesh ** (3 / 2))

    def calc_compactness_2(self):
        self.compactness_2 = 36 * np.pi * (self.vol_mesh ** 2 / self.area_mesh ** 3)

    def calc_spherical_disproportion(self):
        self.spherical_disproportion = self.area_mesh / (36 * np.pi * self.vol_mesh ** 2) ** (1 / 3)

    def calc_sphericity(self):
        self.sphericity = (36 * np.pi * self.vol_mesh ** 2) ** (1 / 3) / self.area_mesh

    def calc_asphericity(self):
        self.asphericity = (self.area_mesh ** 3 / (36 * np.pi * self.vol_mesh ** 2)) ** (1 / 3) - 1

    def calc_centre_of_shift(self, image_array):
        dx, dy, dz = self.spacing
        morph_voxels = np.argwhere(self.array_mask)
        morph_voxels_scaled = morph_voxels * [dx, dy, dz]
        com_geom = np.mean(morph_voxels_scaled, axis=0)

        # Indices of voxels in the intensity mask and their corresponding intensities
        intensity_voxels = np.argwhere(~np.isnan(image_array))
        intensities = image_array[intensity_voxels[:, 0], intensity_voxels[:, 1], intensity_voxels[:, 2]]
        # Scale voxel positions by their dimensions
        intensity_voxels_scaled = intensity_voxels * [dx, dy, dz]
        # Calculate intensity-weighted center of mass
        com_gl = np.average(intensity_voxels_scaled, axis=0, weights=intensities)

        self.centre_of_shift = np.linalg.norm(com_geom - com_gl)

    def calc_convex_hull(self):
        self.conv_hull = ConvexHull(self.mesh_verts)

    def calc_max_diameter(self):
        # scaled_indices = self.mesh_verts
        self.max_diameter = 0
        for i in self.conv_hull.vertices:
            for j in self.conv_hull.vertices:
                distance = np.linalg.norm(self.mesh_verts[i] - self.mesh_verts[j])
                self.max_diameter = max(self.max_diameter, distance)

    def calc_pca(self):

        voxel_indices = np.argwhere(self.array_mask == 1)

        # Convert voxel indices to float to allow scaling
        scaled_voxel_indices = voxel_indices.astype(np.float64)

        # Scale the voxel indices according to voxel dimensions
        scaled_voxel_indices *= self.spacing

        # Perform PCA on the scaled indices
        pca = PCA(n_components=3)
        pca.fit(scaled_voxel_indices)

        # Extract the eigenvalues
        self.pca_eigenvalues = pca.explained_variance_

    def calc_major_minor_least_axes_len(self):
        self.major_axis_len = 4 * np.sqrt(self.pca_eigenvalues[0])
        self.minor_axis_len = 4 * np.sqrt(self.pca_eigenvalues[1])
        self.least_axis_len = 4 * np.sqrt(self.pca_eigenvalues[2])

    def calc_elongation(self):
        self.elongation = np.sqrt(self.pca_eigenvalues[1] / self.pca_eigenvalues[0])

    def calc_flatness(self):
        self.flatness = np.sqrt(self.pca_eigenvalues[2] / self.pca_eigenvalues[0])

    def calc_vol_and_area_densities_aabb(self):
        x_dim, y_dim, z_dim = self.spacing
        # Determine the AABB of the ROI
        x_coords, y_coords, z_coords = np.where(self.array_mask == 1)
        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()
        z_min, z_max = z_coords.min(), z_coords.max()

        # Calculate the dimensions of the AABB
        aabb_x_dim = (x_max - x_min + 1) * x_dim
        aabb_y_dim = (y_max - y_min + 1) * y_dim
        aabb_z_dim = (z_max - z_min + 1) * z_dim

        # Calculate the volume of the AABB
        aabb_volume = aabb_x_dim * aabb_y_dim * aabb_z_dim
        self.vol_density_aabb = self.vol_mesh / aabb_volume

        # Calculate the area of the AABB
        aabb_surface_area = 2 * (aabb_x_dim * aabb_y_dim + aabb_x_dim * aabb_z_dim + aabb_y_dim * aabb_z_dim)
        self.area_density_aabb = self.area_mesh / aabb_surface_area

    def calc_vol_density_aee(self):
        self.vol_density_aee = (8 * 3 * self.vol_mesh) / (
                    4 * np.pi * self.major_axis_len * self.minor_axis_len * self.least_axis_len)

    def calc_area_density_aee(self):
        a = self.major_axis_len / 2
        b = self.minor_axis_len / 2
        c = self.least_axis_len / 2

        alpha = np.sqrt(1 - (b ** 2 / a ** 2))
        beta = np.sqrt(1 - (c ** 2 / a ** 2))
        sum_series = 0
        max_nu = 20  # Def by IBSI
        for nu in range(max_nu + 1):
            p_nu = legendre(nu)
            sum_series += ((alpha * beta) ** nu / (1 - (4 * nu ** 2))) * p_nu(
                (alpha ** 2 + beta ** 2) / (2 * alpha * beta))

        area_aee = 4 * np.pi * a * b * sum_series
        self.area_density_aee = self.area_mesh / area_aee

    def calc_vol_density_ch(self):
        self.vol_density_ch = self.vol_mesh / self.conv_hull.volume

    def calc_area_density_ch(self):
        self.area_density_ch = self.area_mesh / self.conv_hull.area

    def calc_integrated_intensity(self, image_array):
        self.integrated_intensity = np.nanmean(image_array) * self.vol_mesh


class LocalIntensityFeatures:

    def __init__(self, image, masked_image, spacing):

        self.array_image = image
        self.array_masked_image = masked_image
        self.spacing = spacing

        # ---------mech-----------
        self.local_intensity_peak = None  # 3.2.1

    def calc_local_intensity_peak(self):  # 3.2.1

        radius_mm = 6.2
        # Find the indices of the maximum intensity voxels
        max_intensity = np.nanmax(self.array_masked_image)
        max_voxels = np.argwhere(self.array_masked_image == max_intensity)
        highest_peak = []
        for voxel in max_voxels:
            distances = np.sqrt(
                ((np.indices(self.array_masked_image.shape).T * self.spacing - voxel * self.spacing) ** 2).sum(axis=3))
            # Create a mask for selected voxels within the sphere radius
            sphere_mask = (distances <= radius_mm)
            # Ensure the mask is applied in all three dimensions
            selected_voxels = self.array_image[sphere_mask.T]

            # Calculate the mean intensity of the selected voxels
            mean_intensity = np.mean(selected_voxels)

            # Update the highest peak if this one is higher
            highest_peak.append(mean_intensity)

        self.local_intensity_peak = max(highest_peak)


class IntensityBasedStatFeatures:
    def __init__(self):  # , image):
        # self.spacing = spacing
        # self.array_image = image
        # self.array_mask = mask
        # ----------------------
        self.array_image_2 = None
        self.mean_intensity = None  # 3.3.1
        self.intensity_variance = None  # 3.3.2
        self.intensity_skewness = None  # 3.3.3
        self.intensity_kurtosis = None  # 3.3.4
        self.median_intensity = None  # 3.3.5
        self.min_intensity = None  # 3.3.6
        self.intensity_10th_percentile = None  # 3.3.7
        self.intensity_90th_percentile = None  # 3.3.8
        self.max_intensity = None  # 3.3.9
        self.intensity_iqr = None  # 3.3.10
        self.intensity_range = None  # 3.3.11
        self.intensity_based_mean_abs_deviation = None  # 3.3.12
        self.intensity_based_robust_mean_abs_deviation = None  # 3.3.13
        self.intensity_based_median_abs_deviation = None  # 3.3.14
        self.intensity_based_variation_coef = None  # 3.3.15
        self.intensity_based_quartile_coef_dispersion = None  # 3.3.16
        self.intensity_based_energy = None  # 3.3.17
        self.root_mean_square_intensity = None  # 3.3.18
        # ----------------------------------------------
        self.mean_discret_intensity = None  # 3.4.1
        self.discret_intensity_variance = None  # 3.4.2
        self.discret_intensity_skewness = None  # 3.4.3
        self.discret_intensity_kurtosis = None  # 3.4.4
        self.median_discret_intensity = None  # 3.4.5
        self.minimum_discret_intensity = None  # 3.4.6
        self.discret_intensity_10th_percentile = None  # 3.4.7
        self.discret_intensity_90th_percentile = None  # 3.4.8
        self.maximum_discret_intensity = None  # 3.4.9
        self.intensity_hist_mode = None  # 3.4.10
        self.discret_intensity_iqr = None  # 3.4.11
        self.discret_intensity_range = None  # 3.4.12
        self.intensity_hist_mean_abs_deviation = None  # 3.4.13
        self.intensity_hist_robust_mean_abs_deviation = None  # 3.4.14
        self.intensity_hist_median_abs_deviation = None  # 3.4.15
        self.intensity_hist_variation_coef = None  # 3.4.16
        self.intensity_hist_quartile_coef_dispersion = None  # 3.4.17
        self.discret_intensity_entropy = None  # 3.4.18
        self.discret_intensity_uniformity = None  # 3.4.19
        self.max_hist_gradient = None  # 3.4.20
        self.max_hist_gradient_intensity = None  # .3.4.21
        self.min_hist_gradient = None  # 3.4.22
        self.min_hist_gradient_intensity = None  # .3.4.23

    def calc_mean_intensity(self, array):  # 3.3.1, 3.4.1
        self.mean_intensity = np.nanmean(array)

    def calc_intensity_variance(self, array):  # 3.3.2, 3.4.2
        self.intensity_variance = np.nanstd(array) ** 2

    def calc_intensity_skewness(self, array):  # 3.3.3, 3.4.3
        self.intensity_skewness = skew(array, axis=None, nan_policy='omit')

    def calc_intensity_kurtosis(self, array):  # 3.3.4, 3.4.4
        self.intensity_kurtosis = kurtosis(array, axis=None, nan_policy='omit')

    def calc_median_intensity(self, array):  # 3.3.5, 3.4.5
        self.median_intensity = np.nanmedian(array)

    def calc_min_intensity(self, array):  # 3.3.6, 3.4.6
        self.min_intensity = np.nanmin(array)

    def calc_intensity_10th_percentile(self, array):  # 3.3.7, 3.4.7
        self.intensity_10th_percentile = np.nanpercentile(array, 10)

    def calc_intensity_90th_percentile(self, array):  # 3.3.8, 3.4.8
        self.intensity_90th_percentile = np.nanpercentile(array, 90)

    def calc_max_intensity(self, array):  # 3.3.9, 3.4.9
        self.max_intensity = np.nanmax(array)

    def calc_intensity_iqr(self, array):  # 3.3.10, 3.4.11
        self.intensity_iqr = iqr(array, nan_policy='omit')

    def calc_intensity_range(self, array):  # 3.3.11, 3.4.12
        self.intensity_range = np.nanmax(array) - np.nanmin(array)

    def calc_intensity_based_mean_abs_deviation(self, array):  # .3.3.12, 3.4.13
        self.intensity_based_mean_abs_deviation = np.nanmean(np.absolute(array - np.nanmean(array)))

    def calc_intensity_based_robust_mean_abs_deviation(self, array):  # 3.3.13, 3.4.14
        self.array_image_2 = array.copy()
        p10 = np.nanpercentile(self.array_image_2, 10)
        p90 = np.nanpercentile(self.array_image_2, 90)
        ind = np.where((self.array_image_2 < p10) | (self.array_image_2 > p90))
        self.array_image_2[ind] = np.nan
        self.intensity_based_robust_mean_abs_deviation = np.nanmean(
            np.absolute(self.array_image_2 - np.nanmean(self.array_image_2)))

    def calc_intensity_based_median_abs_deviation(self, array):  # 3.3.14, 3.4.15
        self.intensity_based_median_abs_deviation = np.nanmean(np.absolute(array - np.nanmedian(array)))

    def calc_intensity_based_variation_coef(self, array):  # 3.3.15, 3.4.16
        self.intensity_based_variation_coef = np.nanstd(array) / np.nanmean(array)

    def calc_intensity_based_quartile_coef_dispersion(self, array):  # 3.3.16, 3.4.17
        p25 = np.nanpercentile(array, 25)
        p75 = np.nanpercentile(array, 75)
        self.intensity_based_quartile_coef_dispersion = (p75 - p25) / (p75 + p25)

    def calc_intensity_based_energy(self, array):  # 3.3.17
        self.intensity_based_energy = np.nansum(array ** 2)

    def calc_root_mean_square_intensity(self, array):  # .3.3.18
        self.root_mean_square_intensity = np.sqrt(np.nanmean(array ** 2))

    def calc_discretised_intensity_mode(self, array):  # 3.4.10
        values, counts = np.unique(array[~np.isnan(array)], return_counts=True)
        max_count_index = np.argmax(counts)
        self.intensity_hist_mode = values[max_count_index]

    def calc_discretised_intensity_entropy(self, array):  # 3.4.18
        values, counts = np.unique(array[~np.isnan(array)], return_counts=True)
        p = counts / np.sum(counts)
        self.discret_intensity_entropy = (-1) * np.sum(p * np.log2(p))

    def calc_discretised_intensity_uniformity(self, array):  # 3.4.19
        values, counts = np.unique(array[~np.isnan(array)], return_counts=True)
        p = counts / np.sum(counts)
        self.discret_intensity_uniformity = np.sum(p * p)

    def calc_max_hist_gradient(self, array):  # 3.4.20
        values, counts = np.unique(array[~np.isnan(array)], return_counts=True)
        self.max_hist_gradient = np.max(np.gradient(counts))

    def calc_max_hist_gradient_intensity(self, array):  # 3.4.21
        values, counts = np.unique(array[~np.isnan(array)], return_counts=True)
        self.max_hist_gradient_intensity = values[np.argmax(np.gradient(counts))]

    def calc_min_hist_gradient(self, array):  # 3.4.22
        values, counts = np.unique(array[~np.isnan(array)], return_counts=True)
        self.min_hist_gradient = np.min(np.gradient(counts))

    def calc_min_hist_gradient_intensity(self, array):  # 3.4.23
        values, counts = np.unique(array[~np.isnan(array)], return_counts=True)
        self.min_hist_gradient_intensity = values[np.argmin(np.gradient(counts))]


class IntensityVolumeHistogramFeatures:
    def __init__(self, array):
        self.valid_values = array.ravel()[~np.isnan(array.ravel())]
        self.min_intensity = int(np.min(self.valid_values))
        self.max_intensity = int(np.max(self.valid_values))
        self.fractional_volumes = np.zeros(len(range(self.min_intensity, self.max_intensity + 1)))
        self.intensity_fractions = np.zeros(len(range(self.min_intensity, self.max_intensity + 1)))
        self.intensity = np.zeros(len(range(self.min_intensity, self.max_intensity + 1)))

        self.volume_at_intensity_fraction_x_per_cent = None  # 3.5.1
        self.intensity_at_volume_fraction_x_per_cent = None  # 3.5.2
        self.volume_fraction_diff_intensity_fractions = None  # 3.5.3
        self.intensity_fraction_diff_volume_fractions = None  # 3.5.4

        self._fractions()

    def _fractions(self):
        for i in range(self.min_intensity, self.max_intensity + 1):
            # Calculate νi for each intensity
            self.fractional_volumes[i - 1] = 1 - np.sum(self.valid_values < i) / len(self.valid_values)
            # Calculate γi for each intensity
            self.intensity_fractions[i - 1] = (i - self.min_intensity) / (self.max_intensity - self.min_intensity)
            self.intensity[i - 1] = i

    def calc_volume_at_intensity_fraction(self, x):
        return np.max(self.fractional_volumes[self.intensity_fractions > x / 100])

    def calc_intensity_at_volume_fraction(self, x):
        return np.min(self.intensity[self.fractional_volumes <= x / 100])

    def calc_volume_fraction_diff_intensity_fractions(self):
        return self.calc_volume_at_intensity_fraction(10) - self.calc_volume_at_intensity_fraction(90)

    def calc_intensity_fraction_diff_volume_fractions(self):
        return self.calc_intensity_at_volume_fraction(10) - self.calc_intensity_at_volume_fraction(90)


class GLCM:

    def __init__(self, image, slice_weight=False, slice_median=False):
        self.image = image
        self.slice_weight = slice_weight
        self.slice_median = slice_median
        self.lvl = int(np.nanmax(self.image) + 1)
        self.glcm_2d_matrices = None
        self.glcm_3d_matrix = None

        self.glcm_2d_matrices = []
        self.slice_no_of_roi_voxels = []

        self.joint_max = 0  # 3.6.1
        self.joint_average = 0  # 3.6.2
        self.joint_var = 0  # 3.6.3
        self.joint_entropy = 0  # 3.6.4
        self.dif_average = 0  # 3.6.5
        self.dif_var = 0  # 3.6.6
        self.dif_entropy = 0  # 3.6.7
        self.sum_average = 0  # 3.6.8
        self.sum_var = 0  # 3.6.9
        self.sum_entropy = 0  # 3.6.10
        self.ang_second_moment = 0  # 3.6.11
        self.contrast = 0  # 3.6.12
        self.dissimilarity = 0  # 3.6.13
        self.inv_diff = 0  # 3.6.14
        self.norm_inv_diff = 0  # 3.6.15
        self.inv_diff_moment = 0  # 3.6.16
        self.norm_inv_diff_moment = 0  # 3.6.17
        self.inv_variance = 0  # 3.6.18
        self.cor = 0  # 3.6.19
        self.autocor = 0  # 3.6.20
        self.cluster_tendency = 0  # 3.6.21
        self.cluster_shade = 0  # 3.6.22
        self.cluster_prominence = 0  # 3.6.23
        self.inf_cor_1 = 0  # 3.6.24
        self.inf_cor_2 = 0  # 3.6.25

        self.joint_max_list = []  # 3.6.1
        self.joint_average_list = []  # 3.6.2
        self.joint_var_list = []  # 3.6.3
        self.joint_entropy_list = []  # 3.6.4
        self.dif_average_list = []  # 3.6.5
        self.dif_var_list = []  # 3.6.6
        self.dif_entropy_list = []  # 3.6.7
        self.sum_average_list = []  # 3.6.8
        self.sum_var_list = []  # 3.6.9
        self.sum_entropy_list = []  # 3.6.10
        self.ang_second_moment_list = []  # 3.6.11
        self.contrast_list = []  # 3.6.12
        self.dissimilarity_list = []  # 3.6.13
        self.inv_diff_list = []  # 3.6.14
        self.norm_inv_diff_list = []  # 3.6.15
        self.inv_diff_moment_list = []  # 3.6.16
        self.norm_inv_diff_moment_list = []  # 3.6.17
        self.inv_variance_list = []  # 3.6.18
        self.cor_list = []  # 3.6.19
        self.autocor_list = []  # 3.6.20
        self.cluster_tendency_list = []  # 3.6.21
        self.cluster_shade_list = []  # 3.6.22
        self.cluster_prominence_list = []  # 3.6.23
        self.inf_cor_1_list = []  # 3.6.24
        self.inf_cor_2_list = []  # 3.6.25

    def calc_glc_2d_matrices(self):

        def calc_2_d_glcm_slice(image, direction):

            dx, dy, dz = direction

            glcm_slice = np.zeros((self.lvl, self.lvl), dtype=int)

            rows, cols = image.shape
            for i in range(rows):
                for j in range(cols):
                    if (0 <= i + dx < rows) and (0 <= j + dy < cols):

                        row_pixel = image[i, j]
                        col_pixel = image[i + dx, j + dy]

                        # Check if either pixel is nan, skip if so
                        if np.isnan(row_pixel) or np.isnan(col_pixel):
                            continue

                        # Update GLCM
                        row_pixel, col_pixel = int(row_pixel), int(col_pixel)
                        glcm_slice[row_pixel, col_pixel] += 1

            return glcm_slice

        self.tot_no_of_roi_voxels = np.sum(~np.isnan(self.image))
        for z in range(self.image.shape[2]):
            if not np.all(np.isnan(self.image[:, :, z])):
                self.slice_no_of_roi_voxels.append(np.sum(~np.isnan(self.image[:, :, z])))
                z_slice_list = []
                for direction_2D in [[1, 0, 0], [1, 1, 0], [0, 1, 0], [-1, 1, 0]]:
                    glcm = calc_2_d_glcm_slice(self.image[:, :, z], direction_2D)
                    z_slice_list.append((glcm + glcm.T))

                self.glcm_2d_matrices.append(z_slice_list)
        self.glcm_2d_matrices = np.array(self.glcm_2d_matrices)

    def calc_glc_3d_matrix(self):  # arr, dir_vector, n_bits):

        self.glcm_3d_matrix = []

        for direction_3D in [[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [0, 1, -1],
                             [1, 0, 1], [1, 0, -1], [1, 1, 0], [1, -1, 0], [1, 1, 1],
                             [1, 1, -1], [1, -1, 1], [1, -1, -1]]:
            co_matrix = np.zeros((self.lvl, self.lvl), dtype=np.float64)

            len_arr, len_arr_0, len_arr_0_0 = len(self.image), len(self.image[0]), len(self.image[0][0])
            min_i, min_y, min_x = max(0, -direction_3D[2]), max(0, -direction_3D[1]), max(0, -direction_3D[0])
            max_i, max_y, max_x = min(len_arr, len_arr - direction_3D[2]), min(len_arr_0,
                                                                               len_arr_0 - direction_3D[1]), min(
                len_arr_0_0, len_arr_0_0 - direction_3D[0])

            arr1 = self.image[min_i:max_i, min_y:max_y, min_x:max_x]
            arr2 = self.image[min_i + direction_3D[2]:max_i + direction_3D[2],
                   min_y + direction_3D[1]:max_y + direction_3D[1], min_x + direction_3D[0]:max_x + direction_3D[0]]

            not_nan_mask = np.logical_and(~np.isnan(arr1), ~np.isnan(arr2))

            y_cm_values = arr1[not_nan_mask].astype(int)
            x_cm_values = arr2[not_nan_mask].astype(int)

            np.add.at(co_matrix, (y_cm_values, x_cm_values), 1)
            np.add.at(co_matrix, (x_cm_values, y_cm_values), 1)

            self.glcm_3d_matrix.append(co_matrix)

        self.glcm_3d_matrix = np.array(self.glcm_3d_matrix)

    def calc_glcm_3d_matrix_my(self):
        x, y, z = self.image.shape
        directions = [(0, 0, 1), (0, 1, -1), (0, 1, 0),
                      (0, 1, 1), (1, -1, -1), (1, -1, 0),
                      (1, -1, 1), (1, 0, -1), (1, 0, 0),
                      (1, 0, 1), (1, 1, -1), (1, 1, 0),
                      (1, 1, 1)
                      ]

        self.glcm_3d_matrix = []
        for direction in directions:
            glcm = np.zeros((self.lvl, self.lvl))
            for i in range(x):
                for j in range(y):
                    for k in range(z):
                        if np.isnan(self.image[i, j, k]):
                            continue  # Skip cells with np.nan

                        dx, dy, dz = direction
                        gr_lvl = int(self.image[i, j, k])

                        new_i, new_j, new_k = i + dx, j + dy, k + dz
                        if 0 <= new_i < x and 0 <= new_j < y and 0 <= new_k < z and not np.isnan(
                                self.image[new_i, new_j, new_k]):
                            glcm[gr_lvl, int(self.image[new_i, new_j, new_k])] += 1

            self.glcm_3d_matrix.append(glcm + glcm.T)
        self.glcm_3d_matrix = np.array(self.glcm_3d_matrix)

    def calc_p_minus(self, matrix):
        n_g = len(matrix)
        p_minus = np.zeros(n_g)
        for k in range(n_g - 1):
            for i in range(n_g):
                for j in range(n_g):
                    if abs(i - j) == k:
                        p_minus[k] += matrix[i][j]
        return p_minus

    def calc_p_plus(self, matrix):
        n_g = len(matrix)
        p_plus = np.zeros((2 * n_g - 1))
        for k in range(2, 2 * n_g):
            for i in range(n_g):
                for j in range(n_g):
                    if abs(i + j) == k:
                        p_plus[k] += matrix[i][j]
        return p_plus

    def calc_mu_i_and_sigma_i(self, matrix):
        p_i = np.sum(matrix, axis=0)
        mu_i = 0
        for i in range(len(p_i)):
            mu_i += p_i[i] * i

        sigma_i = 0
        for i in range(len(p_i)):
            sigma_i += (i - mu_i) ** 2 * p_i[i]
        sigma_i = np.sqrt(sigma_i)

        return mu_i, sigma_i

    def calc_correlation(self, matrix):
        i, j = np.indices(matrix.shape)
        mu_i, sigma_i = self.calc_mu_i_and_sigma_i(matrix)

        return (np.sum(matrix * i * j) - mu_i ** 2) / sigma_i ** 2

    def calc_cluster_tendency_shade_prominence(self, matrix, pover):

        mu_i, _ = self.calc_mu_i_and_sigma_i(matrix)
        i, j = np.indices(matrix.shape)

        return np.sum((i + j - 2 * mu_i) ** pover * matrix)

    def calc_information_correlation_1(self, matrix):
        p_i_j = matrix
        non_zero_mask_p_i_j = p_i_j != 0
        hxy = (-1) * np.sum(p_i_j[non_zero_mask_p_i_j] * np.log2(p_i_j[non_zero_mask_p_i_j]))

        p_i = np.sum(matrix, axis=0)
        non_zero_mask_p_i = p_i != 0
        hx = (-1) * np.sum(p_i[non_zero_mask_p_i] * np.log2(p_i[non_zero_mask_p_i]))

        hxy_1 = 0
        for i in range(len(p_i_j)):
            for j in range(len(p_i_j)):
                if p_i[i] != 0 and p_i[j] != 0:
                    hxy_1 += p_i_j[i][j] * np.log2(p_i[i] * p_i[j])
        hxy_1 *= (-1)

        return (hxy - hxy_1) / hx

    def calc_information_correlation_2(self, matrix):
        p_i_j = matrix
        non_zero_mask_p_i_j = p_i_j != 0
        hxy = (-1) * np.sum(p_i_j[non_zero_mask_p_i_j] * np.log2(p_i_j[non_zero_mask_p_i_j]))

        p_i = np.sum(matrix, axis=0)

        hxy_2 = 0
        for i in range(len(p_i_j)):
            for j in range(len(p_i_j)):
                if p_i[i] != 0 and p_i[j] != 0:
                    hxy_2 += p_i[i] * p_i[j] * np.log2(p_i[i] * p_i[j])
        hxy_2 *= (-1)

        return np.sqrt(1 - np.exp(-2 * (hxy_2 - hxy)))

    def calc_joint_average(self, matrix):
        i, j = np.indices(matrix.shape)
        return np.sum(matrix * i)

    def calc_joint_var(self, matrix, mu):
        i, j = np.indices(matrix.shape)
        return np.sum(matrix * (i - mu) ** 2)

    def calc_joint_entropy(self, matrix):
        non_zero_mask = matrix != 0
        return (-1) * np.sum(matrix[non_zero_mask] * np.log2(matrix[non_zero_mask]))

    def calc_diff_average(self, p_minus):
        k = np.indices(p_minus.shape)
        diff_average = np.sum(p_minus * k)
        return diff_average

    def calc_dif_var(self, p_minus, mu):
        k = np.indices(p_minus.shape)
        diff_var = np.sum(p_minus * (k - mu) ** 2)
        return diff_var

    def calc_diff_entropy(self, p_minus):
        non_zero_mask = p_minus != 0
        return (-1) * np.sum(p_minus[non_zero_mask] * np.log2(p_minus[non_zero_mask]))

    def calc_sum_average(self, p_plus):
        k = np.indices(p_plus.shape)
        sum_average = np.sum(p_plus * k)
        return sum_average

    def calc_sum_var(self, p_plus, mu):
        k = np.indices(p_plus.shape)
        sum_var = np.sum(p_plus * (k - mu) ** 2)
        return sum_var

    def calc_sum_entropy(self, p_plus):
        non_zero_mask = p_plus != 0
        return (-1) * np.sum(p_plus[non_zero_mask] * np.log2(p_plus[non_zero_mask]))

    def calc_second_moment(self, matrix):
        return np.sum(matrix * matrix)

    def calc_contrast(self, matrix):
        i, j = np.indices(matrix.shape)
        return np.sum(matrix * (i - j) ** 2)

    def calc_dissimilarity(self, matrix):
        i, j = np.indices(matrix.shape)
        return np.sum(matrix * abs(i - j))

    def calc_inverse_diff(self, matrix):
        i, j = np.indices(matrix.shape)
        return np.sum(matrix / (1 + abs(i - j)))

    def calc_norm_inv_diff(self, matrix):
        n_g = len(matrix) - 1
        i, j = np.indices(matrix.shape)
        return np.sum(matrix / (1 + abs(i - j) / n_g))

    def calc_inv_diff_moment(self, p_minus):
        k = np.indices(p_minus.shape)
        return np.sum(p_minus / (1 + k ** 2))

    def calc_norm_inv_diff_moment(self, p_minus):
        k = np.indices(p_minus.shape)
        n_g = len(p_minus) - 1
        return np.sum(p_minus / (1 + (k / n_g) ** 2))

    def calc_inv_variance(self, p_minus):
        k = np.indices(p_minus.shape)
        non_zero_mask = k != 0
        return np.sum(p_minus[1::] / (k[non_zero_mask] ** 2))

    def calc_autocor(self, matrix):
        i, j = np.indices(matrix.shape)
        return np.sum(matrix * i * j)

    def calc_2d_averaged_glcm_features(self):

        number_of_slices = self.glcm_2d_matrices.shape[0]
        number_of_directions = self.glcm_2d_matrices.shape[1]
        weights = []
        for i in range(number_of_slices):
            for j in range(number_of_directions):
                glcm_slice = self.glcm_2d_matrices[i][j] / np.sum(self.glcm_2d_matrices[i][j])
                weight = 1
                if self.slice_weight:
                    weight = self.slice_no_of_roi_voxels[i] / self.tot_no_of_roi_voxels
                weights.append(weight)

                self.joint_max_list.append(np.max(glcm_slice))
                glcm_ij_joint_average = self.calc_joint_average(glcm_slice)
                self.joint_average_list.append(glcm_ij_joint_average)
                self.joint_var_list.append(self.calc_joint_var(glcm_slice, glcm_ij_joint_average))
                self.joint_entropy_list.append(self.calc_joint_entropy(glcm_slice))

                p_minus = self.calc_p_minus(glcm_slice)
                glcm_ij_dif_average = self.calc_diff_average(p_minus)
                self.dif_average_list.append(glcm_ij_dif_average)
                self.dif_var_list.append(self.calc_dif_var(p_minus, glcm_ij_dif_average))
                self.dif_entropy_list.append(self.calc_diff_entropy(p_minus))

                p_plus = self.calc_p_plus(glcm_slice)
                glcm_ij_sum_average = self.calc_sum_average(p_plus)
                self.sum_average_list.append(glcm_ij_sum_average)
                self.sum_var_list.append(self.calc_sum_var(p_plus, glcm_ij_sum_average))
                self.sum_entropy_list.append(self.calc_sum_entropy(p_plus))

                self.ang_second_moment_list.append(self.calc_second_moment(glcm_slice))
                self.contrast_list.append(self.calc_contrast(glcm_slice))
                self.dissimilarity_list.append(self.calc_dissimilarity(glcm_slice))
                self.inv_diff_list.append(self.calc_inverse_diff(glcm_slice))
                self.norm_inv_diff_list.append(self.calc_norm_inv_diff(glcm_slice))
                self.inv_diff_moment_list.append(self.calc_inv_diff_moment(p_minus))
                self.norm_inv_diff_moment_list.append(self.calc_norm_inv_diff_moment(p_minus))
                self.inv_variance_list.append(self.calc_inv_variance(p_minus))

                self.cor_list.append(self.calc_correlation(glcm_slice))
                self.autocor_list.append(self.calc_autocor(glcm_slice))
                self.cluster_tendency_list.append(self.calc_cluster_tendency_shade_prominence(glcm_slice, 2))
                self.cluster_shade_list.append(self.calc_cluster_tendency_shade_prominence(glcm_slice, 3))
                self.cluster_prominence_list.append(self.calc_cluster_tendency_shade_prominence(glcm_slice, 4))

                self.inf_cor_1_list.append(self.calc_information_correlation_1(glcm_slice))
                self.inf_cor_2_list.append(self.calc_information_correlation_2(glcm_slice))

        if self.slice_median and not self.slice_weight:
            self.joint_max = np.median(self.joint_max_list)
            self.joint_average = np.median(self.joint_average_list)
            self.joint_var = np.median(self.joint_var_list)
            self.joint_entropy = np.median(self.joint_entropy_list)
            self.dif_average = np.median(self.dif_average_list)
            self.dif_var = np.median(self.dif_var_list)
            self.dif_entropy = np.median(self.dif_entropy_list)
            self.sum_average = np.median(self.sum_average_list)
            self.sum_var = np.median(self.sum_var_list)
            self.sum_entropy = np.median(self.sum_entropy_list)
            self.ang_second_moment = np.median(self.ang_second_moment_list)
            self.contrast = np.median(self.contrast_list)
            self.dissimilarity = np.median(self.dissimilarity_list)
            self.inv_diff = np.median(self.inv_diff_list)
            self.norm_inv_diff = np.median(self.norm_inv_diff_list)
            self.inv_diff_moment = np.median(self.inv_diff_moment_list)
            self.norm_inv_diff_moment = np.median(self.norm_inv_diff_moment_list)
            self.inv_variance = np.median(self.inv_variance_list)
            self.cor = np.median(self.cor_list)
            self.autocor = np.median(self.autocor_list)
            self.cluster_tendency = np.median(self.cluster_tendency_list)
            self.cluster_shade = np.median(self.cluster_shade_list)
            self.cluster_prominence = np.median(self.cluster_prominence_list)
            self.inf_cor_1 = np.median(self.inf_cor_1_list)
            self.inf_cor_2 = np.median(self.inf_cor_2_list)

        elif not self.slice_median:
            self.joint_max = np.average(self.joint_max_list, weights=weights)
            self.joint_average = np.average(self.joint_average_list, weights=weights)
            self.joint_var = np.average(self.joint_var_list, weights=weights)
            self.joint_entropy = np.average(self.joint_entropy_list, weights=weights)
            self.dif_average = np.average(self.dif_average_list, weights=weights)
            self.dif_var = np.average(self.dif_var_list, weights=weights)
            self.dif_entropy = np.average(self.dif_entropy_list, weights=weights)
            self.sum_average = np.average(self.sum_average_list, weights=weights)
            self.sum_var = np.average(self.sum_var_list, weights=weights)
            self.sum_entropy = np.average(self.sum_entropy_list, weights=weights)
            self.ang_second_moment = np.average(self.ang_second_moment_list, weights=weights)
            self.contrast = np.average(self.contrast_list, weights=weights)
            self.dissimilarity = np.average(self.dissimilarity_list, weights=weights)
            self.inv_diff = np.average(self.inv_diff_list, weights=weights)
            self.norm_inv_diff = np.average(self.norm_inv_diff_list, weights=weights)
            self.inv_diff_moment = np.average(self.inv_diff_moment_list, weights=weights)
            self.norm_inv_diff_moment = np.average(self.norm_inv_diff_moment_list, weights=weights)
            self.inv_variance = np.average(self.inv_variance_list, weights=weights)
            self.cor = np.average(self.cor_list, weights=weights)
            self.autocor = np.average(self.autocor_list, weights=weights)
            self.cluster_tendency = np.average(self.cluster_tendency_list, weights=weights)
            self.cluster_shade = np.average(self.cluster_shade_list, weights=weights)
            self.cluster_prominence = np.average(self.cluster_prominence_list, weights=weights)
            self.inf_cor_1 = np.average(self.inf_cor_1_list, weights=weights)
            self.inf_cor_2 = np.average(self.inf_cor_2_list, weights=weights)
        else:
            print('Weighted median not supported. Aborted!')
            return

    def calc_2d_slice_merged_glcm_features(self):

        number_of_slices = self.glcm_2d_matrices.shape[0]
        weights = []

        averaged_glcm = np.sum(self.glcm_2d_matrices, axis=1)
        for slice_id in range(number_of_slices):
            glcm_slice = averaged_glcm[slice_id] / np.sum(averaged_glcm[slice_id])
            weight = 1
            if self.slice_weight:
                weight = self.slice_no_of_roi_voxels[slice_id] / self.tot_no_of_roi_voxels
            weights.append(weight)

            self.joint_max_list.append(np.max(glcm_slice))
            glcm_i_joint_average = self.calc_joint_average(glcm_slice)
            self.joint_average_list.append(glcm_i_joint_average)
            self.joint_var_list.append(self.calc_joint_var(glcm_slice, glcm_i_joint_average))
            self.joint_entropy_list.append(self.calc_joint_entropy(glcm_slice))

            p_minus = self.calc_p_minus(glcm_slice)
            glcm_i_dif_average = self.calc_diff_average(p_minus)
            self.dif_average_list.append(glcm_i_dif_average)
            self.dif_var_list.append(self.calc_dif_var(p_minus, glcm_i_dif_average))
            self.dif_entropy_list.append(self.calc_diff_entropy(p_minus))

            p_plus = self.calc_p_plus(glcm_slice)
            glcm_i_sum_average = self.calc_sum_average(p_plus)
            self.sum_average_list.append(glcm_i_sum_average)
            self.sum_var_list.append(self.calc_sum_var(p_plus, glcm_i_sum_average))
            self.sum_entropy_list.append(self.calc_sum_entropy(p_plus))

            self.ang_second_moment_list.append(self.calc_second_moment(glcm_slice))
            self.contrast_list.append(self.calc_contrast(glcm_slice))
            self.dissimilarity_list.append(self.calc_dissimilarity(glcm_slice))
            self.inv_diff_list.append(self.calc_inverse_diff(glcm_slice))
            self.norm_inv_diff_list.append(self.calc_norm_inv_diff(glcm_slice))
            self.inv_diff_moment_list.append(self.calc_inv_diff_moment(p_minus))
            self.norm_inv_diff_moment_list.append(self.calc_norm_inv_diff_moment(p_minus))
            self.inv_variance_list.append(self.calc_inv_variance(p_minus))

            self.cor_list.append(self.calc_correlation(glcm_slice))
            self.autocor_list.append(self.calc_autocor(glcm_slice))
            self.cluster_tendency_list.append(self.calc_cluster_tendency_shade_prominence(glcm_slice, 2))
            self.cluster_shade_list.append(self.calc_cluster_tendency_shade_prominence(glcm_slice, 3))
            self.cluster_prominence_list.append(self.calc_cluster_tendency_shade_prominence(glcm_slice, 4))

            self.inf_cor_1_list.append(self.calc_information_correlation_1(glcm_slice))
            self.inf_cor_2_list.append(self.calc_information_correlation_2(glcm_slice))

        if self.slice_median and not self.slice_weight:
            self.joint_max = np.median(self.joint_max_list)
            self.joint_average = np.median(self.joint_average_list)
            self.joint_var = np.median(self.joint_var_list)
            self.joint_entropy = np.median(self.joint_entropy_list)
            self.dif_average = np.median(self.dif_average_list)
            self.dif_var = np.median(self.dif_var_list)
            self.dif_entropy = np.median(self.dif_entropy_list)
            self.sum_average = np.median(self.sum_average_list)
            self.sum_var = np.median(self.sum_var_list)
            self.sum_entropy = np.median(self.sum_entropy_list)
            self.ang_second_moment = np.median(self.ang_second_moment_list)
            self.contrast = np.median(self.contrast_list)
            self.dissimilarity = np.median(self.dissimilarity_list)
            self.inv_diff = np.median(self.inv_diff_list)
            self.norm_inv_diff = np.median(self.norm_inv_diff_list)
            self.inv_diff_moment = np.median(self.inv_diff_moment_list)
            self.norm_inv_diff_moment = np.median(self.norm_inv_diff_moment_list)
            self.inv_variance = np.median(self.inv_variance_list)
            self.cor = np.median(self.cor_list)
            self.autocor = np.median(self.autocor_list)
            self.cluster_tendency = np.median(self.cluster_tendency_list)
            self.cluster_shade = np.median(self.cluster_shade_list)
            self.cluster_prominence = np.median(self.cluster_prominence_list)
            self.inf_cor_1 = np.median(self.inf_cor_1_list)
            self.inf_cor_2 = np.median(self.inf_cor_2_list)

        elif not self.slice_median:
            self.joint_max = np.average(self.joint_max_list, weights=weights)
            self.joint_average = np.average(self.joint_average_list, weights=weights)
            self.joint_var = np.average(self.joint_var_list, weights=weights)
            self.joint_entropy = np.average(self.joint_entropy_list, weights=weights)
            self.dif_average = np.average(self.dif_average_list, weights=weights)
            self.dif_var = np.average(self.dif_var_list, weights=weights)
            self.dif_entropy = np.average(self.dif_entropy_list, weights=weights)
            self.sum_average = np.average(self.sum_average_list, weights=weights)
            self.sum_var = np.average(self.sum_var_list, weights=weights)
            self.sum_entropy = np.average(self.sum_entropy_list, weights=weights)
            self.ang_second_moment = np.average(self.ang_second_moment_list, weights=weights)
            self.contrast = np.average(self.contrast_list, weights=weights)
            self.dissimilarity = np.average(self.dissimilarity_list, weights=weights)
            self.inv_diff = np.average(self.inv_diff_list, weights=weights)
            self.norm_inv_diff = np.average(self.norm_inv_diff_list, weights=weights)
            self.inv_diff_moment = np.average(self.inv_diff_moment_list, weights=weights)
            self.norm_inv_diff_moment = np.average(self.norm_inv_diff_moment_list, weights=weights)
            self.inv_variance = np.average(self.inv_variance_list, weights=weights)
            self.cor = np.average(self.cor_list, weights=weights)
            self.autocor = np.average(self.autocor_list, weights=weights)
            self.cluster_tendency = np.average(self.cluster_tendency_list, weights=weights)
            self.cluster_shade = np.average(self.cluster_shade_list, weights=weights)
            self.cluster_prominence = np.average(self.cluster_prominence_list, weights=weights)
            self.inf_cor_1 = np.average(self.inf_cor_1_list, weights=weights)
            self.inf_cor_2 = np.average(self.inf_cor_2_list, weights=weights)
        else:
            print('Weighted median not supported. Aborted!')
            return

    def calc_2_5d_merged_glcm_features(self):
        glcm = np.sum(np.sum(self.glcm_2d_matrices, axis=1), axis=0)

        glcm = glcm / np.sum(glcm)

        self.joint_max = np.max(glcm)
        glcm_joint_average = self.calc_joint_average(glcm)
        self.joint_average = glcm_joint_average
        self.joint_var = self.calc_joint_var(glcm, glcm_joint_average)
        self.joint_entropy = self.calc_joint_entropy(glcm)

        p_minus = self.calc_p_minus(glcm)
        glcm_dif_average = self.calc_diff_average(p_minus)
        self.dif_average = glcm_dif_average
        self.dif_var = self.calc_dif_var(p_minus, glcm_dif_average)
        self.dif_entropy = self.calc_diff_entropy(p_minus)

        p_plus = self.calc_p_plus(glcm)
        glcm_sum_average = self.calc_sum_average(p_plus)
        self.sum_average = glcm_sum_average
        self.sum_var = self.calc_sum_var(p_plus, glcm_sum_average)
        self.sum_entropy = self.calc_sum_entropy(p_plus)

        self.ang_second_moment = self.calc_second_moment(glcm)
        self.contrast = self.calc_contrast(glcm)
        self.dissimilarity = self.calc_dissimilarity(glcm)
        self.inv_diff = self.calc_inverse_diff(glcm)
        self.norm_inv_diff = self.calc_norm_inv_diff(glcm)
        self.inv_diff_moment = self.calc_inv_diff_moment(p_minus)
        self.norm_inv_diff_moment = self.calc_norm_inv_diff_moment(p_minus)
        self.inv_variance = self.calc_inv_variance(p_minus)

        self.cor = self.calc_correlation(glcm)
        self.autocor = self.calc_autocor(glcm)
        self.cluster_tendency = self.calc_cluster_tendency_shade_prominence(glcm, 2)
        self.cluster_shade = self.calc_cluster_tendency_shade_prominence(glcm, 3)
        self.cluster_prominence = self.calc_cluster_tendency_shade_prominence(glcm, 4)

        self.inf_cor_1 = self.calc_information_correlation_1(glcm)
        self.inf_cor_2 = self.calc_information_correlation_2(glcm)

    def calc_2_5d_direction_merged_glcm_features(self):
        number_of_directions = self.glcm_2d_matrices.shape[1]

        averaged_glcm = np.sum(self.glcm_2d_matrices, axis=0)  # / number_of_slices

        for i in range(number_of_directions):
            M_i = averaged_glcm[i] / np.sum(averaged_glcm[i])
            self.joint_max += np.max(M_i)
            glcm_i_joint_average = self.calc_joint_average(M_i)
            self.joint_average += glcm_i_joint_average
            self.joint_var += self.calc_joint_var(M_i, glcm_i_joint_average)
            self.joint_entropy += self.calc_joint_entropy(M_i)

            p_minus = self.calc_p_minus(M_i)
            glcm_i_dif_average = self.calc_diff_average(p_minus)
            self.dif_average += glcm_i_dif_average
            self.dif_var += self.calc_dif_var(p_minus, glcm_i_dif_average)
            self.dif_entropy += self.calc_diff_entropy(p_minus)

            p_plus = self.calc_p_plus(M_i)
            glcm_i_sum_average = self.calc_sum_average(p_plus)
            self.sum_average += glcm_i_sum_average
            self.sum_var += self.calc_sum_var(p_plus, glcm_i_sum_average)
            self.sum_entropy += self.calc_sum_entropy(p_plus)

            self.ang_second_moment += self.calc_second_moment(M_i)
            self.contrast += self.calc_contrast(M_i)
            self.dissimilarity += self.calc_dissimilarity(M_i)
            self.inv_diff += self.calc_inverse_diff(M_i)
            self.norm_inv_diff += self.calc_norm_inv_diff(M_i)
            self.inv_diff_moment += self.calc_inv_diff_moment(p_minus)
            self.norm_inv_diff_moment += self.calc_norm_inv_diff_moment(p_minus)
            self.inv_variance += self.calc_inv_variance(p_minus)

            self.cor += self.calc_correlation(M_i)
            self.autocor += self.calc_autocor(M_i)
            self.cluster_tendency += self.calc_cluster_tendency_shade_prominence(M_i, 2)
            self.cluster_shade += self.calc_cluster_tendency_shade_prominence(M_i, 3)
            self.cluster_prominence += self.calc_cluster_tendency_shade_prominence(M_i, 4)

            self.inf_cor_1 += self.calc_information_correlation_1(M_i)
            self.inf_cor_2 += self.calc_information_correlation_2(M_i)

        self.joint_max /= number_of_directions
        self.joint_average /= number_of_directions
        self.joint_var /= number_of_directions
        self.joint_entropy /= number_of_directions

        self.dif_average /= number_of_directions
        self.dif_var /= number_of_directions
        self.dif_entropy /= number_of_directions
        self.sum_average /= number_of_directions
        self.sum_var /= number_of_directions
        self.sum_entropy /= number_of_directions

        self.ang_second_moment /= number_of_directions
        self.contrast /= number_of_directions
        self.dissimilarity /= number_of_directions
        self.inv_diff /= number_of_directions
        self.norm_inv_diff /= number_of_directions
        self.inv_diff_moment /= number_of_directions
        self.norm_inv_diff_moment /= number_of_directions
        self.inv_variance /= number_of_directions

        self.cor /= number_of_directions
        self.autocor /= number_of_directions
        self.cluster_tendency /= number_of_directions
        self.cluster_shade /= number_of_directions
        self.cluster_prominence /= number_of_directions

        self.inf_cor_1 /= number_of_directions
        self.inf_cor_2 /= number_of_directions

    def calc_3d_averaged_glcm_features(self):

        nuber_of_dir_3D = 13

        for glcm_i in self.glcm_3d_matrix:
            norm = np.sum(glcm_i)
            glcm_i = glcm_i / norm
            self.joint_max += np.max(glcm_i)
            glcm_i_joint_average = self.calc_joint_average(glcm_i)
            self.joint_average += glcm_i_joint_average
            self.joint_var += self.calc_joint_var(glcm_i, glcm_i_joint_average)
            self.joint_entropy += self.calc_joint_entropy(glcm_i)

            p_minus = self.calc_p_minus(glcm_i)
            glcm_i_dif_average = self.calc_diff_average(p_minus)
            self.dif_average += glcm_i_dif_average
            self.dif_var += self.calc_dif_var(p_minus, glcm_i_dif_average)
            self.dif_entropy += self.calc_diff_entropy(p_minus)

            p_plus = self.calc_p_plus(glcm_i)
            glcm_i_sum_average = self.calc_sum_average(p_plus)
            self.sum_average += glcm_i_sum_average
            self.sum_var += self.calc_sum_var(p_plus, glcm_i_sum_average)
            self.sum_entropy += self.calc_sum_entropy(p_plus)

            self.ang_second_moment += self.calc_second_moment(glcm_i)
            self.contrast += self.calc_contrast(glcm_i)
            self.dissimilarity += self.calc_dissimilarity(glcm_i)
            self.inv_diff += self.calc_inverse_diff(glcm_i)
            self.norm_inv_diff += self.calc_norm_inv_diff(glcm_i)
            self.inv_diff_moment += self.calc_inv_diff_moment(p_minus)
            self.norm_inv_diff_moment += self.calc_norm_inv_diff_moment(p_minus)
            self.inv_variance += self.calc_inv_variance(p_minus)

            self.cor += self.calc_correlation(glcm_i)
            self.autocor += self.calc_autocor(glcm_i)
            self.cluster_tendency += self.calc_cluster_tendency_shade_prominence(glcm_i, 2)
            self.cluster_shade += self.calc_cluster_tendency_shade_prominence(glcm_i, 3)
            self.cluster_prominence += self.calc_cluster_tendency_shade_prominence(glcm_i, 4)

            self.inf_cor_1 += self.calc_information_correlation_1(glcm_i)
            self.inf_cor_2 += self.calc_information_correlation_2(glcm_i)

        self.joint_max /= nuber_of_dir_3D
        self.joint_average /= nuber_of_dir_3D
        self.joint_var /= nuber_of_dir_3D
        self.joint_entropy /= nuber_of_dir_3D

        self.dif_average /= nuber_of_dir_3D
        self.dif_var /= nuber_of_dir_3D
        self.dif_entropy /= nuber_of_dir_3D
        self.sum_average /= nuber_of_dir_3D
        self.sum_var /= nuber_of_dir_3D
        self.sum_entropy /= nuber_of_dir_3D

        self.ang_second_moment /= nuber_of_dir_3D
        self.contrast /= nuber_of_dir_3D
        self.dissimilarity /= nuber_of_dir_3D
        self.inv_diff /= nuber_of_dir_3D
        self.norm_inv_diff /= nuber_of_dir_3D
        self.inv_diff_moment /= nuber_of_dir_3D
        self.norm_inv_diff_moment /= nuber_of_dir_3D
        self.inv_variance /= nuber_of_dir_3D

        self.cor /= nuber_of_dir_3D
        self.autocor /= nuber_of_dir_3D
        self.cluster_tendency /= nuber_of_dir_3D
        self.cluster_shade /= nuber_of_dir_3D
        self.cluster_prominence /= nuber_of_dir_3D

        self.inf_cor_1 /= nuber_of_dir_3D
        self.inf_cor_2 /= nuber_of_dir_3D

    def calc_3d_merged_glcm_features(self):

        M = np.sum(self.glcm_3d_matrix, axis=0)
        M = M / np.sum(M)

        self.joint_max = np.max(M)
        self.joint_average = self.calc_joint_average(M)
        self.joint_var = self.calc_joint_var(M, self.joint_average)
        self.joint_entropy = self.calc_joint_entropy(M)

        p_minus = self.calc_p_minus(M)
        M_dif_average = self.calc_diff_average(p_minus)
        self.dif_average = M_dif_average
        self.dif_var = self.calc_dif_var(p_minus, M_dif_average)
        self.dif_entropy = self.calc_diff_entropy(p_minus)

        p_plus = self.calc_p_plus(M)
        M_sum_average = self.calc_sum_average(p_plus)
        self.sum_average = M_sum_average
        self.sum_var = self.calc_sum_var(p_plus, M_sum_average)
        self.sum_entropy = self.calc_sum_entropy(p_plus)

        self.ang_second_moment = self.calc_second_moment(M)
        self.contrast = self.calc_contrast(M)
        self.dissimilarity = self.calc_dissimilarity(M)
        self.inv_diff = self.calc_inverse_diff(M)
        self.norm_inv_diff = self.calc_norm_inv_diff(M)
        self.inv_diff_moment = self.calc_inv_diff_moment(p_minus)
        self.norm_inv_diff_moment = self.calc_norm_inv_diff_moment(p_minus)
        self.inv_variance = self.calc_inv_variance(p_minus)

        self.cor = self.calc_correlation(M)
        self.autocor = self.calc_autocor(M)
        self.cluster_tendency = self.calc_cluster_tendency_shade_prominence(M, 2)
        self.cluster_shade = self.calc_cluster_tendency_shade_prominence(M, 3)
        self.cluster_prominence = self.calc_cluster_tendency_shade_prominence(M, 4)

        self.inf_cor_1 = self.calc_information_correlation_1(M)
        self.inf_cor_2 = self.calc_information_correlation_2(M)


class GLRLM_GLSZM_GLDZM_NGLDM:
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

        self.short_runs_emphasis = 0
        self.long_runs_emphasis = 0
        self.low_grey_level_run_emphasis = 0
        self.high_gr_lvl_emphasis = 0
        self.short_low_gr_lvl_emphasis = 0
        self.short_high_gr_lvl_emphasis = 0
        self.long_low_gr_lvl_emphasis = 0
        self.long_high_gr_lvl_emphasis = 0
        self.non_uniformity = 0
        self.norm_non_uniformity = 0
        self.length_non_uniformity = 0
        self.norm_length_non_uniformity = 0
        self.percentage = 0
        self.gr_lvl_var = 0
        self.length_var = 0
        self.entropy = 0
        self.energy = 0

        self.short_runs_emphasis_list = []
        self.long_runs_emphasis_list = []
        self.low_grey_level_run_emphasis_list = []
        self.high_gr_lvl_emphasis_list = []
        self.short_low_gr_lvl_emphasis_list = []
        self.short_high_gr_lvl_emphasis_list = []
        self.long_low_gr_lvl_emphasis_list = []
        self.long_high_gr_lvl_emphasis_list = []
        self.non_uniformity_list = []
        self.norm_non_uniformity_list = []
        self.length_non_uniformity_list = []
        self.norm_length_non_uniformity_list = []
        self.percentage_list = []
        self.gr_lvl_var_list = []
        self.length_var_list = []
        self.entropy_list = []
        self.energy_list = []

    def reset_fields(self):

        self.short_runs_emphasis = 0
        self.long_runs_emphasis = 0
        self.low_grey_level_run_emphasis = 0
        self.high_gr_lvl_emphasis = 0
        self.short_low_gr_lvl_emphasis = 0
        self.short_high_gr_lvl_emphasis = 0
        self.long_low_gr_lvl_emphasis = 0
        self.long_high_gr_lvl_emphasis = 0
        self.non_uniformity = 0
        self.norm_non_uniformity = 0
        self.length_non_uniformity = 0
        self.norm_length_non_uniformity = 0
        self.percentage = 0
        self.gr_lvl_var = 0
        self.length_var = 0
        self.entropy = 0
        self.energy = 0

        self.short_runs_emphasis_list = []
        self.long_runs_emphasis_list = []
        self.low_grey_level_run_emphasis_list = []
        self.high_gr_lvl_emphasis_list = []
        self.short_low_gr_lvl_emphasis_list = []
        self.short_high_gr_lvl_emphasis_list = []
        self.long_low_gr_lvl_emphasis_list = []
        self.long_high_gr_lvl_emphasis_list = []
        self.non_uniformity_list = []
        self.norm_non_uniformity_list = []
        self.length_non_uniformity_list = []
        self.norm_length_non_uniformity_list = []
        self.percentage_list = []
        self.gr_lvl_var_list = []
        self.length_var_list = []
        self.entropy_list = []
        self.energy_list = []

    def calc_glrl_2d_matrices(self):

        x, y, z = self.image.shape
        directions = [(0, 1), (1, -1), (1, 0), (1, 1)]

        self.glrlm_2D_matrices = []
        self.no_of_roi_voxels = []
        for z_slice_index in self.range_z:
            z_slice_list = []
            z_slice = self.image[:, :, z_slice_index]
            # if np.sum(~np.isnan(z_slice)) != 0:
            self.no_of_roi_voxels.append(np.sum(~np.isnan(z_slice)))
            for direction in directions:
                rlm = np.zeros((self.lvl, max(x, y)))
                visited_array = np.zeros((x, y), dtype=bool)
                for i in self.range_x:
                    for j in self.range_y:
                        if visited_array[i, j] or np.isnan(z_slice[i, j]):
                            continue

                        dx, dy = direction
                        run_len = 1
                        gr_lvl = int(z_slice[i, j])
                        visited_array[i, j] = True

                        new_i, new_j = i + dx, j + dy
                        while (0 <= new_i < x and 0 <= new_j < y and not np.isnan(z_slice[new_i, new_j])
                               and z_slice[new_i, new_j] == gr_lvl and not visited_array[new_i, new_j]):
                            run_len += 1
                            visited_array[new_i, new_j] = True
                            new_i += dx
                            new_j += dy

                        rlm[gr_lvl, run_len - 1] += 1

                z_slice_list.append(rlm)

            self.glrlm_2D_matrices.append(z_slice_list)
        self.glrlm_2D_matrices = np.array(self.glrlm_2D_matrices).astype(np.int64)

    def calc_glrl_3d_matrix(self):

        x, y, z = self.image.shape
        directions = [
            (0, 0, 1), (0, 1, -1), (0, 1, 0),
            (0, 1, 1), (1, -1, -1), (1, -1, 0),
            (1, -1, 1), (1, 0, -1), (1, 0, 0),
            (1, 0, 1), (1, 1, -1), (1, 1, 0),
            (1, 1, 1)]

        self.glrlm_3D_matrix = []
        for direction in directions:
            rlm = np.zeros((self.lvl, max(x, y, z)))
            visited_array = np.zeros(self.image.shape, dtype=bool)
            for i in self.range_x:
                for j in self.range_y:
                    for k in self.range_z:
                        if visited_array[i, j, k] or np.isnan(self.image[i, j, k]):
                            continue  # Skip already visited cells or cells with np.nan

                        dx, dy, dz = direction
                        run_len = 1
                        gr_lvl = int(self.image[i, j, k])
                        visited_array[i, j, k] = True

                        new_i, new_j, new_k = i + dx, j + dy, k + dz
                        while 0 <= new_i < x and 0 <= new_j < y and 0 <= new_k < z and not np.isnan(
                                self.image[new_i, new_j, new_k]) and self.image[new_i, new_j, new_k] == gr_lvl and not \
                                visited_array[new_i, new_j, new_k]:
                            run_len += 1
                            visited_array[new_i, new_j, new_k] = True
                            new_i += dx
                            new_j += dy
                            new_k += dz

                        rlm[gr_lvl, run_len - 1] += 1

            self.glrlm_3D_matrix.append(rlm)
        self.glrlm_3D_matrix = np.array(self.glrlm_3D_matrix).astype(np.int64)

    def calc_glsz_gldz_3d_matrices(self, mask):

        flattened_array = self.image.flatten()
        _, counts = np.unique(flattened_array[~np.isnan(flattened_array)], return_counts=True)
        max_region_size = np.max(counts)

        def calc_dist_map_3d(image_orig):
            image = image_orig.copy()
            # image[np.isnan(image)] = 0
            larger_array = np.zeros((image.shape[0] + 2, image.shape[1] + 2, image.shape[2] + 2))
            larger_array[1:-1, 1:-1, 1:-1] = image
            distance_map = distance_transform_cdt(larger_array, metric='taxicab')[1:-1, 1:-1, 1:-1].astype(float)

            return distance_map

        dist_map = calc_dist_map_3d(mask)
        glszm = np.zeros((self.lvl, max_region_size), dtype=int)
        gldzm = np.zeros((self.lvl, np.max(self.image.shape)), dtype=int)

        def find_connected_region_3d(start, intensity):
            stack = [start]
            size = 0
            min_dist = np.inf
            while stack:
                x, y, z = stack.pop()
                if 0 <= x < self.image.shape[0] and 0 <= y < self.image.shape[1] and 0 <= z < self.image.shape[2]:
                    if visited[x, y, z] == 0 and self.image[x, y, z] == intensity:
                        visited[x, y, z] = 1
                        size += 1
                        min_dist = min(min_dist, dist_map[x, y, z])
                        for dx in [-1, 0, 1]:
                            for dy in [-1, 0, 1]:
                                for dz in [-1, 0, 1]:
                                    stack.append((x + dx, y + dy, z + dz))
            return size, min_dist

        visited = np.zeros_like(self.image, dtype=int)
        for x in self.range_x:
            for y in self.range_y:
                for z in self.range_z:
                    if visited[x, y, z] == 0 and not np.isnan(self.image[x, y, z]):
                        intensity = int(self.image[x, y, z])
                        size, min_dist = find_connected_region_3d((x, y, z), intensity)
                        if size > 0:
                            glszm[intensity, size - 1] += 1
                            gldzm[intensity, int(min_dist) - 1] += 1

        self.glszm_3D_matrix = glszm.astype(np.int64)
        self.gldzm_3D_matrix = gldzm.astype(np.int64)

    def calc_glsz_gldz_2d_matrices(self, mask):

        max_region_size_list = []
        for z_slice in self.image.T:
            if np.sum(~np.isnan(z_slice)) != 0:
                flattened_array = z_slice.flatten()
                _, counts = np.unique(flattened_array[~np.isnan(flattened_array)], return_counts=True)
                max_region_size_list.append(np.max(counts))
        max_region_size = max(max_region_size_list)

        def calc_dist_map_2d(image_orig):
            image = image_orig.copy()
            larger_array = np.zeros((image.shape[0] + 2, image.shape[1] + 2))
            larger_array[1:-1, 1:-1] = image
            distance_map = distance_transform_cdt(larger_array, metric='taxicab')[1:-1, 1:-1].astype(float)

            return distance_map

        def find_connected_region_2d(start, intensity):
            stack = [start]
            size = 0
            min_dist = np.inf
            while stack:
                x, y = stack.pop()
                if 0 <= x < z_slice.shape[0] and 0 <= y < z_slice.shape[1]:
                    if visited[x, y] == 0 and z_slice[x, y] == intensity:
                        visited[x, y] = 1
                        size += 1
                        min_dist = min(min_dist, dist_map[x, y])
                        for dx in [-1, 0, 1]:
                            for dy in [-1, 0, 1]:
                                stack.append((x + dx, y + dy))
            return size, min_dist

        self.glszm_2D_matrices = []
        self.gldzm_2D_matrices = []
        self.no_of_roi_voxels = []
        for z_slice_index in self.range_z:
            z_slice = self.image[:, :, z_slice_index]
            # if np.sum(~np.isnan(z_slice)) != 0:
            z_slice_mask = mask[:, :, z_slice_index]
            self.no_of_roi_voxels.append(np.sum(~np.isnan(z_slice)))
            dist_map = calc_dist_map_2d(z_slice_mask)
            glszm = np.zeros((self.lvl, max_region_size), dtype=int)
            gldzm = np.zeros((self.lvl, np.max(self.image.shape)), dtype=int)
            visited = np.zeros((self.image.shape[0], self.image.shape[1]), dtype=int)
            for x in self.range_x:
                for y in self.range_y:
                    if visited[x, y] == 0 and not np.isnan(z_slice[x, y]):
                        intensity = int(z_slice[x, y])
                        size, min_dist = find_connected_region_2d((x, y), intensity)
                        if size > 0:
                            glszm[intensity, size - 1] += 1
                            gldzm[intensity, int(min_dist) - 1] += 1
            self.glszm_2D_matrices.append(glszm.astype(np.int64))
            self.gldzm_2D_matrices.append(gldzm.astype(np.int64))

        self.glszm_2D_matrices = np.array(self.glszm_2D_matrices)
        self.gldzm_2D_matrices = np.array(self.gldzm_2D_matrices)

    def calc_ngld_3d_matrix(self):

        ngldm = np.zeros((self.lvl, 27))

        valid_offsets = [(x, y, z) for x in range(-1, 2) for y in range(-1, 2) for z in range(-1, 2) if
                         (z, y, x) != (0, 0, 0)]
        for lvl in range(self.lvl):
            x_indices, y_indices, z_indices = np.where(self.image == lvl)
            for x, y, z in zip(x_indices, y_indices, z_indices):
                j_k = 0
                for off in valid_offsets:
                    neighbors = (x + off[0], y + off[1], z + off[2])
                    if all(0 <= n < sz for n, sz in zip(neighbors, self.image.shape)) and not np.isnan(
                            self.image[neighbors]) and self.image[neighbors] == lvl:
                        j_k += 1
                ngldm[int(lvl), int(j_k)] += 1
        self.ngldm_3D_matrix = ngldm

    def calc_ngld_2d_matrices(self):
        self.ngldm_2d_matrices = []
        self.no_of_roi_voxels = []

        def calc_ngldm_slice(array):
            ngldm = np.zeros((self.lvl, 9))
            valid_offsets = [(x, y) for x in range(-1, 2) for y in range(-1, 2) if (y, x) != (0, 0)]
            for lvl in range(self.lvl):
                x_indices, y_indices = np.where(array == lvl)
                for x, y in zip(x_indices, y_indices):
                    j_k = 0
                    for off in valid_offsets:
                        neighbors = (x + off[0], y + off[1])
                        if all(0 <= n < sz for n, sz in zip(neighbors, array.shape)) and not np.isnan(
                                array[neighbors]) and array[neighbors] == lvl:
                            j_k += 1
                    ngldm[int(lvl), int(j_k)] += 1
            return ngldm

        for z_slice_index in range(self.image.shape[2]):
            z_slice = self.image[:, :, z_slice_index]
            if np.nansum(z_slice) != 0:
                self.no_of_roi_voxels.append(np.sum(~np.isnan(z_slice)))
                self.ngldm_2d_matrices.append(calc_ngldm_slice(z_slice))
        self.ngldm_2d_matrices = np.array(self.ngldm_2d_matrices)

    def calc_short_emphasis(self, m):

        Ns = np.sum(m)
        _, j = np.indices(m.shape)

        return np.sum(m / (j + 1) ** 2) / Ns

    def calc_long_emphasis(self, m):

        Ns = np.sum(m)
        _, j = np.indices(m.shape)

        return np.sum(m * (j + 1) ** 2) / Ns

    def calc_low_gr_lvl_emphasis(self, M):

        Ns = np.sum(M)
        i, _ = np.indices(M.shape)
        mask = i != 0

        return np.sum(M[mask] / (i[mask]) ** 2) / Ns

    def calc_high_gr_lvl_emphasis(self, m):

        Ns = np.sum(m)
        i, _ = np.indices(m.shape)

        return np.sum(m * i ** 2) / Ns

    def calc_short_low_gr_lvl_emphasis(self, M):

        Ns = np.sum(M)
        i, j = np.indices(M.shape)
        mask = i != 0

        M_j = M[mask] / (i[mask] ** 2)

        return np.sum(M_j / ((j[mask] + 1) ** 2)) / Ns

    def calc_short_high_gr_lvl_emphasis(self, M):

        Ns = np.sum(M)
        i, j = np.indices(M.shape)

        return np.sum((i ** 2 * M) / ((j + 1)) ** 2) / Ns

    def calc_long_low_gr_lvl_emphasis(self, M):

        Ns = np.sum(M)
        i, j = np.indices(M.shape)
        mask = i != 0

        return np.sum((M[mask] * (j[mask] + 1) ** 2) / (i[mask]) ** 2) / Ns

    def calc_long_high_gr_lvl_emphasis(self, M):

        n_s = np.sum(M)
        i, j = np.indices(M.shape)

        return np.sum(M * (j + 1) ** 2 * i ** 2) / n_s

    def calc_non_uniformity(self, M):

        Ns = np.sum(M)

        return np.sum(np.sum(M, axis=1) ** 2) / Ns

    def calc_norm_non_uniformity(self, M):

        Ns = np.sum(M)

        return np.sum(np.sum(M, axis=1) ** 2) / Ns ** 2

    def calc_length_non_uniformity(self, M):

        Ns = np.sum(M)

        return np.sum(np.sum(M, axis=0) ** 2) / Ns

    def calc_norm_length_non_uniformity(self, M):

        Ns = np.sum(M)

        return np.sum(np.sum(M, axis=0) ** 2) / Ns ** 2

    def calc_percentage(self, M, Nv):

        Ns = np.sum(M)

        return Ns / Nv

    def calc_gr_lvl_var(self, M):

        Ns = np.sum(M)
        i, _ = np.indices(M.shape)
        mu = np.sum(M * i / Ns)

        return np.sum((i - mu) ** 2 * (M / Ns))

    def calc_length_var(self, M):

        Ns = np.sum(M)
        _, j = np.indices(M.shape)
        mu = np.sum(M * j / Ns)

        return np.sum((j - mu) ** 2 * (M / Ns))

    def calc_entropy(self, M):

        Ns = np.sum(M)
        mask = M != 0

        return np.sum((M[mask] / Ns) * np.log2((M[mask] / Ns))) * (-1)

    def calc_energy(self, M):

        Ns = np.sum(M)
        mask = M != 0

        return np.sum((M[mask] / Ns) ** 2)

    def calc_2d_averaged_glrlm_features(self):

        number_of_slices = self.glrlm_2D_matrices.shape[0]
        number_of_directions = self.glrlm_2D_matrices.shape[1]
        weights = []

        for i in range(number_of_slices):
            for j in range(number_of_directions):
                M_ij = self.glrlm_2D_matrices[i][j]
                weight = 1
                if self.slice_weight:
                    weight = self.no_of_roi_voxels[i] / self.tot_no_of_roi_voxels
                weights.append(weight)

                self.short_runs_emphasis_list.append(self.calc_short_emphasis(M_ij))
                self.long_runs_emphasis_list.append(self.calc_long_emphasis(M_ij))
                self.low_grey_level_run_emphasis_list.append(self.calc_low_gr_lvl_emphasis(M_ij))
                self.high_gr_lvl_emphasis_list.append(self.calc_high_gr_lvl_emphasis(M_ij))
                self.short_low_gr_lvl_emphasis_list.append(self.calc_short_low_gr_lvl_emphasis(M_ij))
                self.short_high_gr_lvl_emphasis_list.append(self.calc_short_high_gr_lvl_emphasis(M_ij))
                self.long_low_gr_lvl_emphasis_list.append(self.calc_long_low_gr_lvl_emphasis(M_ij))
                self.long_high_gr_lvl_emphasis_list.append(self.calc_long_high_gr_lvl_emphasis(M_ij))
                self.non_uniformity_list.append(self.calc_non_uniformity(M_ij))
                self.norm_non_uniformity_list.append(self.calc_norm_non_uniformity(M_ij))
                self.length_non_uniformity_list.append(self.calc_length_non_uniformity(M_ij))
                self.norm_length_non_uniformity_list.append(self.calc_norm_length_non_uniformity(M_ij))
                self.percentage_list.append(self.calc_percentage(M_ij, self.no_of_roi_voxels[i]))
                self.gr_lvl_var_list.append(self.calc_gr_lvl_var(M_ij))
                self.length_var_list.append(self.calc_length_var(M_ij))
                self.entropy_list.append(self.calc_entropy(M_ij))

        if self.slice_median and not self.slice_weight:
            self.short_runs_emphasis = np.median(self.short_runs_emphasis_list)
            self.long_runs_emphasis = np.median(self.long_runs_emphasis_list)
            self.low_grey_level_run_emphasis = np.median(self.low_grey_level_run_emphasis_list)
            self.high_gr_lvl_emphasis = np.median(self.high_gr_lvl_emphasis_list)
            self.short_low_gr_lvl_emphasis = np.median(self.short_low_gr_lvl_emphasis_list)
            self.short_high_gr_lvl_emphasis = np.median(self.short_high_gr_lvl_emphasis_list)
            self.long_low_gr_lvl_emphasis = np.median(self.long_low_gr_lvl_emphasis_list)
            self.long_high_gr_lvl_emphasis = np.median(self.long_high_gr_lvl_emphasis_list)
            self.non_uniformity = np.median(self.non_uniformity_list)
            self.norm_non_uniformity = np.median(self.norm_non_uniformity_list)
            self.length_non_uniformity = np.median(self.length_non_uniformity_list)
            self.norm_length_non_uniformity = np.median(self.norm_length_non_uniformity_list)
            self.percentage = np.median(self.percentage_list)
            self.gr_lvl_var = np.median(self.gr_lvl_var_list)
            self.length_var = np.median(self.length_var_list)
            self.entropy = np.median(self.entropy_list)
        elif not self.slice_median:
            self.short_runs_emphasis = np.average(self.short_runs_emphasis_list, weights=weights)
            self.long_runs_emphasis = np.average(self.long_runs_emphasis_list, weights=weights)
            self.low_grey_level_run_emphasis = np.average(self.low_grey_level_run_emphasis_list, weights=weights)
            self.high_gr_lvl_emphasis = np.average(self.high_gr_lvl_emphasis_list, weights=weights)
            self.short_low_gr_lvl_emphasis = np.average(self.short_low_gr_lvl_emphasis_list, weights=weights)
            self.short_high_gr_lvl_emphasis = np.average(self.short_high_gr_lvl_emphasis_list, weights=weights)
            self.long_low_gr_lvl_emphasis = np.average(self.long_low_gr_lvl_emphasis_list, weights=weights)
            self.long_high_gr_lvl_emphasis = np.average(self.long_high_gr_lvl_emphasis_list, weights=weights)
            self.non_uniformity = np.average(self.non_uniformity_list, weights=weights)
            self.norm_non_uniformity = np.average(self.norm_non_uniformity_list, weights=weights)
            self.length_non_uniformity = np.average(self.length_non_uniformity_list, weights=weights)
            self.norm_length_non_uniformity = np.average(self.norm_length_non_uniformity_list, weights=weights)
            self.percentage = np.average(self.percentage_list, weights=weights)
            self.gr_lvl_var = np.average(self.gr_lvl_var_list, weights=weights)
            self.length_var = np.average(self.length_var_list, weights=weights)
            self.entropy = np.average(self.entropy_list, weights=weights)

    def calc_2d_slice_merged_glrlm_features(self):

        number_of_slices = self.glrlm_2D_matrices.shape[0]
        number_of_directions = self.glrlm_2D_matrices.shape[1]
        averaged_M = np.sum(self.glrlm_2D_matrices, axis=1)  # / number_of_directions
        weights = []

        for i in range(number_of_slices):
            M_i = averaged_M[i]
            weight = 1
            if self.slice_weight:
                weight = self.no_of_roi_voxels[i] / self.tot_no_of_roi_voxels
            weights.append(weight)

            self.short_runs_emphasis_list.append(self.calc_short_emphasis(M_i))
            self.long_runs_emphasis_list.append(self.calc_long_emphasis(M_i))
            self.low_grey_level_run_emphasis_list.append(self.calc_low_gr_lvl_emphasis(M_i))
            self.high_gr_lvl_emphasis_list.append(self.calc_high_gr_lvl_emphasis(M_i))
            self.short_low_gr_lvl_emphasis_list.append(self.calc_short_low_gr_lvl_emphasis(M_i))
            self.short_high_gr_lvl_emphasis_list.append(self.calc_short_high_gr_lvl_emphasis(M_i))
            self.long_low_gr_lvl_emphasis_list.append(self.calc_long_low_gr_lvl_emphasis(M_i))
            self.long_high_gr_lvl_emphasis_list.append(self.calc_long_high_gr_lvl_emphasis(M_i))
            self.non_uniformity_list.append(self.calc_non_uniformity(M_i))
            self.norm_non_uniformity_list.append(self.calc_norm_non_uniformity(M_i))
            self.length_non_uniformity_list.append(self.calc_length_non_uniformity(M_i))
            self.norm_length_non_uniformity_list.append(self.calc_norm_length_non_uniformity(M_i))
            self.percentage_list.append(
                self.calc_percentage(M_i, self.no_of_roi_voxels[i]) * (1 / number_of_directions))
            self.gr_lvl_var_list.append(self.calc_gr_lvl_var(M_i))
            self.length_var_list.append(self.calc_length_var(M_i))
            self.entropy_list.append(self.calc_entropy(M_i))

        if self.slice_median and not self.slice_weight:
            self.short_runs_emphasis = np.median(self.short_runs_emphasis_list)
            self.long_runs_emphasis = np.median(self.long_runs_emphasis_list)
            self.low_grey_level_run_emphasis = np.median(self.low_grey_level_run_emphasis_list)
            self.high_gr_lvl_emphasis = np.median(self.high_gr_lvl_emphasis_list)
            self.short_low_gr_lvl_emphasis = np.median(self.short_low_gr_lvl_emphasis_list)
            self.short_high_gr_lvl_emphasis = np.median(self.short_high_gr_lvl_emphasis_list)
            self.long_low_gr_lvl_emphasis = np.median(self.long_low_gr_lvl_emphasis_list)
            self.long_high_gr_lvl_emphasis = np.median(self.long_high_gr_lvl_emphasis_list)
            self.non_uniformity = np.median(self.non_uniformity_list)
            self.norm_non_uniformity = np.median(self.norm_non_uniformity_list)
            self.length_non_uniformity = np.median(self.length_non_uniformity_list)
            self.norm_length_non_uniformity = np.median(self.norm_length_non_uniformity_list)
            self.percentage = np.median(self.percentage_list)
            self.gr_lvl_var = np.median(self.gr_lvl_var_list)
            self.length_var = np.median(self.length_var_list)
            self.entropy = np.median(self.entropy_list)
        elif not self.slice_median:
            self.short_runs_emphasis = np.average(self.short_runs_emphasis_list, weights=weights)
            self.long_runs_emphasis = np.average(self.long_runs_emphasis_list, weights=weights)
            self.low_grey_level_run_emphasis = np.average(self.low_grey_level_run_emphasis_list, weights=weights)
            self.high_gr_lvl_emphasis = np.average(self.high_gr_lvl_emphasis_list, weights=weights)
            self.short_low_gr_lvl_emphasis = np.average(self.short_low_gr_lvl_emphasis_list, weights=weights)
            self.short_high_gr_lvl_emphasis = np.average(self.short_high_gr_lvl_emphasis_list, weights=weights)
            self.long_low_gr_lvl_emphasis = np.average(self.long_low_gr_lvl_emphasis_list, weights=weights)
            self.long_high_gr_lvl_emphasis = np.average(self.long_high_gr_lvl_emphasis_list, weights=weights)
            self.non_uniformity = np.average(self.non_uniformity_list, weights=weights)
            self.norm_non_uniformity = np.average(self.norm_non_uniformity_list, weights=weights)
            self.length_non_uniformity = np.average(self.length_non_uniformity_list, weights=weights)
            self.norm_length_non_uniformity = np.average(self.norm_length_non_uniformity_list, weights=weights)
            self.percentage = np.average(self.percentage_list, weights=weights)
            self.gr_lvl_var = np.average(self.gr_lvl_var_list, weights=weights)
            self.length_var = np.average(self.length_var_list, weights=weights)
            self.entropy = np.average(self.entropy_list, weights=weights)

    def calc_2_5d_merged_glrlm_features(self):

        number_of_directions = self.glrlm_2D_matrices.shape[1]
        glrlm = np.sum(np.sum(self.glrlm_2D_matrices, axis=1), axis=0)

        self.short_runs_emphasis = self.calc_short_emphasis(glrlm)
        self.long_runs_emphasis = self.calc_long_emphasis(glrlm)
        self.low_grey_level_run_emphasis = self.calc_low_gr_lvl_emphasis(glrlm)
        self.high_gr_lvl_emphasis = self.calc_high_gr_lvl_emphasis(glrlm)
        self.short_low_gr_lvl_emphasis = self.calc_short_low_gr_lvl_emphasis(glrlm)
        self.short_high_gr_lvl_emphasis = self.calc_short_high_gr_lvl_emphasis(glrlm)
        self.long_low_gr_lvl_emphasis = self.calc_long_low_gr_lvl_emphasis(glrlm)
        self.long_high_gr_lvl_emphasis = self.calc_long_high_gr_lvl_emphasis(glrlm)
        self.non_uniformity = self.calc_non_uniformity(glrlm)
        self.norm_non_uniformity = self.calc_norm_non_uniformity(glrlm)
        self.length_non_uniformity = self.calc_length_non_uniformity(glrlm)
        self.norm_length_non_uniformity = self.calc_norm_length_non_uniformity(glrlm)
        self.percentage = self.calc_percentage(glrlm, np.sum(self.no_of_roi_voxels)) / number_of_directions
        self.gr_lvl_var = self.calc_gr_lvl_var(glrlm)
        self.length_var = self.calc_length_var(glrlm)
        self.entropy = self.calc_entropy(glrlm)

    def calc_2_5d_direction_merged_glrlm_features(self):

        number_of_directions = self.glrlm_2D_matrices.shape[1]
        averaged_glrlm = np.sum(self.glrlm_2D_matrices, axis=0)

        for i in range(number_of_directions):
            glrlm_i = averaged_glrlm[i]
            self.short_runs_emphasis += self.calc_short_emphasis(glrlm_i)
            self.long_runs_emphasis += self.calc_long_emphasis(glrlm_i)
            self.low_grey_level_run_emphasis += self.calc_low_gr_lvl_emphasis(glrlm_i)
            self.high_gr_lvl_emphasis += self.calc_high_gr_lvl_emphasis(glrlm_i)
            self.short_low_gr_lvl_emphasis += self.calc_short_low_gr_lvl_emphasis(glrlm_i)
            self.short_high_gr_lvl_emphasis += self.calc_short_high_gr_lvl_emphasis(glrlm_i)
            self.long_low_gr_lvl_emphasis += self.calc_long_low_gr_lvl_emphasis(glrlm_i)
            self.long_high_gr_lvl_emphasis += self.calc_long_high_gr_lvl_emphasis(glrlm_i)
            self.non_uniformity += self.calc_non_uniformity(glrlm_i)
            self.norm_non_uniformity += self.calc_norm_non_uniformity(glrlm_i)
            self.length_non_uniformity += self.calc_length_non_uniformity(glrlm_i)
            self.norm_length_non_uniformity += self.calc_norm_length_non_uniformity(glrlm_i)
            self.percentage += self.calc_percentage(glrlm_i, np.sum(self.no_of_roi_voxels))
            self.gr_lvl_var += self.calc_gr_lvl_var(glrlm_i)
            self.length_var += self.calc_length_var(glrlm_i)
            self.entropy += self.calc_entropy(glrlm_i)

        self.short_runs_emphasis /= number_of_directions
        self.long_runs_emphasis /= number_of_directions
        self.low_grey_level_run_emphasis /= number_of_directions
        self.high_gr_lvl_emphasis /= number_of_directions
        self.short_low_gr_lvl_emphasis /= number_of_directions
        self.short_high_gr_lvl_emphasis /= number_of_directions
        self.long_low_gr_lvl_emphasis /= number_of_directions
        self.long_high_gr_lvl_emphasis /= number_of_directions
        self.non_uniformity /= number_of_directions
        self.norm_non_uniformity /= number_of_directions
        self.length_non_uniformity /= number_of_directions
        self.norm_length_non_uniformity /= number_of_directions
        self.percentage /= number_of_directions
        self.gr_lvl_var /= number_of_directions
        self.length_var /= number_of_directions
        self.entropy /= number_of_directions

    def calc_3d_averaged_glrlm_features(self):

        number_of_directions = self.glrlm_3D_matrix.shape[0]
        for i in range(number_of_directions):
            M_i = self.glrlm_3D_matrix[i]

            self.short_runs_emphasis += self.calc_short_emphasis(M_i)
            self.long_runs_emphasis += self.calc_long_emphasis(M_i)
            self.low_grey_level_run_emphasis += self.calc_low_gr_lvl_emphasis(M_i)
            self.high_gr_lvl_emphasis += self.calc_high_gr_lvl_emphasis(M_i)
            self.short_low_gr_lvl_emphasis += self.calc_short_low_gr_lvl_emphasis(M_i)
            self.short_high_gr_lvl_emphasis += self.calc_short_high_gr_lvl_emphasis(M_i)
            self.long_low_gr_lvl_emphasis += self.calc_long_low_gr_lvl_emphasis(M_i)
            self.long_high_gr_lvl_emphasis += self.calc_long_high_gr_lvl_emphasis(M_i)
            self.non_uniformity += self.calc_non_uniformity(M_i)
            self.norm_non_uniformity += self.calc_norm_non_uniformity(M_i)
            self.length_non_uniformity += self.calc_length_non_uniformity(M_i)
            self.norm_length_non_uniformity += self.calc_norm_length_non_uniformity(M_i)
            self.percentage += self.calc_percentage(M_i, self.tot_no_of_roi_voxels)
            self.gr_lvl_var += self.calc_gr_lvl_var(M_i)
            self.length_var += self.calc_length_var(M_i)
            self.entropy += self.calc_entropy(M_i)

        self.short_runs_emphasis /= number_of_directions
        self.long_runs_emphasis /= number_of_directions
        self.low_grey_level_run_emphasis /= number_of_directions
        self.high_gr_lvl_emphasis /= number_of_directions
        self.short_low_gr_lvl_emphasis /= number_of_directions
        self.short_high_gr_lvl_emphasis /= number_of_directions
        self.long_low_gr_lvl_emphasis /= number_of_directions
        self.long_high_gr_lvl_emphasis /= number_of_directions
        self.non_uniformity /= number_of_directions
        self.norm_non_uniformity /= number_of_directions
        self.length_non_uniformity /= number_of_directions
        self.norm_length_non_uniformity /= number_of_directions
        self.percentage /= number_of_directions
        self.gr_lvl_var /= number_of_directions
        self.length_var /= number_of_directions
        self.entropy /= number_of_directions

    def calc_3d_merged_glrlm_features(self):

        number_of_directions = self.glrlm_3D_matrix.shape[0]
        M = np.sum(self.glrlm_3D_matrix, axis=0)

        self.short_runs_emphasis = self.calc_short_emphasis(M)
        self.long_runs_emphasis = self.calc_long_emphasis(M)
        self.low_grey_level_run_emphasis = self.calc_low_gr_lvl_emphasis(M)
        self.high_gr_lvl_emphasis = self.calc_high_gr_lvl_emphasis(M)
        self.short_low_gr_lvl_emphasis = self.calc_short_low_gr_lvl_emphasis(M)
        self.short_high_gr_lvl_emphasis = self.calc_short_high_gr_lvl_emphasis(M)
        self.long_low_gr_lvl_emphasis = self.calc_long_low_gr_lvl_emphasis(M)
        self.long_high_gr_lvl_emphasis = self.calc_long_high_gr_lvl_emphasis(M)
        self.non_uniformity = self.calc_non_uniformity(M)
        self.norm_non_uniformity = self.calc_norm_non_uniformity(M)
        self.length_non_uniformity = self.calc_length_non_uniformity(M)
        self.norm_length_non_uniformity = self.calc_norm_length_non_uniformity(M)
        self.percentage = self.calc_percentage(M, self.tot_no_of_roi_voxels) / number_of_directions
        self.gr_lvl_var = self.calc_gr_lvl_var(M)
        self.length_var = self.calc_length_var(M)
        self.entropy = self.calc_entropy(M)

    def calc_2d_glszm_features(self):

        number_of_slices = self.glszm_2D_matrices.shape[0]
        weights = []

        for i in range(number_of_slices):
            glszm_slice = self.glszm_2D_matrices[i]
            weight = 1
            if self.slice_weight:
                weight = self.no_of_roi_voxels[i] / self.tot_no_of_roi_voxels
            weights.append(weight)

            self.short_runs_emphasis_list.append(self.calc_short_emphasis(glszm_slice))
            self.long_runs_emphasis_list.append(self.calc_long_emphasis(glszm_slice))
            self.low_grey_level_run_emphasis_list.append(self.calc_low_gr_lvl_emphasis(glszm_slice))
            self.high_gr_lvl_emphasis_list.append(self.calc_high_gr_lvl_emphasis(glszm_slice))
            self.short_low_gr_lvl_emphasis_list.append(self.calc_short_low_gr_lvl_emphasis(glszm_slice))
            self.short_high_gr_lvl_emphasis_list.append(self.calc_short_high_gr_lvl_emphasis(glszm_slice))
            self.long_low_gr_lvl_emphasis_list.append(self.calc_long_low_gr_lvl_emphasis(glszm_slice))
            self.long_high_gr_lvl_emphasis_list.append(self.calc_long_high_gr_lvl_emphasis(glszm_slice))
            self.non_uniformity_list.append(self.calc_non_uniformity(glszm_slice))
            self.norm_non_uniformity_list.append(self.calc_norm_non_uniformity(glszm_slice))
            self.length_non_uniformity_list.append(self.calc_length_non_uniformity(glszm_slice))
            self.norm_length_non_uniformity_list.append(self.calc_norm_length_non_uniformity(glszm_slice))
            self.percentage_list.append(self.calc_percentage(glszm_slice, self.no_of_roi_voxels[i]))
            self.gr_lvl_var_list.append(self.calc_gr_lvl_var(glszm_slice))
            self.length_var_list.append(self.calc_length_var(glszm_slice))
            self.entropy_list.append(self.calc_entropy(glszm_slice))

        if self.slice_median and not self.slice_weight:
            self.short_runs_emphasis = np.median(self.short_runs_emphasis_list)
            self.long_runs_emphasis = np.median(self.long_runs_emphasis_list)
            self.low_grey_level_run_emphasis = np.median(self.low_grey_level_run_emphasis_list)
            self.high_gr_lvl_emphasis = np.median(self.high_gr_lvl_emphasis_list)
            self.short_low_gr_lvl_emphasis = np.median(self.short_low_gr_lvl_emphasis_list)
            self.short_high_gr_lvl_emphasis = np.median(self.short_high_gr_lvl_emphasis_list)
            self.long_low_gr_lvl_emphasis = np.median(self.long_low_gr_lvl_emphasis_list)
            self.long_high_gr_lvl_emphasis = np.median(self.long_high_gr_lvl_emphasis_list)
            self.non_uniformity = np.median(self.non_uniformity_list)
            self.norm_non_uniformity = np.median(self.norm_non_uniformity_list)
            self.length_non_uniformity = np.median(self.length_non_uniformity_list)
            self.norm_length_non_uniformity = np.median(self.norm_length_non_uniformity_list)
            self.percentage = np.median(self.percentage_list)
            self.gr_lvl_var = np.median(self.gr_lvl_var_list)
            self.length_var = np.median(self.length_var_list)
            self.entropy = np.median(self.entropy_list)
        elif not self.slice_median:
            self.short_runs_emphasis = np.average(self.short_runs_emphasis_list, weights=weights)
            self.long_runs_emphasis = np.average(self.long_runs_emphasis_list, weights=weights)
            self.low_grey_level_run_emphasis = np.average(self.low_grey_level_run_emphasis_list, weights=weights)
            self.high_gr_lvl_emphasis = np.average(self.high_gr_lvl_emphasis_list, weights=weights)
            self.short_low_gr_lvl_emphasis = np.average(self.short_low_gr_lvl_emphasis_list, weights=weights)
            self.short_high_gr_lvl_emphasis = np.average(self.short_high_gr_lvl_emphasis_list, weights=weights)
            self.long_low_gr_lvl_emphasis = np.average(self.long_low_gr_lvl_emphasis_list, weights=weights)
            self.long_high_gr_lvl_emphasis = np.average(self.long_high_gr_lvl_emphasis_list, weights=weights)
            self.non_uniformity = np.average(self.non_uniformity_list, weights=weights)
            self.norm_non_uniformity = np.average(self.norm_non_uniformity_list, weights=weights)
            self.length_non_uniformity = np.average(self.length_non_uniformity_list, weights=weights)
            self.norm_length_non_uniformity = np.average(self.norm_length_non_uniformity_list, weights=weights)
            self.percentage = np.average(self.percentage_list, weights=weights)
            self.gr_lvl_var = np.average(self.gr_lvl_var_list, weights=weights)
            self.length_var = np.average(self.length_var_list, weights=weights)
            self.entropy = np.average(self.entropy_list, weights=weights)

    def calc_2d_gldzm_features(self):

        number_of_slices = self.gldzm_2D_matrices.shape[0]
        weights = []

        for i in range(number_of_slices):
            M = self.gldzm_2D_matrices[i]
            weight = 1
            if self.slice_weight:
                weight = self.no_of_roi_voxels[i] / self.tot_no_of_roi_voxels
            weights.append(weight)

            self.short_runs_emphasis_list.append(self.calc_short_emphasis(M))
            self.long_runs_emphasis_list.append(self.calc_long_emphasis(M))
            self.low_grey_level_run_emphasis_list.append(self.calc_low_gr_lvl_emphasis(M))
            self.high_gr_lvl_emphasis_list.append(self.calc_high_gr_lvl_emphasis(M))
            self.short_low_gr_lvl_emphasis_list.append(self.calc_short_low_gr_lvl_emphasis(M))
            self.short_high_gr_lvl_emphasis_list.append(self.calc_short_high_gr_lvl_emphasis(M))
            self.long_low_gr_lvl_emphasis_list.append(self.calc_long_low_gr_lvl_emphasis(M))
            self.long_high_gr_lvl_emphasis_list.append(self.calc_long_high_gr_lvl_emphasis(M))
            self.non_uniformity_list.append(self.calc_non_uniformity(M))
            self.norm_non_uniformity_list.append(self.calc_norm_non_uniformity(M))
            self.length_non_uniformity_list.append(self.calc_length_non_uniformity(M))
            self.norm_length_non_uniformity_list.append(self.calc_norm_length_non_uniformity(M))
            self.percentage_list.append(self.calc_percentage(M, self.no_of_roi_voxels[i]))
            self.gr_lvl_var_list.append(self.calc_gr_lvl_var(M))
            self.length_var_list.append(self.calc_length_var(M))
            self.entropy_list.append(self.calc_entropy(M))

        if self.slice_median and not self.slice_weight:
            self.short_runs_emphasis = np.median(self.short_runs_emphasis_list)
            self.long_runs_emphasis = np.median(self.long_runs_emphasis_list)
            self.low_grey_level_run_emphasis = np.median(self.low_grey_level_run_emphasis_list)
            self.high_gr_lvl_emphasis = np.median(self.high_gr_lvl_emphasis_list)
            self.short_low_gr_lvl_emphasis = np.median(self.short_low_gr_lvl_emphasis_list)
            self.short_high_gr_lvl_emphasis = np.median(self.short_high_gr_lvl_emphasis_list)
            self.long_low_gr_lvl_emphasis = np.median(self.long_low_gr_lvl_emphasis_list)
            self.long_high_gr_lvl_emphasis = np.median(self.long_high_gr_lvl_emphasis_list)
            self.non_uniformity = np.median(self.non_uniformity_list)
            self.norm_non_uniformity = np.median(self.norm_non_uniformity_list)
            self.length_non_uniformity = np.median(self.length_non_uniformity_list)
            self.norm_length_non_uniformity = np.median(self.norm_length_non_uniformity_list)
            self.percentage = np.median(self.percentage_list)
            self.gr_lvl_var = np.median(self.gr_lvl_var_list)
            self.length_var = np.median(self.length_var_list)
            self.entropy = np.median(self.entropy_list)
        elif not self.slice_median:
            self.short_runs_emphasis = np.average(self.short_runs_emphasis_list, weights=weights)
            self.long_runs_emphasis = np.average(self.long_runs_emphasis_list, weights=weights)
            self.low_grey_level_run_emphasis = np.average(self.low_grey_level_run_emphasis_list, weights=weights)
            self.high_gr_lvl_emphasis = np.average(self.high_gr_lvl_emphasis_list, weights=weights)
            self.short_low_gr_lvl_emphasis = np.average(self.short_low_gr_lvl_emphasis_list, weights=weights)
            self.short_high_gr_lvl_emphasis = np.average(self.short_high_gr_lvl_emphasis_list, weights=weights)
            self.long_low_gr_lvl_emphasis = np.average(self.long_low_gr_lvl_emphasis_list, weights=weights)
            self.long_high_gr_lvl_emphasis = np.average(self.long_high_gr_lvl_emphasis_list, weights=weights)
            self.non_uniformity = np.average(self.non_uniformity_list, weights=weights)
            self.norm_non_uniformity = np.average(self.norm_non_uniformity_list, weights=weights)
            self.length_non_uniformity = np.average(self.length_non_uniformity_list, weights=weights)
            self.norm_length_non_uniformity = np.average(self.norm_length_non_uniformity_list, weights=weights)
            self.percentage = np.average(self.percentage_list, weights=weights)
            self.gr_lvl_var = np.average(self.gr_lvl_var_list, weights=weights)
            self.length_var = np.average(self.length_var_list, weights=weights)
            self.entropy = np.average(self.entropy_list, weights=weights)

    def calc_2_5d_glszm_features(self):

        glszm = np.sum(self.glszm_2D_matrices, axis=0)

        self.short_runs_emphasis = self.calc_short_emphasis(glszm)
        self.long_runs_emphasis = self.calc_long_emphasis(glszm)
        self.low_grey_level_run_emphasis = self.calc_low_gr_lvl_emphasis(glszm)
        self.high_gr_lvl_emphasis = self.calc_high_gr_lvl_emphasis(glszm)
        self.short_low_gr_lvl_emphasis = self.calc_short_low_gr_lvl_emphasis(glszm)
        self.short_high_gr_lvl_emphasis = self.calc_short_high_gr_lvl_emphasis(glszm)
        self.long_low_gr_lvl_emphasis = self.calc_long_low_gr_lvl_emphasis(glszm)
        self.long_high_gr_lvl_emphasis = self.calc_long_high_gr_lvl_emphasis(glszm)
        self.non_uniformity = self.calc_non_uniformity(glszm)
        self.norm_non_uniformity = self.calc_norm_non_uniformity(glszm)
        self.length_non_uniformity = self.calc_length_non_uniformity(glszm)
        self.norm_length_non_uniformity = self.calc_norm_length_non_uniformity(glszm)
        self.percentage = self.calc_percentage(glszm, np.sum(self.no_of_roi_voxels))
        self.gr_lvl_var = self.calc_gr_lvl_var(glszm)
        self.length_var = self.calc_length_var(glszm)
        self.entropy = self.calc_entropy(glszm)

    def calc_2_5d_gldzm_features(self):

        M = np.sum(self.gldzm_2D_matrices, axis=0)

        self.short_runs_emphasis = self.calc_short_emphasis(M)
        self.long_runs_emphasis = self.calc_long_emphasis(M)
        self.low_grey_level_run_emphasis = self.calc_low_gr_lvl_emphasis(M)
        self.high_gr_lvl_emphasis = self.calc_high_gr_lvl_emphasis(M)
        self.short_low_gr_lvl_emphasis = self.calc_short_low_gr_lvl_emphasis(M)
        self.short_high_gr_lvl_emphasis = self.calc_short_high_gr_lvl_emphasis(M)
        self.long_low_gr_lvl_emphasis = self.calc_long_low_gr_lvl_emphasis(M)
        self.long_high_gr_lvl_emphasis = self.calc_long_high_gr_lvl_emphasis(M)
        self.non_uniformity = self.calc_non_uniformity(M)
        self.norm_non_uniformity = self.calc_norm_non_uniformity(M)
        self.length_non_uniformity = self.calc_length_non_uniformity(M)
        self.norm_length_non_uniformity = self.calc_norm_length_non_uniformity(M)
        self.percentage = self.calc_percentage(M, np.sum(self.no_of_roi_voxels))
        self.gr_lvl_var = self.calc_gr_lvl_var(M)
        self.length_var = self.calc_length_var(M)
        self.entropy = self.calc_entropy(M)

    def calc_3d_glszm_features(self):

        M = self.glszm_3D_matrix

        self.short_runs_emphasis = self.calc_short_emphasis(M)
        self.long_runs_emphasis = self.calc_long_emphasis(M)
        self.low_grey_level_run_emphasis = self.calc_low_gr_lvl_emphasis(M)
        self.high_gr_lvl_emphasis = self.calc_high_gr_lvl_emphasis(M)
        self.short_low_gr_lvl_emphasis = self.calc_short_low_gr_lvl_emphasis(M)
        self.short_high_gr_lvl_emphasis = self.calc_short_high_gr_lvl_emphasis(M)
        self.long_low_gr_lvl_emphasis = self.calc_long_low_gr_lvl_emphasis(M)
        self.long_high_gr_lvl_emphasis = self.calc_long_high_gr_lvl_emphasis(M)
        self.non_uniformity = self.calc_non_uniformity(M)
        self.norm_non_uniformity = self.calc_norm_non_uniformity(M)
        self.length_non_uniformity = self.calc_length_non_uniformity(M)
        self.norm_length_non_uniformity = self.calc_norm_length_non_uniformity(M)
        self.percentage = self.calc_percentage(M, self.tot_no_of_roi_voxels)
        self.gr_lvl_var = self.calc_gr_lvl_var(M)
        self.length_var = self.calc_length_var(M)
        self.entropy = self.calc_entropy(M)

    def calc_3d_gldzm_features(self):

        ngdzm = self.gldzm_3D_matrix.astype(np.int64)

        self.short_runs_emphasis = self.calc_short_emphasis(ngdzm)
        self.long_runs_emphasis = self.calc_long_emphasis(ngdzm)
        self.low_grey_level_run_emphasis = self.calc_low_gr_lvl_emphasis(ngdzm)
        self.high_gr_lvl_emphasis = self.calc_high_gr_lvl_emphasis(ngdzm)
        self.short_low_gr_lvl_emphasis = self.calc_short_low_gr_lvl_emphasis(ngdzm)
        self.short_high_gr_lvl_emphasis = self.calc_short_high_gr_lvl_emphasis(ngdzm)
        self.long_low_gr_lvl_emphasis = self.calc_long_low_gr_lvl_emphasis(ngdzm)
        self.long_high_gr_lvl_emphasis = self.calc_long_high_gr_lvl_emphasis(ngdzm)
        self.non_uniformity = self.calc_non_uniformity(ngdzm)
        self.norm_non_uniformity = self.calc_norm_non_uniformity(ngdzm)
        self.length_non_uniformity = self.calc_length_non_uniformity(ngdzm)
        self.norm_length_non_uniformity = self.calc_norm_length_non_uniformity(ngdzm)
        self.percentage = self.calc_percentage(ngdzm, self.tot_no_of_roi_voxels)
        self.gr_lvl_var = self.calc_gr_lvl_var(ngdzm)
        self.length_var = self.calc_length_var(ngdzm)
        self.entropy = self.calc_entropy(ngdzm)

    def calc_2d_ngldm_features(self):

        number_of_slices = self.ngldm_2d_matrices.shape[0]
        weights = []

        for i in range(number_of_slices):
            ngldm_matrix = self.ngldm_2d_matrices[i]
            weight = 1
            if self.slice_weight:
                weight = self.no_of_roi_voxels[i] / self.tot_no_of_roi_voxels
            weights.append(weight)

            self.short_runs_emphasis_list.append(self.calc_short_emphasis(ngldm_matrix))
            self.long_runs_emphasis_list.append(self.calc_long_emphasis(ngldm_matrix))
            self.low_grey_level_run_emphasis_list.append(self.calc_low_gr_lvl_emphasis(ngldm_matrix))
            self.high_gr_lvl_emphasis_list.append(self.calc_high_gr_lvl_emphasis(ngldm_matrix))
            self.short_low_gr_lvl_emphasis_list.append(self.calc_short_low_gr_lvl_emphasis(ngldm_matrix))
            self.short_high_gr_lvl_emphasis_list.append(self.calc_short_high_gr_lvl_emphasis(ngldm_matrix))
            self.long_low_gr_lvl_emphasis_list.append(self.calc_long_low_gr_lvl_emphasis(ngldm_matrix))
            self.long_high_gr_lvl_emphasis_list.append(self.calc_long_high_gr_lvl_emphasis(ngldm_matrix))
            self.non_uniformity_list.append(self.calc_non_uniformity(ngldm_matrix))
            self.norm_non_uniformity_list.append(self.calc_norm_non_uniformity(ngldm_matrix))
            self.length_non_uniformity_list.append(self.calc_length_non_uniformity(ngldm_matrix))
            self.norm_length_non_uniformity_list.append(self.calc_norm_length_non_uniformity(ngldm_matrix))
            self.percentage_list.append(self.calc_percentage(ngldm_matrix, self.no_of_roi_voxels[i]))
            self.gr_lvl_var_list.append(self.calc_gr_lvl_var(ngldm_matrix))
            self.length_var_list.append(self.calc_length_var(ngldm_matrix))
            self.entropy_list.append(self.calc_entropy(ngldm_matrix))
            self.energy_list.append(self.calc_energy(ngldm_matrix))

        if self.slice_median and not self.slice_weight:
            self.short_runs_emphasis = np.median(self.short_runs_emphasis_list)
            self.long_runs_emphasis = np.median(self.long_runs_emphasis_list)
            self.low_grey_level_run_emphasis = np.median(self.low_grey_level_run_emphasis_list)
            self.high_gr_lvl_emphasis = np.median(self.high_gr_lvl_emphasis_list)
            self.short_low_gr_lvl_emphasis = np.median(self.short_low_gr_lvl_emphasis_list)
            self.short_high_gr_lvl_emphasis = np.median(self.short_high_gr_lvl_emphasis_list)
            self.long_low_gr_lvl_emphasis = np.median(self.long_low_gr_lvl_emphasis_list)
            self.long_high_gr_lvl_emphasis = np.median(self.long_high_gr_lvl_emphasis_list)
            self.non_uniformity = np.median(self.non_uniformity_list)
            self.norm_non_uniformity = np.median(self.norm_non_uniformity_list)
            self.length_non_uniformity = np.median(self.length_non_uniformity_list)
            self.norm_length_non_uniformity = np.median(self.norm_length_non_uniformity_list)
            self.percentage = np.median(self.percentage_list)
            self.gr_lvl_var = np.median(self.gr_lvl_var_list)
            self.length_var = np.median(self.length_var_list)
            self.entropy = np.median(self.entropy_list)
            self.energy = np.median(self.energy_list)
        elif not self.slice_median:
            self.short_runs_emphasis = np.average(self.short_runs_emphasis_list, weights=weights)
            self.long_runs_emphasis = np.average(self.long_runs_emphasis_list, weights=weights)
            self.low_grey_level_run_emphasis = np.average(self.low_grey_level_run_emphasis_list, weights=weights)
            self.high_gr_lvl_emphasis = np.average(self.high_gr_lvl_emphasis_list, weights=weights)
            self.short_low_gr_lvl_emphasis = np.average(self.short_low_gr_lvl_emphasis_list, weights=weights)
            self.short_high_gr_lvl_emphasis = np.average(self.short_high_gr_lvl_emphasis_list, weights=weights)
            self.long_low_gr_lvl_emphasis = np.average(self.long_low_gr_lvl_emphasis_list, weights=weights)
            self.long_high_gr_lvl_emphasis = np.average(self.long_high_gr_lvl_emphasis_list, weights=weights)
            self.non_uniformity = np.average(self.non_uniformity_list, weights=weights)
            self.norm_non_uniformity = np.average(self.norm_non_uniformity_list, weights=weights)
            self.length_non_uniformity = np.average(self.length_non_uniformity_list, weights=weights)
            self.norm_length_non_uniformity = np.average(self.norm_length_non_uniformity_list, weights=weights)
            self.percentage = np.average(self.percentage_list, weights=weights)
            self.gr_lvl_var = np.average(self.gr_lvl_var_list, weights=weights)
            self.length_var = np.average(self.length_var_list, weights=weights)
            self.entropy = np.average(self.entropy_list, weights=weights)
            self.energy = np.average(self.energy_list, weights=weights)

    def calc_2_5d_ngldm_features(self):

        ngld_matrix = np.sum(self.ngldm_2d_matrices, axis=0)

        self.short_runs_emphasis = self.calc_short_emphasis(ngld_matrix)
        self.long_runs_emphasis = self.calc_long_emphasis(ngld_matrix)
        self.low_grey_level_run_emphasis = self.calc_low_gr_lvl_emphasis(ngld_matrix)
        self.high_gr_lvl_emphasis = self.calc_high_gr_lvl_emphasis(ngld_matrix)
        self.short_low_gr_lvl_emphasis = self.calc_short_low_gr_lvl_emphasis(ngld_matrix)
        self.short_high_gr_lvl_emphasis = self.calc_short_high_gr_lvl_emphasis(ngld_matrix)
        self.long_low_gr_lvl_emphasis = self.calc_long_low_gr_lvl_emphasis(ngld_matrix)
        self.long_high_gr_lvl_emphasis = self.calc_long_high_gr_lvl_emphasis(ngld_matrix)
        self.non_uniformity = self.calc_non_uniformity(ngld_matrix)
        self.norm_non_uniformity = self.calc_norm_non_uniformity(ngld_matrix)
        self.length_non_uniformity = self.calc_length_non_uniformity(ngld_matrix)
        self.norm_length_non_uniformity = self.calc_norm_length_non_uniformity(ngld_matrix)
        self.percentage = self.calc_percentage(ngld_matrix, np.sum(self.no_of_roi_voxels))
        self.gr_lvl_var = self.calc_gr_lvl_var(ngld_matrix)
        self.length_var = self.calc_length_var(ngld_matrix)
        self.entropy = self.calc_entropy(ngld_matrix)
        self.energy = self.calc_energy(ngld_matrix)

    def calc_3d_ngldm_features(self):

        ngldm = self.ngldm_3D_matrix

        self.short_runs_emphasis = self.calc_short_emphasis(ngldm)
        self.long_runs_emphasis = self.calc_long_emphasis(ngldm)
        self.low_grey_level_run_emphasis = self.calc_low_gr_lvl_emphasis(ngldm)
        self.high_gr_lvl_emphasis = self.calc_high_gr_lvl_emphasis(ngldm)
        self.short_low_gr_lvl_emphasis = self.calc_short_low_gr_lvl_emphasis(ngldm)
        self.short_high_gr_lvl_emphasis = self.calc_short_high_gr_lvl_emphasis(ngldm)
        self.long_low_gr_lvl_emphasis = self.calc_long_low_gr_lvl_emphasis(ngldm)
        self.long_high_gr_lvl_emphasis = self.calc_long_high_gr_lvl_emphasis(ngldm)
        self.non_uniformity = self.calc_non_uniformity(ngldm)
        self.norm_non_uniformity = self.calc_norm_non_uniformity(ngldm)
        self.length_non_uniformity = self.calc_length_non_uniformity(ngldm)
        self.norm_length_non_uniformity = self.calc_norm_length_non_uniformity(ngldm)
        self.percentage = self.calc_percentage(ngldm, self.tot_no_of_roi_voxels)
        self.gr_lvl_var = self.calc_gr_lvl_var(ngldm)
        self.length_var = self.calc_length_var(ngldm)
        self.entropy = self.calc_entropy(ngldm)
        self.energy = self.calc_energy(ngldm)


class NGTDM:
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
        ngtdm = np.zeros((self.lvl, 2))

        valid_offsets = [(z, y, x) for z in range(-1, 2) for y in range(-1, 2) for x in range(-1, 2) if
                         (z, y, x) != (0, 0, 0)]

        for lvl in range(self.lvl):
            s_i = 0
            n_i = 0
            z_indices, y_indices, x_indices = np.where(self.image == lvl)
            for z, y, x in zip(z_indices, y_indices, x_indices):
                x_k = []
                for off in valid_offsets:
                    neighbors = (z + off[0], y + off[1], x + off[2])
                    if all(0 <= n < sz for n, sz in zip(neighbors, self.image.shape)) and not np.isnan(
                            self.image[neighbors]):
                        x_k.append(self.image[neighbors])
                if x_k:
                    n_i += 1
                    s_i += abs(lvl - np.mean(x_k))
            ngtdm[lvl, 0] = n_i
            ngtdm[lvl, 1] = s_i

        self.ngtd_3d_matrix = ngtdm

    def calc_ngtd_2d_matrices(self):

        def calc_slice_ngtdm(matrix, n_bits):
            ngtdm_slice = np.zeros((n_bits, 2))

            valid_offsets = [(x, y) for x in range(-1, 2) for y in range(-1, 2) if (x, y) != (0, 0)]

            for lvl in range(n_bits):
                s_i = 0
                n_i = 0
                x_indices, y_indices = np.where(matrix == lvl)
                for x, y in zip(x_indices, y_indices):
                    x_k = []
                    for off in valid_offsets:
                        neighbors = (x + off[0], y + off[1])
                        if all(0 <= n < sz for n, sz in zip(neighbors, matrix.shape)) and not np.isnan(
                                matrix[neighbors]):
                            x_k.append(matrix[neighbors])
                    if x_k:
                        n_i += 1
                        s_i += abs(lvl - np.mean(x_k))

                ngtdm_slice[lvl, 0] = n_i
                ngtdm_slice[lvl, 1] = s_i

            return ngtdm_slice

        for z_slice_index in self.range_z:
            z_slice = self.image[:, :, z_slice_index]
            if np.sum(~np.isnan(z_slice)) != 0:
                self.slice_no_of_roi_voxels.append(np.sum(~np.isnan(z_slice)))
                self.ngtd_2d_matrices.append(calc_slice_ngtdm(z_slice, self.lvl))
        self.ngtd_2d_matrices = np.array(self.ngtd_2d_matrices)

    def calc_coarseness(self, matrix):
        n = np.sum(matrix[:, 0])
        denum = 0
        for i in range(matrix.shape[0]):
            denum += matrix[i, 0] * matrix[i, 1]
        return n / denum

    def calc_contrast(self, matrix):
        n = np.sum(matrix[:, 0])
        n_g = np.sum(matrix[:, 0] != 0)
        s_1 = 0
        s_2 = 0
        for i in range(matrix.shape[0]):
            s_2 += matrix[i, 1]
            for j in range(matrix.shape[0]):
                s_1 += (matrix[i, 0] * matrix[j, 0] * (i - j) ** 2) / (n ** 2)
        return (s_1 * s_2) / (n_g * (n_g - 1) * np.sum(matrix[:, 0]))

    def calc_busyness(self, matrix):
        n = np.sum(matrix[:, 0])
        num = 0
        denum = 0
        for i in range(matrix.shape[0]):
            num += (matrix[i, 0] * matrix[i, 1]) / n
            for j in range(matrix.shape[0]):
                if matrix[i, 0] != 0 and matrix[j, 0] != 0:
                    denum += abs(i * matrix[i, 0] - j * matrix[j, 0]) / n
        return num / denum

    def calc_complexity(self, matrix):
        n = np.sum(matrix[:, 0])
        sum_compl = 0
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[0]):
                if matrix[i, 0] != 0 and matrix[j, 0] != 0:
                    num = ((matrix[i, 0] * matrix[i, 1] + matrix[j, 0] * matrix[j, 1]) * abs(i - j)) / n
                    denum = (matrix[i, 0] + matrix[j, 0]) / n
                    sum_compl += num / denum
        return sum_compl / np.sum(matrix[:, 0])

    def calc_strength(self, matrix):
        n = np.sum(matrix[:, 0])
        num = 0
        denum = 0
        for i in range(matrix.shape[0]):
            denum += matrix[i, 1]
            for j in range(matrix.shape[0]):
                if matrix[i, 0] != 0 and matrix[j, 0] != 0:
                    num += ((matrix[i, 0] + matrix[j, 0]) * (i - j) ** 2) / n
        return num / denum

    def calc_2d_ngtdm_features(self):

        number_of_slices = self.ngtd_2d_matrices.shape[0]
        weights = []
        for i in range(number_of_slices):
            ngtdm_slice = self.ngtd_2d_matrices[i]
            weight = 1
            if self.slice_weight:
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
