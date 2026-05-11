import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import ConvexHull
from scipy.special import legendre
from skimage import measure

from ..exceptions import DataStructureError
from .base import BaseFeatureGroup

def _pca_eigenvalues(points: np.ndarray) -> np.ndarray:
    """Return eigenvalues of the covariance matrix of ``points`` sorted descending."""

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Expected an (n_samples, 3) array of points")

    n_samples = points.shape[0]
    if n_samples < 3:
        raise ValueError("At least three points are required to compute PCA")

    points = points.astype(np.float64, copy=False)
    centered = points - np.mean(points, axis=0, keepdims=True)
    cov = np.cov(centered, rowvar=False, bias=False)
    eigenvalues, _ = np.linalg.eigh(cov)
    order = np.argsort(eigenvalues)[::-1]

    return eigenvalues[order]


class MorphologicalFeatures:
    """Morphological and shape descriptors for a 3D region of interest.

    Features describe ROI volume, surface area, compactness, principal axes,
    convex-hull density, and related shape measures. Calculations use physical
    voxel spacing so outputs are expressed in image-world units where
    applicable.

    Parameters
    ----------
    spacing : sequence of float
        Physical voxel spacing along the three array axes.
    """
    def __init__(self, spacing):
        self.spacing = spacing
        self.unit_vol = self.spacing[0] * self.spacing[1] * self.spacing[2]

    def get_params(self):
        """Return the configuration parameters of this morphology calculator.

        Returns
        -------
        dict
            Parameter names mapped to their configured values.
        """
        return {
            'spacing': self.spacing,
        }

    def get_feature_names(self):
        """Return the morphology feature names produced by this calculator.

        Returns
        -------
        list of str
            Feature names defined for the morphology family.
        """
        return list(MORPHOLOGY_FEATURE_NAMES)

    def _calc_mesh(self, mask_array):
        mesh_verts, mesh_faces, _, _ = measure.marching_cubes(mask_array, level=0.5)
        return mesh_verts * self.spacing, mesh_faces

    @staticmethod
    def _calc_vol_and_area_mesh(mesh_verts, mesh_faces):
        faces = np.asarray(mesh_faces)
        verts = np.asarray(mesh_verts)
        a, b, c = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
        cross_bc = np.cross(b, c)
        cross_ba_ca = np.cross(b - a, c - a)
        vol_mesh = abs(np.einsum('ij,ij->i', a, cross_bc).sum() / 6)
        area_mesh = np.linalg.norm(cross_ba_ca, axis=1).sum() / 2
        return vol_mesh, area_mesh

    def _calc_vol_count(self, mask_array):
        return np.sum(mask_array) * self.unit_vol

    @staticmethod
    def _calc_surf_to_vol_ratio(area_mesh, vol_mesh):
        return area_mesh / vol_mesh

    @staticmethod
    def _calc_compactness_1(vol_mesh, area_mesh):
        return vol_mesh / (np.pi ** (1 / 2) * area_mesh ** (3 / 2))

    @staticmethod
    def _calc_compactness_2(vol_mesh, area_mesh):
        return 36 * np.pi * (vol_mesh ** 2 / area_mesh ** 3)

    @staticmethod
    def _calc_spherical_disproportion(area_mesh, vol_mesh):
        return area_mesh / (36 * np.pi * vol_mesh ** 2) ** (1 / 3)

    @staticmethod
    def _calc_sphericity(vol_mesh, area_mesh):
        return (36 * np.pi * vol_mesh ** 2) ** (1 / 3) / area_mesh

    @staticmethod
    def _calc_asphericity(area_mesh, vol_mesh):
        return (area_mesh ** 3 / (36 * np.pi * vol_mesh ** 2)) ** (1 / 3) - 1

    def _calc_centre_of_shift(self, mask_array, image_array):
        dx, dy, dz = self.spacing
        morph_voxels = np.argwhere(mask_array)
        morph_voxels_scaled = morph_voxels * [dx, dy, dz]
        com_geom = np.mean(morph_voxels_scaled, axis=0)

        intensity_voxels = np.argwhere(~np.isnan(image_array))
        intensities = image_array[intensity_voxels[:, 0], intensity_voxels[:, 1], intensity_voxels[:, 2]]
        intensity_voxels_scaled = intensity_voxels * [dx, dy, dz]
        com_gl = np.average(intensity_voxels_scaled, axis=0, weights=intensities)
        return np.linalg.norm(com_geom - com_gl)

    @staticmethod
    def _calc_convex_hull(mesh_verts):
        return ConvexHull(mesh_verts)

    @staticmethod
    def _calc_max_diameter(conv_hull, mesh_verts):
        hull_verts = mesh_verts[conv_hull.vertices]
        if hull_verts.shape[0] < 2:
            return 0
        return np.max(pdist(hull_verts))

    def _calc_pca(self, mask_array):
        voxel_indices = np.argwhere(mask_array == 1)
        scaled_voxel_indices = voxel_indices.astype(np.float64)
        scaled_voxel_indices *= self.spacing
        return _pca_eigenvalues(scaled_voxel_indices)

    @staticmethod
    def _calc_major_minor_least_axes_len(pca_eigenvalues):
        return (
            4 * np.sqrt(pca_eigenvalues[0]),
            4 * np.sqrt(pca_eigenvalues[1]),
            4 * np.sqrt(pca_eigenvalues[2]),
        )

    @staticmethod
    def _calc_elongation(pca_eigenvalues):
        if pca_eigenvalues[0] == 0:
            raise DataStructureError(f"PCA eigenvalue is zero. ")
        return np.sqrt(pca_eigenvalues[1] / pca_eigenvalues[0])

    @staticmethod
    def _calc_flatness(pca_eigenvalues):
        if pca_eigenvalues[0] == 0:
            raise DataStructureError(f"PCA eigenvalue is zero. ")
        return np.sqrt(pca_eigenvalues[2] / pca_eigenvalues[0])

    def _calc_vol_and_area_densities_aabb(self, mask_array, vol_mesh, area_mesh):
        x_dim, y_dim, z_dim = self.spacing
        x_coords, y_coords, z_coords = np.where(mask_array == 1)
        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()
        z_min, z_max = z_coords.min(), z_coords.max()

        aabb_x_dim = (x_max - x_min + 1) * x_dim
        aabb_y_dim = (y_max - y_min + 1) * y_dim
        aabb_z_dim = (z_max - z_min + 1) * z_dim

        aabb_volume = aabb_x_dim * aabb_y_dim * aabb_z_dim
        aabb_surface_area = 2 * (aabb_x_dim * aabb_y_dim + aabb_x_dim * aabb_z_dim + aabb_y_dim * aabb_z_dim)
        return vol_mesh / aabb_volume, area_mesh / aabb_surface_area

    @staticmethod
    def _calc_vol_density_aee(vol_mesh, major_axis_len, minor_axis_len, least_axis_len):
        if major_axis_len == 0 or minor_axis_len == 0 or least_axis_len == 0:
            raise DataStructureError(f"One of the axis is zero. ")
        return (8 * 3 * vol_mesh) / (4 * np.pi * major_axis_len * minor_axis_len * least_axis_len)

    @staticmethod
    def _calc_area_density_aee(area_mesh, major_axis_len, minor_axis_len, least_axis_len):
        a = major_axis_len / 2
        b = minor_axis_len / 2
        c = least_axis_len / 2

        alpha = np.sqrt(1 - (b ** 2 / a ** 2))
        beta = np.sqrt(1 - (c ** 2 / a ** 2))
        if alpha == 0 or beta == 0:
            raise DataStructureError(f"Alpha or beta in area density (AEE) is zero.")
        sum_series = 0
        max_nu = 20  # Def by IBSI
        for nu in range(max_nu + 1):
            p_nu = legendre(nu)
            sum_series += ((alpha * beta) ** nu / (1 - (4 * nu ** 2))) * p_nu(
                (alpha ** 2 + beta ** 2) / (2 * alpha * beta))

        area_aee = 4 * np.pi * a * b * sum_series
        return area_mesh / area_aee

    @staticmethod
    def _calc_vol_density_ch(vol_mesh, conv_hull):
        return vol_mesh / conv_hull.volume

    @staticmethod
    def _calc_area_density_ch(area_mesh, conv_hull):
        return area_mesh / conv_hull.area

    @staticmethod
    def _calc_integrated_intensity(image_array, vol_mesh):
        return np.nanmean(image_array) * vol_mesh

    def calculate_features(self, mask_array, image_array):
        """Calculate morphology features for prepared mask and intensity arrays.

        Parameters
        ----------
        mask_array : numpy.ndarray
            Prepared binary ROI mask array.
        image_array : numpy.ndarray
            Prepared intensity image aligned with ``mask_array`` where voxels
            outside the ROI can be represented by ``NaN``.

        Returns
        -------
        dict
            Mapping of morphology feature names to calculated values.
        """
        mask_array = np.asarray(mask_array)
        image_array = np.asarray(image_array)

        mesh_verts, mesh_faces = self._calc_mesh(mask_array)
        vol_mesh, area_mesh = self._calc_vol_and_area_mesh(mesh_verts, mesh_faces)
        vol_count = self._calc_vol_count(mask_array)
        surf_to_vol_ratio = self._calc_surf_to_vol_ratio(area_mesh, vol_mesh)
        compactness_1 = self._calc_compactness_1(vol_mesh, area_mesh)
        compactness_2 = self._calc_compactness_2(vol_mesh, area_mesh)
        spherical_disproportion = self._calc_spherical_disproportion(area_mesh, vol_mesh)
        sphericity = self._calc_sphericity(vol_mesh, area_mesh)
        asphericity = self._calc_asphericity(area_mesh, vol_mesh)
        centre_of_shift = self._calc_centre_of_shift(mask_array, image_array)
        conv_hull = self._calc_convex_hull(mesh_verts)
        max_diameter = self._calc_max_diameter(conv_hull, mesh_verts)
        pca_eigenvalues = self._calc_pca(mask_array)
        major_axis_len, minor_axis_len, least_axis_len = self._calc_major_minor_least_axes_len(pca_eigenvalues)
        elongation = self._calc_elongation(pca_eigenvalues)
        flatness = self._calc_flatness(pca_eigenvalues)
        vol_density_aabb, area_density_aabb = self._calc_vol_and_area_densities_aabb(mask_array, vol_mesh, area_mesh)
        vol_density_aee = self._calc_vol_density_aee(vol_mesh, major_axis_len, minor_axis_len, least_axis_len)
        area_density_aee = self._calc_area_density_aee(area_mesh, major_axis_len, minor_axis_len, least_axis_len)
        vol_density_ch = self._calc_vol_density_ch(vol_mesh, conv_hull)
        area_density_ch = self._calc_area_density_ch(area_mesh, conv_hull)
        integrated_intensity = self._calc_integrated_intensity(image_array, vol_mesh)

        values = [
            vol_mesh,
            vol_count,
            area_mesh,
            surf_to_vol_ratio,
            compactness_1,
            compactness_2,
            spherical_disproportion,
            sphericity,
            asphericity,
            centre_of_shift,
            max_diameter,
            major_axis_len,
            minor_axis_len,
            least_axis_len,
            elongation,
            flatness,
            vol_density_aabb,
            area_density_aabb,
            vol_density_aee,
            area_density_aee,
            vol_density_ch,
            area_density_ch,
            integrated_intensity,
        ]
        return dict(zip(MORPHOLOGY_FEATURE_NAMES, values))

class MorphologyCorrelationFeatures:
    """Spatial autocorrelation descriptors for a 3D region of interest.

    Moran's I and Geary's C summarize how intensity values vary with physical
    distance between ROI voxels. They are optional morphology-related features
    and require both a morphology mask and aligned intensity image.

    Parameters
    ----------
    spacing : sequence of float
        Physical voxel spacing along the three array axes.
    """

    def __init__(self, spacing):
        self.spacing = spacing

    def get_params(self):
        """Return the configuration parameters of this morphology correlation calculator.

        Returns
        -------
        dict
            Parameter names mapped to their configured values.
        """
        return {
            'spacing': self.spacing,
        }

    def get_feature_names(self):
        """Return the morphology correlation feature names produced by this calculator.

        Returns
        -------
        list of str
            Feature names defined for the morphology correlation family.
        """
        return list(MORPHOLOGY_CORRELATION_FEATURE_NAMES)

    def _valid_intensity_data(self, mask_array, image_array):
        indices = np.argwhere(mask_array)
        scaled_indices = indices * self.spacing
        intensities = image_array[indices[:, 0], indices[:, 1], indices[:, 2]]
        valid = ~np.isnan(intensities)
        if np.sum(valid) < 2:
            return None, None
        return scaled_indices[valid], intensities[valid]

    def _calc_moran_i(self, mask_array, image_array):
        scaled_indices, intensities = self._valid_intensity_data(mask_array, image_array)
        if scaled_indices is None:
            return np.nan

        n = len(intensities)
        mu = np.mean(intensities)
        distances = squareform(pdist(scaled_indices))
        weights = np.zeros_like(distances)
        nonzero_mask = distances > 0
        if np.any(distances[nonzero_mask] == 0):
            raise DataStructureError(f"There is a zero distance in Moran I.")
        weights[nonzero_mask] = 1.0 / distances[nonzero_mask]
        s0 = np.sum(weights)
        diff = intensities - mu
        diff_outer = np.outer(diff, diff)
        numerator = np.sum(weights * diff_outer)
        denominator = np.sum(diff ** 2)
        if denominator == 0:
            raise DataStructureError(f"There determinator is zero in Moran I.")
        return (n / s0) * (numerator / denominator)

    def _calc_geary_c(self, mask_array, image_array):
        scaled_indices, intensities = self._valid_intensity_data(mask_array, image_array)
        if scaled_indices is None:
            return np.nan

        n = len(intensities)
        mu = np.mean(intensities)
        distances = squareform(pdist(scaled_indices))
        weights = np.zeros_like(distances)
        nonzero_mask = distances > 0
        if np.any(distances[nonzero_mask] == 0):
            raise DataStructureError(f"There is a zero distance in Geary C.")
        weights[nonzero_mask] = 1.0 / distances[nonzero_mask]
        s0 = np.sum(weights)
        diff_matrix = np.subtract.outer(intensities, intensities)
        squared_diff = diff_matrix ** 2
        numerator = np.sum(weights * squared_diff)
        denominator = np.sum((intensities - mu) ** 2)
        if denominator == 0:
            raise DataStructureError(f"There determinator is zero in Geary C.")
        return ((n - 1) / (2 * s0)) * (numerator / denominator)

    def calculate_features(self, mask_array, image_array):
        """Calculate morphology correlation features for prepared mask and intensity arrays.

        Parameters
        ----------
        mask_array : numpy.ndarray
            Prepared binary ROI mask array.
        image_array : numpy.ndarray
            Prepared intensity image aligned with ``mask_array`` where voxels
            outside the ROI can be represented by ``NaN``.

        Returns
        -------
        dict
            Mapping of morphology correlation feature names to calculated values.
        """
        mask_array = np.asarray(mask_array)
        image_array = np.asarray(image_array)
        return {
            'morph_moran_i': self._calc_moran_i(mask_array, image_array),
            'morph_geary_c': self._calc_geary_c(mask_array, image_array),
        }



MORPHOLOGY_FEATURE_NAMES = (
    'morph_volume',
    'morph_vol_approx',
    'morph_area_mesh',
    'morph_av',
    'morph_comp_1',
    'morph_comp_2',
    'morph_sph_dispr',
    'morph_sphericity',
    'morph_asphericity',
    'morph_com',
    'morph_diam',
    'morph_pca_maj_axis',
    'morph_pca_min_axis',
    'morph_pca_least_axis',
    'morph_pca_elongation',
    'morph_pca_flatness',
    'morph_vol_dens_aabb',
    'morph_area_dens_aabb',
    'morph_vol_dens_aee',
    'morph_area_dens_aee',
    'morph_vol_dens_conv_hull',
    'morph_area_dens_conv_hull',
    'morph_integ_int',
)


MORPHOLOGY_CORRELATION_FEATURE_NAMES = (
    'morph_moran_i',
    'morph_geary_c',
)


class MorphologyFeatureGroup(BaseFeatureGroup):
    family = 'morphology'
    requirements = frozenset({'base_masks'})

    def supports(self, context):
        return not context.is_slice_2d

    def default_enabled(self, context):
        return self.supports(context)

    def output_names(self, context):
        return MORPHOLOGY_FEATURE_NAMES

    def feature_aliases(self, context):
        return {name: name for name in self.output_names(context)}

    def calculate(self, context, prepared_data):
        masks = prepared_data.require_base_masks()
        morphology = MorphologicalFeatures(
            masks.morphological_mask.spacing[::-1],
        )
        return morphology.calculate_features(
            masks.morphological_mask.array,
            masks.intensity_mask.array,
        )


class MorphologyCorrelationFeatureGroup(BaseFeatureGroup):
    family = 'morphology_correlation'
    requirements = frozenset({'base_masks'})

    def supports(self, context):
        return not context.is_slice_2d

    def default_enabled(self, context):
        return self.supports(context) and context.calc_morph_moran_i_and_geary_c_features

    def output_names(self, context):
        return MORPHOLOGY_CORRELATION_FEATURE_NAMES

    def feature_aliases(self, context):
        return {name: name for name in self.output_names(context)}

    def calculate(self, context, prepared_data):
        masks = prepared_data.require_base_masks()
        morphology = MorphologyCorrelationFeatures(
            masks.morphological_mask.spacing[::-1],
        )
        return morphology.calculate_features(
            masks.morphological_mask.array,
            masks.intensity_mask.array,
        )
