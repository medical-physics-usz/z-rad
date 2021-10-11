import logging
import os

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pydicom as dc

from structure import Structures
from utils import new_directory
from hdbet.run import run_hd_bet


def linear_norm(struct_norm1, struct_norm2, input_image_path, rs, slices, x_ct, y_ct, pixel_spacing, onlyfiles, rows,
                columns):
    """Get the normalization coefficients for MR image based on two normal structures (for example muscle and white
    matter) and linear function.
    """
    # False for the wavelet and local argument, none for dim
    struct1 = Structures(rs, [struct_norm1], slices, x_ct, y_ct, pixel_spacing, len(slices), False, None, False)
    norm1_contour_x = struct1.Xcontour
    norm1_contour_y = struct1.Ycontour

    struct2 = Structures(rs, [struct_norm2], slices, x_ct, y_ct, pixel_spacing, len(slices), False, None, False)
    norm2_contour_x = struct2.Xcontour
    norm2_contour_y = struct2.Ycontour

    # to read only slices where there is a contour
    ind = []
    for f in range(len(norm1_contour_x)):
        if norm1_contour_x[f] != []:
            ind.append(f)
    for f in range(len(norm2_contour_x)):
        if norm2_contour_x[f] != []:
            ind.append(f)

    zmin = np.min(ind)
    zmax = np.max(ind)

    onlyfiles = onlyfiles[zmin:zmax + 1]
    norm1_contour_x = norm1_contour_x[zmin:zmax + 1]
    norm1_contour_y = norm1_contour_y[zmin:zmax + 1]
    norm2_contour_x = norm2_contour_x[zmin:zmax + 1]
    norm2_contour_y = norm2_contour_y[zmin:zmax + 1]

    im_matrix = []  # list containing the images matrix
    for i in onlyfiles:
        data = dc.read_file(input_image_path + i).PixelData
        data16 = np.array(np.fromstring(data, dtype=np.int16))  # converting to decimal
        # recalculating for rows x columns
        a = []
        for j in range(rows):
            a.append(data16[j * columns:(j + 1) * columns])
        a = np.array(a)
        im_matrix.append(np.array(a))
    im_matrix = np.array(im_matrix)

    v1 = []  # values for structure 1
    v2 = []  # values for structure 2

    for i in range(len(norm1_contour_x)):  # slices
        for j in range(len(norm1_contour_x[i])):  # sub-structures in the slice
            for k in range(len(norm1_contour_x[i][j])):
                v1.append(im_matrix[i][norm1_contour_y[i][j][k]][norm1_contour_x[i][j][k]])

    for i in range(len(norm2_contour_x)):  # slices
        for j in range(len(norm2_contour_x[i])):  # sub-structures in the slice
            for k in range(len(norm2_contour_x[i][j])):
                v2.append(im_matrix[i][norm2_contour_y[i][j][k]][norm2_contour_x[i][j][k]])

    f1 = np.mean(v1)
    f2 = np.mean(v2)

    # find coefficients of a linear function
    fa = (800 - 300) / (f1 - f2)
    fb = 800 - f1 * fa

    return fa, fb


def zscore_norm(roi):
    mean = np.nanmean(roi)
    std = np.nanstd(roi)
    return mean, std


def hist_matching_norm(new_brain, name, path_save, path_template):
    """
    Return modified source array so that the cumulative density function of
    its values matches the cumulative density function of the template.
    """
    src_values, src_unique_indices, src_counts = np.unique(new_brain.ravel(), return_inverse=True, return_counts=True)
    # src_values - unique values
    # src_unique_indices - indices per voxel to given unique value, eash nan voxel is considered a unique value and has
    # to be handled separately
    # src_counts - counts how many time thie unique value occurs
    #        example
    #        a = array([[nan,  2.,  3.,  4.,  4.],
    #        [ 4.,  4.,  4.,  5., nan]])
    #        np.unique(a.ravel(), return_inverse=True, return_counts=True)
    #        (array([ 2.,  3.,  4.,  5., nan, nan]),
    #         array([4, 0, 1, 2, 2, 2, 2, 2, 3, 5], dtype=int64),
    #         array([1, 1, 5, 1, 1, 1], dtype=int64))

    # read template
    standard_brain = np.load(path_template)

    tmpl_values, tmpl_counts = np.unique(standard_brain.ravel(), return_counts=True)

    # check for nan
    src_nan = np.min(np.where(np.isnan(src_values)))
    src_counts = src_counts[: src_nan]

    tmpl_nan = np.min(np.where(np.isnan(tmpl_values)))
    tmpl_counts = tmpl_counts[: tmpl_nan]
    tmpl_values = tmpl_values[: tmpl_nan]

    # calculate normalized quantiles for each array
    src_quantiles = np.cumsum(src_counts) / (new_brain.size - len(np.where(np.isnan(new_brain))[0]))
    tmpl_quantiles = np.cumsum(tmpl_counts) / (standard_brain.size - len(np.where(np.isnan(standard_brain))[0]))

    interp_a_values = np.interp(src_quantiles, tmpl_quantiles, tmpl_values)

    src_new_values = np.zeros(len(src_unique_indices))
    src_new_values[:] = np.nan
    for i, sind in enumerate(src_unique_indices):
        if sind < src_nan:
            src_new_values[i] = interp_a_values[sind]

    a = src_new_values.reshape(new_brain.shape)
    # test segmentation
    save_dir = os.path.join(path_save, 'brain_mask_hist', name)
    new_directory(save_dir)
    for n in range(len(a) // 8 + 1):
        fig = plt.figure(10, figsize=(20, 20))
        fig.text(0.5, 0.95, name)
        for j in range(8):
            axes = fig.add_subplot(3, 3, j + 1)
            axes.set_title(8 * n + j)
            try:
                im = axes.imshow(a[8 * n + j], cmap=plt.get_cmap('jet'))
            except IndexError:
                break
        axes = fig.add_subplot(3, 3, 9)
        try:
            fig.colorbar(im)
        except UnboundLocalError:
            pass
        fig.savefig(os.path.join(save_dir, f'{n+1}.png'))
        plt.close()

    affine_trans = np.zeros((4, 4))
    affine_trans[0, 0] = -1 * 0.5
    affine_trans[1, 1] = -1 * 0.5
    affine_trans[2, 2] = 0.5
    affine_trans[3, 3] = 1.
    affine_trans[0, 3] = -1 * 100
    affine_trans[1, 3] = -1 * 100
    affine_trans[2, 3] = 100
    im_matrix_nifti = nib.Nifti1Image(a, affine=affine_trans)
    nib.save(im_matrix_nifti, os.path.join(save_dir, 'norm.nii.gz'))
    return src_new_values.reshape(new_brain.shape)


def _skull_stripping(im_matrix, transform, path_save, name):
    im_matrix_nifti = nib.Nifti1Image(im_matrix, affine=transform)
    im_matrix_nifti_dir = os.path.join(path_save, 'nifti')
    im_matrix_nifti_filename = f'{name}.nii.gz'
    im_matrix_nifti_filepath = os.path.join(im_matrix_nifti_dir, im_matrix_nifti_filename)
    new_directory(im_matrix_nifti_dir)
    nib.save(im_matrix_nifti, im_matrix_nifti_filepath)
    del im_matrix_nifti

    brain_nifti_dir = os.path.join(path_save, 'brain_nifti')
    brain_nifti_filename = f'{name}.nii.gz'
    brain_nifti_filepath = os.path.join(brain_nifti_dir, brain_nifti_filename)
    new_directory(brain_nifti_dir)

    # settings as from https://github.com/MIC-DKFZ/HD-BET/blob/master/HD_BET/hd-bet
    mode = 'fast'  # fast\' or \'accurate
    config_file = 'config.py'  # this points to hdbet/config.py
    device = 'cpu'
    pp = True
    tta = False  # Only for testing, set this to True later
    save_mask = True
    overwrite_existing = False

    run_hd_bet(im_matrix_nifti_filepath, brain_nifti_filepath, mode, config_file, device, pp, tta, save_mask,
               overwrite_existing)

    brain_mask_nifti_filename = f'{name}_mask.nii.gz'
    brain_mask_nifti_filepath = os.path.join(brain_nifti_dir, brain_mask_nifti_filename)
    mask = nib.load(brain_mask_nifti_filepath)
    mask_data = mask.get_fdata()
    mask_data[np.where(mask_data == 0)] = np.nan
    a = im_matrix * mask_data

    # test segmentation
    brain_mask_png_dir = os.path.join(path_save, 'brain_png', name)
    new_directory(brain_mask_png_dir)
    n_subplots_horizontal = 3
    n_subplots_vertical = 3
    n_subplots = n_subplots_horizontal * n_subplots_vertical
    for n in range(len(a) // n_subplots + 1):
        fig = plt.figure(10, figsize=(20, 20))
        fig.text(0.5, 0.95, name)
        for j in range(n_subplots):
            axes = fig.add_subplot(n_subplots_horizontal, n_subplots_vertical, j + 1)
            axes.set_title(n_subplots * n + j)
            try:
                im = axes.imshow(a[n_subplots * n + j], cmap=plt.get_cmap('viridis'), vmin=0, vmax=np.nanmax(a))
            except IndexError:
                break
        try:
            fig.subplots_adjust(right=0.8)
            cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
            fig.colorbar(im, cax=cbar_ax)
        except UnboundLocalError:
            pass

        brain_mask_png_filename = f'{n+1}.png'
        brain_mask_png_filepath = os.path.join(brain_mask_png_dir, brain_mask_png_filename)
        fig.savefig(brain_mask_png_filepath)
        plt.close()
    return mask_data


def mask_brain(patient_position, pixel_spacing, x_ct, y_ct, z_ct, im_matrix, path_skull, im_name, path_save):
    if path_skull == '':
        if patient_position == 'HFS':
            affine_trans = np.zeros((4, 4))
            affine_trans[0, 0] = -1 * pixel_spacing
            affine_trans[1, 1] = -1 * pixel_spacing
            affine_trans[2, 2] = pixel_spacing
            affine_trans[3, 3] = 1.
            affine_trans[0, 3] = -1 * x_ct
            affine_trans[1, 3] = -1 * y_ct
            affine_trans[2, 3] = z_ct
            mask_data = _skull_stripping(im_matrix, affine_trans, path_save, im_name)
        else:
            logger = logging.getLogger("MR read")
            logger.info("Scan position not HFS, the skull stripping was not tested")
            raise TypeError
    else:
        try:
            mask = nib.load(os.path.join(path_skull, im_name, '_mask.nii.gz'))
        except OSError:
            mask = nib.load(os.path.join(path_skull, im_name, '_mask.nii'))
        mask_data = mask.get_fdata()
        mask_data[np.where(mask_data == 0)] = np.nan
    return mask_data


def mask_roi(norm_contour_x, norm_contour_y, matrix_shape, outer_value):
    # creating the matrix to fill it with points of the structure, y rows, x columns
    mask_data = np.zeros(matrix_shape)
    mask_data[:, :, :] = outer_value
    for i in range(len(norm_contour_x)):  # slices
        for j in range(len(norm_contour_x[i])):  # sub-structures in the slice
            for k in range(len(norm_contour_x[i][j])):
                mask_data[i][norm_contour_y[i][j][k]][norm_contour_x[i][j][k]] = 1
    return mask_data
