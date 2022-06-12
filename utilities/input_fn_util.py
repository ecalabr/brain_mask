"""
Utility functions for use by input_functions.py
"""

import os
from glob import glob
import numpy as np
import nibabel as nib
import scipy.ndimage as ndi
import tensorflow as tf
import scipy.ndimage
from utilities.normalizers import Normalizers


##############################################
# Local data utilities
##############################################
def _byte_convert(byte_data):
    if isinstance(byte_data, bytes):
        return byte_data.decode()
    if isinstance(byte_data, dict):
        return dict(map(_byte_convert, byte_data.items()))
    if isinstance(byte_data, tuple):
        return map(_byte_convert, byte_data)
    if isinstance(byte_data, (np.ndarray, list)):
        return list(map(_byte_convert, byte_data))

    return byte_data


def _load_single_study(study_dir, file_prefixes, data_format, out_res=None, out_dims=None, res_order=2, plane=None,
                       norm=None, norm_mode='zero_mean'):
    """
    Image data I/O function for use in tensorflow Dataset map function. Takes a study directory and file prefixes and
    returns a 4D numpy array containing the image data. Performs optional slice trimming in z and normalization.
    :param study_dir: (str) the full path to the study directory
    :param file_prefixes: (str, list(str)) the file prefixes for the images to be loaded
    :param data_format: (str) the desired tensorflow data format. Must be either 'channels_last' or 'channels_first'
    :param out_res: list(int) the desired output resolution. If none, no resampling will be done.
    :param out_dims: list(int) the desired output dimensions. If none, no cropping/padding will be done.
    :param out_dims: int the spline order to be used for resampling.
    :param plane: (str) The plane to load data in. Must be a string in ['ax', 'cor', 'sag']
    :param norm: (bool) whether or not to perform per dataset normalization
    :param norm_mode: (str) The method for normalization, used by normalize function.
    :return: output - a 4D numpy array containing the image data
    """

    # sanity checks
    if not os.path.isdir(study_dir):
        raise ValueError("Specified study_dir does not exist")
    if data_format not in ['channels_last', 'channels_first']:
        raise ValueError("data_format invalid")
    images = [glob(study_dir + '/*' + contrast + '.nii.gz')[0] for contrast in file_prefixes]
    if not images:
        raise ValueError("No matching image files found for file prefixes: " + str(images))

    # get data dims
    nii = nib.load(images[0])
    orig_dims = nii.shape

    # handle out_res
    if out_res:
        if len(out_res) == 1:
            out_res = list(out_res) * 3
        res_dims = [round((i / j) * k) for i, j, k in zip(nii.header.get_zooms()[0:3], out_res, orig_dims)]
    else:
        res_dims = orig_dims

    # handle out_dims
    if out_dims:
        alloc_dims = out_dims
    else:
        alloc_dims = res_dims

    # preallocate
    data = np.empty(list(alloc_dims) + [len(images)], dtype=np.float32)

    # load images and concatenate into a 4d numpy array
    for ind, image in enumerate(images):
        # first nii is already loaded
        if ind > 0:
            nii = nib.load(images[ind])
        # get float data
        img = nii.get_fdata()
        # handle 4d data - take only first image in 4th dim
        if len(img.shape) > 3:
            img = np.squeeze(img[:, :, :, 0])
        # handle resampling
        if out_res:
            # determine zooms
            zooms = [i/j for i, j in zip(nii.header.get_zooms()[0:3], out_res)]
            # only resample if desired resolution is different from current resolution
            if not zooms == [1, 1, 1]:
                img = scipy.ndimage.zoom(img, zooms, order=res_order)
        # handle cropping
        if not list(alloc_dims) == list(img.shape)[0:3]:
            img = _crop_pad_image(img, alloc_dims)
        # handle normalization
        if norm:
            data[..., ind] = Normalizers(norm_mode)(img)
        else:
            data[..., ind] = img

    # permute to desired plane in format [x, y, z, channels] for tensorflow
    if plane == 'ax':
        pass
    elif plane == 'cor':
        data = np.transpose(data, axes=(0, 2, 1, 3))
    elif plane == 'sag':
        data = np.transpose(data, axes=(1, 2, 0, 3))
    else:
        raise ValueError("Did not understand specified plane: " + str(plane))

    # handle channels first data format
    if data_format == 'channels_first':
        data = np.transpose(data, axes=(3, 0, 1, 2))

    return data


def _expand_region(input_dims, region_bbox, delta):
    """
    Symmetrically expands a given 3D region bounding box by delta in each dim without exceeding original image dims
    If delta is a single int then each dim is expanded by this amount. If a list or tuple, of ints then dims are
    expanded to match the size of each int in the list respectively.
    :param input_dims: (list or tuple of ints) the original image dimensions
    :param region_bbox: (list or tuple of ints) the region bounding box to expand
    :param delta: (int or list/tuple of ints) the amount to expand each dimension by.
    :return: (list or tuple of ints) the expanded bounding box
    """

    # if delta is actually a list, then refer to expand_region_dims, make sure its same length as input_dims
    if isinstance(delta, (list, tuple, np.ndarray)):
        if not len(delta) == len(input_dims):
            raise ValueError("When a list is passed, parameter mask_dilate must have one val for each dim in mask")
        return _expand_region_dims(input_dims, region_bbox, delta)

    # if delta is zero return
    if delta == 0:
        return region_bbox

    # determine how much to add on each side of the bounding box
    deltas = np.array([-int(np.floor(delta / 2.)), int(np.ceil(delta / 2.))] * 3)

    # use deltas to get a new bounding box
    tmp_bbox = np.array(region_bbox) + deltas

    # make sure there are not values outside of the original image
    new_bbox = []
    for i, item in enumerate(tmp_bbox):
        if i % 2 == 0:  # for even indices, make sure there are no negatives
            if item < 0:
                item = 0
        else:  # for odd indices, make sure they do not exceed original dims
            if item > input_dims[int(round((i - 1) / 2))]:
                item = input_dims[int(round((i - 1) / 2))]
        new_bbox.append(item)

    return new_bbox


def _expand_region_dims(input_dims, region_bbox, out_dims):
    """
    Symmetrically expands a given 3D region bounding box to the specified output size
    :param input_dims: (list or tuple of ints) the original image dimensions
    :param region_bbox: (list or tuple of ints) the region bounding box to expand
    :param out_dims: (list or tuple of ints) the desired output dimensions.
    :return: (list or tuple of ints) the expanded bounding box
    """

    # determine region dimensions
    region_dims = [region_bbox[1] - region_bbox[0], region_bbox[3] - region_bbox[2], region_bbox[5] - region_bbox[4]]
    # region_dims = [region_bbox[x] - region_bbox[x - 1] for x in range(len(region_bbox))[1::2]]

    # determine the delta in each dimension - exclude negatives
    deltas = [x - y for x, y in zip(out_dims, region_dims)]
    deltas = [0 if d < 0 else d for d in deltas]

    # determine how much to add on each side of the bounding box
    pre_inds = np.array([-int(np.floor(d / 2.)) for d in deltas])
    post_inds = np.array([int(np.ceil(d / 2.)) for d in deltas])
    deltas = np.empty((pre_inds.size + post_inds.size,), dtype=pre_inds.dtype)
    deltas[0::2] = pre_inds
    deltas[1::2] = post_inds

    # use deltas to get a new bounding box
    tmp_bbox = np.array(region_bbox) + deltas

    # make sure there are not values outside of the original image
    new_bbox = []
    for i, item in enumerate(tmp_bbox):
        if i % 2 == 0:  # for even indices, make sure there are no negatives
            if item < 0:
                item = 0
        else:  # for odd indices, make sure they do not exceed original dims
            if item > input_dims[int(round((i - 1) / 2))]:
                item = input_dims[int(round((i - 1) / 2))]
        new_bbox.append(item)

    return new_bbox


def _create_affine(theta=None, phi=None, psi=None):
    """
    Creates a 3D rotation affine matrix given three rotation angles.
    :param theta: (float) The theta angle in radians. If None, a random angle is chosen.
    :param phi: (float) The phi angle in radians. If None, a random angle is chosen.
    :param psi: (float) The psi angle in radians. If None, a random angle is chosen.
    :return: (np.ndarray) a 3x3 affine rotation matrix.
    """

    # return identitiy if all angles are zero
    if all(val == 0. for val in [theta, phi, psi]):
        return np.eye(3)

    # define angles
    if theta is None:
        theta = np.random.random() * (np.pi / 2.)
    if phi is None:
        phi = np.random.random() * (np.pi / 2.)
    if psi is None:
        psi = np.random.random() * (np.pi / 2.)

    # define affine array
    affine = np.asarray([
        [np.cos(theta) * np.cos(psi),
         -np.cos(phi) * np.sin(psi) + np.sin(phi) * np.sin(theta) * np.cos(psi),
         np.sin(phi) * np.sin(psi) + np.cos(phi) * np.sin(theta) * np.cos(psi)],

        [np.cos(theta) * np.sin(psi),
         np.cos(phi) * np.cos(psi) + np.sin(phi) * np.sin(theta) * np.sin(psi),
         -np.sin(phi) * np.cos(psi) + np.cos(phi) * np.sin(theta) * np.sin(psi)],

        [-np.sin(theta),
         np.sin(phi) * np.cos(theta),
         np.cos(phi) * np.cos(theta)]
    ])

    return affine


def _affine_transform(image, affine, offset=None, order=1):
    """
    Apply a 3D rotation affine transform to input_img with specified offset and spline interpolation order.
    :param image: (np.ndarray) The input image.
    :param affine: (np.ndarray) The 3D affine array of shape [3, 3]
    :param offset: (np.ndarray) The offset to apply to the image after rotation, should be shape [3,]
    :param order: (int) The spline interpolation order. Must be 0-5
    :return: The input image after applying the affine rotation and offset.
    """

    # sanity checks
    if not isinstance(image, np.ndarray):
        raise TypeError("Input image should be np.ndarray but is: " + str(type(image)))
    # define affine if it doesn't exist
    if affine is None:
        affine = _create_affine()
    if not isinstance(affine, np.ndarray):
        raise TypeError("Affine should be np.ndarray but is: " + str(type(affine)))
    if not affine.shape == (3, 3):
        raise ValueError("Affine should have shape (3, 3)")
    # define offset if it doesn't exist
    if offset is None:
        center = np.array(image.shape)
        offset = center - np.dot(affine, center)
    if not isinstance(offset, np.ndarray):
        raise TypeError("Offset should be np.ndarray but is: " + str(type(offset)))
    if not offset.shape == (3,):
        raise ValueError("Offset should have shape (3,)")

    # Apply affine
    # handle 4d
    if len(image.shape) > 3:
        # make 4d identity matrix and replace xyz component with 3d affine
        affine4d = np.eye(4)
        affine4d[:3, :3] = affine
        offset4d = np.append(offset, 0.)
        image = ndi.interpolation.affine_transform(image, affine4d, offset=offset4d, order=order, output=np.float32)
    # handle 3d
    else:
        image = ndi.interpolation.affine_transform(image, affine, offset=offset, order=order, output=np.float32)

    return image


def _affine_transform_roi(image, roi, labels=None, affine=None, dilate=None, order=1):
    """
    Function for affine transforming and extracting an roi from a set of input images and labels
    :param image: (np.ndarray) a 3d or 4d input image
    :param roi: (np.ndarray) a 3d mask or segmentation roi
    :param labels: (np.ndarray) a set of 3d labels (optional)
    :param affine: (np.ndarray) an affine transform. if None, a random transform is created
    :param dilate: (np.ndarray) an int or list of 3 ints. If int, each dim is symmetrically expanded by that amount. If
    an array of 3 ints, then each dimension is expanded to the value of the corresponding int.
    :param order: (int) the spline order used for label interpolation
    :return: (np.ndarray) The transformed and cropped images, rois and optionally labels
    """

    # sanity checks
    if affine is None:
        affine = _create_affine()

    # get input tight bbox shape
    roi = np.squeeze(roi)  # squeeze out singleton channel dimension from data loading
    roi_bbox = _nonzero_slice_inds3d(roi)

    # dilate if necessary
    if dilate is not None:
        roi_bbox = _expand_region(roi.shape, roi_bbox, dilate)

    # determine output shape
    in_shape = [roi_bbox[1] - roi_bbox[0], roi_bbox[3] - roi_bbox[2], roi_bbox[5] - roi_bbox[4]]
    out_x = in_shape[0] * np.abs(affine[0, 0]) + in_shape[1] * np.abs(affine[0, 1]) + in_shape[2] * np.abs(affine[0, 2])
    out_y = in_shape[0] * np.abs(affine[1, 0]) + in_shape[1] * np.abs(affine[1, 1]) + in_shape[2] * np.abs(affine[1, 2])
    out_z = in_shape[0] * np.abs(affine[2, 0]) + in_shape[1] * np.abs(affine[2, 1]) + in_shape[2] * np.abs(affine[2, 2])
    out_shape = [int(np.ceil(out_x)), int(np.ceil(out_y)), int(np.ceil(out_z))]

    # determine affine transform offset
    inv_af = affine.T
    c_in = np.array(in_shape) * 0.5 + np.array(roi_bbox[::2])
    c_out = np.array(out_shape) * 0.5
    offset = c_in - c_out.dot(inv_af)

    # Apply affine to roi
    roi = ndi.interpolation.affine_transform(roi, affine, offset=offset, order=0, output=np.float32,
                                             output_shape=out_shape)

    # Apply affine to image, accouting for possible 4d image
    if len(image.shape) > 3:
        # make 4d identity matrix and replace xyz component with 3d affine
        affine4d = np.eye(4)
        affine4d[:3, :3] = affine
        offset4d = np.append(offset, 0.)
        out_shape4d = np.append(out_shape, image.shape[-1])
        image = ndi.interpolation.affine_transform(image, affine4d, offset=offset4d, order=1, output=np.float32,
                                                   output_shape=out_shape4d)
        out_shape4d = np.append(out_shape, labels.shape[-1])
        labels = ndi.interpolation.affine_transform(labels, affine4d, offset=offset4d, order=order, output=np.float32,
                                                    output_shape=out_shape4d)
    else:
        image = ndi.interpolation.affine_transform(image, affine, offset=offset, order=1, output=np.float32,
                                                   output_shape=out_shape)
        labels = ndi.interpolation.affine_transform(labels, affine, offset=offset, order=order, output=np.float32,
                                                    output_shape=out_shape)

    # crop to rotated input ROI, adjusting for dilate
    nzi = _nonzero_slice_inds3d(roi)
    if dilate is not None:
        nzi = _expand_region(roi.shape, nzi, dilate)
    roi = roi[nzi[0]:nzi[1], nzi[2]:nzi[3], nzi[4]:nzi[5]]
    image = image[nzi[0]:nzi[1], nzi[2]:nzi[3], nzi[4]:nzi[5]]
    labels = labels[nzi[0]:nzi[1], nzi[2]:nzi[3], nzi[4]:nzi[5]]

    return image, roi, labels


def _nonzero_slice_inds3d(input_numpy):
    """
    Takes numpy array and returns slice indices of first and last nonzero pixels in 3d
    :param input_numpy: (np.ndarray) a numpy array containing image data.
    :return: inds - a list of 2 indices per dimension corresponding to the first and last nonzero slices in the array
    """

    # sanity checks
    if type(input_numpy) is not np.ndarray:
        raise ValueError("Input must be numpy array")

    # finds inds of first and last nonzero pixel in x
    vector = np.max(np.max(input_numpy, axis=2), axis=1)
    nz = np.nonzero(vector)[0]
    xinds = [nz[0], nz[-1]]

    # finds inds of first and last nonzero pixel in y
    vector = np.max(np.max(input_numpy, axis=0), axis=1)
    nz = np.nonzero(vector)[0]
    yinds = [nz[0], nz[-1]]

    # finds inds of first and last nonzero pixel in z
    vector = np.max(np.max(input_numpy, axis=0), axis=0)
    nz = np.nonzero(vector)[0]
    zinds = [nz[0], nz[-1]]

    # perpare return
    inds = [xinds[0], xinds[1], yinds[0], yinds[1], zinds[0], zinds[1]]

    return inds


def _crop_pad_image(input_data, out_dims):
    """
    Crops and/or pads an input image to the specified dimensions.
    :param input_data: (np.ndarray) the image data to be padded
    :param out_dims: (list(int)) the desired output dimensions for each axis.
    :return: (np.ndarray) the zero padded image
    """

    # sanity checks
    if type(input_data) is not np.ndarray:
        raise ValueError("Input must be a numpy array")

    # determine pad widths
    crop_pads = []
    for i, o in zip(input_data.shape, out_dims):
        crop_pad = [int(np.ceil((o - i) / 2.)), int(np.floor((o - i) / 2.))]
        crop_pads.append(crop_pad)

    # seperate crops and pads
    crops = [[abs(elem) for elem in item] if sum(item) < 0 else [0, 0] for item in crop_pads]
    pads = [item if sum(item) > 0 else [0, 0] for item in crop_pads]

    # crop array
    if np.sum(crops) > 0:
        orig_dims = input_data.shape
        input_data = input_data[crops[0][0]:orig_dims[0] - crops[0][1],
                                crops[1][0]:orig_dims[1] - crops[1][1],
                                crops[2][0]:orig_dims[2] - crops[2][1]]

    # pad array with zeros (default)
    if np.sum(pads) > 0:
        input_data = np.pad(input_data, pads, 'constant')

    return input_data


def _tf_pad_up(tensor, patch_size, strides):
    """
    Function to symmetrically zero pad a tensor up to a specific shape
    Args:
        tensor: (tf.tensor) the input tensor to pad
        patch_size: (list of 3 ints) the desired output shape for x y z respectively
        strides: (list of 3 ints) the strides for the patches
    Returns: (tf.tensor) the zero padded tensor with x y z shape that can be patched with VALID spacing

    """
    im_s = tf.cast(tf.shape(tensor), tf.float32)  # get full shape of data tensor
    strides = tf.cast(strides, tf.float32)  # cast strides to float
    patch_size = tf.cast(patch_size, tf.float32)  # cast patch to float
    out_s = (tf.math.ceil((im_s - patch_size) / strides) * strides) + patch_size  # determine smallest patchable shape
    out_s = tf.where(out_s < patch_size, patch_size, out_s)  # out shape shouldn't be smaller than patch size
    before = tf.cast(tf.math.floor((out_s - im_s) / 2)[..., tf.newaxis], tf.int32)  # before padding
    after = tf.cast(tf.math.ceil((out_s - im_s) / 2)[..., tf.newaxis], tf.int32)  # after padding
    pads = tf.reshape(tf.concat([before, after], axis=-1), [5, 2])  # concatenate pads
    return tf.pad(tensor, pads, 'CONSTANT', constant_values=0)


def _tf_patch_3d(tensor, ksizes, strides, padding='VALID'):
    tensor = _tf_pad_up(tensor, ksizes, strides)  # pad up to at least a multiple of patch size for VALID patching
    tensor = tf.extract_volume_patches(tensor, ksizes=ksizes, strides=strides, padding=padding)
    return tensor


##############################################
# TENSORFLOW MAP FUNCTIONS
##############################################
def tf_patches_3d(data, labels, patch_size, data_format, data_chan, label_chan=1, overlap=1):
    """
    Extract 3D patches from a data array with overlap if desired
    :param data: (numpy array) the data tensorflow tensor
    :param labels: (numpy array) the labels tensorflow tensor
    :param patch_size: (list or tupe of ints) the patch dimensions
    :param data_format: (str) either channels_last or channels_first - the tensorflow data format
    :param data_chan: (int) the number of channels in the feature data
    :param label_chan: (int) the number of channels in the label data
    :param overlap: (int or list/tuple of ints) the divisor for patch strides - determines the patch overlap in x, y
    :return: returns tensorflow tensor patches
    """

    # sanity checks
    if not len(patch_size) == 3:
        raise ValueError("Patch size must be shape 3 to use 3D patch function but is: " + str(patch_size))

    # handle overlap int vs list/tuple
    if not isinstance(overlap, (np.ndarray, int, list, tuple)):
        raise ValueError("Overlap must be a list, tuple, array, or int.")
    if isinstance(overlap, int):
        overlap = [overlap] * 3

    # handle channels first by temporarily converting to channels last
    if data_format == 'channels_first':
        data = tf.transpose(a=data, perm=[0, 2, 3, 4, 1])
        labels = tf.transpose(a=labels, perm=[0, 2, 3, 4, 1])

    # get sliding window size and strides based on user params
    ksizes = [1] + patch_size + [1]
    strides = [1, patch_size[0] / overlap[0], patch_size[1] / overlap[1], patch_size[2] / overlap[2], 1]
    strides = [int(round(item)) for item in strides]

    # make patches
    data = _tf_patch_3d(data, ksizes, strides, padding='VALID')
    data = tf.reshape(data, [-1] + patch_size + [data_chan])
    labels = _tf_patch_3d(labels, ksizes, strides, padding='VALID')
    labels = tf.reshape(labels, [-1] + patch_size + [label_chan])

    # handle channels first
    if data_format == 'channels_first':
        data = tf.transpose(a=data, perm=[0, 4, 1, 2, 3])
        labels = tf.transpose(a=labels, perm=[0, 4, 1, 2, 3])

    return data, labels


def filter_zero_patches(labels, data_format, thresh=0.05):
    """
    Filters out patches that contain mostly zeros in the label data. Works for 3D and 2D patches.
    :param labels: (tf.tensor) containing labels data (uses only first channel currently)
    :param data_format: (str) either 'channels_first' or 'channels_last' - the tensorflow data format
    :param thresh: (float) the threshold percentage for keeping patches. Default is 5%.
    :return: Returns tf.bool False if less than threshold, else returns tf.bool True
    """
    if float(thresh) == 0.:
        return tf.constant(True, dtype=tf.bool)

    if data_format == 'channels_last':
        # handle channels last
        labels = labels[:, :, :, 0]
    else:
        # handle channels first
        labels = labels[0, :, :, :]

    # make threshold a tf tensor for comparisson
    thr = tf.constant(thresh, dtype=tf.float32)

    return tf.less(thr, tf.math.count_nonzero(labels, dtype=tf.float32) / tf.size(input=labels, out_type=tf.float32))


##############################################
# Low level data loaders
##############################################
class LoadRoiMulticonAndLabels3D:
    """
    Patch loader generates 3D patch data for images and labels given a list of 3D input NiFTI images a mask.
    Performs optional data augmentation with affine rotation in 3D.
    Data is cropped to the nonzero bounding box for the mask file before patches are generated.
    """
    def __init__(self, params):
        self.params = params

    def load(self, study_dir):

        # convert bytes to strings and get relevant params
        study_dir = _byte_convert(study_dir)
        params = self.params
        feat_prefx = params.data_prefix
        label_prefx = params.label_prefix
        mask_prefx = params.mask_prefix
        load_res = params.resample_spacing
        load_shape = params.load_shape
        dilate = params.mask_dilate
        plane = params.data_plane
        dfmt = params.data_format
        aug = params.augment_train_data
        interp = params.label_interp
        dnorm = params.norm_data
        lnorm = params.norm_labels
        mode = params.norm_mode

        # sanity checks
        if plane not in ['ax', 'cor', 'sag']:
            raise ValueError("Did not understand specified plane: " + str(plane))
        if dfmt not in ['channels_last', 'channels_first']:
            raise ValueError("Did not understand specified data_fmt: " + str(plane))

        # load data
        data = _load_single_study(study_dir, feat_prefx, data_format=dfmt, plane=plane, norm=dnorm, norm_mode=mode,
                                  out_res=load_res, out_dims=load_shape)

        # load labels
        labels = _load_single_study(study_dir, label_prefx, data_format=dfmt, plane=plane, norm=lnorm, norm_mode=mode,
                                    out_res=load_res, out_dims=load_shape, res_order=interp)

        # load the mask and handle None mask argument (in which case whole image is used)
        if mask_prefx:
            mask = _load_single_study(study_dir, mask_prefx, data_format=dfmt, plane=plane, norm=False, norm_mode=mode,
                                      out_res=load_res, out_dims=load_shape, res_order=1)
        else:
            mask_dims = data.shape[:-1] if dfmt == "channels_last" else data.shape[1:]
            mask = np.ones(mask_dims)

        # center the ROI in the image usine affine, with optional rotation for data augmentation
        if aug:  # if augmenting, select random rotation values (+/- 30 deg) for each of x, y, and z axes
            posneg = 1 if np.random.random() < 0.5 else -1
            theta = np.random.random() * (np.pi / 6.) * posneg  # rotation in yz plane
            posneg = 1 if np.random.random() < 0.5 else -1
            phi = np.random.random() * (np.pi / 6.) * posneg  # rotation in xz plane
            posneg = 1 if np.random.random() < 0.5 else -1
            psi = np.random.random() * (np.pi / 6.) * posneg  # rotation in xy plane
        else:  # if not augmenting, no rotation is applied, and affine is used only for offset to center the mask ROI
            theta = 0.
            phi = 0.
            psi = 0.

        # make affine, calculate offset using mask center of mass of binirized mask, get nonzero bbox of mask
        affine = _create_affine(theta=theta, phi=phi, psi=psi)

        # apply affines to mask, data, labels
        data, mask, labels = _affine_transform_roi(data, mask, labels, affine, dilate, interp)

        # add batch and channel dims as necessary to get to [batch, x, y, z, channel]
        data = np.expand_dims(data, axis=0)  # add a batch dimension of 1
        labels = np.expand_dims(labels, axis=0)  # add a batch and channel dimension of 1

        # handle different planes
        if plane == 'ax':
            pass
        elif plane == 'cor':
            data = np.transpose(data, axes=[0, 1, 3, 2, 4])
            labels = np.transpose(labels, axes=[0, 1, 3, 2, 4])
        elif plane == 'sag':
            data = np.transpose(data, axes=[0, 2, 3, 1, 4])
            labels = np.transpose(labels, axes=[0, 2, 3, 1, 4])
        else:
            raise ValueError("Did not understand specified plane: " + str(plane))

        # handle channels first data format
        if dfmt == 'channels_first':
            data = np.transpose(data, axes=[0, 4, 1, 2, 3])
            labels = np.transpose(labels, axes=[0, 4, 1, 2, 3])

        return data.astype(np.float32), labels.astype(np.float32)


class LoadMulticonAndLabels3D:
    """
    Load multicontrast image data and labels without cropping or otherwise adjusting size.
    For use with testing.
    """
    def __init__(self, params):
        self.params = params

    def load(self, study_dir):

        # get relevant params
        study_dir = _byte_convert(study_dir)
        params = self.params
        feat_prefx = params.data_prefix
        label_prefx = params.label_prefix
        dfmt = params.data_format
        plane = params.data_plane
        dnorm = params.norm_data
        lnorm = params.norm_labels
        mode = params.norm_mode
        load_res = params.resample_spacing
        load_shape = params.load_shape
        interp = params.label_interp

        # sanity checks
        if not os.path.isdir(study_dir):
            raise ValueError("Specified study_directory does not exist")
        if dfmt not in ['channels_last', 'channels_first']:
            raise ValueError("data_format invalid")

        # load multi-contrast data and normalize, no slice trimming for infer data
        data = _load_single_study(study_dir, feat_prefx, data_format=dfmt, plane=plane, norm=dnorm, norm_mode=mode,
                                  out_res=load_res, out_dims=load_shape)
        labels = _load_single_study(study_dir, label_prefx, data_format=dfmt, plane=plane, norm=lnorm, norm_mode=mode,
                                    out_res=load_res, out_dims=load_shape, res_order=interp)

        # generate batch size==1 format such that format is [1, x, y, z, c] or [1, c, x, y, z]
        data = np.expand_dims(data, axis=0)
        labels = np.expand_dims(labels, axis=0)

        return data, labels


class LoadMulticon3D:
    """
    Load multicontrast image data without cropping or otherwise adjusting size. For use with inference/prediction.
    """
    def __init__(self, params):
        self.params = params

    def load(self, study_dir):

        # get relevant params
        study_dir = _byte_convert(study_dir)
        params = self.params
        feat_prefx = params.data_prefix
        data_fmt = params.data_format
        plane = params.data_plane
        norm = params.norm_data
        norm_mode = params.norm_mode
        chan_dim = len(params.label_prefix)
        load_res = params.resample_spacing
        load_shape = params.load_shape

        # sanity checks
        if not os.path.isdir(study_dir):
            raise ValueError("Specified study_directory does not exist")
        if data_fmt not in ['channels_last', 'channels_first']:
            raise ValueError("data_format invalid")

        # load multi-contrast data and normalize, no slice trimming for infer data
        data = _load_single_study(study_dir, feat_prefx, data_format=data_fmt, plane=plane, norm=norm,
                                  norm_mode=norm_mode, out_res=load_res, out_dims=load_shape)

        # generate batch size==1 format such that format is [1, x, y, z, c] or [1, c, x, y, z]
        data = np.expand_dims(data, axis=0)

        # retun an empty array for "labels" so that this will work with other functions
        labels = np.empty(data.shape[0:4] + (chan_dim,), dtype=np.float32)

        return data, labels


##############################################
# PATCH RECONSTRUCTION FUNCTIONS
##############################################
def reconstruct_infer_patches_3d(predictions, infer_dir, params):
    """
    Function for reconstructing the input 3D volume from the output predictions after 3D image patch prediction
    :param predictions: (tf.tensor) - the output of 3D patch prediction
    :param infer_dir: (str) - the directory containing the inferance data
    :param params: (obj) - the parameter object derived from the param file
    :return: (np.ndarray) - returns the reconstructed image generated by reversing ExtractVolumePatches
    """

    # define params - converting all to python native variables as they may be imported as numpy
    patch_size = params.infer_dims
    overlap = params.infer_patch_overlap

    # for sliding window 3d slabs - must be same as in _tf_patches_3d_infer above
    ksizes = [1] + patch_size + [1]
    strides = [1, patch_size[0] / overlap[0], patch_size[1] / overlap[1], patch_size[2] / overlap[2], 1]

    # define necessary functions
    def extract_patches(x):
        return _tf_patch_3d(x, ksizes, strides, padding='VALID')

    def extract_patches_inverse(x, y):
        with tf.GradientTape(persistent=True) as tape:
            _x = tf.zeros_like(x)
            tape.watch(_x)
            _y = extract_patches(_x)
        # get gradient
        grad = tape.gradient(_y, _x)
        # Divide by grad, to "average" together the overlapping patches
        # otherwise they would simply sum up
        return tape.gradient(_y, _x, output_gradients=y) / grad

    # load original data as a dummy and convert channel dim size to match output [batch, x, y, z, channel]
    data, _ = LoadMulticon3D(params).load(infer_dir)
    data = np.zeros((data.shape[0:4] + (params.output_filters,)), dtype=np.float32)

    # get shape of patches as they would have been generated during inference
    dummy_patches = extract_patches(data)

    # reshape predictions to original patch shape
    predictions = tf.reshape(predictions, tf.shape(input=dummy_patches))

    # reconstruct
    reconstructed = extract_patches_inverse(data, predictions)
    output = np.squeeze(reconstructed.numpy())

    return output
