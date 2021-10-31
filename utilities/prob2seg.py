""" converts one or more 4D probabilty image(s) into a binary mask using softmax argmax """

import nibabel as nib
import os
import argparse
import numpy as np
import scipy.ndimage
from glob import glob
import logging


# define functions
def softmax(x, theta=1.0, axis=None):
    # make X at least 2d
    y = np.atleast_2d(x)
    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)
    # multiply y against the theta parameter,
    y = y * float(theta)
    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis=axis), axis)
    # exponentiate y
    y = np.exp(y)
    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis=axis), axis)
    # finally: divide elementwise
    p = y / ax_sum
    # flatten if X was 1D
    if len(x.shape) == 1:
        p = p.flatten()
    return p


def get_largest_component(img):
    s = scipy.ndimage.generate_binary_structure(3, 2)  # iterate structure
    labeled_array, numpatches = scipy.ndimage.label(img, s)  # labeling
    sizes = scipy.ndimage.sum(img, labeled_array, range(1, numpatches + 1))
    sizes_list = [sizes[i] for i in range(len(sizes))]
    sizes_list.sort()
    if len(sizes) == 1:
        out_img = img
    else:
        max_size = sizes_list[-1]
        max_label = np.where(sizes == max_size)[0] + 1
        component = labeled_array == max_label
        out_img = component
    return out_img


def get_largest_n_components(img, n=1):
    s = scipy.ndimage.generate_binary_structure(3, 2)  # iterate structure
    labeled_array, numpatches = scipy.ndimage.label(img, s)  # labeling
    sizes = scipy.ndimage.sum(img, labeled_array, range(1, numpatches + 1))
    sizes_list = [sizes[i] for i in range(len(sizes))]
    sizes_list.sort()
    # if n labels is less than n segs, then just use entire image > 0
    if len(sizes) <= n:
        out_img = img > 0.
    # if there are more than n labels, then just get the largest n
    else:
        biggest_sizes = sizes_list[-n:]
        max_labels = np.where(np.isin(sizes, biggest_sizes))[0] + 1
        components = np.isin(labeled_array, max_labels)
        out_img = components
    return out_img


def robpow(a, n):
    return np.sign(a) * (np.abs(a)) ** n


def create_superellipse_mask(shape, o=2.5, v=0.9):
    # unpack shape
    w, h, d = shape

    # get center
    center = np.array((int(w / 2), int(h / 2), int(d / 2)))

    # determine ellipsoid
    x, y, z = np.ogrid[:w, :h, :d]
    ellipsoid = (
                (np.abs((x - center[0]) / (center[0])) ** o) +
                (np.abs((y - center[1]) / (center[1])) ** o) +
                (np.abs((z - center[2]) / (center[2])) ** o)
                ) ** (1 / o)

    # make elliptical mask
    mask = ellipsoid > v

    return mask


def convert_prob(files, nii_out_path, clean, thresh=0.5, n_seg=1, zero_periph=True, order=2.5):
    # set up logger
    logger = logging.getLogger("brain_mask")

    # announce files
    logger.info("Found the following probability file: {}".format(' '.join(files)))

    # load data
    data = []
    nii = []

    # loop through files
    if isinstance(files, str):
        files = [files]
    for f in files:
        nii = nib.load(f)
        data.append(nii.get_fdata())

    # convert to array
    data = np.squeeze(np.array(data))

    # determine if softmax needed
    if np.max(data) > 1. or np.min(data) < 0.:
        data = [softmax(arr, axis=-1) for arr in data]

    # handle single vs multiple predictions that need to be averaged
    if len(data.shape) > 4:
        # average
        data = np.mean(data, axis=0)
    # handle multple prediction channels vs just one
    if len(data.shape) > 3:
        # argmax
        data = np.argmax(data, axis=-1)
    else:
        data = data > thresh

    # zero out array periphery
    if zero_periph:
        ellipse_mask = create_superellipse_mask(data.shape, o=order)
        data[ellipse_mask] = 0

    # binary morph ops
    try:
        struct = scipy.ndimage.generate_binary_structure(3, 2)  # rank 3, connectivity 2
        struct = scipy.ndimage.iterate_structure(struct, 2)  # iterate structure to 5x5x5
        data = scipy.ndimage.morphology.binary_erosion(data, structure=struct)  # erosion
        data = get_largest_n_components(data, n_seg)  # largest connected component
        data = scipy.ndimage.morphology.binary_dilation(data, structure=struct)  # dilation
        data = scipy.ndimage.morphology.binary_fill_holes(data)  # fill holes
        data = scipy.ndimage.morphology.binary_closing(data, structure=struct)  # final closing
    except Exception:
        logger.error("Mask generation failed... Skipping...")
        if clean:
            for f in files:
                os.remove(f)
        return nii_out_path

    # make output nii
    nii_out = nib.Nifti1Image(data.astype(float), nii.affine, nii.header)
    nib.save(nii_out, nii_out_path)

    # if output is created, and cleanup is true, then clean
    if os.path.isfile(nii_out_path) and clean:
        for f in files:
            os.remove(f)

    return nii_out_path


# executed  as script
if __name__ == '__main__':

    # parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default=None,
                        help="Path to data image(s). Can be an image or a directory.")
    parser.add_argument('--name', default='',
                        help="Name of data image(s). Any string within the data filename. " +
                             "Leave blank for all images, which will be averaged.")
    parser.add_argument('--outname', default='binary_mask.nii.gz',
                        help="Name of output image")
    parser.add_argument('--outpath', default=None,
                        help="Output path")
    parser.add_argument('--clean', action="store_true", default=False,
                        help="Delete inputs after conversion")

    # check input arguments
    args = parser.parse_args()
    data_in_path = args.data
    assert data_in_path, "Must specify input data using --data"
    namestr = args.name
    outname = args.outname
    outpath = args.outpath
    if '.nii.gz' not in outname:
        outname = outname.split('.')[0] + '.nii.gz'
    my_files = []
    data_root = []
    if os.path.isfile(data_in_path):
        my_files = [data_in_path]
        data_root = os.path.dirname(data_in_path)
    elif os.path.isdir(data_in_path):
        my_files = glob(data_in_path + '/*' + namestr + '*.nii*')
        data_root = data_in_path
    else:
        raise ValueError("No data found at {}".format(data_in_path))

    # handle outpath creation and make sure output file is not an input
    if outpath and os.path.isdir(outpath):
        data_root = outpath
    my_nii_out_path = os.path.join(data_root, outname)
    if my_nii_out_path in my_files:
        my_files.remove(my_nii_out_path)

    # announce
    if not my_files:
        raise ValueError("No data found at {}".format(data_in_path))

    # do work
    final_nii_out = convert_prob(my_files, my_nii_out_path, args.clean)
