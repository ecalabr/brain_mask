"""
Allows a user to choose a specific normalization function using a string in parameter file
"""

import numpy as np
import scipy.stats as stats


# define globals
EPSILON = 1e-10


class Normalizers:
    methods = {}

    def __init__(self, m):
        self.method = m
        if self.method not in self.methods:
            raise ValueError(
                "Specified normalization method: '{}' is not an available method: {}".format(self.method,
                                                                                             self.methods))

    def __call__(self, input_img):
        if not isinstance(input_img, np.ndarray):
            raise TypeError("Input image should be np.ndarray but is: " + str(type(input_img)))
        return self.methods[self.method](input_img)

    @classmethod
    def register_method(cls, name):
        def decorator(m):
            cls.methods[name] = m
            return m
        return decorator


# handle unit mode
@Normalizers.register_method("unit")
def unit(img):
    # perform normalization to [0, 1]
    img *= 1.0 / (np.max(img) + EPSILON)
    return img


# handle mean zscore
@Normalizers.register_method("zscore")
def zscore(img):
    # perform z score normalization to 0 mean, unit std
    nonzero_bool = img != 0.
    mean = np.mean(img[nonzero_bool], axis=None)
    std = np.std(img[nonzero_bool], axis=None) + EPSILON
    img = np.where(nonzero_bool, ((img - mean) / std), 0.)
    return img


# handle mean stdev
@Normalizers.register_method("mean_stdev")
def mean_stdev(img):
    # constants
    new_mean = 1000.
    new_std = 200.
    # perform normalization to specified mean, stdev
    nonzero_bool = img != 0.
    mean = np.mean(img[nonzero_bool], axis=None)
    std = np.std(img[nonzero_bool], axis=None) + EPSILON
    img = np.where(nonzero_bool, ((img - mean) / (std / new_std)) + new_mean, 0.)
    return img


# handle median interquartile range
@Normalizers.register_method("med_iqr")
def med_iqr(img, new_med=0., new_stdev=1.):
    # perform normalization to median, normalized interquartile range
    # uses factor of 0.7413 to normalize interquartile range to standard deviation
    nonzero_bool = img != 0.
    med = np.median(img[nonzero_bool], axis=None)
    niqr = stats.iqr(img[nonzero_bool], axis=None) * 0.7413 + EPSILON
    img = np.where(nonzero_bool, ((img - med) / (niqr / new_stdev)) + new_med, 0.)
    return img
