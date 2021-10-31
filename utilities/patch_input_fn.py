import random
import tensorflow as tf
from utilities.input_fn_util import *
import json
import logging
import numpy as np
import os
from glob import glob


def _patch_input_fn_3d(params, mode, train_dirs, eval_dirs, infer_dir=None):
    # generate input dataset objects for the different training modes

    # train mode - uses patches, patch filtering, batching, data augmentation, and shuffling - works on train_dirs
    if mode == 'train':
        # variable setup
        data_dirs = tf.constant(train_dirs)
        data_chan = len(params.data_prefix)
        weighted = False if isinstance(params.mask_weights, np.bool) and not params.mask_weights else True
        # defined the fixed py_func params, the study directory will be passed separately by the iterator
        py_func_params = [params.data_prefix,
                          params.label_prefix,
                          params.mask_prefix,
                          params.mask_dilate,
                          params.data_plane,
                          params.data_format,
                          params.augment_train_data,
                          params.label_interp,
                          params.norm_data,
                          params.norm_labels,
                          params.norm_mode,
                          params.mask_weights]  # param makes loader return weights as last channel in labels data]
        # create tensorflow dataset variable from data directories
        dataset = tf.data.Dataset.from_tensor_slices(data_dirs)
        # randomly shuffle directory order
        dataset = dataset.shuffle(buffer_size=len(data_dirs))
        # map data directories to the data using a custom python function
        dataset = dataset.map(
            lambda x: tf.numpy_function(load_roi_multicon_and_labels_3d,
                                        [x] + py_func_params,
                                        (tf.float32, tf.float32)),
            num_parallel_calls=params.num_threads)  # tf.data.experimental.AUTOTUNE)
        # map each dataset to a series of patches
        dataset = dataset.map(
            lambda x, y: tf_patches_3d(x, y, params.train_dims, params.data_format, data_chan,
                                       weighted=weighted,
                                       overlap=params.train_patch_overlap),
            num_parallel_calls=params.num_threads)  # tf.data.experimental.AUTOTUNE)
        # flatten out dataset so that each entry is a single patch and associated label
        dataset = dataset.flat_map(lambda x, y: tf.data.Dataset.from_tensor_slices((x, y)))
        # filter out zero patches
        if params.filter_zero > 0.:
            dataset = dataset.filter(lambda x, y: filter_zero_patches(
                y, params.data_format, params.dimension_mode, params.filter_zero))
        # shuffle a set number of exampes
        dataset = dataset.shuffle(buffer_size=params.shuffle_size)
        # generate batch data
        dataset = dataset.batch(params.batch_size, drop_remainder=True)
        # prefetch with experimental autotune
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        # repeat dataset infinitely so that dataset doesn't exhaust prematurely during fit
        dataset = dataset.repeat()

    # eval mode - uses patches and batches but no patch filtering, data augmentation, or shuffling, works on eval_dirs
    elif mode == 'eval':
        # variable setup
        data_dirs = tf.constant(eval_dirs)
        data_chan = len(params.data_prefix)
        weighted = False if isinstance(params.mask_weights, np.bool) and not params.mask_weights else True
        # defined the fixed py_func params, the study directory will be passed separately by the iterator
        py_func_params = [params.data_prefix,
                          params.label_prefix,
                          params.mask_prefix,
                          params.mask_dilate,
                          params.data_plane,
                          params.data_format,
                          False,  # no data augmentation for eval mode
                          params.label_interp,
                          params.norm_data,
                          params.norm_labels,
                          params.norm_mode,
                          params.mask_weights]  # param makes loader return weights as last channel in labels data]
        # create tensorflow dataset variable from data directories
        dataset = tf.data.Dataset.from_tensor_slices(data_dirs)
        # map data directories to the data using a custom python function
        dataset = dataset.map(
            lambda x: tf.numpy_function(load_roi_multicon_and_labels_3d,
                                        [x] + py_func_params,
                                        (tf.float32, tf.float32)),
            num_parallel_calls=params.num_threads)  # tf.data.experimental.AUTOTUNE)
        # map each dataset to a series of patches
        dataset = dataset.map(
            lambda x, y: tf_patches_3d(x, y, params.train_dims, params.data_format, data_chan,
                                       weighted=weighted,
                                       overlap=params.train_patch_overlap),
            num_parallel_calls=params.num_threads)  # tf.data.experimental.AUTOTUNE)
        # flatten out dataset so that each entry is a single patch and associated label
        dataset = dataset.flat_map(lambda x, y: tf.data.Dataset.from_tensor_slices((x, y)))
        # generate batch data
        dataset = dataset.batch(params.batch_size, drop_remainder=True)
        # prefetch with experimental autotune
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    # infer mode - does not use patches (patch function uses infer_dims), batches, or shuffling - works on infer_dir
    elif mode == 'infer':
        if not infer_dir:
            assert ValueError("Must specify inference directory for inference mode")
        dirs = tf.constant(infer_dir)
        # define dims of inference
        data_dims = list(params.infer_dims)
        chan_size = len(params.data_prefix)
        # defined the fixed py_func params, the study directory will be passed separately by the iterator
        py_func_params = [params.data_prefix, params.data_format, params.data_plane, params.norm_data,
                          params.norm_mode]
        # create tensorflow dataset variable from data directories
        dataset = tf.data.Dataset.from_tensor_slices(dirs)
        # map data directories to the data using a custom python function
        dataset = dataset.map(
            lambda x: tf.numpy_function(load_multicon_preserve_size_3d,
                                        [x] + py_func_params,
                                        tf.float32),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        # map each dataset to a series of patches based on infer inputs
        dataset = dataset.map(
            lambda x: tf_patches_3d_infer(x, data_dims, chan_size, params.data_format, params.infer_patch_overlap),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        # flat map so that each tensor is a single slice
        dataset = dataset.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(x))
        # generate a batch of data
        dataset = dataset.batch(batch_size=1, drop_remainder=True)
        # automatic prefetching to improve efficiency
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    # error if not train, eval, or infer
    else:
        raise ValueError("Specified mode does not exist: " + mode)

    return dataset


# utility function to get all study subdirectories in a given parent data directory
# returns shuffled directory list using user defined randomization seed
# saves a copy of output to study_dirs_list.json in study directory
def get_study_dirs(params, change_basedir=None):

    # Study dirs json filename setup
    study_dirs_filepath = os.path.join(params.model_dir, 'study_dirs_list.json')

    # load study dirs file if it already exists for consistent training
    if os.path.isfile(study_dirs_filepath):
        logging.info("Loading existing study directories file: {}".format(study_dirs_filepath))
        with open(study_dirs_filepath) as f:
            study_dirs = json.load(f)

        # handle change_basedir argument
        if change_basedir:
            # get rename list of directories
            study_dirs = [os.path.join(change_basedir, os.path.basename(os.path.dirname(item))) for item in study_dirs]
            if not all([os.path.isdir(d) for d in study_dirs]):
                logging.error("Using change basedir argument in get_study_dirs but not all study directories exist")
                # get list of missing files
                missing = []
                for item in study_dirs:
                    if not os.path.isdir(item):
                        missing.append(item)
                raise FileNotFoundError("Missing the following data directories: {}".format(', '.join(missing)))

        # make sure that study directories loaded from file actually exist and warn/error if some/all do not
        valid_study_dirs = []
        for study in study_dirs:
            # get list of all expected files via glob
            files = [glob("{}/*{}.nii.gz".format(study, item)) for item in params.data_prefix + params.label_prefix]
            # check that a file was found and that file exists in each case
            if all(files) and all([os.path.isfile(f[0]) for f in files]):
                valid_study_dirs.append(study)
        # case, no valid study dirs
        if not valid_study_dirs:
            logging.error("study_dirs_list.json exists in the model directory but does not contain valid directories")
            raise ValueError("No valid study directories in study_dirs_list.json")
        # case, less valid study dirs than found in study dirs file
        elif len(valid_study_dirs) < len(study_dirs):
            logging.warning("Some study directories listed in study_dirs_list.json are missing or incomplete")
        # case, all study dirs in study dirs file are valid
        else:
            logging.info("All directories listed in study_dirs_list.json are present and complete")
        study_dirs = valid_study_dirs

    # if study dirs file does not exist, then determine study directories and create study_dirs_list.json
    else:
        logging.info("Determining train/test split based on params and available study directories in data directory")
        # get all valid subdirectories in data_dir
        study_dirs = [item for item in glob(params.data_dir + '/*/') if os.path.isdir(item)]
        # make sure all necessary files are present in each folder
        study_dirs = [study for study in study_dirs if all(
            [glob('{}/*{}.nii.gz'.format(study, item)) and os.path.isfile(glob('{}/*{}.nii.gz'.format(study, item))[0])
             for item in params.data_prefix + params.label_prefix])]
        # error if no valid study dirs found
        assert len(study_dirs) >= 1, "No valid Study directories found in data directory: {} using prefixes: {}".format(
            params.data_dir, params.data_prefix + params.label_prefix)

        # study dirs sorted in alphabetical order for reproducible results
        study_dirs.sort()

        # randomly shuffle input directories for training using a user defined randomization seed
        random.Random(params.random_state).shuffle(study_dirs)

        # save directory list to json file so it can be loaded in future
        with open(study_dirs_filepath, 'w+', encoding='utf-8') as f:
            json.dump(study_dirs, f, ensure_ascii=False, indent=4)  # save study dir list for consistency

    return study_dirs


# split list of all valid study directories into a train and test batch based on train fraction
def train_test_split(study_dirs, params):
    # first train fraction is train dirs, last 1-train fract is test dirs
    # assumes study dirs is already shuffled and/or stratified as wanted
    train_dirs = study_dirs[0:int(np.floor(params.train_fract * len(study_dirs)))]
    eval_dirs = study_dirs[int(np.floor(params.train_fract * len(study_dirs))):]

    return train_dirs, eval_dirs


# patch input function for 2d or 3d
def patch_input_fn(params, mode, infer_dir=None):

    # set global random seed for tensorflow
    tf.random.set_seed(params.random_state)

    # handle inference mode
    if mode == 'infer':
        infer_dir = tf.constant([infer_dir])
        train_dirs = []
        eval_dirs = []

    # handle train and eval modes
    else:
        # get valid study directories
        study_dirs = get_study_dirs(params)

        # split study directories into train and test sets
        train_dirs, eval_dirs = train_test_split(study_dirs, params)

    # return data structure
    return _patch_input_fn_3d(params, mode, train_dirs, eval_dirs, infer_dir)
