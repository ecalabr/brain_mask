"""
Utility functions for general use
"""

import yaml
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
from glob import glob
import random


def load_param_file(yaml_path):
    """
    Function loads hyperparameters from a yaml or yml file.
    Example:
    params = load_param_file(yaml_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    """

    required_params = [
        "data_dir",
        "model_dir",
        "overwrite",
        "random_state",
        "data_prefix",
        "label_prefix",
        "mask_prefix",
        "mask_dilate",
        "filter_zero",
        "input_function",
        "data_plane",
        "train_dims",
        "train_patch_overlap",
        "infer_dims",
        "infer_patch_overlap",
        "augment_train_data",
        "label_interp",
        "metrics",
        "norm_data",
        "norm_labels",
        "norm_mode",
        "model_name",
        "base_filters",
        "output_filters",
        "layer_layout",
        "final_layer",
        "kernel_size",
        "data_format",
        "activation",
        "mixed_precision",
        "dist_strat",
        "shuffle_size",
        "batch_size",
        "num_threads",
        "samples_per_epoch",
        "train_fract",
        "learning_rate",
        "learning_rate_decay",
        "loss",
        "optimizer",
        "num_epochs",
        "dropout_rate"
    ]

    optional_params = []

    class Params:
        def __init__(self, my_yaml_path):
            self.update(my_yaml_path)  # load parameters
            self.check()  # check parameters
            self.params_path = my_yaml_path  # path to param file stored here

        def save(self, my_yaml_path):
            """Saves parameters to yml file"""
            with open(my_yaml_path, 'w') as f:
                yaml.dump(self.__dict__, f, indent=4)

        def update(self, my_yaml_path):
            """Loads parameters from yml file"""
            with open(my_yaml_path) as f:
                params = yaml.load(f, Loader=yaml.SafeLoader)
                self.__dict__.update(params)

        def check(self):
            """Checks that all required parameters are defined in params.yml file"""
            members = [attr for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("__")]
            if any([param not in members for param in required_params]):
                missing = [param for param in required_params if param not in members]
                raise ValueError("Missing the following parameter(s) in parameter file: {}".format(" ".join(missing)))
            # check for unused params in param file
            all_params = required_params + optional_params
            unused = [param for param in members if param not in all_params]
            if unused:
                raise ValueError("The following parameters in param file are not used: {}".format(" ".join(unused)))
    return Params(yaml_path)


def set_logger(log_path, level=logging.INFO):
    """
    Sets the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file.
    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
        level: (int) the logging level
    """
    logger = logging.getLogger()
    logger.setLevel(level)

    if not logger.handlers:
        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)

        # Logging to a file
        if log_path and os.path.isdir(os.path.dirname(log_path)):
            file_handler = logging.FileHandler(log_path)
            file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
            logger.addHandler(file_handler)
            logger.info("Created log file at {}".format(log_path))
        else:
            logger.info("Logging to console only")

    return logger


# utility function to get all study subdirectories in a given parent data directory
# returns shuffled directory list using user defined randomization seed
# saves a copy of output to study_dirs_list.yml in study directory
def get_study_dirs(params, change_basedir=None):

    # Study dirs yml filename setup
    study_dirs_filepath = os.path.join(params.model_dir, 'study_dirs_list.yml')

    # load study dirs file if it already exists for consistent training
    if os.path.isfile(study_dirs_filepath):
        logging.info("Loading existing study directories file: {}".format(study_dirs_filepath))
        with open(study_dirs_filepath) as f:
            study_dirs = yaml.load(f, Loader=yaml.SafeLoader)

        # check that study_dirs_list format is correct
        if not all([item in study_dirs.keys() for item in ["train", "eval", "test"]]):
            raise ValueError("study_dirs_list.yml must contain entries for train, eval, and test")

        # loop through train, eval, test
        for key in ["train", "eval", "test"]:

            # handle change_basedir argument
            if change_basedir:
                # get rename list of directories
                study_dirs[key] = [os.path.join(change_basedir, os.path.basename(os.path.dirname(item))) for item in
                                   study_dirs[key]]
                if not all([os.path.isdir(d) for d in study_dirs[key]]):
                    logging.error("Using change basedir argument but not all {} directories exist".format(key))
                    # get list of missing files
                    missing = []
                    for item in study_dirs[key]:
                        if not os.path.isdir(item):
                            missing.append(item)
                    raise FileNotFoundError("Missing the following {} directories: {}".format(', '.join(missing), key))

            # make sure that study directories loaded from file actually exist and warn/error if some/all do not
            valid_study_dirs = []
            for study in study_dirs[key]:
                # get list of all expected files via glob
                files = [glob("{}/*{}.nii.gz".format(study, item)) for item in params.data_prefix + params.label_prefix]
                # check that a file was found and that file exists in each case
                if all(files) and all([os.path.isfile(f[0]) for f in files]):
                    valid_study_dirs.append(study)
            # case, no valid study dirs
            if not valid_study_dirs:
                logging.info(" -No valid {} directories found".format(key))
            # case, less valid study dirs than found in study dirs file
            elif len(valid_study_dirs) < len(study_dirs[key]):
                logging.warning(
                    " -Some {} directories listed in study_dirs_list.yml are missing or incomplete".format(key))
            # case, all study dirs in study dirs file are valid
            else:
                logging.info(" -All {} directories listed in study_dirs_list.yml are present and complete".format(key))
            study_dirs[key] = valid_study_dirs

    # if study dirs file does not exist, then determine study directories and create study_dirs_list.yml
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

        # do train eval split
        train_dirs, eval_dirs = train_test_split(study_dirs, params)
        study_dirs = {"train": train_dirs, "eval": eval_dirs, "test": []}

        # save directory list to yml file so it can be loaded in future
        with open(study_dirs_filepath, 'w+', encoding='utf-8') as f:
            yaml.dump(study_dirs, f)  # save study dir list for consistency

    return study_dirs


# split list of all valid study directories into a train and test batch based on train fraction
def train_test_split(study_dirs, params):
    # first train fraction is train dirs, last 1-train fract is test dirs
    # assumes study dirs is already shuffled and/or stratified as wanted
    train_dirs = study_dirs[0:int(np.floor(params.train_fract * len(study_dirs)))]
    eval_dirs = study_dirs[int(np.floor(params.train_fract * len(study_dirs))):]

    return train_dirs, eval_dirs


def display_tf_dataset(dataset_data, data_format, data_dims):
    """
    Displays tensorflow dataset output images and labels/regression images.
    :param dataset_data: (tf.tensor) output from tf dataset function containing images and labels/regression image
    :param data_format: (str) the desired tensorflow data format. Must be either 'channels_last' or 'channels_first'
    :param data_dims: (list or tuple of ints) the data dimensions that come out of the input function
    :return: displays images for 3 seconds then continues
    """

    # make figure
    fig = plt.figure(figsize=(10, 4))

    # define close event and create timer
    def close_event():
        plt.close()
    timer = fig.canvas.new_timer(interval=4000)
    timer.add_callback(close_event)

    # handle 2d case
    if len(data_dims) == 2:
        # image data
        image_data = dataset_data[0]  # dataset_data[0]
        if len(image_data.shape) > 3:
            image_data = np.squeeze(image_data[0, :, :, :])  # handle batch data
        nplots = image_data.shape[0] + 1 if data_format == 'channels_first' else image_data.shape[2] + 1
        channels = image_data.shape[0] if data_format == 'channels_first' else image_data.shape[2]
        for z in range(channels):
            ax = fig.add_subplot(1, nplots, z + 1)
            data_img = np.swapaxes(np.squeeze(image_data[z, :, :]), 0,
                                   1) if data_format == 'channels_first' else np.squeeze(
                image_data[:, :, z])
            ax.imshow(data_img, cmap='gray')
            ax.set_title('Data Image ' + str(z + 1))

        # label data
        label_data = dataset_data[1]  # dataset_data[1]
        if len(label_data.shape) > 3:
            label_data = np.squeeze(label_data[0, :, :, :])  # handle batch data
        ax = fig.add_subplot(1, nplots, nplots)
        label_img = np.swapaxes(np.squeeze(label_data), 0, 1) if data_format == 'channels_first' else np.squeeze(
            label_data)
        ax.imshow(label_img, cmap='gray')
        ax.set_title('Labels')

    # handle 3d case
    if len(data_dims) == 3:

        # load image data
        image_data = dataset_data[0]  # dataset_data[0]

        # handle channels first and batch data
        if len(image_data.shape) > 4:
            if data_format == 'channels_first':
                image_data = np.transpose(image_data, [0, 2, 3, 4, 1])
            image_data = np.squeeze(image_data[0, :, :, :, :])  # handle batch data
        else:
            if data_format == 'channels_first':
                image_data = np.transpose(image_data, [1, 2, 3, 0])

        # determine n plots and channels
        nplots = image_data.shape[-1] + 1
        channels = image_data.shape[-1]

        # loop through channels
        for z in range(channels):
            ax = fig.add_subplot(1, nplots, z + 1)
            data_img = np.squeeze(image_data[:, :, :, z])
            # concatenate along z to make 1 2d image per slab
            data_img = np.reshape(np.transpose(data_img), [data_img.shape[0] * data_img.shape[2], data_img.shape[1]])
            ax.imshow(data_img, cmap='gray')
            ax.set_title('Data Image ' + str(z + 1))

        # load label data
        label_data = dataset_data[1]  # dataset_data[1]

        # handle channels first and batch data
        if len(label_data.shape) > 4:
            if data_format == 'channels_first':
                label_data = np.transpose(label_data, [0, 2, 3, 4, 1])
            label_data = label_data[0, :, :, :, :]  # handle batch data by taking only first element of batch
        else:
            if data_format == 'channels_first':
                label_data = np.transpose(label_data, [1, 2, 3, 0])

        # add labels to fig
        ax = fig.add_subplot(1, nplots, nplots)
        label_img = np.squeeze(label_data)
        inds = [label_img.shape[0] * label_img.shape[2], label_img.shape[1]]
        label_img = np.reshape(np.transpose(label_img), inds)
        ax.imshow(label_img, cmap='gray')
        ax.set_title('Labels')

    # start timer and show plot
    timer.start()
    plt.show()

    return
