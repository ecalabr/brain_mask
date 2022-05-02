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

    required_params = {
        "data_dir":  {"type": str},
        "model_dir":  {"type": str},
        "overwrite":  {"type": bool},
        "random_state":  {"type": int},
        "data_prefix":  {"type": list, "subtype": str},
        "label_prefix":  {"type": list, "subtype": str},
        "mask_prefix":  {"type": list, "subtype": str},
        "mask_dilate":  {"type": list, "subtype": int, "length": [1, 2, 3]},
        "filter_zero":  {"type": (int, float)},
        "resample_spacing": {"type": list, "subtype": (int, float), "length": [0, 1, 2, 3]},
        "load_shape": {"type": list, "subtype": int, "length": [0, 1, 2, 3]},
        "input_function":  {"type": str},
        "data_plane":  {"type": str},
        "train_dims":  {"type": list, "subtype": int, "length": [2, 3]},
        "train_patch_overlap":  {"type": list, "subtype": (int, float), "length": [2, 3]},
        "infer_dims":  {"type": list, "subtype": int, "length": [2, 3]},
        "infer_patch_overlap":  {"type": list, "subtype": (int, float), "length": [2, 3]},
        "augment_train_data":  {"type": bool},
        "label_interp":  {"type": int},
        "metrics":  {"type": list, "subtype": str},
        "norm_data":  {"type": bool},
        "norm_labels":  {"type": bool},
        "norm_mode":  {"type": str},
        "model_name":  {"type": str},
        "base_filters":  {"type": int},
        "output_filters":  {"type": int},
        "layer_layout":  {"type": list, "subtype": int},
        "final_layer":  {"type": str},
        "kernel_size":  {"type": list, "subtype": int, "length": [2, 3]},
        "data_format":  {"type": str},
        "activation":  {"type": str},
        "mixed_precision":  {"type": bool},
        "dist_strat":  {"type": str},
        "shuffle_size":  {"type": int},
        "batch_size":  {"type": int},
        "num_threads":  {"type": int},
        "samples_per_epoch":  {"type": int},
        "train_fract":  {"type": float},
        "test_fract": {"type": float},
        "learning_rate":  {"type": list, "subtype": (int, float)},
        "learning_rate_decay":  {"type": str},
        "loss":  {"type": str},
        "optimizer":  {"type": str},
        "num_epochs":  {"type": int},
        "dropout_rate":  {"type": float}
    }

    optional_params = {}

    class Params:
        def __init__(self, my_yaml_path):
            self.update(my_yaml_path)  # load parameters
            self.check()  # check parameters
            # handle "same" argument for model_dir
            if self.model_dir == 'same':  # this allows the model dir to be inferred from params.json file path
                self.model_dir = os.path.dirname(my_yaml_path)
            if not os.path.isdir(self.model_dir):
                raise FileNotFoundError(f"Specified model_dir {self.model_dir} does not exist!")
            self.params_path = my_yaml_path  # path to param file stored here
            self.saved_state = {}

        def save(self, my_yaml_path):
            """Saves parameters to yml file"""
            with open(my_yaml_path, 'w') as f:
                yaml.dump(self.__dict__, f, indent=4)

        def save_state(self):
            """Save current params state for restoring later"""
            attrs = [attr for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("__")]
            # dont cache saved_state
            if "saved_state" in attrs:
                attrs.remove("saved_state")
            self.saved_state = {attr: getattr(self, attr) for attr in attrs}

        def load_state(self):
            """Loads parameter state saved in self.save_state"""
            for attr in self.saved_state:
                setattr(self, attr, self.saved_state[attr])
            self.saved_state = {}

        def update(self, my_yaml_path):
            """Loads parameters from yml file"""
            with open(my_yaml_path) as f:
                params = yaml.load(f, Loader=yaml.SafeLoader)
                self.__dict__.update(params)

        def check(self):
            """Checks that all required parameters are defined in params.yml file"""
            # get all attributes
            members = [attr for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("__")]
            # check for missing required params
            if any([param not in members for param in required_params]):
                missing = [param for param in required_params if param not in members]
                raise ValueError("Missing the following parameter(s) in parameter file: {}".format(" ".join(missing)))
            # check params for type, length, subtype, etc
            for attr in members:
                # check for primary type
                t = required_params[attr]["type"]
                if not isinstance(getattr(self, attr), t):
                    raise ValueError(
                        "Required parameter {} must be {} but is {}".format(attr, t, type(getattr(self, attr))))
                # check for length
                if "length" in required_params[attr]:
                    length = len(getattr(self, attr))
                    req_length = required_params[attr]["length"]
                    if length not in req_length:
                        raise ValueError(
                            "Required parameter {} must have length {} but length is {}".format(attr, length,
                                                                                                req_length))
                # check for secondary type
                if "subtype" in required_params[attr]:
                    elems = [elem for elem in getattr(self, attr)]
                    req_subtypes = required_params[attr]["subtype"]
                    if not all([isinstance(elem, req_subtypes) for elem in elems]):
                        subtypes = [type(elem) for elem in elems]
                        raise ValueError(
                            "Every element in required parameter {} must be {} but types are {}".format(attr,
                                                                                                        req_subtypes,
                                                                                                        subtypes))

            # check for unused params in param file
            all_params = {**required_params, **optional_params}
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
# throughout this funcion train=training, val=training validation, eval=evaluation (testing) data for use in evaluate.py
def get_study_dirs(params, change_basedir=None):

    # Study dirs yml filename setup
    study_dirs_filepath = os.path.join(params.model_dir, 'study_dirs_list.yml')

    # load study dirs file if it already exists for consistent training
    if os.path.isfile(study_dirs_filepath):
        logging.info("Loading existing study directories file: {}".format(study_dirs_filepath))
        with open(study_dirs_filepath) as f:
            study_dirs = yaml.load(f, Loader=yaml.SafeLoader)

        # check that study_dirs_list format is correct
        if not all([item in study_dirs.keys() for item in ["train", "val", "eval"]]):
            raise ValueError("study_dirs_list.yml must contain entries for train, val, and eval")

        # loop through train, val, eval
        for key in ["train", "val", "eval"]:

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
        logging.info("Determining train/val/eval split based on params and study directories in data directory")
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

        # do train/val/eval split
        train_dirs, val_dirs, eval_dirs = train_val_eval_split(study_dirs, params)
        study_dirs = {"train": train_dirs, "val": val_dirs, "eval": eval_dirs}

        # save directory list to yml file so it can be loaded in future
        with open(study_dirs_filepath, 'w+', encoding='utf-8') as f:
            yaml.dump(study_dirs, f)  # save study dir list for consistency

    return study_dirs


# split list of all valid study directories into a train and test batch based on train fraction
def train_val_eval_split(study_dirs, params):
    # assumes study dirs is already shuffled and/or stratified as wanted
    # check for impossible splits
    if params.test_fract >= (1 - params.train_fract):
        raise ValueError("Parameter 'test_fract' must be less than 1-'train_fract'. Please adjust the parameter file.")
    # get indices of different fractions
    train_end_ind = int(np.floor(params.train_fract * len(study_dirs)))
    val_start_ind = train_end_ind
    eval_start_ind = int(np.ceil((1 - params.test_fract) * len(study_dirs)))

    train_dirs = study_dirs[0:train_end_ind]
    val_dirs = study_dirs[val_start_ind:eval_start_ind]
    eval_dirs = study_dirs[eval_start_ind:]

    return train_dirs, val_dirs, eval_dirs


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
