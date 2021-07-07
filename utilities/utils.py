""" General utility functions """

import json
import logging
import matplotlib.pyplot as plt
import numpy as np


class Params:
    """
    Class that loads hyperparameters from a json file.
    Example:
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    """

    # declare attributes as None initially. All attributes defined here must be fed values from params.json.
    data_dir = None
    model_dir = None
    overwrite = None
    random_state = None

    data_prefix = None
    label_prefix = None
    mask_prefix = None
    mask_dilate = None  # must have same number of dims as mask
    filter_zero = None  # set the threshold for filtering out patches where labels is mostly zero

    data_plane = None
    train_dims = None
    train_patch_overlap = None
    infer_dims = None
    infer_patch_overlap = None
    augment_train_data = None
    label_interp = None
    mask_weights = None
    metrics = None

    norm_data = None
    norm_labels = None
    norm_mode = None

    model_name = None
    base_filters = None
    output_filters = None
    layer_layout = None
    final_layer = None
    kernel_size = None
    data_format = None
    activation = None
    mixed_precision = None
    dist_strat = None

    shuffle_size = None
    batch_size = None
    num_threads = None
    samples_per_epoch = None
    train_fract = None
    learning_rate = None
    learning_rate_decay = None
    loss = None
    optimizer = None
    num_epochs = None
    dropout_rate = None

    def __init__(self, json_path):
        self.update(json_path)
        self.check()

    def save(self, json_path):
        """Saves parameters to json file"""
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def check(self):
        """Checks that all required parameters are defined in params.json file"""
        member_val = [getattr(self, attr) for attr in dir(self) if
                      not callable(getattr(self, attr)) and not attr.startswith("__")]
        members = [attr for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("__")]
        if any([member is None for member in member_val]):
            raise ValueError(
                "Missing the following parameter(s) in params.json: "
                + " ".join([attr for attr in members if getattr(self, attr) is None]))

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']`"""
        return self.__dict__


def set_logger(log_path):
    """
    Sets the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def save_dict_to_json(d, json_path):
    """
    Saves dict of floats in json file
    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


def display_tf_dataset(dataset_data, data_format, data_dims, weighted=False):
    """
    Displays tensorflow dataset output images and labels/regression images.
    :param dataset_data: (tf.tensor) output from tf dataset function containing images and labels/regression image
    :param data_format: (str) the desired tensorflow data format. Must be either 'channels_last' or 'channels_first'
    :param data_dims: (list or tuple of ints) the data dimensions that come out of the input function
    :param weighted: (bool) whether or not the labels slice includes weights as the final channel dimension
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
        if weighted:
            nplots += 1

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

        # handle weights
        weights = None
        if weighted:
            weights = label_data[..., [-1]]  # last channel is weights
            label_data = label_data[..., [0]]  # use first channel for labes

        # add to fig
        if weighted:
            # handle labels first
            ax = fig.add_subplot(1, nplots, nplots-1)
            label_img = np.squeeze(label_data)
            inds = [label_img.shape[0] * label_img.shape[2], label_img.shape[1]]
            label_img = np.reshape(np.transpose(label_img), inds)
            ax.imshow(label_img, cmap='gray')
            ax.set_title('Labels')
            # finally handle weights
            ax = fig.add_subplot(1, nplots, nplots)
            weight_img = np.reshape(np.transpose(weights), inds)
            ax.imshow(weight_img, cmap='gray')
            ax.set_title('Weights')
        else:
            # handle labels only
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
