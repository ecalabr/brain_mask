"""
Allows user to choose a specific input function using a string
"""

from utilities.input_fn_util import LoadRoiMulticonAndLabels3D, LoadMulticon3D, LoadMulticonAndLabels3D
from utilities.input_fn_util import tf_patches_3d, filter_zero_patches
import tensorflow as tf
import os


# class to select input function
class InputFunctions:
    functions = {}

    def __init__(self, params):
        self.params = params
        self.function = params.input_function
        if self.function not in self.functions:
            raise ValueError(
                "Specified input function: '{}' is not one "
                "of the defined input functions: {}".format(self.function, self.functions))

    def get_dataset(self, data_dirs, mode):
        if not all([os.path.isdir(d) for d in data_dirs]):
            raise FileNotFoundError("Some of the specified data directories do not exist!")
        if not mode or not isinstance(mode, str):
            raise ValueError("Mode argument must be a string!")
        return self.functions[self.function](self.params)(data_dirs, mode)

    @classmethod
    def register_subclass(cls, name):
        def decorator(subclass):
            cls.functions[name] = subclass
            return subclass
        return decorator


@InputFunctions.register_subclass("PatchInputFn3D")
class PatchInputFn3D:
    def __init__(self, params):
        self.params = params

    def __call__(self, data_dirs, mode):

        # error if mode is not specified
        assert mode, "Mode not specified for get_dataset call"

        # set variables that are the same for all modes
        data_chan = len(self.params.data_prefix)
        label_chan = self.params.output_filters
        dfmt = self.params.data_format
        filt_zero = self.params.filter_zero
        threads = self.params.num_threads
        shuffle_size = self.params.shuffle_size
        repeat = False  # infinitely repeat dataset

        # mode switch
        # train mode - loads data and labels, shuffles, batches, patches, augments, and filters
        if mode.lower() == "train":
            # define mode specific fixed variables
            dims = self.params.train_dims
            overlap = self.params.train_patch_overlap
            shuffle = True
            batch = self.params.batch_size
            loader = LoadRoiMulticonAndLabels3D
            repeat = True  # infinitely repeat dataset

        # eval mode - loads data and labels, batches, and patches - no shuffling or augmenting, filtering
        elif mode.lower() == "eval":
            dims = self.params.train_dims
            overlap = self.params.train_patch_overlap
            self.params.augment_train_data = False  # no augmentation for this mode
            self.params.filter_zero = 0.  # no filtering for this mode
            shuffle = False
            batch = self.params.batch_size
            loader = LoadRoiMulticonAndLabels3D

        # test mode - loads data and labels - no shuffling, batching, patching, augmenting, or filtering
        elif mode.lower() == "test":
            dims = self.params.infer_dims
            overlap = self.params.infer_patch_overlap
            self.params.augment_train_data = False  # no augmentation for this mode
            self.params.filter_zero = 0.  # no filtering for this mode
            shuffle = False
            batch = 1  # no batching for this mode
            loader = LoadMulticonAndLabels3D

        # infer mode - loads data only (dummy labels) - no shuffling, batching, patching, augmenting, or filtering
        elif mode.lower() == "infer":
            dims = self.params.infer_dims
            overlap = self.params.infer_patch_overlap
            self.params.augment_train_data = False  # no augmentation for this mode
            self.params.filter_zero = 0.  # no filtering for this mode
            shuffle = False
            batch = 1  # no batching for this mode
            loader = LoadMulticon3D

        # unknown mode
        else:
            raise ValueError(f"Specified mode does not exist: {mode}")

        # create tensorflow dataset variable from data directories (all modes)
        dataset = tf.data.Dataset.from_tensor_slices(data_dirs)
        # randomly shuffle directory order (only if shuffle is true)
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(data_dirs))
        # map data directories to the data using a custom python function using specified loader
        dataset = dataset.map(lambda x: tf.numpy_function(loader(self.params).load, [x], (tf.float32, tf.float32)),
                              num_parallel_calls=threads)
        # map each dataset to a series of patches - if dims are larger than data, a padded array is returned (no patch)
        dataset = dataset.map(lambda x, y: tf_patches_3d(x, y, dims, dfmt, data_chan, label_chan, overlap),
                              num_parallel_calls=threads)
        # flatten out dataset so that each entry is a single patch and associated label
        dataset = dataset.flat_map(lambda x, y: tf.data.Dataset.from_tensor_slices((x, y)))
        # filter out zero patches
        if filt_zero > 0.:
            dataset = dataset.filter(lambda x, y: filter_zero_patches(y, dfmt, filt_zero))
        # shuffle a set number of exampes (only if shuffle is true
        if shuffle:
            dataset = dataset.shuffle(buffer_size=shuffle_size)
        # generate batch data - if batch size is 1, drop remainder has no effect
        dataset = dataset.batch(batch, drop_remainder=True)
        # prefetch with experimental autotune
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        # repeat dataset infinitely for train mode only
        if repeat:
            dataset = dataset.repeat()

        return dataset
