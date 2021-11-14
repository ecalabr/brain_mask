"""
Allows user to choose a specific input function using a string
"""

from utilities.input_fn_util import *


# patch input function for 2d or 3d
class InputFunctions:
    functions = {}

    def __init__(self, params):
        self.params = params
        self.function = params.input_function
        if self.function not in self.functions:
            raise ValueError(
                "Specified input function: '{}' is not one "
                "of the defined input functions: {}".format(self.function, self.functions))

    def __call__(self):
        return self.functions[self.function](self.params)

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

    def __call__(self, mode, data_dirs):
        # define fixed variables
        params = self.params
        if not isinstance(data_dirs, list):
            data_dirs = [data_dirs]
        data_chan = len(params.data_prefix)

        # train mode - uses patches, patch filtering, batching, data augmentation, and shuffling - works on train_dirs
        if mode == 'train':
            # create tensorflow dataset variable from data directories
            dataset = tf.data.Dataset.from_tensor_slices(data_dirs)
            # randomly shuffle directory order
            dataset = dataset.shuffle(buffer_size=len(data_dirs))
            # map data directories to the data using a custom python function
            dataset = dataset.map(
                lambda x: tf.numpy_function(LoadRoiMulticonAndLabels3D(params).load, [x], (tf.float32, tf.float32)),
                num_parallel_calls=params.num_threads)
            # map each dataset to a series of patches
            dataset = dataset.map(
                lambda x, y: tf_patches_3d(x, y, params.train_dims, params.data_format, data_chan,
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

        # eval mode - uses patches & batches but no patch filtering, data augmentation, or shuffling, works on eval_dirs
        elif mode == 'eval':
            # turn off data augmentation for eval mode
            params.augment_train_data = False
            # create tensorflow dataset variable from data directories
            dataset = tf.data.Dataset.from_tensor_slices(data_dirs)
            # map data directories to the data using a custom python function
            dataset = dataset.map(
                lambda x: tf.numpy_function(LoadRoiMulticonAndLabels3D(params).load, [x], (tf.float32, tf.float32)),
                num_parallel_calls=params.num_threads)  # tf.data.experimental.AUTOTUNE)
            # map each dataset to a series of patches
            dataset = dataset.map(
                lambda x, y: tf_patches_3d(x, y, params.train_dims, params.data_format, data_chan,
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
            # define dims of inference
            infer_dims = list(params.infer_dims)
            # create tensorflow dataset variable from data directories
            dataset = tf.data.Dataset.from_tensor_slices(data_dirs)
            # map data directories to the data using a custom python function
            dataset = dataset.map(
                lambda x: tf.numpy_function(LoadMulticonPreserveSize3D(params).load, [x], tf.float32),
                num_parallel_calls=tf.data.experimental.AUTOTUNE)
            # map each dataset to a series of patches based on infer inputs
            dataset = dataset.map(
                lambda x: tf_patches_3d_infer(x, infer_dims, data_chan, params.data_format, params.infer_patch_overlap),
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
