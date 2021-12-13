"""
Network layers for use by networks.py.
"""

import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Conv3D, Dropout, PReLU
import tensorflow.keras.initializers as initializers
from utilities.activations import Activations
import functools


def bneck_resid3d(x, params):

    # get relevant fixed params
    filt = params.base_filters
    dfmt = params.data_format
    dropout = params.dropout_rate
    ksize = params.kernel_size
    policy = params.policy

    # create shortcut
    shortcut = x

    # perform first 1x1x1 conv for bottleneck block using 1/4 filters
    x = BatchNormalization(axis=-1 if dfmt == 'channels_last' else 1)(x)
    x = Activations(params)(x)
    x = Conv3D(int(round(filt/4)), kernel_size=[1, 1, 1], padding='same', data_format=dfmt, dtype=policy)(x)

    # perform 3x3x3 conv for bottleneck block with bn and activation using 1/4 filters (optionally strided)
    x = BatchNormalization(axis=-1 if dfmt == 'channels_last' else 1)(x)
    x = Activations(params)(x)
    x = Conv3D(int(round(filt/4)), ksize, padding='same', data_format=dfmt, dtype=policy)(x)

    # perform second 1x1x1 conv with full filters (no strides)
    x = BatchNormalization(axis=-1 if dfmt == 'channels_last' else 1)(x)
    x = Activations(params)(x)
    x = Conv3D(filt, kernel_size=[1, 1, 1], padding='same', data_format=dfmt, dtype=policy)(x)

    # optional dropout layer
    if dropout > 0.:
        x = Dropout(rate=dropout)(x)

    # fuse shortcut with tensor output, transforming filter number as needed
    if x.shape[-1] == shortcut.shape[-1]:
        x = tf.add(x, shortcut)
    else:
        shortcut = Conv3D(filt, kernel_size=[1, 1, 1], padding='same', data_format=dfmt, dtype=policy)(shortcut)
        x = tf.add(x, shortcut)

    return x


class Deconvolution3D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, output_shape, subsample):
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = (1,) + subsample + (1,)
        self.output_shape_ = output_shape
        super(Deconvolution3D, self).__init__()

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape):
        assert len(input_shape) == 5
        self.input_shape_ = input_shape
        w_shape = self.kernel_size + (self.filters, input_shape[4],)
        self.W = self.add_weight(shape=w_shape,
                                 initializer=functools.partial(initializers.glorot_uniform()),
                                 name='{}_W'.format(self.name))
        self.b = self.add_weight(shape=(1, 1, 1, self.filters,), initializer='zero', name='{}_b'.format(self.name))
        self.built = True

    def compute_output_shape(self, _):
        return (None,) + self.output_shape_[1:]

    def call(self, x, _):
        return tf.nn.conv3d_transpose(x, self.W, output_shape=self.output_shape_,
                                      strides=self.strides, padding='SAME', name=self.name) + self.b

    def get_config(self):
        base_config = super(Deconvolution3D, self).get_config().copy()
        base_config['output_shape'] = self.output_shape_
        return base_config


def downward_layer(input_layer, n_convolutions, n_output_channels):
    inl = input_layer

    for _ in range(n_convolutions):
        inl = PReLU()(
            Conv3D(filters=(n_output_channels // 2), kernel_size=5,
                   padding='same', kernel_initializer='he_normal')(inl)
        )
    add_l = tf.add(inl, input_layer)
    downsample = Conv3D(filters=n_output_channels, kernel_size=2, strides=2,
                        padding='same', kernel_initializer='he_normal')(add_l)
    downsample = PReLU()(downsample)
    return downsample, add_l


def upward_layer(input0, input1, n_convolutions, batch_size, n_output_channels):
    merged = tf.concat([input0, input1], axis=4)
    inl = merged
    for _ in range(n_convolutions):
        inl = PReLU()(
            Conv3D((n_output_channels * 4), kernel_size=5,
                   padding='same', kernel_initializer='he_normal')(inl)
        )
    add_l = tf.add(inl, merged)
    shape = add_l.get_shape().as_list()
    new_shape = (batch_size, shape[1] * 2, shape[2] * 2, shape[3] * 2, n_output_channels)
    # noinspection PyCallingNonCallable
    upsample = Deconvolution3D(n_output_channels, (2, 2, 2), new_shape, subsample=(2, 2, 2))(add_l)
    return PReLU()(upsample)
