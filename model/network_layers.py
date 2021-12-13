"""
Network layers for use by networks.py.
"""

import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Conv3D, Conv3DTranspose, Dropout, PReLU
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


def vnet_conv3d_block(layer_input, num_conv, params):

    # get relevant fixed params
    filt = params.base_filters
    dfmt = params.data_format
    dropout = params.dropout_rate
    ksize = params.kernel_size
    policy = params.policy

    # loop through num conv
    x = layer_input
    for i in range(num_conv):
        x = Conv3D(filt, kernel_size=ksize, padding='same', data_format=dfmt, dtype=policy)(x)
        if i == num_conv - 1:
            x = tf.add(x, layer_input)
        x = Activations(params)(x)
        # x = BatchNormalization(axis=-1 if dfmt == 'channels_last' else 1)(x)
        x = Dropout(rate=dropout)(x)

    return x


def vnet_conv3d_block_2(layer_input, fine_feat, num_conv, params):

    # get relevant fixed params
    filt = params.base_filters
    dfmt = params.data_format
    dropout = params.dropout_rate
    ksize = params.kernel_size
    policy = params.policy

    # concat
    x = tf.concat([layer_input, fine_feat], axis=-1)

    # check for 1 conv
    if num_conv == 1:
        x = Conv3D(filt * 2, kernel_size=ksize, padding='same', data_format=dfmt, dtype=policy)(x)
        x = tf.add(x, x)
        x = Activations(params)(x)
        # x = BatchNormalization(axis=-1 if dfmt == 'channels_last' else 1)(x)
        x = Dropout(rate=dropout)(x)
        return x

    # base conv
    x = Conv3D(filt * 2, kernel_size=ksize, padding='same', data_format=dfmt, dtype=policy)(x)
    x = Activations(params)(x)
    # x = BatchNormalization(axis=-1 if dfmt == 'channels_last' else 1)(x)
    x = Dropout(rate=dropout)(x)

    # loop through num_conv
    for i in range(num_conv):
        x = Conv3D(filt, kernel_size=ksize, padding='same', data_format=dfmt, dtype=policy)(x)
        if i == num_conv - 1:
            x = tf.add(x, layer_input)
        x = Activations(params)(x)
        # x = BatchNormalization(axis=-1 if dfmt == 'channels_last' else 1)(x)
        x = Dropout(rate=dropout)(x)

    return x


def vnet_down_conv3d(x, params):

    # get relevant fixed params
    filt = params.base_filters
    dfmt = params.data_format
    ksize = [2, 2, 2]
    strides = [2, 2, 2]
    policy = params.policy

    # downsample
    x = Conv3D(filt * 2, ksize, strides=strides, padding='same', data_format=dfmt, dtype=policy)(x)
    x = Activations(params)(x)

    return x


def vnet_up_conv3d(x, params):

    # get relevant fixed params
    filt = params.base_filters
    dfmt = params.data_format
    ksize = [2, 2, 2]
    strides = [2, 2, 2]
    policy = params.policy

    # upsample
    x = Conv3DTranspose(filt // 2, ksize, strides=strides, padding='same', data_format=dfmt, dtype=policy)(x)
    x = Activations(params)(x)

    return x
