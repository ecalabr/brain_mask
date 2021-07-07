import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Conv3D, Dropout
from utilities.activations import activation_layer


def bneck_resid3d(x, filt, ksize, dfmt, dropout, act, policy='float32'):

    # create shortcut
    shortcut = x

    # perform first 1x1x1 conv for bottleneck block using 1/4 filters
    x = BatchNormalization(axis=-1 if dfmt == 'channels_last' else 1)(x)
    x = activation_layer(act)(x)
    x = Conv3D(int(round(filt/4)), kernel_size=[1, 1, 1], padding='same', data_format=dfmt, dtype=policy)(x)

    # perform 3x3x3 conv for bottleneck block with bn and activation using 1/4 filters (optionally strided)
    x = BatchNormalization(axis=-1 if dfmt == 'channels_last' else 1)(x)
    x = activation_layer(act)(x)
    x = Conv3D(int(round(filt/4)), ksize, padding='same', data_format=dfmt, dtype=policy)(x)

    # perform second 1x1x1 conv with full filters (no strides)
    x = BatchNormalization(axis=-1 if dfmt == 'channels_last' else 1)(x)
    x = activation_layer(act)(x)
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
