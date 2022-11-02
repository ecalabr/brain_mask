"""
Allows user to choose a specific network using a string in parameter file
"""

import tensorflow as tf
from tensorflow.keras.layers import Conv3D, Conv3DTranspose, Input
from tensorflow.keras.models import Model
from model.network_layers import bneck_resid3d, vnet_conv3d_block, vnet_conv3d_block_2, vnet_up_conv3d, vnet_down_conv3d


class Networks:
    models = {}

    def __init__(self, params):
        self.params = params
        self.model = params.model_name
        if self.model not in self.models:
            raise ValueError(
                "Specified model type: '{}' is not one of the available types: {}".format(self.model, self.models))

    def __call__(self):
        return self.models[self.model](self.params)()

    @classmethod
    def register_method(cls, name):
        def decorator(model):
            cls.models[name] = model
            return model

        return decorator


# 3D unet with bottleneck residual blocks, conv downsample, conv transpose upsample
# long range concat skips, batch norm, dropout
@Networks.register_method("Unet3dBneck")
class Unet3dBneck:
    def __init__(self, params):
        self.params = params

    def __call__(self):
        # define fixed params
        dfmt = self.params.data_format
        ksize = self.params.kernel_size
        chan = len(self.params.data_prefix)
        train_dims = self.params.train_dims + [chan] if dfmt == 'channels_last' else [chan] + self.params.train_dims
        batch_size = self.params.batch_size
        output_filt = self.params.output_filters
        policy = self.params.policy
        max_filters = 512

        # additional setup for network construction
        skips = []
        horz_layers = self.params.layer_layout[-1]
        unet_layout = self.params.layer_layout[:-1]

        # input layer
        inputs = Input(shape=train_dims, batch_size=batch_size, dtype='float32')

        # initial convolution layer
        x = Conv3D(self.params.base_filters, ksize, padding='same', data_format=dfmt, dtype=policy)(inputs)

        # unet encoder limb with residual bottleneck blocks
        for n, n_layers in enumerate(unet_layout):
            # horizontal layers
            for layer in range(n_layers):
                # residual blocks with activation and batch norm
                x = bneck_resid3d(x, self.params)

            # create skip connection
            skips.append(tf.identity(x))

            # downsample block
            self.params.base_filters = self.params.base_filters * 2  # double filters before downsampling
            filters = self.params.base_filters if self.params.base_filters <= max_filters else max_filters
            x = Conv3D(filters, ksize, strides=[2, 2, 2], padding='same', data_format=dfmt,
                       dtype=policy)(x)

        # unet horizontal (bottom) bottleneck blocks
        for layer in range(horz_layers):
            x = bneck_resid3d(x, self.params)

        # reverse layout and skip connections for decoder limb
        skips.reverse()
        unet_layout.reverse()

        # unet decoder limb with residual bottleneck blocks
        for n, n_layers in enumerate(unet_layout):

            # upsample block
            self.params.base_filters = int(round(self.params.base_filters / 2))  # half filters before upsampling
            filters = self.params.base_filters if self.params.base_filters <= max_filters else max_filters
            x = Conv3DTranspose(filters, ksize, strides=[2, 2, 2], padding='same', data_format=dfmt,
                                dtype=policy)(x)

            # fuse skip connections with concatenation of features
            x = tf.concat([x, skips[n]], axis=-1 if dfmt == 'channels_last' else 1)

            # horizontal blocks
            for layer in range(n_layers):
                x = bneck_resid3d(x, self.params)

        # output layer - always force float32 on final layer reguardless of mixed precision
        if self.params.final_layer == "conv":
            x = Conv3D(filters=output_filt, kernel_size=[1, 1, 1], padding='same', data_format=dfmt, dtype='float32')(x)
        elif self.params.final_layer == "sigmoid":
            x = Conv3D(filters=output_filt, kernel_size=[1, 1, 1], padding='same', data_format=dfmt, dtype='float32')(x)
            x = tf.nn.sigmoid(x)
        elif self.params.final_layer == "softmax":
            x = Conv3D(filters=output_filt, kernel_size=[1, 1, 1], padding='same', data_format=dfmt, dtype='float32')(x)
            x = tf.nn.softmax(x, axis=-1 if dfmt == 'channels_last' else 1)
        else:
            assert ValueError("Specified final layer is not implemented: {}".format(self.params.final_layer))

        return Model(inputs=inputs, outputs=x)


# Vnet - https://github.com/amorimdiogo/VNet/blob/master/vnet.py
@Networks.register_method("Vnet")
class Vnet:
    def __init__(self, params):
        self.params = params

    def __call__(self):
        # define fixed params
        dfmt = self.params.data_format
        chan = len(self.params.data_prefix)
        train_dims = self.params.train_dims + [chan] if dfmt == 'channels_last' else [chan] + self.params.train_dims
        batch_size = self.params.batch_size
        output_filt = self.params.output_filters
        horz_layers = self.params.layer_layout[-1]
        vnet_layout = self.params.layer_layout[:-1]

        # input layer
        inputs = Input(shape=train_dims, batch_size=batch_size, dtype='float32')
        x = inputs

        # features list
        features = []

        # loop through levels encoder
        for n, n_layers in enumerate(vnet_layout):
            # conv blocks
            x = vnet_conv3d_block(x, n_layers, self.params)
            features.append(x)
            # downsample
            x = vnet_down_conv3d(x, self.params)

        # bottom level
        x = vnet_conv3d_block(x, horz_layers, self.params)

        # loop through levels decoder
        vnet_layout.reverse()
        features.reverse()
        for n, n_layers in enumerate(vnet_layout):
            # upsample
            x = vnet_up_conv3d(x, self.params)
            # conv blocks
            x = vnet_conv3d_block_2(x, features[n], n_layers, self.params)

        # output layer - always force float32 on final layer reguardless of mixed precision
        if self.params.final_layer == "conv":
            x = Conv3D(filters=output_filt, kernel_size=[1, 1, 1], padding='same', data_format=dfmt, dtype='float32')(x)
        elif self.params.final_layer == "sigmoid":
            x = Conv3D(filters=output_filt, kernel_size=[1, 1, 1], padding='same', data_format=dfmt, dtype='float32')(x)
            x = tf.nn.sigmoid(x)
        elif self.params.final_layer == "softmax":
            x = Conv3D(filters=output_filt, kernel_size=[1, 1, 1], padding='same', data_format=dfmt, dtype='float32')(x)
            x = tf.nn.softmax(x, axis=-1 if dfmt == 'channels_last' else 1)
        else:
            assert ValueError("Specified final layer is not implemented: {}".format(self.params.final_layer))

        return Model(inputs=inputs, outputs=x)
