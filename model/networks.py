"""
Allows user to choose a specific network using a string in parameter file
"""

import tensorflow as tf
from tensorflow.keras.layers import Conv3D, Conv3DTranspose, Input, PReLU
from tensorflow.keras.models import Model
from model.network_layers import bneck_resid3d, downward_layer, Deconvolution3D, upward_layer


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
            x = Conv3D(self.params.base_filters, ksize, strides=[2, 2, 2], padding='same', data_format=dfmt,
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
            x = Conv3DTranspose(self.params.base_filters, ksize, strides=[2, 2, 2], padding='same', data_format=dfmt,
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
        # ksize = self.params.kernel_size
        chan = len(self.params.data_prefix)
        train_dims = self.params.train_dims + [chan] if dfmt == 'channels_last' else [chan] + self.params.train_dims
        batch_size = self.params.batch_size
        output_filt = self.params.output_filters
        policy = self.params.policy
        init = 'he_normal'

        # Layer 1
        inputs = Input(shape=train_dims, batch_size=batch_size, dtype='float32')
        conv1 = Conv3D(16, kernel_size=5, strides=1, padding='same', kernel_initializer=init, data_format=dfmt,
                       dtype=policy)(inputs)
        conv1 = PReLU()(conv1)
        repeat1 = tf.concat(16 * [inputs], axis=-1)
        add1 = tf.add(conv1, repeat1)
        down1 = Conv3D(32, 2, strides=2, padding='same', kernel_initializer=init, data_format=dfmt, dtype=policy)(add1)
        down1 = PReLU()(down1)

        # Layer 2,3,4
        down2, add2 = downward_layer(down1, 2, 64)
        down3, add3 = downward_layer(down2, 3, 128)
        down4, add4 = downward_layer(down3, 3, 256)

        # Layer 5
        # !Mudar kernel_size=(5, 5, 5) quando imagem > 64!
        conv_5_1 = Conv3D(256, kernel_size=(5, 5, 5), padding='same', kernel_initializer=init, data_format=dfmt,
                          dtype=policy)(down4)
        conv_5_1 = PReLU()(conv_5_1)
        conv_5_2 = Conv3D(256, kernel_size=(5, 5, 5), padding='same', kernel_initializer=init, data_format=dfmt,
                          dtype=policy)(conv_5_1)
        conv_5_2 = PReLU()(conv_5_2)
        conv_5_3 = Conv3D(256, kernel_size=(5, 5, 5), padding='same', kernel_initializer=init, data_format=dfmt,
                          dtype=policy)(conv_5_2)
        conv_5_3 = PReLU()(conv_5_3)
        add5 = tf.add(conv_5_3, down4)
        aux_shape = add5.shape
        # noinspection PyCallingNonCallable
        upsample_5 = Deconvolution3D(128, (2, 2, 2), (batch_size, aux_shape[1] * 2, aux_shape[2] * 2,
                                                      aux_shape[3] * 2, 128), subsample=(2, 2, 2))(add5)
        upsample_5 = PReLU()(upsample_5)

        # Layer 6,7,8
        upsample_6 = upward_layer(upsample_5, add4, 3, batch_size, 64)
        upsample_7 = upward_layer(upsample_6, add3, 3, batch_size, 32)
        upsample_8 = upward_layer(upsample_7, add2, 2, batch_size, 16)

        # Layer 9
        merged_9 = tf.concat([upsample_8, add1], axis=4)
        conv_9_1 = Conv3D(32, kernel_size=(5, 5, 5), padding='same', kernel_initializer=init, data_format=dfmt,
                          dtype=policy)(merged_9)
        conv_9_1 = PReLU()(conv_9_1)
        add_9 = tf.add(conv_9_1, merged_9)
        conv_9_2 = Conv3D(1, kernel_size=(1, 1, 1), padding='same', kernel_initializer=init, data_format=dfmt,
                          dtype=policy)(add_9)
        x = PReLU()(conv_9_2)

        # output layer - always force float32 on final layer reguardless of mixed precision
        if self.params.final_layer == "conv":
            x = Conv3D(filters=output_filt, kernel_size=[1, 1, 1], padding='same', data_format=dfmt, dtype='float32',
                       kernel_initializer=init)(x)
        elif self.params.final_layer == "sigmoid":
            x = Conv3D(filters=output_filt, kernel_size=[1, 1, 1], padding='same', data_format=dfmt, dtype='float32',
                       kernel_initializer=init)(x)
            x = tf.nn.sigmoid(x)
        elif self.params.final_layer == "softmax":
            x = Conv3D(filters=output_filt, kernel_size=[1, 1, 1], padding='same', data_format=dfmt, dtype='float32',
                       kernel_initializer=init)(x)
            x = tf.nn.softmax(x, axis=-1 if dfmt == 'channels_last' else 1)
        else:
            assert ValueError("Specified final layer is not implemented: {}".format(self.params.final_layer))

        return Model(inputs=inputs, outputs=x)
