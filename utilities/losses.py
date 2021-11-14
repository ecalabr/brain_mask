"""
Allows user to choose a specific loss function using a string in parameter file
"""

import tensorflow as tf


class Losses:
    methods = {}

    def __init__(self, params):
        self.method = params.loss
        if self.method not in self.methods:
            raise ValueError(
                "Specified loss method: '{}' is not an available method: {}".format(self.method, self.methods))

    def __call__(self):
        return self.methods[self.method]

    @classmethod
    def register_method(cls, name):
        def decorator(method):
            cls.methods[name] = method
            return method
        return decorator


# generalized DICE loss for 2D and 3D networks
@Losses.register_method("dice_loss")
def dice_loss(y_true, y_pred):
    numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=(1, 2, 3))
    denominator = tf.reduce_sum(y_true + y_pred, axis=(1, 2, 3))
    return 1 - numerator / denominator


# combo dice and binary cross entropy for use with mirrored strategy
@Losses.register_method("combo_loss3d_mirrored")
def combo_loss3d_mirrored(y_true, y_pred):
    def dice_l(y_t, y_p):
        numerator = 2 * tf.reduce_sum(y_t * y_p, axis=(1, 2, 3, 4))
        denominator = tf.reduce_sum(y_t + y_p, axis=(1, 2, 3, 4))
        return tf.reshape(1 - numerator / denominator, (-1, 1, 1, 1))

    return tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.SUM)(y_true, y_pred) + dice_l(y_true,
                                                                                                                y_pred)


# combo dice and binary cross entropy
@Losses.register_method("combo_loss3d")
def combo_loss3d(y_true, y_pred):
    def dice_l(y_t, y_p):
        numerator = 2 * tf.reduce_sum(y_t * y_p, axis=(1, 2, 3, 4))
        denominator = tf.reduce_sum(y_t + y_p, axis=(1, 2, 3, 4))
        return tf.reshape(1 - numerator / denominator, (-1, 1, 1, 1))

    return tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.AUTO)(y_true, y_pred) + dice_l(y_true,
                                                                                                                 y_pred)
