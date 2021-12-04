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


# generalized DICE loss
@Losses.register_method("dice")
def dice(y_t, y_p):
    numerator = 2 * tf.reduce_sum(y_t * y_p, axis=(1, 2, 3, 4))
    denominator = tf.reduce_sum(y_t + y_p, axis=(1, 2, 3, 4))
    return tf.reshape(1 - (numerator / denominator), (-1, 1, 1, 1))


# sum of dice and binary cross entropy for use with mirrored strategy
@Losses.register_method("combo_loss3d_mirrored")
def combo_loss3d_mirrored(y_t, y_p):
    return tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.SUM)(y_t, y_p) + dice(y_t, y_p)


# sum of dice and binary cross entropy
@Losses.register_method("combo_loss3d")
def combo_loss3d(y_t, y_p):
    return tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.AUTO)(y_t, y_p) + dice(y_t, y_p)
