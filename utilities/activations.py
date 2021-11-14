"""
Allows user to choose a specific activation using a string in parameter file
"""

import tensorflow as tf


class Activations:
    methods = {}

    def __init__(self, params):
        self.act = params.activation
        if self.act not in self.methods:
            raise ValueError(
                "Specified activation method: '{}' is not an available method: {}".format(self.act, self.methods))

    def __call__(self, x):
        return self.methods[self.act]()(x)

    @classmethod
    def register_method(cls, name):
        def decorator(method):
            cls.methods[name] = method
            return method
        return decorator


# methods:
# regular relu
@Activations.register_method("relu")
def relu():
    return tf.keras.layers.ReLU(
        max_value=None,
        negative_slope=0,
        threshold=0
    )


# leaky relu
@Activations.register_method("leaky_relu")
def leaky_relu():
    return tf.keras.layers.LeakyReLU(
        alpha=0.3
    )


# prelu
@Activations.register_method("prelu")
def prelu():
    return tf.keras.layers.PReLU(
        alpha_initializer='zeros',
        alpha_regularizer=None,
        alpha_constraint=None,
        shared_axes=None
    )
