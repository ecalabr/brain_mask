"""
Allows user to choose a specific optimizer using a string in parameter file
"""

import tensorflow as tf


class Optimizers:
    methods = {}

    def __init__(self, params):
        self.optimizer = params.optimizer
        if self.optimizer not in self.methods:
            raise ValueError(
                "Specified optimizer: '{}' is not an available method: {}".format(self.optimizer, self.methods))

    def __call__(self):
        return self.methods[self.optimizer]()

    @classmethod
    def register_method(cls, name):
        def decorator(method):
            cls.methods[name] = method
            return method
        return decorator


# regular adam
@Optimizers.register_method("adam")
def adam():
    return tf.keras.optimizers.Adam(
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
        amsgrad=False,
        name='Adam'
    )


# AMSgrad variant of adam
@Optimizers.register_method("amsgrad")
def amsgrad():
    return tf.keras.optimizers.Adam(
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
        amsgrad=True,
        name='Adam'
    )


# stochastic gradient descent
@Optimizers.register_method("sgd")
def sgd():
    return tf.keras.optimizers.SGD(
        learning_rate=0.01,
        momentum=0.0,
        nesterov=False,
        name='SGD'
    )
