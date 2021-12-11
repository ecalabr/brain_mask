"""
Allows a user to choose a specific evaluation metric using a string in parameter file
"""

import tensorflow as tf


class Metrics:
    methods = {}

    def __init__(self, metrics, params):
        self.metrics = metrics
        self.params = params
        for metric in self.metrics:
            if metric not in self.methods:
                raise ValueError("Specified metric: '{}' is not an available method: {}".format(metric, self.methods))

    def get_metrics(self):
        return [self.methods[metric](self.params) for metric in self.metrics]

    @classmethod
    def register_method(cls, name):
        def decorator(m):
            cls.methods[name] = m
            return m
        return decorator


# handle bce metric
@Metrics.register_method("bce")
def bce(_):
    return tf.keras.metrics.BinaryCrossentropy(from_logits=False)


# handle dice metric
@Metrics.register_method("dice")
def dice_metric(_):
    def dice(y_t, y_p):
        numerator = 2 * tf.reduce_sum(y_t * y_p, axis=(1, 2, 3, 4))
        denominator = tf.reduce_sum(y_t + y_p, axis=(1, 2, 3, 4))
        return tf.reshape((numerator / denominator), (-1, 1, 1, 1))
    return dice


@Metrics.register_method("binary_dice")
def binary_dice_metric(_):
    def binary_dice(y_t, y_p):
        y_p = tf.cast(y_p > 0.5, tf.float32)
        return dice_metric(_)(y_t, y_p)
    return binary_dice
