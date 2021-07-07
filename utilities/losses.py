import tensorflow as tf


# get built in locals
start_globals = list(globals().keys())


# generalized DICE loss for 2D and 3D networks
def dice_loss(y_true, y_pred):
    numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=(1, 2, 3))
    denominator = tf.reduce_sum(y_true + y_pred, axis=(1, 2, 3))
    return 1 - numerator / denominator


# combo dice and binary cross entropy
def combo_loss3d(y_true, y_pred):
    def dice_l(y_t, y_p):
        numerator = 2 * tf.reduce_sum(y_t * y_p, axis=(1, 2, 3, 4))
        denominator = tf.reduce_sum(y_t + y_p, axis=(1, 2, 3, ))
        return tf.reshape(1 - numerator / denominator, (-1, 1, 1, 1))

    return tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.SUM)(y_true, y_pred) + dice_l(y_true,
                                                                                                                y_pred)


def loss_picker(params):

    # sanity checks
    if not isinstance(params.loss, str):
        raise ValueError("Loss method parameter must be a string")

    # check for specified loss method and error if not found
    if params.loss in globals():
        loss_fn = globals()[params.loss]
    else:
        methods = [k for k in globals().keys() if k not in start_globals]
        raise NotImplementedError(
            "Specified loss method: '{}' is not one of the available methods: {}".format(params.loss, methods))

    return loss_fn
