import tensorflow as tf


# get built in locals
start_globals = list(globals().keys())


# regular adam
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
def sgd():
    return tf.keras.optimizers.SGD(
        learning_rate=0.01,
        momentum=0.0,
        nesterov=False,
        name='SGD'
    )


def optimizer_picker(params):

    # get optimizer string
    optimizer = params.optimizer

    # sanity checks
    if not isinstance(optimizer, str):
        raise ValueError("Optimizer parameter must be a string")

    # check for specified loss method and error if not found
    if optimizer.lower() in globals():
        return globals()[optimizer]()
    else:
        methods = [k for k in globals().keys() if k not in start_globals]
        raise NotImplementedError("Specified optimizer: '{}' is not an available method: {}".format(optimizer, methods))
