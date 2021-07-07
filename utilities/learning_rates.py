from tensorflow.keras.optimizers.schedules import ExponentialDecay


# get built in locals
start_globals = list(globals().keys())


# constant (no decay)
def constant(learn_rate):
    if isinstance(learn_rate, (list, tuple)):
        learn_rate = learn_rate[0]
    # simple function that always returns learn_rate regardless of epoch
    def constant_lr(_epoch):
        return learn_rate
    return constant_lr


# simple stepwise learning rate decay with lead-in
def simple_step(learn_rate):
    if not isinstance(learn_rate, (list, tuple)):
        raise ValueError("Simple step decay requres three values: starting learning rate, lead-in epochs, and factor")
    init_learn_rate = learn_rate[0]
    epochs = learn_rate[1]
    factor = learn_rate[2]
    # simple function that always returns learn_rate regardless of epoch
    def step_lr(_epoch):
        if _epoch < epochs:
            lr = init_learn_rate
        else:
            lr = init_learn_rate * (factor ** (_epoch - epochs))
        return lr
    return step_lr


# exponential decay
def exponential(learn_rate):
    if not isinstance(learn_rate, (list, tuple)):
        raise ValueError("Exponential decay requres three values: starting learning rate, steps, and decay factor")
    start_lr = learn_rate[0]
    steps = learn_rate[1]
    decay = learn_rate[2]
    learning_rate_sced = ExponentialDecay(start_lr, steps, decay, staircase=True)
    return learning_rate_sced


# manual schedule
def manual(learn_rate):
    def manual_lr(_epoch):
        lr = learn_rate[_epoch - 1]
        return lr
    return manual_lr


def learning_rate_picker(init_learn_rate, decay_method):

    # sanity checks
    if not isinstance(init_learn_rate, (float, list, tuple)):
        raise ValueError("Learning rate must be a float or list/tuple")
    if not isinstance(decay_method, str):
        raise ValueError("Learning rate decay parameter must be a string")

    # check for specified loss method and error if not found
    if decay_method in globals():
        learning_rate_sched = globals()[decay_method](init_learn_rate)
    else:
        # get list of available normalization modes
        methods = [k for k in globals().keys() if k not in start_globals]
        raise NotImplementedError(
            "Specified learning rate method: '{}' is not an available method: {}".format(decay_method, methods))

    return learning_rate_sched
