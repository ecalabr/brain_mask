"""
Allows user to choose a specific learning rate schedule using a string in parameter file
"""

from tensorflow.keras.optimizers.schedules import ExponentialDecay


class LearningRates:
    methods = {}

    def __init__(self, params):
        self.method = params.learning_rate_decay
        self.lr = params.learning_rate
        if self.method not in self.methods:
            raise ValueError(
                "Specified learning rate method: '{}' is not an available method: {}".format(self.method, self.methods))

    def __call__(self):
        return self.methods[self.method](self.lr)

    @classmethod
    def register_method(cls, name):
        def decorator(method):
            cls.methods[name] = method
            return method
        return decorator


# constant (no decay)
@LearningRates.register_method("constant")
def constant(learn_rate):
    if isinstance(learn_rate, (list, tuple)):
        learn_rate = learn_rate[0]

    # simple function that always returns learn_rate regardless of epoch
    def constant_lr(_epoch):
        return learn_rate
    return constant_lr


# simple stepwise learning rate decay with lead-in
@LearningRates.register_method("simple_step")
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
@LearningRates.register_method("exponential")
def exponential(learn_rate):
    if not isinstance(learn_rate, (list, tuple)):
        raise ValueError("Exponential decay requres three values: starting learning rate, steps, and decay factor")
    start_lr = learn_rate[0]
    steps = learn_rate[1]
    decay = learn_rate[2]
    learning_rate_sced = ExponentialDecay(start_lr, steps, decay, staircase=True)
    return learning_rate_sced


# manual schedule
@LearningRates.register_method("manual")
def manual(learn_rate):
    def manual_lr(_epoch):
        lr = learn_rate[_epoch - 1]
        return lr
    return manual_lr
