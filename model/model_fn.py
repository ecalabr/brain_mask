"""
Creates a tensorflow model based on a set of parameters
"""

from model.networks import Networks
from utilities.losses import Losses
from utilities.optimizers import Optimizers
from utilities.metrics import Metrics
import tensorflow as tf
import os
from contextlib import redirect_stdout
from tensorflow.keras import mixed_precision


# model function
def model_fn(params):

    # get metrics
    metrics = params.metrics
    if not isinstance(metrics, (list, tuple)):
        metrics = list(metrics)
    metrics = Metrics(metrics, params).get_metrics()

    # handle distribution strategy
    if not hasattr(params, 'strategy'):
        if params.dist_strat and params.dist_strat.lower() == 'mirrored':
            params.strategy = tf.distribute.MirroredStrategy()
        else:
            params.strategy = tf.distribute.get_strategy()
        # set global batch size to batch size * num replicas
        params.batch_size = params.batch_size * params.strategy.num_replicas_in_sync

    # handle mixed precision
    if params.mixed_precision:  # enable mixed precision and warn user
        print("WARNING: using tensorflow mixed precision... This could lead to numeric instability in some cases.")
        params.policy = mixed_precision.Policy('mixed_float16')
        # warn if batch size and/or nfilters is not a multpile of 8
        if not params.base_filters % 8 == 0:
            print("WARNING: parameter base_filters is not a multiple of 8, which will not use tensor cores.")
        if not params.batch_size % 8 == 0:
            print("WARNING: parameter batch_size is not a multiple of 8, which will not use tensor cores.")
    else:  # if not using mixed precision, then assume float32
        params.policy = mixed_precision.Policy('float32')
    # set default policy, subsequent per layer dtype can be specified
    mixed_precision.set_global_policy(params.policy)  # default policy for layers

    # Define model and loss using loss picker function
    with params.strategy.scope():  # use distribution strategy scope
        model = Networks(params)()
        loss = Losses(params)()
        optimzer = Optimizers(params)()
        model.compile(optimizer=optimzer, loss=loss, metrics=metrics)

    # save text representation of graph
    model_info_dir = os.path.join(params.model_dir, 'model')
    if not os.path.isdir(model_info_dir):
        os.mkdir(model_info_dir)
    model_sum = os.path.join(model_info_dir, 'model_summary.txt')
    if not os.path.isfile(model_sum):
        with open(model_sum, 'w+') as f:
            with redirect_stdout(f):
                model.summary()

    # save graphical representation of graph
    model_im = os.path.join(model_info_dir, 'model_graphic.png')
    if not os.path.isfile(model_im):
        tf.keras.utils.plot_model(
            model, to_file=model_im, show_shapes=False, show_layer_names=True,
            rankdir='TB', expand_nested=False, dpi=96)

    return model
