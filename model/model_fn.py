from model.net_builder import net_builder
from utilities.losses import loss_picker
from utilities.optimizers import optimizer_picker
import tensorflow as tf
import os
from contextlib import redirect_stdout


def model_fn(params):

    # metrics
    metrics = params.metrics
    if not isinstance(metrics, (list, tuple)):
        metrics = list(metrics)

    # handle distribution strategy if not already defined
    if not hasattr(params, 'strategy'):
        if params.dist_strat and params.dist_strat.lower() == 'mirrored':
            params.strategy = tf.distribute.MirroredStrategy()
        else:
            params.strategy = tf.distribute.get_strategy()
        # set global batch size to batch size * num replicas
        params.batch_size = params.batch_size * params.strategy.num_replicas_in_sync

    # Define model and loss using loss picker function
    with params.strategy.scope():  # use distribution strategy scope
        model = net_builder(params)
        loss = loss_picker(params)
        optimzer = optimizer_picker(params)
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
