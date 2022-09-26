"""Train the model"""

import argparse
from glob import glob
import logging
import os
# set tensorflow logging level before importing things that contain tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0 = INFO, 1 = WARN, 2 = ERROR, 3 = FATAL
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint, TensorBoard, CSVLogger
from utilities.utils import load_param_file, set_logger, get_study_dirs
from utilities.input_functions import InputFunctions
from utilities.learning_rates import LearningRates
from model.model_fn import model_fn


# define functions
def train(params):
    # logging
    train_logger = logging.getLogger()
    # Make sure data directory exists
    if not os.path.isdir(params.data_dir):
        raise ValueError(f"Specified data directory does not exist: {params.data_dir}")
    train_logger.info(f"Using data directory {params.data_dir}")

    # determine distribution strategy for multi GPU training
    if params.dist_strat.lower() == 'mirrored':
        train_logger.info("Using Mirrored distribution strategy")
        params.strategy = tf.distribute.MirroredStrategy()
        # adjust batch size and learning rate to compensate for mirrored replicas
        # batch size is multiplied by num replicas
        params.batch_size = params.batch_size * params.strategy.num_replicas_in_sync
        train_logger.info(
            f"Batch size adjusted to {params.batch_size} for {params.strategy.num_replicas_in_sync} replicas")
        # initial learning rate is multiplied by squre root of replicas
        # params.learning_rate[0] = params.learning_rate[0] * np.sqrt(params.strategy.num_replicas_in_sync)
        # train_logger.info(
        #     "Initial learning rate adjusted by a factor of {} (root {} for {} replicas)".format(
        #         np.sqrt(params.strategy.num_replicas_in_sync), params.strategy.num_replicas_in_sync,
        #         params.strategy.num_replicas_in_sync))
    else:
        params.strategy = tf.distribute.get_strategy()

    # determine checkpoint directories and determine current epoch
    checkpoint_path = os.path.join(params.model_dir, 'checkpoints')
    latest_ckpt = None
    if not os.path.isdir(checkpoint_path):
        os.mkdir(checkpoint_path)
    checkpoints = glob(checkpoint_path + '/*.hdf5')
    if checkpoints and not params.overwrite:
        latest_ckpt = max(checkpoints, key=os.path.getctime)
        completed_epochs = int(os.path.splitext(os.path.basename(latest_ckpt).split('epoch_')[1])[0].split('_')[0])
        train_logger.info(f"Checkpoint exists for epoch {completed_epochs}")
    else:
        completed_epochs = 0

    # get model inputs - for both training and validation
    study_dirs = get_study_dirs(params)  # returns a dict of "train", "val", and "eval"
    # first check if there is an existing dataset directory containing a pre-generated dataset from generate_dataset.py
    dataset_dir = os.path.join(params.model_dir, 'dataset')
    train_data_dir = os.path.join(dataset_dir, 'train')
    val_data_dir = os.path.join(dataset_dir, 'val')
    # if both train and val datasets exist, then load
    if all([os.path.isdir(item) for item in [train_data_dir, val_data_dir]]):
        train_logger.info(f"Loading existing training and and validation datasets from {dataset_dir}")
        train_inputs = tf.data.experimental.load(train_data_dir)
        val_inputs = tf.data.experimental.load(val_data_dir)
        # determine train dataset cardinality and use this instead of params.samples_per_epoch
        cardinality = tf.data.experimental.cardinality(train_inputs).numpy()
        train_logger.info("Determining samples per epoch based on pre-generated dataset cardinality")
        # unknown cardinality = -2, infinite cardinality = -1
        if cardinality > 0:
            orig = params.samples_per_epoch
            params.samples_per_epoch = cardinality * params.batch_size
            train_logger.info(f"Adjusted parameter samples_per_epoch from {orig} to {params.samples_per_epoch}")
        else:
            train_logger.info("Train dataset cardinality could not be determined, using value from parameter file")
    # otherwise generate the data on the fly using the input function specified in parameter file
    else:
        train_logger.info("No dataset folder found, training and validation data will be generated on the fly")
        input_fn = InputFunctions(params)
        train_inputs = input_fn.get_dataset(data_dirs=study_dirs["train"], mode="train")
        val_inputs = input_fn.get_dataset(data_dirs=study_dirs["val"], mode="val")

    # Check for existing model and load if exists, otherwise create from scratch
    if latest_ckpt and not params.overwrite:
        train_logger.info("Creating the model to resume checkpoint")
        model = model_fn(params)  # recreating model from scratech may be neccesary if custom loss function is used
        train_logger.info(f"Loading model weights checkpoint file {latest_ckpt}")
        model.load_weights(latest_ckpt)
    else:
        # Define the model from scratch
        train_logger.info("Creating the model...")
        model = model_fn(params)

    # SET CALLBACKS FOR TRAINING FUNCTION

    # define learning rate schedule callback for model
    learning_rate = LearningRateScheduler(LearningRates(params)())

    # checkpoint save callback
    if not os.path.isdir(checkpoint_path):
        os.mkdir(checkpoint_path)
    # save validation loss in name if validation files are passed, else use train loss
    if params.train_fract < 1.:
        ckpt = os.path.join(checkpoint_path, 'epoch_{epoch:02d}_valloss_{val_loss:.4f}.hdf5')
    else:
        ckpt = os.path.join(checkpoint_path, 'epoch_{epoch:02d}_trainloss_{loss:.4f}.hdf5')
    checkpoint = ModelCheckpoint(
        ckpt,
        monitor='val_loss',
        verbose=1,
        save_weights_only=False,
        save_best_only=False,
        mode='auto',
        save_freq='epoch')

    # tensorboard callback
    tensorboard = TensorBoard(
        log_dir=params.model_dir,
        histogram_freq=0,
        write_graph=True,
        write_images=False,
        update_freq=(params.samples_per_epoch // params.batch_size) // 100,  # write losses/metrics 100x per epoch
        profile_batch=2,
        embeddings_freq=0,
        embeddings_metadata=None)

    # training metric logging callback
    if not hasattr(params, "train_dir"):
        params.train_dir = os.path.join(params.model_dir, 'train')
    csv_logger = CSVLogger(
        os.path.join(params.train_dir, 'train_metrics.csv'),
        separator=',',
        append=True)

    # combine callbacks for the model
    train_callbacks = [learning_rate, checkpoint, tensorboard, csv_logger]

    # TRAINING
    train_logger.info(f"Training for {params.num_epochs} total epochs starting at epoch {completed_epochs + 1}")
    model.fit(
        # train inputs are repeated infinitely and an epoch is a user define number of steps
        # this is because model.fit will error with variable epoch size, which can happen with data aumgentation methods
        train_inputs.repeat(),
        epochs=params.num_epochs,
        initial_epoch=completed_epochs,
        steps_per_epoch=params.samples_per_epoch // params.batch_size,
        callbacks=train_callbacks,
        validation_data=val_inputs,
        shuffle=False,
        verbose=1)
    epochs_trained = params.num_epochs - completed_epochs
    train_logger.info(f"Successfully trained model for {epochs_trained} epochs ({params.num_epochs} total epochs)")


# executed  as script
if __name__ == '__main__':
    # parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--param_file', default=None, type=str,
                        help="Path to params.json")
    parser.add_argument('-l', '--logging', default=2, type=int, choices=[1, 2, 3, 4, 5],
                        help="Set logging level: 1=DEBUG, 2=INFO, 3=WARN, 4=ERROR, 5=CRITICAL")
    parser.add_argument('-x', '--overwrite', default=False,
                        help="Overwrite existing data.",
                        action='store_true')

    # Load the parameters from the experiment params.json file in model_dir
    args = parser.parse_args()
    assert args.param_file, "Must specify a parameter file using --param_file"
    assert os.path.isfile(args.param_file), f"No json configuration file found at {args.param_file}"

    # load params from param file
    my_params = load_param_file(args.param_file)

    # set global random seed for tensorflow operations
    tf.random.set_seed(my_params.random_state)

    # handle logging argument
    train_dir = os.path.join(my_params.model_dir, 'train')
    my_params.train_dir = train_dir  # add train dir to params for later use
    if not os.path.isdir(train_dir):
        os.mkdir(train_dir)
    log_path = os.path.join(train_dir, 'train.log')
    if os.path.isfile(log_path) and args.overwrite:
        os.remove(log_path)
    logger = set_logger(log_path, level=args.logging * 10)
    logger.info(f"Using model directory {my_params.model_dir}")
    logger.info(f"Using TensorFlow version {tf.__version__}")

    # do work
    train(my_params)
