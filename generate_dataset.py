import argparse
import logging
import os
import time
# set tensorflow logging level before importing things that contain tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0 = INFO, 1 = WARN, 2 = ERROR, 3 = FATAL
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import tensorflow as tf
from utilities.utils import load_param_file, set_logger, get_study_dirs
from utilities.input_functions import InputFunctions


# generates at TF dataset and writes to disk
def generate_dataset(params):
    # logging
    dataset_logger = logging.getLogger()

    # generate directories
    study_dirs = get_study_dirs(params)  # returns a dict of "train", "val", and "eval"

    # generate dataset objects for model inputs
    input_fn = InputFunctions(params)

    # handle eval dataset
    if study_dirs["eval"]:
        start = time.time()
        dataset_logger.info("Saving evaluation dataset...")
        eval_dataset_dir = os.path.join(params.dataset_dir, "eval")
        eval_inputs = input_fn.get_dataset(data_dirs=study_dirs["eval"], mode="eval")
        eval_inputs.repeat(count=0)
        tf.data.experimental.save(eval_inputs, eval_dataset_dir)
        end = time.time()
        dataset_logger.info(f"- elapsed time is {(end - start)/60:0.2f} minutes")
    else:
        dataset_logger.info("No evaluation data found")

    # handle val dataset
    if study_dirs["val"]:
        start = time.time()
        dataset_logger.info("Saving validation dataset...")
        val_dataset_dir = os.path.join(params.dataset_dir, "val")
        val_inputs = input_fn.get_dataset(data_dirs=study_dirs["val"], mode="val")
        val_inputs.repeat(count=0)
        tf.data.experimental.save(val_inputs, val_dataset_dir)
        end = time.time()
        dataset_logger.info(f"- elapsed time is {(end - start) / 60:0.2f} minutes")
    else:
        dataset_logger.info("No validation data found")

    # handle train dataset
    if study_dirs["train"]:
        start = time.time()
        dataset_logger.info("Saving training dataset...")
        train_dataset_dir = os.path.join(params.dataset_dir, "train")
        train_inputs = input_fn.get_dataset(data_dirs=study_dirs["train"], mode="train")
        train_inputs.repeat(count=0)
        tf.data.experimental.save(train_inputs, train_dataset_dir)
        end = time.time()
        dataset_logger.info(f"- elapsed time is {(end - start) / 60:0.2f} minutes")
    else:
        dataset_logger.info("No training data found")


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
    assert os.path.isfile(args.param_file), "No json configuration file found at {}".format(args.param_file)

    # load params from param file
    my_params = load_param_file(args.param_file)

    # set global random seed for tensorflow operations
    tf.random.set_seed(my_params.random_state)

    # determine dataset directory and create it if it doesn't exist
    my_params.dataset_dir = os.path.join(my_params.model_dir, 'dataset')
    if not os.path.isdir(my_params.dataset_dir):
        os.mkdir(my_params.dataset_dir)

    # handle logging argument
    log_path = os.path.join(my_params.dataset_dir, 'dataset.log')
    if os.path.isfile(log_path) and args.overwrite:
        os.remove(log_path)
    logger = set_logger(log_path, level=args.logging * 10)
    logger.info(f"Using dataset directory {my_params.dataset_dir}")
    logger.info(f"Using TensorFlow version {tf.__version__}")

    # handle overwrite
    if len(os.listdir(my_params.dataset_dir)) > 0 and not args.overwrite:
        logger.warning("Dataset directory is not empty and overwrite argument is false!")
        raise FileExistsError()

    # do work
    generate_dataset(my_params)
