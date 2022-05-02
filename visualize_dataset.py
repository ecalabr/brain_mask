import argparse
import os
import logging
# set tensorflow logging level before importing things that contain tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0 = INFO, 1 = WARN, 2 = ERROR, 3 = FATAL
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import tensorflow as tf
from utilities.utils import load_param_file, set_logger, get_study_dirs
from utilities.input_functions import InputFunctions
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib


# make a png montage
def png_montage(dataset, batches, out_dir, mode):

    # logging
    png_logger = logging.getLogger()

    # warn for large number of batches
    lots_o_batches = 100
    if batches > lots_o_batches:
        wait_for_conf = True
        while wait_for_conf:
            response = input(
                f"Making a PNG montage with >{lots_o_batches} batches may take a lot of resources, "
                "are you sure you want to proceed? [Y/n]\n>> ")
            if response == 'n':
                exit()
            elif response == 'Y':
                wait_for_conf = False

    # get first batch to figure out shapes of inputs
    data_iter = iter(dataset.repeat())
    elem = next(data_iter)

    # check tuple length
    inputs_per_batch = len(elem)
    input_shapes = [elem[i].get_shape().as_list() for i in range(inputs_per_batch)]
    batch_size = input_shapes[0][0]
    chan_sizes = [el.get_shape().as_list()[-1] for el in elem]

    # prepare subplot system for montage
    # each column will contain all data for one batch
    cols = inputs_per_batch * batch_size
    # each row will be a new batch
    rows = batches
    # adjust width ratios for different number of channels per input
    fig, ax = plt.subplots(rows, cols, gridspec_kw={'width_ratios': chan_sizes * batch_size})
    # set plot size based on expected total size
    total_width = sum([input_shapes[i][1] * chan_sizes[i] for i in range(inputs_per_batch)]) * batch_size
    total_heigh = input_shapes[0][2] * batches
    dpi = 300
    width_in = total_width / dpi
    height_in = total_heigh / dpi
    plt.subplots_adjust(wspace=0., hspace=0.)

    # batch loop
    for b in range(batches):
        # batch element loop
        for e in range(batch_size):
            # input loop
            for i in range(inputs_per_batch):
                # concatenate channels
                img = np.concatenate([np.swapaxes(elem[i][e, :, :, input_shapes[i][3] // 2, chan], 0, 1) for chan in
                                      range(input_shapes[i][4])], axis=1)
                # get current subplot
                current_ax = ax[b, (e * inputs_per_batch) + i]
                current_ax.imshow(img, cmap='gray')
                current_ax.invert_yaxis()
                current_ax.text(0., 0.01, f"b{b}i{i}e{e}", fontsize=3,  color="yellow")
                current_ax.axis('off')
        # get next element
        elem = next(data_iter)
    # save
    fig = plt.gcf()
    fig.set_size_inches(width_in, height_in)
    outname = os.path.join(out_dir, f"{mode}-inputs_{batches}-batches.png")
    fig.savefig(outname, bbox_inches='tight', dpi=dpi)
    png_logger.info(f"Wrote PNG montage for {batches} {mode} batches to: {outname}")


# save niftis
def save_niftis(dataset, batches, vis_dir, mode):

    # logging
    nifti_logger = logging.getLogger()

    # make output subdirectory
    out_dir = os.path.join(vis_dir, f"{mode}_niis_{batches}_batches")
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    # make infinite dataset iterator and get first element to determine number of inputs
    data_iter = iter(dataset.repeat())
    elem = next(data_iter)
    inputs_per_batch = len(elem)
    input_shapes = [elem[i].get_shape().as_list() for i in range(inputs_per_batch)]
    batch_size = input_shapes[0][0]

    # determine zfill length for number of batches
    zb = len(str(batches))
    ze = len(str(batch_size))
    zi = len(str(inputs_per_batch))

    # loop through batches
    for b in range(batches):
        # batch element loop
        for e in range(batch_size):
            # input loop
            for i in range(inputs_per_batch):
                # get image
                img = np.squeeze(elem[i][e, :, :, :, :])
                # make outname
                outname = os.path.join(out_dir, f"b{str(b).zfill(zb)}_e{str(e).zfill(ze)}_i{str(i).zfill(zi)}.nii.gz")
                # save
                nib.save(nib.Nifti1Image(img, np.eye(4)), outname)
    nifti_logger.info(f"Wrote nifti files for {batches} {mode} batches to: {out_dir}")


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
    parser.add_argument('-m', '--mode', default="train", choices=["train", "val", "eval", "infer"],
                        help="Data mode to visualize")
    parser.add_argument('-i', '--infer_dirs', default=None, type=list, nargs='+',
                        help="List of directories for inference mode")
    parser.add_argument('-g', '--png', default=False, action="store_true",
                        help="Output a png montage")
    parser.add_argument('-n', '--nifti', default=False, action="store_true",
                        help="Output individual nifti files")
    parser.add_argument('-b', '--batches', default=25, type=int,
                        help="Number of batches to visualize")

    # Load the parameters from the experiment params.json file in model_dir
    args = parser.parse_args()
    assert args.param_file, "Must specify a parameter file using --param_file"
    assert os.path.isfile(args.param_file), "No json configuration file found at {}".format(args.param_file)

    # load params from param file
    my_params = load_param_file(args.param_file)

    # determine visualization directory and create it if it doesn't exist
    my_params.vis_dir = os.path.join(my_params.model_dir, 'dataset_visualization')
    if not os.path.isdir(my_params.vis_dir):
        os.mkdir(my_params.vis_dir)

    # handle logging argument
    log_path = os.path.join(my_params.vis_dir, 'vis.log')
    if os.path.isfile(log_path) and args.overwrite:
        os.remove(log_path)
    logger = set_logger(log_path, level=args.logging * 10)
    logger.info(f"Using dataset visualization directory {my_params.vis_dir}")
    logger.info(f"Using TensorFlow version {tf.__version__}")

    # set up dataset directory
    dataset_dir = os.path.join(my_params.model_dir, 'dataset')

    # handle train
    if args.mode == 'train':
        train_data_dir = os.path.join(dataset_dir, 'train')
        if os.path.isdir(train_data_dir):
            logger.info(f"Loading existing training dataset from {dataset_dir}")
            input_dataset = tf.data.experimental.load(train_data_dir)
        else:
            logger.info(f"No dataset folder found, training data will be generated on the fly")
            study_dirs = get_study_dirs(my_params)  # returns a dict of "train", "val", and "eval"
            input_fn = InputFunctions(my_params)
            input_dataset = input_fn.get_dataset(data_dirs=study_dirs["train"], mode="train")

    # handle val
    elif args.mode == 'val':
        val_data_dir = os.path.join(dataset_dir, 'val')
        if os.path.isdir(val_data_dir):
            logger.info(f"Loading existing validation dataset from {dataset_dir}")
            input_dataset = tf.data.experimental.load(val_data_dir)
        else:
            logger.info(f"No dataset folder found, validation data will be generated on the fly")
            study_dirs = get_study_dirs(my_params)  # returns a dict of "train", "val", and "eval"
            input_fn = InputFunctions(my_params)
            input_dataset = input_fn.get_dataset(data_dirs=study_dirs["val"], mode="val")

    # handle eval
    elif args.mode == 'eval':
        eval_data_dir = os.path.join(dataset_dir, 'eval')
        if os.path.isdir(eval_data_dir):
            logger.info(f"Loading existing evaluation dataset from {dataset_dir}")
            input_dataset = tf.data.experimental.load(eval_data_dir)
        else:
            logger.info(f"No dataset folder found, evaluation data will be generated on the fly")
            study_dirs = get_study_dirs(my_params)  # returns a dict of "train", "val", and "eval"
            input_fn = InputFunctions(my_params)
            input_dataset = input_fn.get_dataset(data_dirs=study_dirs["eval"], mode="eval")

    # handle infer
    elif args.mode == 'infer':
        assert args.infer_dirs, "Must specify infer_dirs with the infer argument"
        if not all(os.path.isdir(item) for item in args.infer_dirs):
            raise FileNotFoundError("One or more directories in infer_dirs argument does not exist!")
        input_dataset = InputFunctions(my_params).get_dataset(data_dirs=args.infer_dirs, mode="infer")

    # unknown mode
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    # handle png
    if args.png:
        png_montage(input_dataset, args.batches, my_params.vis_dir, mode=args.mode)

    # handle nifti
    if args.nifti:
        save_niftis(input_dataset, args.batches, my_params.vis_dir, mode=args.mode)
