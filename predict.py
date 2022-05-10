"""Use trained model for prediction"""

import argparse
import logging
import os
from glob import glob
import nibabel as nib
import numpy as np
# set tensorflow logging to FATAL before importing things with tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # 0 = INFO, 1 = WARN, 2 = ERROR, 3 = FATAL
tf_logger = logging.getLogger('tensorflow')
tf_logger.setLevel(logging.WARN)
import tensorflow as tf
from utilities.input_fn_util import reconstruct_infer_patches_3d
from utilities.utils import load_param_file, set_logger
from utilities.input_functions import InputFunctions
from model.model_fn import model_fn


# take raw predictions and convert to nifti file
def predictions_2_nii(predictions, infer_dir, out_dir, params, mask=None, thresh=None):
    # set up logging
    pred_logger = logging.getLogger()

    # load one of the original images to restore original shape and to use for masking
    nii1 = nib.load(glob(infer_dir + '/*' + params.data_prefix[0] + '*.nii.gz')[0])
    affine = nii1.affine
    name_prefix = os.path.basename(infer_dir[0:-1] if infer_dir.endswith('/') else infer_dir)

    # 3D inference
    predictions = reconstruct_infer_patches_3d(predictions, infer_dir, params)

    # mask predictions based on provided mask
    if mask:
        mask_nii = glob(infer_dir + '/*' + mask + '.nii.gz')[0]
        mask_img = nib.load(mask_nii).get_fdata() > 0
        predictions = np.squeeze(predictions) * mask_img

    # optional probability thresholding
    if thresh:
        predictions = predictions > thresh

    # convert to nifti format and save
    model_name = os.path.basename(params.model_dir)
    nii_out = os.path.join(out_dir, name_prefix + '_predictions_' + model_name + '.nii.gz')
    img = nib.Nifti1Image(predictions.astype(np.float32), affine)
    pred_logger.info("Saving predictions to: " + nii_out)
    nib.save(img, nii_out)
    if not os.path.isfile(nii_out):
        raise ValueError("Output nii could not be created at {}".format(nii_out))

    return nii_out


# make the prediction mode
def pred_model(params, checkpoint='last', cpu=False):
    # set up logging
    pred_logger = logging.getLogger()

    # handle force cpu
    if cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # load latest checkpoint
    checkpoint_path = os.path.join(params.model_dir, 'checkpoints')
    checkpoints = glob(checkpoint_path + '/*.hdf5')
    if checkpoints:
        # load best or last checkpoint
        # determine last by timestamp
        if checkpoint == 'last':
            ckpt = max(checkpoints, key=os.path.getctime)
        # determine best by minimum loss value in filename
        elif checkpoint == 'best':
            try:
                vals = [float(item[0:-5].split('_')[-1]) for item in checkpoints]
                ckpt = checkpoints[np.argmin(vals)]
            except:
                line1 = "Could not determine 'best' checkpoint based on checkpoint filenames. "
                line2 = "Use 'last' or pass a specific checkpoint filename to the checkpoint argument."
                pred_logger.error(line1 + line2)
                raise ValueError(line1 + line2)
        elif os.path.isfile(os.path.join(my_params.model_dir, "checkpoints/{}.hdf5".format(checkpoint))):
            ckpt = os.path.join(my_params.model_dir, "checkpoints/{}.hdf5".format(checkpoint))
        else:
            raise ValueError("Did not understand checkpoint value: {}".format(args.checkpoint))
        # net_builder input layer uses train_dims, so set these to infer dims to allow different size inference
        params.train_dims = params.infer_dims
        # batch size for inference is hard-coded to 1
        params.batch_size = 1
        # recreate the model using infer dims as input dims
        pred_logger.info("Creating the model")
        model = model_fn(params)
        # load weights from last checkpoint
        pred_logger.info("Loading '{}' checkpoint from {}...".format(checkpoint, ckpt))
        model.load_weights(ckpt)
    else:
        raise ValueError("No model checkpoints found at {}".format(checkpoint_path))

    return model


# predict a batch of input directories
def predict(params, model, pred_dirs, out_dir, mask=None, overwrite=False, thresh=None):
    # set up logging
    pred_logger = logging.getLogger()

    # infer directories in a loop
    niis_out = []
    for pred_dir in pred_dirs:
        # define expected output file name to check if output prediction already exists
        model_name = os.path.basename(params.model_dir)
        name_prefix = os.path.basename(pred_dir)
        pred_out = os.path.join(out_dir, name_prefix + '_predictions_' + model_name + '.nii.gz')
        # if output doesn't already exist, then predict and make nii
        if not os.path.isfile(pred_out) or overwrite:
            # Create the inference dataset structure
            infer_inputs = InputFunctions(params).get_dataset(data_dirs=[pred_dir], mode="infer")
            # predict
            predictions = model.predict(infer_inputs)
            # save nii
            pred_out = predictions_2_nii(predictions, pred_dir, out_dir, params, mask=mask, thresh=thresh)
        else:
            pred_logger.info("Predictions already exist and will not be overwritten: {}".format(pred_out))
        # update list of output niis
        niis_out.append(pred_out)

    return niis_out


# executed  as script
if __name__ == '__main__':

    # parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--param_file', default=None,
                        help="Path to params.json")
    parser.add_argument('-d', '--data_dir', default=None,
                        help="Path to directory to generate inference from")
    parser.add_argument('-c', '--checkpoint', default='last', choices=["best", "last"],
                        help="Can be 'best', 'last', or an hdf5 filename in the checkpoints subdirectory of model_dir")
    parser.add_argument('-m', '--mask', default=None,
                        help="Optionally specify a filename prefix for a mask to mask the predictions")
    parser.add_argument('-o', '--out_dir', default=None,
                        help="Optionally specify output directory")
    parser.add_argument('-s', '--spec_direc', default=None,
                        help="Optionally specify a specifing single directory for inference (overrides -d)")
    parser.add_argument('-f', '--force_cpu', default=False,
                        help="Disable GPU and force all computation to be done on CPU",
                        action='store_true')
    parser.add_argument('-t', '--threshold', default=None, type=float,
                        help="Threshold probability output to create a binary mask")
    parser.add_argument('-l', '--logging', default=2, type=int, choices=[1, 2, 3, 4, 5],
                        help="Set logging level: 1=DEBUG, 2=INFO, 3=WARN, 4=ERROR, 5=CRITICAL")
    parser.add_argument('-x', '--overwrite', default=False,
                        help="Overwrite existing data.",
                        action='store_true')
    args = parser.parse_args()

    # handle logging argument
    log_path = None
    if args.out_dir:
        log_path = os.path.join(args.out_dir, 'predict.log')
        if os.path.isfile(log_path) and args.overwrite:
            os.remove(log_path)
    logger = set_logger(log_path, level=args.logging * 10)
    logging.info("Using parameter file {}".format(args.param_file))
    logging.info("Using TensorFlow version {}".format(tf.__version__))

    # handle param argument
    assert args.param_file, "Must specify param file using --param_file"
    assert os.path.isfile(args.param_file), "No json configuration file found at {}".format(args.param_file)
    my_params = load_param_file(args.param_file)
    # turn of distributed strategy and mixed precision
    my_params.dist_strat = None
    my_params.mixed_precision = False

    # handle out_dir argument
    if args.out_dir:
        assert os.path.isdir(args.out_dir), "Specified output directory does not exist: {}".format(args.out_dir)
    else:
        args.out_dir = os.path.join(os.path.dirname(args.param_file), 'prediction')
        if not os.path.isdir(args.out_dir):
            os.mkdir(args.out_dir)

    # determine directories for inference
    if not any([args.data_dir, args.spec_direc]):
        raise ValueError("Must specify directory for inference with -d (parent directory) or -s (single directory)")
    study_dirs = []
    # handle specific directory argument
    if args.spec_direc:
        if os.path.isdir(args.spec_direc) and all([glob(args.spec_direc + '/*{}.nii.gz'.format(item)) for item in
                                                   my_params.data_prefix]):
            study_dirs = [args.spec_direc]
            logger.info("Generating predictions for single directory: {}".format(args.spec_direc))
        else:
            logger.error("Specified directory does not have the required files: {}".format(args.spec_direc))
            exit()
    # if specific directory is not specified, then use data_dir argument
    else:
        # get all subdirectories in data_dir
        study_dirs = [item for item in glob(args.data_dir + '/*/') if os.path.isdir(item)]
    # make sure all necessary files are present in each folder
    study_dirs = [study for study in study_dirs if all(
        [glob('{}/*{}.nii.gz'.format(study, item)) and os.path.isfile(glob('{}/*{}.nii.gz'.format(study, item))[0])
         for item in my_params.data_prefix])]
    # study dirs sorted in alphabetical order for reproducible results
    study_dirs.sort()
    # make sure there are study dirs found
    if not study_dirs:
        logger.error("No valid study directories found in parent data directory {}".format(args.data_dir))
        exit()

    # handle checkpoint argument
    if args.checkpoint not in ['best', 'last']:  # checks if user wants to use the best or last checkpoint
        # strip hdf5 extension if present
        args.checkpoint = args.checkpoint.split('.hdf5')[0] if args.checkpoint.endswith('.hdf5') else args.checkpoint
        # check for existing checkpoint corresponding to user specified checkpoint argument string
        if not os.path.isfile(os.path.join(my_params.model_dir, "checkpoints/{}.hdf5".format(args.checkpoint))):
            raise ValueError("Did not understand checkpoint value: {}".format(args.checkpoint))

    # handle mask argument
    if args.mask:
        mask_niis = [glob(study_dir + '/*' + args.mask + '.nii.gz')[0] for study_dir in study_dirs]
        if not all(os.path.isfile(item) for item in mask_niis):
            raise ValueError("Specified mask prefix is not present for all studies in data_dir: {}".format(args.mask))

    # handle force cpu argument
    if args.force_cpu:
        logger.info("Forcing CPU (GPU disabled)")
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # handle threshold argument
    if args.threshold:
        if not 0. < args.threshold < 1.:
            raise ValueError(f"Threshold must be a float between 0 and 1 but is {args.threshold}")

    # make predictions
    my_model = pred_model(my_params, checkpoint=args.checkpoint, cpu=args.force_cpu)
    pred = predict(my_params, my_model, study_dirs, args.out_dir,
                   mask=args.mask,
                   overwrite=args.overwrite,
                   thresh=args.threshold)
