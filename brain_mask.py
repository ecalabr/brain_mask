""" hacky script for making a bunch of brain masks using a CNN """

import argparse
import os
from glob import glob
from utilities.prob2seg import convert_prob
import logging
from utilities.utils import Params
from predict import predict


# set tensorflow logging to FATAL
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = INFO, 1 = WARN, 2 = ERROR, 3 = FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)
import tensorflow as tf

# limit tensorflow memory growth at start
physical_devices = tf.config.list_physical_devices('GPU')
for phys_dev in physical_devices:
    try:
        tf.config.experimental.set_memory_growth(phys_dev, True)
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass


# define function to make a batch of brain masks from a list of directories
def batch_mask(infer_direcs, param_file, out_dir, suffix, overwrite=False, thresh=0.5, clean=False):
    # handle out_dir = None
    if out_dir is None:
        out_dir = infer_direcs
    else:
        out_dir = [out_dir] * len(infer_direcs)
    # ensure that infer_direcs is list
    if not isinstance(infer_direcs, (list, tuple)):
        infer_direcs = [infer_direcs]
    # initiate outputs
    outnames = []
    # load params
    params = Params(param_file)
    # turn off mixed precision and distribution
    params.dist_strat = "none"
    params.mixed_precision = False
    # get model dir
    if params.model_dir == 'same':  # this allows the model dir to be inferred from params.json file path
        params.model_dir = os.path.dirname(param_file)
    if not os.path.isdir(params.model_dir):
        raise ValueError("Specified model directory does not exist")
    # run inference and post-processing for each infer_dir
    for i, direc in enumerate(infer_direcs):
        print("Processing the following directory: {}".format(direc))
        # make sure all required files exist in data directory, if not, skip
        skip = 0
        for suf in params.data_prefix:
            if not glob(direc + "/*{}.nii.gz".format(suf)):
                print("Directory {} is missing required file: {} and will be skipped...".format(direc, suf))
                skip = 1
        if skip:
            continue

        # run predict on one directory and get the output probabilities
        prob = predict(params, [direc], out_dir[i], mask=None, checkpoint='last')  # direc must be list for predict fn

        # convert probs to mask with cleanup
        idno = os.path.basename(direc.rsplit('/', 1)[0] if direc.endswith('/') else direc)
        nii_out_path = os.path.join(direc, idno + "_" + suffix + ".nii.gz")
        if os.path.isfile(nii_out_path) and not overwrite:
            print("Mask file already exists at {}".format(nii_out_path))
        else:
            if prob:
                nii_out_path = convert_prob(prob, nii_out_path, clean=clean, thresh=thresh)

                # report
                if os.path.isfile(nii_out_path):
                    print("Created mask file at: {}".format(nii_out_path))
                else:
                    raise ValueError("No mask output file found at: {}".format(direc))

                # add to outname list
                outnames.append(nii_out_path)

    # release memory at end of task
    tf.keras.backend.clear_session()

    return outnames


# executed  as script
if __name__ == '__main__':
    # parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default="t1-t1c-t2-flair",
                        help="Path to params.json")
    parser.add_argument('-i', '--input_dir', default=None,
                        help="Input folder or files", required=True)
    parser.add_argument('-o', '--out_dir', default=None,
                        help="Filename suffix for output mask")
    parser.add_argument('-s', '--out_suffix', default="brain_mask",
                        help="Filename suffix for output mask")
    parser.add_argument('-t', '--thresh', default=0.5,
                        help="Probability threshold for predictions")
    parser.add_argument('-p', '--prob', default=False,
                        help="Output probabilities in addition to mask.",
                        action='store_true')
    parser.add_argument('-x', '--overwrite', default=False,
                        help="Overwrite existing data.",
                        action='store_true')
    parser.add_argument('-f', '--force_cpu', default=False,
                        help="Disable GPU and force all computation to be done on CPU",
                        action='store_true')
    args = parser.parse_args()

    # handle input argument
    assert args.input_dir, "Must specify input directory using -i or --input_dir"
    assert os.path.isdir(args.input_dir), "Specified input directory does not exist: {}".format(args.input_dir)
    if glob(args.input_dir + "/*.nii.gz"):
        data_dirs = [args.input_dir]
    elif glob(args.input_dir + "/*/*.nii.gz"):
        data_dirs = [y for y in list(set(os.path.dirname(x) for x in glob(args.input_dir + "/*/*.nii.gz")))]
    else:
        raise FileNotFoundError("No image data (.nii.gz) found in input directory!")
    print("Found the following data directories to process:\n{}".format("\n".join(data_dirs)))

    # handle model argument
    mod_nf = "Model argument must be one of:'t1', 't1c', 't2', 'flair', 'dwi', 't1-t1c-t2-flair', 't1-t1c-t2-flair-dwi'"
    assert args.model in ["t1", "t1c", "t2", "flair", "dwi", "t1-t1c-t2-flair", "t1-t1c-t2-flair-dwi"], mod_nf
    script_dir = os.path.dirname(os.path.realpath(__file__))
    my_param_file = os.path.join(script_dir, "trained_models/{}/{}.json".format(args.model, args.model))
    print("Using the following parameter file: {}".format(my_param_file))

    # handle out_dir argument
    if args.out_dir:
        assert os.path.isdir(args.out_dir), "Specified output directory does not exist: {}".format(args.out_dir)

    # handle thresh argument
    if not isinstance(args.thresh, float):
        try:
            args.thresh = float(args.thresh)
        except:
            raise ValueError("Could not cast thresh argument to float: {}".format(args.thresh))

    # handle suffix argument
    if not isinstance(args.out_suffix, str):
        raise ValueError('Suffix argument must be a string: {}'.format(args.out_suffix))
    else:
        if args.out_suffix.endswith('.nii.gz'):
            args.out_suffix = args.out_suffix.split('.nii.gz')[0]

    # handle force cpu argument
    if args.force_cpu:
        logging.info("Forcing CPU (GPU disabled)")
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # do work
    output_names = batch_mask(data_dirs,
                              my_param_file,
                              args.out_dir,
                              args.out_suffix,
                              overwrite=args.overwrite,
                              thresh=args.thresh,
                              clean=not args.prob)
