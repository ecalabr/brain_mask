""" Wrapper script for making masks from images using CNN """

import argparse
import os
from glob import glob
from utilities.prob2seg import convert_prob
import logging
from utilities.utils import Params
from predict import predict

# set up tensorflow logging before import
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # 0 = INFO, 1 = WARN, 2 = ERROR, 3 = FATAL
tf_logger = logging.getLogger('tensorflow')
tf_logger.setLevel(logging.WARN)
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
def batch_mask(infer_direcs, param_file, out_dir, suffix, overwrite=False, thresh=0.5, clean=False, names=None, n_seg=1,
               zero_periph=True, zp_order=2.5):
    # set up logging
    logger = logging.getLogger("brain_mask")
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
    failed = []
    # load params
    params = Params(param_file)
    # turn off mixed precision and distribution
    params.dist_strat = "none"
    params.mixed_precision = False
    # handle names argument
    if names:
        if not isinstance(names, (list, tuple)):
            names = [names]
        assert len(names) == len(params.data_prefix), "Length of specified suffixes is {} but should be {}".format(
            len(names), len(params.data_prefix))
        logger.info("Using user specified suffixes for data as follows:")
        for n, o in zip(names, params.data_prefix):
            logger.info("{} --> {}".format(o, n))
        params.data_prefix = names
    # get model dir
    if params.model_dir == 'same':  # this allows the model dir to be inferred from params.json file path
        params.model_dir = os.path.dirname(param_file)
    if not os.path.isdir(params.model_dir):
        raise ValueError("Specified model directory does not exist")
    # run inference and post-processing for each infer_dir
    for i, direc in enumerate(infer_direcs):
        logger.info("Processing the following directory: {}".format(direc))
        # make sure all required files exist in data directory, if not, skip
        skip = 0
        for suf in params.data_prefix:
            if not glob(direc + "/*{}.nii.gz".format(suf)):
                logger.info("Directory {} is missing required file: {} and will be skipped...".format(direc, suf))
                skip = 1
        if skip:
            continue
        # determine mask output path
        idno = os.path.basename(direc.rsplit('/', 1)[0] if direc.endswith('/') else direc)
        nii_out_path = os.path.join(direc, idno + "_" + suffix + ".nii.gz")
        # if overwrite not set, make sure output doesn't exist before proceeding
        if not overwrite:
            if os.path.isfile(nii_out_path):
                logger.info("Mask file already exists at {} and overwrite argument is false".format(nii_out_path))
                continue

        # run predict on one directory and get the output probabilities
        prob = predict(params, [direc], out_dir[i], mask=None, checkpoint='last')  # direc must be list for predict fn

        # convert probs to mask with cleanup
        nii_out_path = convert_prob(prob,
                                    nii_out_path,
                                    clean=clean,
                                    thresh=thresh,
                                    n_seg=n_seg,
                                    zero_periph=zero_periph,
                                    order=zp_order)

        # report
        if os.path.isfile(nii_out_path):
            logger.info("Created mask file at: {}".format(nii_out_path))
            # add to outname list
            outnames.append(nii_out_path)
        else:
            failed.append(direc)

    # release memory at end of task
    tf.keras.backend.clear_session()

    # report failures
    if failed:
        logger.info("Mask generation failed for the following directories:\n{}".format("\n".join(failed)))

    return outnames, failed


# executed  as script
if __name__ == '__main__':
    # parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default="t1-t1c-t2-flair", type=str,
                        help="Path to params.json")
    parser.add_argument('-n', '--names', default=None, nargs="+",
                        help="Optionally specify a list of suffixes to replace the expected suffixes for inputs")
    parser.add_argument('-i', '--input_dir', type=str, required=True,
                        help="Input directory or parent directory containing nifti inputs")
    parser.add_argument('-o', '--out_dir', default=None, type=str,
                        help="Optionally specify an output directory. Default is to use input directories.")
    parser.add_argument('-s', '--out_suffix', default="brain_mask", type=str,
                        help="Filename suffix for output mask")
    parser.add_argument('-t', '--thresh', default=0.5, type=float,
                        help="Probability threshold for predictions.")
    parser.add_argument('-p', '--prob', default=False,
                        help="Output probabilities in addition to mask.",
                        action='store_true')
    parser.add_argument('-x', '--overwrite', default=False,
                        help="Overwrite existing data.",
                        action='store_true')
    parser.add_argument('-f', '--force_cpu', default=False,
                        help="Disable GPU and force all computation to be done on CPU.",
                        action='store_true')
    parser.add_argument('-l', '--logging', default=2, type=int, choices=[1, 2, 3, 4, 5],
                        help="Set logging level: 1=DEBUG, 2=INFO, 3=WARN, 4=ERROR, 5=CRITICAL")
    parser.add_argument('-c', '--components', type=int, default=1,
                        help="Number of connected components in output mask.")
    parser.add_argument('-z', '--zero_periphery', default=True,
                        help="Zero the periphery of probability image (using a superellipse mask) before binarizing."
                             " This can remove unwanted artifacts at the periphery of the image.",
                        action='store_false')
    parser.add_argument('-y', '--zp_order', default=2.5, type=float,
                        help="If using zero_periphery - optionally specify the exponent for the superellipse.")
    parser.add_argument('--start', default=0, type=int,
                        help="Index of directories to start processing at (starting from 0 [default].")
    parser.add_argument('--end', default=-1, type=int,
                        help="Index of directories to end processing at. -1 [default] is the last directory.")
    parser.add_argument('--list', action="store_true", default=False,
                        help="List the directories to be processed in order and exit without doing anything.")
    args = parser.parse_args()

    # handle logging argument
    args.logging = args.logging
    logging.basicConfig()
    bm_logger = logging.getLogger('brain_mask')
    bm_logger.setLevel(args.logging * 10)

    # handle input argument
    assert os.path.isdir(args.input_dir), "Specified input directory does not exist: {}".format(args.input_dir)
    if glob(args.input_dir + "/*.nii.gz"):
        data_dirs = [args.input_dir]
    elif glob(args.input_dir + "/*/*.nii.gz"):
        data_dirs = [y for y in list(set(os.path.dirname(x) for x in glob(args.input_dir + "/*/*.nii.gz")))]
    else:
        raise FileNotFoundError("No image data (.nii.gz) found in input directory!")

    # sort data dirs
    data_dirs = sorted(data_dirs)

    # handle start and end arguments
    if args.end != -1:
        data_dirs = data_dirs[int(args.start):int(args.end) + 1]
    else:
        data_dirs = data_dirs[int(args.start):]
    if isinstance(data_dirs, str):
        data_dirs = [data_dirs]

    # handle list argument
    if args.list:
        for i, item in enumerate(data_dirs, 0):
            print(str(i) + ': ' + item)
        exit()

    # handle model argument
    script_dir = os.path.dirname(os.path.realpath(__file__))
    model_names = sorted(
        [os.path.basename(item.rstrip("/")) for item in glob(script_dir + "/trained_models/*") if os.path.isdir(item)])
    bm_logger.debug("Found the following pre-trained models: {}".format(", ".join(model_names)))
    mod_nf = "Model argument must be one of: {}, but is: {}".format(", ".join(model_names), args.model)
    assert args.model in model_names, mod_nf
    my_param_file = os.path.join(script_dir, "trained_models/{}/{}.json".format(args.model, args.model))
    bm_logger.info("Using the following parameter file: {}".format(my_param_file))

    # handle out_dir argument
    if args.out_dir:
        assert os.path.isdir(args.out_dir), "Specified output directory does not exist: {}".format(args.out_dir)

    # handle suffix argument
    if not isinstance(args.out_suffix, str):
        raise ValueError('Suffix argument must be a string: {}'.format(args.out_suffix))
    else:
        if args.out_suffix.endswith('.nii.gz'):
            args.out_suffix = args.out_suffix.split('.nii.gz')[0]

    # handle force cpu argument
    if args.force_cpu:
        bm_logger.info("Forcing CPU (GPU disabled)")
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # do work
    output_names, fail = batch_mask(data_dirs,
                                    my_param_file,
                                    args.out_dir,
                                    args.out_suffix,
                                    overwrite=args.overwrite,
                                    thresh=args.thresh,
                                    clean=not args.prob,
                                    names=args.names,
                                    n_seg=args.components,
                                    zero_periph=args.zero_periphery,
                                    zp_order=args.zp_order
                                    )
