# Brain Mask
brain_mask is a 3D fully convolutional deep neural network designed to mask a variety of common brain MR images.

# Installation
The following command will clone a copy of brain_mask to your computer using git:
```
git clone https://github.com/ecalabr/brain_mask.git
```

# Data setup
This software expects your image data to be in Nifti format with a specific directory tree.

### Directory tree
The following example starts with a parent directory (data_dir, but it could be any valid directory name).

```
data_dir/
```
This is an example of the base directory for all of the image data that you want to use. All subdirectories in this folder should contain individual patient image data. Please also see the included "example_data" folder, which follows the expected data format.

```
data_dir/123456/
```
This is an example of an individual patient study directory. The directory name is typically a patient ID, but can be any folder name that does not contain the "_" character

```
data_dir/123456/123456_t1c.nii.gz
```
This is an example of a single patient image.

The complete directory tree should look something like this:

```
data_dir
│
└───123456
│   │   123456_t1c.nii.gz
│   │   123456_t1.nii.gz
│   
└───234567
    │   234567_t1c.nii.gz
    │   234567_t1.nii.gz
```

Note that your data may include more or fewer series than those shown here (t1 and t1c).

### Image filename prefix and suffix
The image file name must start with a prefix, which is typcically the patient ID (and is the same as the patient directory name) followed by a "\_" character and then a suffix, which describes the series. Everything in the filename after the first "\_" character will be treated as the prefix and everything after will be treated as the suffix.

For example:
```
example-patient-1234_t1post_fatsat.nii.gz
```
In this case the prefix is "example-patient-1234" and suffix is "t1post_fatsat". This file would live in a folder that was named the same as the prefix (example-patient-1234/).

### G-zipped requirement
All Nifti files are expected to be g-zipped! Uncompressed nifti files will not work without modifying the code considerably.

# Usage
## Make brain masks using pretrained models
The following command will create a brain mask for data in the included "example_data" folder:

NOTE: Example data is already masked to ensure subject anonymity, so the resulting mask may not be correct. This data is included for testing purposes only.
```
python brain_mask.py -m t1-t1c-t2-flair -i example_data
```
The "-m" flag is used to specify the brain masking model, in this case, "t1-t1c-t2-flair" refers to the model trained on the 4 standard contrasts (T1, T1 + contrast, T2, and FLAIR).

The "-i" flag specifies the input directory. The script automatically finds any subdirectories that contain the required images.

### Image suffixes
NOTE: image data must have series suffixes that correspond to the model being used, i.e. "example001_flair.nii.gz".

TO USE DIFFERENT SUFFIXES YOU CAN EITHER:
1. Edit the yaml file in the corresponding "trained_models" directory (data_prefix).
2. Use the "-n" or "-names" argument followed by the list of suffixes, in order, corresponding to the expected model suffixes.

For example, the following command would execute the 4 contrast masking, but instead of looking for the suffixes t1, t1c, t2, and flair, it would look for T1, T1gad, T2, and FLAIR (note that suffixes are case sensitive):

 ```
python brain_mask.py -m t1-t1c-t2-flair -i example_data -n T1 T1gad T2 FLAIR
```

### Additional optional arguments
The "-h" flag will list all arguments and give additional information on what they do.

The "-s" flag can be used to specify an output suffix (default  is "brain_mask").

The "-t" flag is used to set the probability threshold for masking. Values >0.5 will result in more conservative masks, and <0.5 will result in more liberal masks.

The "-p" flag will output the probability image in addition to the mask.

The "-x" flag is used to overwrite existing data.

The "-f" flag is used to force CPU execution in case you cannot use a GPU (this will be slower). This can also be used to overcome GPU memory limitations.

The "-k" flag is used to specify whether to use the 'best' or 'last' model checkpoint.

The "-l" flag is used to specify the logging level.

The "-c" flag is used to specify the number of connected components to include in the output mask (default is 1). The largest n connected components are kept.

The "-z" flag is used to zero out the periphery of the predicted mask probabilities using a 3D superellipse, which may be helpful if there are artifacts at the periphery of the image.

The "-y" flag is used to specify the order of the 3D superellipse used for zeroing the periphery of the probability image.

The "-start", "-end", and "-list" flags can be used to skip processing of certain subdirectories. See "-h" for more details.

### Image dimension considerations
If your input images are larger than 256 cubed, you may want to adjust the "infer_dims" parameter of the yaml file in the corresponding "trained_models" directory. If infer dims is smaller than the input, then masking will be done in chunks and the result will be stitched together. Larger chunks use more memory and are slower. Smaller chunks are faster and use less VRAM (or RAM if using CPU mode), but can result in stitching artifact if smaller than the input image. You can also adjust the "infer_patch_overlap" parameter to reduce stitching artifact, but high overlap will make memory usage grow very rapidly.

## Pre-generating dataset objects (optional)
Optionally, you can use generate_dataset.py to create and save a tensorflow dataset objects to your model directory. If you choose not to generate dataset objects before training, then datasets will be generated on the fly from NIfTI files. Therefore, this step is not strictly necessary.

Pre-generating your datasets may lead to faster data loading times, and therefore faster training, validation, and evaluation.

One potential downside of pre-generated datasets is that each epoch will use exactly the same training data, whereas with on the fly data generation, each epoch has unique data augmentation (assuming data augmentation is used).

If you want to visualize the model inputs from your dataset, see the "Data visualization" section below.

```
python generate_dataset.py -p brain_mask_params.yml
```
This will create a new directory in your model directory (specified in parameter file) called 'dataset' with subdirectories for training data, validation data (if specified), and evaluation data (if specified).

When you run train.py, it will automatically look for a dataset directory in your model directory, and if found, it will attempt to load data from the dataset directory instead of generating it on the fly. 

## Train
The following command will train the network using the parameters specified in the specified param file (see examples in the included "example_data" directory):
```
python train.py -p brain_mask_params.yml
```
For help with parameter files, please refer to a separate markdown file in this subdirectory named "params_help.md".

Training outputs will be located in the model directory as specified in the param file.
 
If you specified a train_fract of < 1.0 in the parameter file, then validation will be performed after each epoch using the data not used for training.

## Evaluate
After training is complete, you can evaluate the model using evaluate.py. In order to run evaluate.py, you either need to have specified a test_fract > 0.0 in the parameter file, or you can manually edit study_dirs_list.yml to include the directories you want to evaluate on (in the 'eval' list).
```
python evaluate.py -p brain_mask_params.yml
```
This will evaluate your model on the testing set, which is specified with the test_fract parameter and is seperate from both training and evaluation data. The output is printed to the console and will consist of the average loss and whatever metrics you specify in the 'metrics' parameter.
 
## Predict
The following command will use the trained T1c model weights "last" checkpoint (-c) to predict for a single patient (-s) with ID 123456:
```
python predict.py -p brain_mask/trained_models/T1c/T1c.yml -s data_dir/123456 -c "last"
```
Note that the output of predict.py is a probability file (after applying the sigmoid logistic function). If you want to generate a binary mask instead, you can use the '-t' or '--threshold' argument to specify the probability threshold to use for generating a binary output:
```
python predict.py -p brain_mask/trained_models/T1c/T1c.yml -s data_dir/123456 -c "last" -t 0.5
```
Valid thresholds are any value in linspace(0, 1, 0.01) i.e. any float between 0 and 1 in increments of 0.01. However, if you really want binary mask outputs, then it may be worthwhile using brain_mask.py, which has additional options for smoothing and cleaning up binary outputs.

## Input visualization
The included script visualize_dataset.py allows you to visualize model inputs for troubleshooting purposes. This will work on either pre-generated datasets or on the fly generated datasets.

```
python generate_dataset.py -p brain_mask_params.yml -m train --png --nifti -b 20
```
For example, the above command will take 20 batches (-b 20) of train data (-m train) and generate both a PNG montage output (--png) and a folder of individual NIfTI outputs (--nifti) that will be exactly the same as the data seen by the model during training.

Note that the PNG montage output only shows the middle slice of each 3D input, while the NIfTI output will include the entire volume.