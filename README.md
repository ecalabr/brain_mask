# Brain Mask
brain_mask is a 3D fully convolutional deep neural network designed to mask a variety of common brain MR images.

# Installation
The following command will clone a copy of brain_mask to your computer using git:
```bash
git clone https://github.com/ecalabr/brain_mask.git
```

# Data directory tree setup
Gadnet expects your image data to be in Nifti format with a specific directory tree. The following example starts with any directory (referred to as data_dir).

```bash
data_dir/
```
This is an example of the base directory for all of the image data that you want to use. All subdirectories in this folder should contain individual patient image data. Please also see the included "example_data" folder, which follows the expected data format.

```bash
data_dir/123456/
```
This is an example of an individual patient study directory. The directory name is typically a patient ID, but can be any folder name that does not contain the "_" character

```bash
data_dir/123456/123456_t1c.nii.gz
```
This is an example of a single patient image. The image file name must start with the patient ID (i.e. the same as the patient directory name) followed by a "_" character and the series suffix - in this case "t1c" for T1-weighted contrast enhanced imaging.

All Nifti files are expected to be g-zipped!

# Usage
## Make brain masks using pretrained models
The following command will create a brain mask for data in the included "example_data" folder:

NOTE: Example data is already masked to ensure subject anonymity, so the resulting mask may not be correct. This data is included for testing purposes only.
```bash
python brain_mask.py -m t1-t1c-t2-flair -i example_data
```
The "-m" flag is used to specify the brain masking model, in this case, "t1-t1c-t2-flair" refers to the model trained on the 4 standard contrasts (T1, T1 + contrast, T2, and FLAIR).

The "-i" flag specifies the input directory. The script automatically finds any subdirectories that contain the required images.

NOTE: image data must have series suffixes that correspond to the model being used, i.e. "example001_flair.nii.gz". If you need to use different suffixes, you can edit the json file in the corresponding "trained_models" directory (data_prefix).

The "-s" flag can be used to specify an output suffix (default  is "brain_mask").

The "-t" flag is used to set the probability threshold for masking. Values >0.5 will result in more conservative masks, and <0.5 will result in more liberal masks.

The "-p" flag will output the probability image in addition to the mask.

The "-x" flag is used to overwrite existing data.

The "-f" flag is used to force CPU execution in case you cannot use a GPU (this will be slower). This can also be used to overcome GPU memory limitations.

NOTE: if your input images are larger than 256 cubed, you may want to adjust the "infer_dims" parameter of the json file in the corresponding "trained_models" directory. If infer dims is smaller than the input, then masking will be done in chunks and the result will be stitched together. Larger chunks use more memory and are slower. Smaller chunks are faster and use less memory, but will result in stitching artifact if smaller than the input image. You can also adjust the "infer_patch_overlap" parameter to reduce stitch artifact, but high overlap will make memory usage grow very rapidly.

## Train
The following command will train the network using the parameters specified in the specified param file (see examples in the included "example_data" directory):
```bash
python train.py -p brain_mask_params.json
```
For help with parameter files, please refer to a separate markdown in this subdirectory named "params_help.md".

Training outputs will be located in the model directory as specified in the param file.
 
## Predict
The following command will use the trained T1c model weights "last" checkpoint (-c) to predict for a single patient (-s) with ID 123456:
```bash
python predict.py -p brain_mask/trained_models/T1c/T1c.json -s data_dir/123456 -c "last"
```
