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
This is an example of the base directory for all of the image data that you want to use. All subdirectories in this folder should contain individual patient image data.

```bash
data_dir/123456/
```
This is an example of an individual patient study directory. The directory name is typically a patient ID, but can be any folder name that does not contain the "_" character

```bash
data_dir/123456/123456_T1gad.nii.gz
```
This is an example of a single patient image. The image file name must start with the patient ID (i.e. the same as the patient directory name) followed by a "_" character. All Nifti files are expected to be g-zipped.

# Usage
## Train
The following command will train the network using the parameters specified in the param file (-p):
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
