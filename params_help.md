# Param files
This guide explains the required parameters for param files. Param files should be written in YAML or JSON.

## Setup parameters
The following parameters pertain to global setup of the model:
```yaml
data_dir: /path/to/data
model_dir: same
overwrite: true
random_state: 42
```
### Explanations
```yaml
data_dir: /path/to/data
```
The full path to the directory contaning the individual study directories to be used for training the model.
```yaml
model_dir: same
```
The full path to the directory for model outputs. If "same" is passed, then the data directory is used for model outputs.
```yaml
overwrite: True
```
Whether or not to overwrite model ouputs if the model is run again. Must be boolean. Note that specifying false will raise an error if a model output already exists in model_dir.
```yaml
random_state: 42
```
An integer to be used as the seed for random operations that need to be reproducible (example = directory shuffling for train/test split

## Data input parameters
These parameters dictate how the input data is handled:
```yaml
data_prefix: [T1, T1gad, T2, FLAIR]
label_prefix: [brain_mask]
mask_prefix: [brain_mask]
mask_dilate: [232, 232, 80]
filter_zero: 0.2
resample_spacing: []
load_shape: []
```
### Explanations
```yaml
data_prefix: [T1, T1gad, T2, FLAIR]
```
The prefixes for the input image files specified as a bracketed list. The expected filename format is directory-name_data-prefix.nii.gz i.e. 01234567_FLAIR.nii.gz.
```yaml
label_prefix: [brain_mask]
```
The prefixes for the label image files specified as a bracketed list. The expected filename format is directory-name_label-prefix.nii.gz i.e. 01234567_labels.nii.gz.
```yaml
mask_prefix: [brain_mask]
```
The prefixes for the input image mask file specified as a bracketed list. The expected filename format is directory-name_mask-prefix.nii.gz i.e. 01234567_brainmask.nii.gz. Can be false, in which case a mask of all 1s is used.
```yaml
mask_dilate: [232, 232, 80]
```
How much to dilate the input mask after cropping the input images to the tight bounding box of the mask. If an integer, then all dimensions are symmetrically expanded by that amount (1/2 before, 1/2 after). If a list of 3 ints, then the input is symmetrically expanded such that its final dimensions are equal to the ints in the list. Each masked input image dimension is expanded to the size of the corresponding value. If the int is 0 or smaller than the corresponding masked input dim, then no expansion is performed in that dimension.
```yaml
filter_zero: 0.2
```
Filters out patches where labels == 0 in most of the patch. filter_zero is a float defining the minimum threshold for label pixels > 0 in the patch.
```yaml
resample_spacing: []
```
When [], this parameter does nothing. Normally this parameter should be an empty list since your NIfTI data should already be in saved in the desired resolution and dimensions. You can use this parameter if you want to resample the NIfTI data to any arbitrary resolution at the time of loading from disk. This parameter will be the image spacing (i.e. resolution) in mm. For example, if you want to resample your NIfTI images to 1 mm x 1 mm x 2 mm at the time of loading, this paramter should be [1, 1, 2]. Note that you can specify a length 1 list (e.g. [1]) if you want all dimensions to be the same resolution/spacing. Note that this is applied immediately after loading the data from disk and can have complex interactions with other parameters that act at later stages of the input function.
```yaml
load_shape: []
```
When [], this parameter does nothing. Normally this parameter should be an empty list since your NIfTI data should already be in saved in the desired resolution and dimensions. You can use this parameter if you want to pad/crop the NIfTI data to any arbitrary shape at the time of loading from disk. This parameter will be the image shape (i.e. dimensions) in voxels. For example, if you want to pad/crop your NIfTI images to 128 x 128 x 64 voxels at the time of loading, this paramter should be [128, 128, 64]. Note that this is applied immediately after resample_spacing if specified (see above), which is immediately after loading the data from disk. This parameter can have complex interactions with both resample_spacing and with other parameters that act at later stages of the input function.

## Training parameters
The following parameters pertain to training and inference inputs:
```yaml
input_function: PatchInputFn3D
data_plane: ax
train_dims: [80, 80, 80]
train_patch_overlap: [4, 4, 4]
infer_dims: [232, 232, 232]
infer_patch_overlap: [1, 1, 1]
augment_train_data: true
label_interp: 1
mask_weights: false
metrics: [mae]
```
### Explanations
```yaml
input_function: PatchInputFn3D
```
A string corresponding to a registered subclass in utilities/input_functions.py. The input function to be used for loading data into the network.
```yaml
data_plane: ax
```
The data plane for the input images. Must be "ax", "cor", or "sag". If ax, the data is not rotated, if cor or sag the data is rotated assuming it was originally in axial orientation.
```yaml
train_dims: [80, 80, 80]
```
The dimensions for the train patch. Number of elements corresponds to dimensionality of the training inputs. This value will be passed to extract_image_patches or extract_volume_patches. If patch training is not desired, simply make the patches bigger than the input images and set overlap to 1.
```yaml
train_patch_overlap: [4, 4, 4]
```
A list of numbers that serve as the divisior for determining the stride passed to extract_image_patches or extract_volume_patches. Number of elements corresponds to the dimensionality of the training inputs. The training dimension is divided by the corresponding overlap value to determine the stride. For stride=1 (maximum overlap) pass the same value as the corresponding train dim.
```yaml
infer_dims: [232, 232, 232]
```
The dimensions for the infer patch. Number of elements corresponds to dimensionality of the infer inputs. This value will be passed to extract_image_patches or extract_volume_patches. If patch inference is not desired, simply make the patches bigger than the input images.
```yaml
infer_patch_overlap: [1, 1, 1]
```
A list of numbers that serve as the divisior for determining the stride passed to extract_image_patches or extract_volume_patches. Number of elements corresponds to the dimensionality of the infer inputs. The infer dimension is divided by the corresponding overlap value to determine the stride. For stride=1 (maximum overlap) pass the same value as the corresponding train dim.
```yaml
augment_train_data: True
```
Whether or not to randomly rotate the training data (rotation amount and axes is determined by the input function). Must be boolean.
```yaml
label_interp: 1
```
The order of interpolation to be used when randomly rotating the label data. This only matters if augment_train_data == yes. 0 = nearest neighbor, 1 = linear, 2 = cubic, etc
```yaml
mask_weights: False
```
If true then mask is used as a weight tensor for the loss function during training. If a list is passed, then the values in the mask are mapped to the values in the list. For example, passing [1, 2, 4] would map mask value 0->1, 1->2, 2->4.
```yaml
metrics: [mae]
```
A string or list of strings corresponding to registered metric methods in utilities/metrics.py. These are the metrics to be collected during training.

## Normalization parameters
The following parameters pertain to data normalization:
```yaml
norm_data: True
norm_labels: False
norm_mode: zscore
```
### Explanations
```yaml
norm_data: True
```
Whether or not to normalize the input data.
```yaml
norm_labels: False
```
Whether or not to normalize the label data.
```yaml
norm_mode: zscore
```
A string corresponding to registered normalization methods in utilities/normalizers.py. Applies to both data and labels (if norm_data and norm_labels are true).

## Model parameters
The following parameters define the model:
```yaml
model_name: Unet3DBneck
base_filters: 32
output_filters: 1
layer_layout: [3, 4, 5]
final_layer: conv
kernel_size: [3, 3, 3]
data_format: channels_last
activation: leaky_relu
mixed_precision: False
dist_strat: mirrored
```
### Explanations
```yaml
model_name: Unet3DBneck
```
A string corresponding to a registered network subclass in model/networks.py. This is the name of the network that will be used.
```yaml
base_filters: 32
```
The base number of filters used in network construction. Should be divisible by 4 when using bottleneck layers. Should also be a multiple of 8 if tensor core usage is desired (relevant for mixed precision).
```yaml
output_filters: 1
```
The number of output filters for the final layer of the network. For example, 2 could be used for a binary categorization task, 1 could be used for a synthetic output task.
```yaml
layer_layout: [3, 4, 5]
```
The layer layout parameter used during network construction. This is used differently by each network in net_builder.py
```yaml
final_layer: conv
```
The final layer to be applied to the model outputs. Can be any in ["sigmoid", "softmax", "conv"]
```yaml
kernel_size: [3, 3, 3]
```
The kernel size used during network construction.
```yaml
data_format: channels_last
```
The tensorflow data format. Must be either "channels_last" or "channels_first". Either should work regardless of the model or input data; however, channels first is incompletely tested.
```yaml
activation: leaky_relu
```
A string corresponding to a registered activation method in utilities/activations.py. The activation function used during network construction.
```yaml
mixed_precision: False
```
Whether or not to use mixed precision float for tensorflow ops to take advantage of Nvidia tensor cores. Only useful if CUDA compute capability is 7.0 or higher.
```yaml
dist_strat: mirrored
```
Specify a distribution strategy for multi-gpu training. Currently only supports "Mirrored" (case insensitive) with everything else resulting in default no distribution.

## Model hyperparameters
The following parameters define additional model hyperparameters:
```yaml
shuffle_size: 1600
batch_size: 8
num_threads: 6
samples_per_epoch: 32000
train_fract: 0.9
test_fract: 0.0
learning_rate: [0.001]
learning_rate_decay: constant
optimizer: adam
loss: MSE
num_epochs: 25
dropout_rate: 0.3
```
### Explanations
```yaml
shuffle_size: 1600
```
The size of the shuffle buffer. This should be much higher than the number of training patches that come from a single image series, or else it won't really be shuffling the data.
```yaml
batch_size: 8
```
The number of patches/samples per training batch. Should be a multiple of 8 for tensor core usage.
```yaml
num_threads: 6
```
The number of threads to use for image input function processing.
```yaml
samples_per_epoch: 32000
```
The number of individual training samples per epoch. This gets divided by batch_size to determine how many training steps there will be per epoch. Note that when loading pre-generated datasets, this value is not used - rather this value will be determined based on the 'cardinality' of the pre-generated dataset.
```yaml
train_fract: 0.9
```
The fraction of input series/directories to be used for training. The rest (minus any held for testing (see below)) are used for training validation.
```yaml
test_fract: 0.0
```
The fraction of input series/directories to be held out for evaluation after training is complete. Can be zero if you don't want to hold out a test set. The remaining fraction (1 - (train_fact + test_fract)) are used for training validation. This cannot be larger than 1 - train_fract. This fraction is used by evaluate.py (but not by train.py).
```yaml
learning_rate: [0.001]
```
The learning rate for the model. This is used differently by the different learning rate functions.
```yaml
learning_rate_decay: constant
```
A string corresponding to a registered learning rate method in utilities/learning_rates.py. The learning rate decay function to use.
```yaml
loss: MSE
```
A string corresponding to a registered loss method in utilities/losses.py. The loss function to use.
```yaml
optimizer: adam
```
A string corresponding to a registered optimizer method in utilities/optimizers.py. The optimizer to use during training. This must be specified in optimizers.py by optimizer_picker.
```yaml
num_epochs:25
```
The number of training epochs to use. I.e. the number of times to iterate through all of the training data.
```yaml
dropout_rate: 0.3
```
The dropout fraction. This is only used if the network speficied in model_name has a dropout function.
