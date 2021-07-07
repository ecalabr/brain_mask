# Param files
This guide explains the required parameters for param files.

## Setup parameters
The following parameters pertain to global setup of the model:
```json
"data_dir": "/data/gliogenomics/gliomarad",
"model_dir": "same",
"overwrite": true,
"random_state": 42,
```
### Explanations
```json
"data_dir": "/data/gliogenomics/gliomarad",
```
The full path to the directory contaning the individual study directories to be used for training the model.
```json
"model_dir": "same",
```
The full path to the directory for model outputs. If "same" is passed, then the data directory is used for model outputs.
```json
"overwrite": true,
```
Whether or not to overwrite model ouputs if the model is run again. Must be boolean. Note that specifying false will raise an error if a model output already exists in model_dir.
```json
"random_state": 42
```
An integer to be used as the seed for random operations that need to be reproducible (example = directory shuffling for train/test split

## Data input parameters
These parameters dictate how the input data is handled:
```json
"data_prefix": ["T1", "T1gad", "T2", "FLAIR"],
"label_prefix": ["brain_mask"],
"mask_prefix": ["brain_mask"],
"mask_dilate": [232, 232, 80],
"filter_zero": 0.2,
```
### Explanations
```json
"data_prefix": ["T1", "T1gad", "T2", "FLAIR"],
```
The prefixes for the input image files specified as a bracketed list. The expected filename format is directory-name_data-prefix.nii.gz i.e. 01234567_FLAIR.nii.gz.
```json
"label_prefix": ["brain_mask"],
```
The prefixes for the label image files specified as a bracketed list. The expected filename format is directory-name_label-prefix.nii.gz i.e. 01234567_labels.nii.gz.
```json
"mask_prefix": ["brain_mask"],
```
The prefixes for the input image mask file specified as a bracketed list. The expected filename format is directory-name_mask-prefix.nii.gz i.e. 01234567_brainmask.nii.gz. Can be false, in which case a mask of all 1s is used.
```json
"mask_dilate": [232, 232, 80],
```
How much to dilate the input mask after cropping the input images to the tight bounding box of the mask. If an integer, then all dimensions are symmetrically expanded by that amount (1/2 before, 1/2 after). If a list of 3 ints, then the input is symmetrically expanded such that its final dimensions are equal to the ints in the list. Each masked input image dimension is expanded to the size of the corresponding value. If the int is 0 or smaller than the corresponding masked input dim, then no expansion is performed in that dimension.
```json
"filter_zero": 0.2,
```
Filters out patches where labels == 0 in most of the patch. filter_zero is a float defining the minimum threshold for label pixels > 0 in the patch.

## Training parameters
The following parameters pertain to training and inference inputs:
```json
"data_plane": "ax",
"train_dims": [80, 80, 80],
"train_patch_overlap": [4, 4, 4],
"infer_dims": [232, 232, 232],
"infer_patch_overlap": [1, 1, 1],
"augment_train_data": true,
"label_interp": 1,
"mask_weights": false,
"metrics": ["mae"],
```
### Explanations
```json
"data_plane": "ax",
```
The data plane for the input images. Must be "ax", "cor", or "sag". If ax, the data is not rotated, if cor or sag the data is rotated assuming it was originally in axial orientation.
```json
"train_dims": [80, 80, 80],
```
The dimensions for the train patch. This must be a list of 2 (if using 2D mode) or 3 (if using 2.5D or 3D mode) ints. This value will be passed to extract_image_patches or extract_volume_patches. If patch training is not desired, simply make the patches bigger than the input images and set overlap to 1. For 2.5D mode, set the z dimension here and make sure the overlap for z is the same number, i.e. complete overlap.
```json
"train_patch_overlap": [4, 4, 4],
```
A list of 2 (if using 2D mode) or 3 (if using 2.5D or 3D mode) ints that serve as the divisior for determining the stride passed to extract_image_patches or extract_volume_patches. The training dimension is divided by the corresponding overlap value to determine the stride. For stride=1 (complete overlap) pass the same value as the corresponding train dim (this is used for 2.5D training to ensure stride 1 in z dimension).
```json
"infer_dims": [232, 232, 232],
```
The dimensions for the infer patch. This must be a list of 2 (if using 2D mode) or 3 (if using 2.5D or 3D mode) ints. This value will be passed to extract_image_patches or extract_volume_patches. If patch inference is not desired, simply make the patches bigger than the input images. Patch inference is only supported for 3D networks. 2D and 2.5D networks should use infer dims that are larger than the inputs.
```json
"infer_patch_overlap": [1, 1, 1],
```
A list of 2 (if using 2D mode) or 3 (if using 2.5D or 3D mode) ints that serve as the divisior for determining the stride passed to extract_image_patches or extract_volume_patches. The inference dimension is divided by the corresponding overlap value to determine the stride. For stride=1 (complete overlap) pass the same value as the corresponding infer dim (this is used for 2.5D inference to ensure stride 1 in z dimension).
```json
"augment_train_data": true,
```
Whether or not to randomly rotate the training data (in the axial plane). Must be boolean.
```json
"label_interp": 1,
```
The order of interpolation to be used when randomly rotating the label data. This only matters if augment_train_data == yes. 0 = nearest neighbor, 1 = linear, 2 = cubic, etc
```json
"mask_weights": false,
```
If true then mask is used as a weight tensor for the loss function during training. If a list is passed, then the values in the mask are mapped to the values in the list. For example, passing [1, 2, 4] would map mask value 0->1, 1->2, 2->4.
```json
"metrics": ["mae"],
```
A string or list of strings corresponding to tf.keras.metrics function names. These are the metrics to be collected during training.

## Normalization parameters
The following parameters pertain to data normalization:
```json
"norm_data": true,
"norm_labels": false,
"norm_mode": "zscore",
```
### Explanations
```json
"norm_data": true,
```
Whether or not to normalize the input data.
```json
"norm_labels": false,
```
Whether or not to normalize the label data.
```json
"norm_mode": "zscore",
```
What normalization mode to use. Applies to both data and labels (if norm_data and norm_labels are true). Names correspond to identifier strings in the normalize function in input_fn_util.py.

## Model parameters
The following parameters define the model:
```json
"model_name": "unet_3d_bneck",
"base_filters": 32,
"output_filters": 1,
"layer_layout": [3, 4, 5],
"final_layer": "conv",
"kernel_size": [3, 3, 3],
"data_format": "channels_last",
"activation": "leaky_relu",
"mixed_precision": false,
"dist_strat": "mirrored",
```
### Explanations
```json
"model_name": "unet_3d_bneck",
```
The name of the network that is going to be trained. This must be defined in net_builder.py
```json
"base_filters": 32,
```
The base number of filters used in network construction. Should be divisible by 4 when using bottleneck layers. Should also be a multiple of 8 for tensor core usage.
```json
"output_filters": 1,
```
The number of output filters for the final layer of the network. For example, 2 could be used for a binary categorization task, 1 could be used for a synthetic output task.
```json
"layer_layout": [3, 4, 5],
```
The layer layout parameter used during network construction. This is used differently by each network in net_builder.py
```json
"final_layer": ["conv"],
```
The final layer to be applied to the model outputs. Can be any in {"sigmoid", "softmax", "conv"}
```json
"kernel_size": [3, 3, 3],
```
The kernel size used during network construction.
```json
"data_format": "channels_last",
```
The tensorflow data format. Must be either "channels_last" or "channels_first". Either should work regardless of the model or input data; however, channels first is incompletely tested.
```json
"activation": "leaky_relu",
```
The name of the activation function used during network construction. This must be defined in losses.py loss_picker function.
```json
"mixed_precision": true,
```
Whether or not to use mixed precision float for tensorflow ops to take advantage of Nvidia tensor cores. Only useful if CUDA compute capability is 7.0 or higher.
```json
"dist_strat": "mirrored",
```
Specify a distribution strategy for multi-gpu training. Currently supports "Mirrored" (case insensitive) with everything else resulting in default no distribution.

## Model hyperparameters
The following parameters define additional model hyperparameters:
```json
"shuffle_size": 1600,
"batch_size": 8,
"num_threads": 6,
"samples_per_epoch": 32000,
"train_fract": 0.9,
"learning_rate": [0.001],
"learning_rate_decay": "constant",
"optimizer": "adam",
"loss": "MSE",
"num_epochs": 25,
"dropout_rate": 0.3
```
### Explanations
```json
"shuffle_size": 1600,
```
The size of the shuffle buffer. This should be much higher than the number of training patches that come from a single image series, or else it won't really be shuffling the data.
```json
"batch_size": 8,
```
The number of patches/samples per training batch. Should be a multiple of 8 for tensor core usage.
```json
"num_threads": 6,
```
The number of threads to use for image input function processing.
```json
"samples_per_epoch": 32000,
```
The number of individual training samples per epoch. This gets divided by batch_size to determine how many training steps there will be per epoch.
```json
"train_fract": 0.9,
```
The fraction of input series/directories to be used for training. The rest are used for evaluation.
```json
"learning_rate": [0.001],
```
The learning rate for the model. This is used differently by the different learning rate functions defined in utils.py
```json
"learning_rate_decay": "constant",
```
The name of the learning rate decay function to use. This must be specified in utils.py learning_rate_picker.
```json
"loss": "MSE",
```
The name of the loss function to use. This must be specified in losses.py loss_picker.
```json
"optimizer": "adam"
```
The name of the optimizer to use. This must be specified in optimizers.py by optimizer_picker.
```json
"num_epochs":25,
```
The number of training epochs to use. I.e. the number of times to iterate through all of the training data.
```json
"dropout_rate": 0.3,
```
The dropout fraction. This is only used if the network speficied in model_name has a dropout function.
