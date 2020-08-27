This folder contains all the sources files of the "frame_field_learning" Python package.

We briefly introduce each script in the following.


## data_transform.py

Contains functions that return transformations applied to input data before feeding it to the network.
It includes pre-processing (whose result is store on disk), 
CPU transforms which are applied on the loaded pre-processed data before being transferred to the GPU, 
and finally GPU transforms which are applied on the GPU for speed (such as data augmentation).


## evaluate.py

Defines the "evaluate" function called by ```main.py``` when ```--mode=eval``` which setups and instantiates an Evaluator object whose evaluate method is then called.


## evaluator.py

Defines the "Evaluator" class used to run inference on a trained model followed by computing some measures and saving all results for all samples of a evaluation fold of a dataset.


## frame_field_utils.py

This script defines all "frame field"-related functions. 
For example the ```framefield_align_error``` function is used to compute the "align", 
"align90" losses as well as the "frame field align" energy for ASm optimization in our paper.

The ```LaplacianPenalty``` class is used for the "smooth" loss to ensure a smooth frame field.

Both ```compute_closest_in_uv``` and ```detect_corners``` are use to detect corners sing the frame field.
They are somewhat redundant but the first is applied on torch tensors while the second is applied on numpy arrays.


## ictnet.py

This script implements ICTNet (https://theictlab.org/lp/2019ICTNet) in PyTorch to try and add frame field learning to ICTNet.
We can use it as a backbone model with the ```ICTNetBackbone``` class.

## inference.py

This script defines the ```inference``` function to run a trained model on one image tile. It is used by the evaluation code.

If the "patch_size" parameter is set in the "eval_params", then the image tile is split in small patches of size "patch_size", 
inference is run on bacth_size patches at a time, 
finally the result tile is stoched together from the results of all patches.
Typically those patches overlap by a few pixels and the result is linearly interpolated in overlapping areas.

If there is no "patch_size" parameter, inference is run directly on the whole image tile.


## inference_from_filepath.py

This script defines the ```inference_from_filepath``` function which is called by ```main.py``` when the ```--in_filepath``` argument is set.
It runs inference + polygonization on the images specified by the ```--in_filepath``` argument and saves the result.

### :warning: TODO: complete this README

