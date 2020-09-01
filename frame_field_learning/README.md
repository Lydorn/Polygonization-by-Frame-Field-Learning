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


## local_utils.py

This script holds several utility functions for the project that do not really belong elsewhere.


## losses.py

The script defines a ```Loss``` class that is the base class to be used for all loss functions. 
It makes it easy to compute normalized losses for easier balancing. 
It thus also defines a ```MultiLoss``` class to combine all necessary losses and apply multiplicative coefficients to their normalized version.

All individual losses (segmentation loss, frame field align loss, frame field smooth loss, etc.) are defined as classes inheriting from the ```Loss``` class.

Lastly the ```build_combined_loss``` function instantiates a ```MultiLoss``` object with all required losses depending on the specified config file.


## measures.py

This script defines the ```iou``` function to compute the Intersection over Union (Iou), also calles the Jaccard index, and 
the ```dice``` function to compute the differentiable version of the IoU in the form of the Dice coefficient.


## model.py

This script defines the ```FrameFieldModel``` class which implements the final network. 
It needs a backbone to be specified model and adds the necessary convolutions for the segmentation and frame field outputs.
Its forward method also performs transforms to be done on the GPU before pushing the result into the backbone. 


## plot_utils.py

This script defines several functions to plot segmentations, frame fields, polygons and polylines for visualization.


## polygonize.py

This script is the entrypoint for all polygonization algorithms. It can then call these polygonization methods:
- Active Contours Model (ACM) from polygonize_acm.py
- Active Skeleton Model (ASM) from polygonize_asm.py
- Simple polygonization (Marching Cubes contour detection + Ramer-Douglas-Peucker simplification) from polygonize_simple.py


## polygonize_acm.py

This script implements the Active Contours Model (ACM) polygonization algorithm. 
It starts with contours detected with Marching Squares which are then optimized on the GPU with PyTorch to align to the frame field (in addition to other objectives, see paper).
It does not handle common walls between adjoining buildings. 
For that reason we have also developed the Active Skeleton Model (ASM), see next section.


## polygonize_asm.py

This script implements the Active Skeleton Model (ASM) polygonization algorithm. 
It starts with a skeleton graph detected from the wall segmentation map. 
It is then optimized on the GPU with PyTorch to align to the frame field (in addition to other objectives, see paper).
It can also be initialized with Marching Squares and essentially implements the ACM as well. 
Thus, the polygonize_acm.py script should be redundant 
(we still keep it around as it follows a different approach to the data structure which can be interesting).


## polygonize_simple.py

This script implements the simple polygonization (Marching Cubes contour detection + Ramer-Douglas-Peucker simplification) used as baseline in the paper.


## polygonize_utils.py

This scripts implements a few utility functions used by several of the above polygonization methods. 
For example, the marching squares contour detection and computation of polygon probability computed from the building interior probability map.


## save_utils.py

This script implements functions for saving results produced by the network and polygonization in various formats.


## train.py

This script defines the ```train``` function which sets up the traiing procedure. 
It instantiates a ```Trainer``` object which will then run the optimization.

## trainer.py

This script implements the ```Trainer``` class which is responsible for training the given model.
It implements multi-GPU training, loss normalization, restarting traiing from a checkpoint, etc.


## tta_utils.py

When performing inference with the model, 
Test Time Augmentation (TTA) can be used to increase the quality of the result.
We augment the input 8 times with right-angled rotations and flips. 
The outputs are then merged with a given aggregation function. 
We implemented several but it seems averaging works best.
The aggregation function is used on the 8 segmentation maps.
However, for the frame field, neither the uv nor the c0, c2 representations can be averaged easily (because of non-linearity).
Thus, we first search for the one segmentation out of the 8 ones that agress the most with the aggregated one.
Then the corresponding frame field is selected.
TTA is performed in the ```forward``` method of ```FrameFieldModel```.


## unet.py

This script implements the ```UNetBackbone``` class for the original U-Net network to be used as a backbone.


## unet_resnet.py

This script is an adapted code from https://github.com/neptune-ai/open-solution-mapping-challenge/blob/master/src/unet_models.py
to use the ResNet network as an encoder for a U-Net. This is used to instantiate the Unet-Resnet101 backbone we use in the paper.


### :warning: TODO: complete this README

