# Introduction

This "paper" folder contains scripts used to generate certain figures/results for our paper.
These scripts use data saved by launching the ```main.py``` with ```--mode=eval``` or with ```--mode=eval_coco```.

We introduce each script in the following.

## convert_stats_to_latex.py

When launching ```main.py``` with ```--mode=eval_coco```, it will compute COCO measures (AP/AR and their variants) and save them in .json files with filenames "*.stats.*.annotation.*.json".
Use this script to print those metrics in LaTex code ready to be copy-pasted into tables.

Run as(any filename matching "*.stats.*.annotation.*.json" under the given ```dirpath``` will be printed):
```
convert_stats_to_latex.py --dirpath <Path to eval directory>
```

## plot_complexity_fidelity.py

Use this script to plot the complexity vs fidelity figures.

Run as (no arguments, will have to change paths directly in the "main" function of the script):
```
plot_complexity_fidelity.py
```

## plot_contour_angle_hist.py

This script is used to compute and plot the "relative angle distribution" histograms from contours detected in segmentation probability maps.
It is an additional measure for building regularity that we propose. It does not need any
ground truth annotation so that results are evaluated on their own. 
We use the simple polygonization method to obtain contours from the segmentation map. The
minimum rotated rectangle is computed for each building. Then the relative angle between each
contour edge and the principal axis of the associated minimum rotated rectangle is computed. For
a collection of contours, we aggregate the data in the form of a distribution of relative angles.
If the distribution is more homogeneous, it means buildings are less regular, i.e. smoother.
Conversely, if the distribution has peaks around certain relative angle values (which are expected
to be 0°, 90°, and 180° for buildings), it means buildings are more regular, with sharper corners
having similar angles.

Run as (no arguments, will have to change paths directly in the "main" function of the script):
```
plot_contour_angle_hist.py
```

## plot_contour_metrics.py

This script is used to plot our "max tangent angle error" metric. 
All computations are done when launching ```main.py``` with ```--mode=eval_coco``` which generates .json files with filenames "test.metrics.test.annotation.poly.*.json".
This script takes as input the path to the "eval directory" where the results for all runs are saved and where these .json files will be found.

Example use (filenames to specific runs and .json result metrics should be changed directly in the "main" function of the script):
```
plot_contour_metrics.py --dirpath <Path to eval directory>
```

## show_result_image.py

This script uses the pycocotools API to plot result contours in [polygon] format on top of test images.

Run as (no arguments, will have to change paths directly in the "main" function of the script):
```
plot_contour_angle_hist.py
```