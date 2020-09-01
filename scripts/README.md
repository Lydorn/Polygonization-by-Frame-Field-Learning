# Introduction

This "scripts" folder contains stand-alone scripts for some useful tasks detailed below.

## mask_to_json.py

Use this script to convert .png segmentation masks from the Open Solution from the CrowdAI challenge 
(https://github.com/neptune-ai/open-solution-mapping-challenge) 
to the COCO .json format with RLE mask encododing.
Run as:
```
mask_to_json.py --mask_dirpath <path to directory with the png masks>  --output_filepath <path to the output .json COCO format annotation file>
```

## plot_framefield.py

Use this script to plot a framefield saved as a .npy file. Can be useful for visualization. 
Explanation about its arguments can be accessed with:
```
mask_to_json.py --help
```

## ply_to_json.py

Use this script to convert .ply segmentation polygons from the paper 
"Li, M., Lafarge, F., Marlet, R.: Approximating shapes in images with low-complexity polygons. In: CVPR (2020)" 
to the COCO .json format with [polygon] mask encoding. In order to fill the score field of each annotation in the COCO format, we also need access to segmentation masks.

Run as
```
ply_to_json.py --ply_dirpath <path to directory with the .ply files> --mask_dirpath <path to directory with the probability masks>  --output_filepath <path to the output .json COCO format annotation file>
```