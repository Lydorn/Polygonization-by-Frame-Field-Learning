### :warning: Under construction

All the code is there. A more detailed README is coming soon...

# Polygonal Building Segmentation by Frame Field Learning

This repo contains the official code for the paper:

**Regularized Building Segmentation by Frame Field Learning**\
[Nicolas Girard](https://www-sop.inria.fr/members/Nicolas.Girard/),
[Dmitriy Smirnov](https://people.csail.mit.edu/smirnov/),
[Justin Solomon](https://people.csail.mit.edu/jsolomon/),
[Yuliya Tarabalka](https://www-sop.inria.fr/members/Yuliya.Tarabalka/)\
IGARSS 2020\
**\[[paper](https://arxiv.org/pdf/2004.14875.pdf) (extended version)\]**

# Introduction

We add a frame field output to an image segmentation neural network to improve segmentation quality 
and provide structural information for a subsequent polygonization step. 
A frame field encodes two directions up to sign at every point of an image. 
To improve segmentation, we train a network to align an output frame field to the tangents of ground truth contours. 
In addition to increasing performance by leveraging the multi-task learning effect, 
our method produces more regular segmentations and is more robust due to the additional learning signal.

# Run

Execute [main.py] script to train a model, test a model or use a model on your own image.
See the help of the main script with:

```python main.py -help```

# Launch inference on one image

Make sure the run folder has the correct structure:

```
frame-field-learning
|-- runs
|   |-- <run_name> | yyyy-mm-dd hh:mm:ss
|   `-- ...
|-- main.py
|-- README.md (this file)
`-- ...
```

Execute the [main.py] script like so (filling values for arguments run_name and in_filepath):
```python main.py --run_name <run_name> --in_filepath <your_image_filepath>```

The outputs will be saved next to the input image

# Using Docker while developing the packages

In order to install the packages in develop mode (or editale mode) i.e. ```pip install -e <package_path>``` 
in the Docker container, execute the setup.sh script inside the Docker container after start-up.


# Git submodules

This project uses various git submodules. You have to pull all for the code to work.

See this tutorial on git submodules used with Python modules in dev mode: https://shunsvineyard.info/2019/12/23/using-git-submodule-and-develop-mode-to-manage-python-projects/

Further useful git submodules commands:

Clone a repository including its submodules:
```
git clone --recursive --jobs 8 <URL to Git repo>
```

If you already have cloned a repository and now want to load it’s submodules:
```
git submodule update --init --recursive --jobs 8
OR
git submodule update --recursive
```

Pull everything, including submodules:
```
git pull --recurse-submodules
```

Add a sudmodule:
```
git submodule add -b <branch_name> <URL to Git repo>
git submodule init
```

Update your submodule --remote fetches new commits in the submodules and updates the working tree to the commit described by the branch:
```
git submodule update --remote
```

The following example shows how to update a submodule to its latest commit in its master branch:
```
# update submodule in the master branch
# skip this if you use --recurse-submodules
# and have the master branch checked out
cd [submodule directory]
git checkout master
git pull

# commit the change in main repo
# to use the latest commit in master of the submodule
cd ..
git add [submodule directory]
git commit -m "move submodule to latest commit in master"

# share your changes
git push
```

Get the update by pulling in the changes and running the submodules update command:
```
# another developer wants to get the changes
git pull

# this updates the submodule to the latest
# commit in master as set in the last example
git submodule update
```

Remove submodule:
```
git rm the_submodule
rm -rf .git/modules/the_submodule
```