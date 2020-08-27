# Introduction

I explain here how to use Singularity with Docker images (on Fedora) if ti is needed.

# Install Singularity on Fedora

I had to install these dependencies:
```
dnf install libarchive-devel
dnf install squashfs-tools
```

Then follow [Quick installaton steps](https://www.sylabs.io/guides/2.6/user-guide/quick_start.html#quick-installation-steps) from Singularity 2.6 docs:
```
git clone https://github.com/sylabs/singularity.git

cd singularity

git fetch --all

git checkout 2.6.0

./autogen.sh

./configure --prefix=/usr/local

make

sudo make install
```

# Build a Singularity image from a local Docker image:

The Docker image has to put on a registry for Singularity to use it. Usually this is the Docker hub registry but you can use a local one too:

Create local registry:
```
docker run -d -p 5000:5000 --restart=always --name registry registry:2
```

Push local docker container to it:
```
docker tag <image_name> localhost:5000/<image_name>
docker push localhost:5000/<image_name>
```

Create a Singularity def file:
```
Bootstrap: docker
Registry: http://localhost:5000
Namespace:
From: <image_name>
```

Build singularity image:
```
sudo SINGULARITY_NOHTTPS=1 singularity build <image_name>.simg Singularity
```

# Send image to cluster

```
scp <image_name>.simg nef-devel:<target_path>
scp frame-field-learning_1.2.simg nef-devel:frame_field_learning/singularity/
```