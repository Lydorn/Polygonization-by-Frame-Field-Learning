# Introduction

This repository contains a Dockerfile with all the required depedencies for this project installed within.

# What is Docker and NVIDIA docker?

From the [official Docker website](https://www.docker.com/what-docker):
> Docker containers wrap a piece of software in a complete filesystem that contains everything needed to run: code, 
runtime, system tools, system libraries – anything that can be installed on a server. This guarantees that the software 
will always run the same, regardless of its environment.

More high-level information about Docker can be found on the 
[Moby project README](https://github.com/moby/moby/blob/master/README.md). Technical documentation can be found: [docs.docker.com](https://docs.docker.com/).

NVIDIA Docker allows docker containers to access the GPU.

While all the necessary instructions/commands to use Docker are given in this README, it is very beneficial to
know what is going on. There's a tutorial playlist of YouTube videos which explains Docker really well:
[Docker tutorials by takacsmark
](https://www.youtube.com/watch?v=Vyp5_F42NGs&list=PLX0Ak4vUBQfC6S8egys9kx6uy6tpw5yDX).

# Install Docker and NVIDIA Docker

## Step 1: Install docker
Follow the [Docker install instructions](https://docs.docker.com/engine/installation).

## Step 2: Install nvidia-docker2
Install nvidia-docker 2.0 by following these
[NVIDIA Docker install instructions](https://github.com/NVIDIA/nvidia-docker).

# Dockerfiles

To build a Dockerfile, make sure your working directory is the same as the Dockerfile you want to build is in.

You can execute the build.sh script:
```
sh build.sh
```

Or enter this command, replacing <name of image> with a name of your choosing:
```
docker build -t <name of image>
```

Then to run a container: enter this command by replacing the various placeholders:
```
docker run --rm -it --init --gpus all --ipc=host --network=host-e NVIDIA_VISIBLE_DEVICES=0 -v <host folder>:<container folder> <name of image>
```

# Useful Docker commands

During building of an image, some intermediary images are created. Those are called dangling images because they are 
not used afterwards. Run this command to remove them (and save some disk space):
```
docker rmi $(docker images -f dangling=true -q)
```

Display all Docker images (the -a option displays intermediary images as well):
```
docker images [-a]
```

Display all running Docker containers (the -a option also displays non-running containers):
```
docker ps [-a]
```

Remove container:
```
docker rm <container name>
```

Remove image:
```
docker rmi <image name>
```

# Troubleshooting

When installing Docker on Fedora, it may happen that the Docker deamon does not launch at computer start. Enter this command to start docker:
```
sudo systemctl start docker
```
