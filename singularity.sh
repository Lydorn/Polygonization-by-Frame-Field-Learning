#!/usr/bin/env bash

module load singularity/2.5.2  # Load module on cluster
singularity shell -B /local -B /data --nv ./singularity/frame-field-learning_1.2.simg