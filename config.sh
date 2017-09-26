#!/bin/bash
# Change the location of your dataset
FACE_DATASET=/var/face_dataset/
FACE_DATASET=${FACE_DATASET%/}

export FACE_DATASET
