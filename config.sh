#!/bin/bash
# Change the location of your dataset
FACE_DATASET=/var/face_dataset/
FACE_DATASET=${FACE_DATASET%/}
export SERVER_URL=http://localhost:5000

export FACE_DATASET
