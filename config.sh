#!/bin/bash
# Change the location of your dataset
FACE_DATASET=/media/mike/HD/our_dataset
SVM_MODEL=$PWD/svm-model
FACENET_MODEL=$PWD/svm-model
FACE_DATASET=${FACE_DATASET%/}
FACENET_MODEL=~/models/facenet/dl-20170511-185253
export SERVER_URL=http://localhost:5000

export FACE_DATASET
