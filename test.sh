#!/bin/bash
source $PWD/config.sh
IMAGE="$FACE_DATASET/Bill Cosby/Bill Cosby-0002.png"
curl -F "image=@$IMAGE" http://localhost:5000/classify
echo 
