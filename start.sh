#!/bin/bash
source $PWD/config.sh

docker run --rm -it \
  -v $SVM_MODEL:/svm-model/ \
  -v $PWD/tmp:/tmp/ \
  -v $PWD/store:/store/ \
  -v $FACE_DATASET:/images/ \
  -v $FACE_DATASET:/face-data \
  -v $FACENET_MODEL:/base-model \
  -v $PWD/src:/app \
  -p 5000:5000 facenet-rest-server
