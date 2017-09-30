#!/bin/bash
source $PWD/config.sh
#docker run --rm -d -v $PWD/tf-model:/tf-model -p 9000:9000 facenet-serving 
docker run --rm -it -v $SVM_MODEL:/svm-model/ -v $FACENET_MODEL:/base-model -v $PWD/src:/app -v $FACE_DATASET:/face-data/ -p 5000:5000 --entrypoint="" facenet-rest-server python /app/train.py
