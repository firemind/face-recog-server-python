#docker run --rm -d -v $PWD/tf-model:/tf-model -p 9000:9000 facenet-serving 
docker run --rm -it -v $PWD/facenet-model:/base-model -v $PWD/src:/facenet -p 5000:5000 facenet-rest-server
