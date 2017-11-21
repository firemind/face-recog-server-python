# Face Recognition Server

## Requirements

* Install [Docker](https://www.docker.com/community-edition)
* Install [Docker-Compose](https://docs.docker.com/compose/)
* Download [Pre-Trained Facenet Model](https://drive.google.com/file/d/0B5MzpY9kBtDVOTVnU3NIaUdySFE) and place it in ~/models/facenet/dl-20170511-185253/ (configurable in docker-compose.yml)

## Install & run

```bash
git clone https://github.com/firemind/face-recog-server-python
cd face-recog-server-python/
docker-compose up
```

## Architecture

![Diagram](https://github.com/firemind/face-recog-server/raw/master/figures/face_recog_full.png "Architecture")


## Documentation

![Wiki](https://github.com/firemind/facial-recognition-sa-2017/wiki)


## Running Test Scripts

### Test

```bash
cd test
./test_tracking.sh
```


## Acknowledgements

Face embeddings are calculated using the Tensorflow-based facenet https://github.com/davidsandberg/facenet
