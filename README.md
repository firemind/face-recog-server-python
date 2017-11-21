# Face Recognition Server

## Requirements
Install [Docker](https://www.docker.com/community-edition)
Install [Docker-Compose](https://docs.docker.com/compose/)

## Install & run

```bash
git clone https://github.com/firemind/face-recog-server-python
cd face-recog-server-python/
docker-compose up
```

## Architecture

![Diagram](https://github.com/firemind/face-recog-server-python/raw/master/face_recog_server.png "Architecture")

## Documentation

![Wiki](https://github.com/firemind/facial-recognition-sa-2017/wiki)


## Running Test Scripts

### Configure

Adjust `config.sh` to match the location of your dataset and server URL.


### Test

```bash
cd test
./test_tracking.sh
```
