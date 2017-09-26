# Face Recognition Server

## Requirements
Install [Git LFS](https://git-lfs.github.com/)
Install [Docker](https://www.docker.com/community-edition)

## Configure

Change in `config.sh` `FACE_DATASET` to your dataset location

## Install & run

```bash
git clone https://github.com/firemind/face-recog-server-python
cd face-recog-server-python/
./build.sh
./start.sh
```

## Test

```bash
./test.sh
```

## Train

```bash
./train.sh
```
