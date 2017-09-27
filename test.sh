#!/bin/bash
source $PWD/config.sh
#IMAGE="/media/mike/HD/our_dataset_verification/Random/Random.jpg"

function request {
  image=$1
  echo $image
  curl -F "image=@$image" $SERVER_URL/classify
  echo 
}
export -f request

#find /media/mike/HD/our_dataset_verification/ -type f -exec bash -c 'request "$0"' "{}" \;
find test/ -type f -exec bash -c 'request "$0"' "{}" \;
