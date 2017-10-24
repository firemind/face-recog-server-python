#!/bin/bash
source $PWD/../config.sh
#IMAGE="/media/mike/HD/our_dataset_verification/Random/Random.jpg"

function request {
  NAME="Alissa White-Gluz"
  image=$1
  echo $image
  curl -F "image=@$image" -F "label=$NAME" -F "align=true" $SERVER_URL/store
  echo 
}
export -f request

#find /media/mike/HD/our_dataset_verification/ -type f -exec bash -c 'request "$0"' "{}" \;
find store -type f -exec bash -c 'request "$0"' "{}" \;
