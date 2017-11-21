#!/bin/bash

function request {
  image=$1
  echo $image
  curl -F "image=@$image" -F "data={\"location\":\"cam1\", \"positions\":[{\"width\":100,\"height\":100,\"left\":10,\"top\":10}]}" http://localhost:3000/track
  echo 
}
export -f request

#find /media/mike/HD/our_dataset_verification/ -type f -exec bash -c 'request "$0"' "{}" \;
find tracking/ -type f -exec bash -c 'request "$0"' "{}" \;
