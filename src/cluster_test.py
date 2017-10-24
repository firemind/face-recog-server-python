from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import facenet
import os
import sys
import math
import pickle
from sklearn.svm import SVC
import time

from face_mind import FaceMind
from face_tracker import FaceTracker

face_mind = FaceMind()
face_tracker = FaceTracker()

def main(args):
  np.random.seed(seed=666)

  dataset = facenet.get_dataset(args.data_dir)

  paths, labels = facenet.get_image_paths_and_labels(dataset)

  # images = facenet.load_data(paths, False, False, face_mind.image_size)
  emb_array = face_mind.request_embedding(paths)
  print("Calculated Embeddings")
  for idx, emb in enumerate(emb_array):
    face_tracker.track(paths[idx], emb)

  for idx, label in enumerate(labels):
    pred = face_tracker.get_history(emb_array[idx])
    print(label)
    print(pred)


def parse_arguments(argv):
  parser = argparse.ArgumentParser()

  parser.add_argument('--data_dir', type=str,
                      help='Path to the data directory containing aligned LFW face patches.', default="/media/mike/HD/our_dataset_verification/")

  return parser.parse_args(argv)


if __name__ == '__main__':
  main(parse_arguments(sys.argv[1:]))
