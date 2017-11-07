from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import argparse
import os
import sys
from face_mind import FaceMind

face_mind = FaceMind()

def main():
  np.random.seed(seed=args.seed)

  if os.path.isfile(args.classifier_filename):
    face_mind.load_classifier(args.classifier_filename)
  else:
    face_mind.train(args.data_dir)
    face_mind.save_classifier(args.classifier_filename)


def parse_arguments(argv):
  parser = argparse.ArgumentParser()

  parser.add_argument('--classifier_filename',
                      help='Classifier model file name as a pickle (.pkl) file. ' +
                           'For training this is the output and for classification this is an input.',
                      default="/svm-model/model.pkl")
  parser.add_argument('--batch_size', type=int,
                      help='Number of images to process in a batch.', default=90)
  parser.add_argument('--seed', type=int,
                      help='Random seed.', default=666)
  parser.add_argument('--min_nrof_images_per_class', type=int,
                      help='Only include classes with at least this number of images in the dataset', default=20)
  parser.add_argument('--nrof_train_images_per_class', type=int,
                      help='Use this number of images from each class for training and the rest for testing',
                      default=10)
  parser.add_argument('--data_dir', type=str,
                    help='Path to the data directory containing aligned LFW face patches.', default="/face-data/")

  return parser.parse_args(argv)


if __name__ == '__main__':
  args = parse_arguments(sys.argv[1:])
  main()
