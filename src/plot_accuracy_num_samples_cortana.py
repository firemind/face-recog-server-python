from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from Tkinter import *
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle
import random
import time
import numpy as np
import sys
import argparse


def main():
  np.random.seed(seed=1234)
  max_samples=10
  verify_on=10
  test_sizes = range(1, max_samples+1)
  results = []
  dataset = facenet.get_dataset(args.data_dir)
  for test_size in test_sizes:
    train_set, test_set = split_dataset(dataset, test_size+verify_on, test_size)
    face_mind.train_on_dataset(train_set)
    paths, labels = facenet.get_image_paths_and_labels(test_set)
    # SVM model fitting
    print("Verifying on %i paths" % len(paths))
    res = face_mind.classify_all(paths)
    correct = 0.0
    for i in range(0,len(res)):
      prediction, score = res[i]
      if prediction == face_mind.class_names[labels[i]]:
        correct += 1

    acc = correct/len(res)
    print("Accuracy for %i: %f" % (test_size, acc))
    results.append(acc)

  print("|Num Samples| "+" | ".join(map(str,test_sizes))+"|")
  print("|-----------|"+"---|"*len(test_sizes))
  print("| Accuracy  | "+" | ".join(map(str,results))+"|")
  red_patch = mpatches.Patch(color='red', label='Cortana')
  plt.legend(handles=[red_patch])
  plt.plot(test_sizes, results, 'ro')
  plt.show()

def split_dataset(dataset, min_nrof_images_per_class, nrof_train_images_per_class):
  train_set = []
  test_set = []
  for cls in dataset:
    paths = cls.image_paths
    # Remove classes with less than min_nrof_images_per_class
    if len(paths)>=min_nrof_images_per_class:
      np.random.shuffle(paths)
      train_set.append(facenet.ImageClass(cls.name, paths[:nrof_train_images_per_class]))
      test_set.append(facenet.ImageClass(cls.name, paths[nrof_train_images_per_class:min_nrof_images_per_class]))
  return train_set, test_set

def parse_arguments(argv):
  parser = argparse.ArgumentParser()

  parser.add_argument('--model', type=str,
                      help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file',
                      default="/base-model")
  parser.add_argument('--data_dir', type=str,
                      help='Path to the data directory containing aligned LFW face patches.', default="/face-data/")

  return parser.parse_args(argv)


if __name__ == '__main__':
  args = parse_arguments(sys.argv[1:])
  main()
