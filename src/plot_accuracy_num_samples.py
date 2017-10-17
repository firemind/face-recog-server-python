from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
from Tkinter import *
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle
from sklearn.svm import SVC
from sklearn import linear_model
import random
import time
import tensorflow as tf
import numpy as np
import sys
import argparse
import facenet
from face_mind import FaceMind

tf.Graph().as_default()
sess = tf.Session()

face_mind = FaceMind(sess)

loss_types =['log']

def main():
  global sess
  with sess as sess:
    np.random.seed(seed=1234)
    max_samples=7
    verify_on=3
    faceapi_results = load_faceapi_results(args.face_api_result)
    face_mind.load_model(args.model)
    test_sizes = range(1, max_samples+1)
    results = []
    results_all = {}
    dataset = facenet.get_dataset(args.data_dir)
    for test_size in test_sizes:
      train_set, test_set = split_dataset(dataset, verify_on, test_size)
      face_mind.train_on_dataset(train_set)
      paths, labels = facenet.get_image_paths_and_labels(test_set)
      # SVM model fitting
      face_mind.model = SVC(kernel='linear', probability=True)
      face_mind.fit()
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

      for t in loss_types:
        # linear model fitting
        face_mind.model = linear_model.SGDClassifier(loss=t)
        face_mind.fit()
        res = face_mind.classify_all(paths)
        correct = 0.0
        for i in range(0,len(res)):
          prediction, score = res[i]
          if prediction == face_mind.class_names[labels[i]]:
            correct += 1
        acc = correct/len(res)
        results_all.setdefault(t, [])
        results_all[t].append(acc)

    print("|Num Samples       | "+" | ".join(map(str,test_sizes))+"|")
    print("|------------------|"+"---|"*len(test_sizes))
    print("| Accuracy SVM     | "+" | ".join(map(str,results))+"|")
    print("| Accuracy LOG     | "+" | ".join(map(str,results_all['log']))+"|")
    print("| Accuracy FaceAPI | "+" | ".join(map(str,faceapi_results))+"|")
    red_patch = mpatches.Patch(color='red', label='SVC (Facenet)')
    blue_patch = mpatches.Patch(color='blue', label='SGD (Log)')
    green_patch = mpatches.Patch(color='green', label='FaceAPI')
    plt.legend(handles=[red_patch, blue_patch,green_patch])
    plt.plot(test_sizes, results, 'ro', test_sizes, results_all['log'], 'bo', test_sizes, faceapi_results, 'go')
    plt.show()

def load_faceapi_results(filename):
  with open(filename, 'r') as f:
    result = json.load(f)
  val = []
  for r in result:
    val.append(r["successrate"])
  return val

def split_dataset(dataset, nrof_test_images_per_class, nrof_train_images_per_class):
  train_set = []
  test_set = []
  for cls in dataset:
    paths = cls.image_paths
    # Remove classes with less than min_nrof_images_per_class
    if len(paths)>=nrof_test_images_per_class + nrof_train_images_per_class:
      # np.random.shuffle(paths)
      train_set.append(facenet.ImageClass(cls.name, paths[nrof_test_images_per_class:nrof_test_images_per_class + nrof_train_images_per_class]))
      test_set.append(facenet.ImageClass(cls.name, paths[:nrof_test_images_per_class]))
  return train_set, test_set

def parse_arguments(argv):
  parser = argparse.ArgumentParser()

  parser.add_argument('--model', type=str,
                      help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file',
                      default="/base-model")
  parser.add_argument('--face_api_result', type=str,
                      help="Path to result.json from cortana.py")
  parser.add_argument('--data_dir', type=str,
                      help='Path to the data directory containing aligned LFW face patches.', default="/face-data/")

  return parser.parse_args(argv)


if __name__ == '__main__':
  args = parse_arguments(sys.argv[1:])
  main()
