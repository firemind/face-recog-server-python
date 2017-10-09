from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from Tkinter import *
import matplotlib.pyplot as plt
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
loss_types =['log', 'modified_huber']

def main():
  global sess
  with sess as sess:
    np.random.seed(seed=1234)
    num_samples=10
    verify_on=10
    face_mind.load_model(args.model)
    test_sizes = [2,3,4,5,10,15,20,30,40,50,60,70]
    results = []
    results_all = {}
    dataset = facenet.get_dataset(args.data_dir)
    for test_size in test_sizes:
      train_set, test_set = split_dataset(dataset, verify_on, num_samples, test_size)
      face_mind.train_on_dataset(train_set)
      paths, labels = facenet.get_image_paths_and_labels(test_set)
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

    print("|Num Samples   | "+" | ".join(map(str,test_sizes))+"|")
    print("|--------------|"+"---|"*len(test_sizes))
    print("| Accuracy SVM | "+" | ".join(map(str,results))+"|")
    print("| Accuracy LOG | "+" | ".join(map(str,results_all['log']))+"|")
    print("| Accuracy HUB | "+" | ".join(map(str,results_all['modified_huber']))+"|")
    plt.plot(test_sizes, results, 'ro', test_sizes, results_all['log'], 'bo', test_sizes, results_all['modified_huber'], 'go')
    plt.show()

def split_dataset(dataset, nrof_varification_images_per_class, nrof_train_images_per_class, nrof_classes):
  train_set = []
  test_set = []
  classes_added=0
  for cls in dataset:
    paths = cls.image_paths
    # Remove classes with less than min_nrof_images_per_class
    if len(paths)>=nrof_varification_images_per_class+nrof_train_images_per_class:
      np.random.shuffle(paths)
      train_set.append(facenet.ImageClass(cls.name, paths[:nrof_train_images_per_class]))
      test_set.append(facenet.ImageClass(cls.name, paths[nrof_train_images_per_class:(nrof_train_images_per_class+nrof_varification_images_per_class)]))
      classes_added+=1
      if classes_added >= nrof_classes:
        break
  print("Train set %i and Test set %i" % (len(train_set), len(test_set)))
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
