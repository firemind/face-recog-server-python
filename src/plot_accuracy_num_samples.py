from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#from Tkinter import *
#import matplotlib.pyplot as plt
import pickle
from sklearn.svm import SVC
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


#image = face_mind.align(image)
#misc.imsave(image_path, image)


def main():
  global sess
  with sess as sess:
    np.random.seed(seed=1234)
    max_samples=10
    verify_on=10
    face_mind.load_model(args.model)
    test_sizes = range(1, max_samples+1)
    results = []
    dataset = facenet.get_dataset(args.data_dir)
    for test_size in test_sizes:
      train_set, test_set = split_dataset(dataset, test_size+verify_on, test_size)
      face_mind.train_on_dataset(train_set)
      paths, labels = facenet.get_image_paths_and_labels(test_set)
      print("Verifying on %i paths" % len(paths))
      res = face_mind.classify_all(paths)
      correct = 0.0
      for i in range(0,len(res)):
        prediction, score = res[i]
        if prediction == face_mind.class_names[labels[i]]:
          correct += 1
        else:
          print("wrong %s != %s" % (prediction, face_mind.class_names[labels[i]]))

      print("Accuracy for %i: %f" % (test_size, correct/len(res)))
      #accuracy = np.mean(np.equal(best_class_indices, labels))

      #indices = random.sample(, test_size-1)
      #test_emb_array = []
      #test_labels = []
      #for i in indices:
        #test_emb_array.append(emb_array[i])
        #test_labels.append(labels[i])
      #start = time.time()
      #model.fit(test_emb_array, test_labels)
      #results.append(time.time()-start)


    #plt.plot(test_sizes, results, 'ro')
    #plt.show()

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
