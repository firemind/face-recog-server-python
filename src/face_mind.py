import json
import math
import os
import pickle

import numpy as np
# from sklearn.svm import SVC
from sklearn import linear_model

from embedding_service import facenet
import httplib, urllib

class FaceMind:
  """A simple example class"""
  def __init__(self):
    self.margin = 44
    self.image_size = 160

  def save_classifier(self, classifier_filename_exp):
    print('Saving classifier model to file "%s"' % classifier_filename_exp)
    with open(classifier_filename_exp, 'wb') as outfile:
      #pickle.dump((model, class_names), outfile)
      pickle.dump((self.emb_array, self.labels, self.class_names), outfile)

  def load_classifier(self, classifier_filename_exp):

    classifier_filename_exp = os.path.expanduser(classifier_filename_exp)
    print('Loaded classifier model from file "%s"' % classifier_filename_exp)
    with open(classifier_filename_exp, 'rb') as infile:
      #(self.model, self.class_names) = pickle.load(infile)
      (self.emb_array, self.labels, self.class_names) = pickle.load(infile)
    self._set_model()
    self.fit()

  def classify(self, image_path):
    return self.classify_all([image_path])[0]

  def classify_all(self, image_paths):
    # Get input and output tensors
    images = facenet.load_data(image_paths, False, False, self.image_size)
    emb_array = self.request_embedding(images)
    predictions = self.model.predict_proba(emb_array)
    best_class_indices = np.argmax(predictions, axis=1)
    best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
    return map(lambda i: [self.class_names[best_class_indices[i]], best_class_probabilities[i]], range(len(best_class_indices)))

  def request_embedding(self, images):
    params = urllib.urlencode({'images': json.dumps( images.tolist())})
    conn = httplib.HTTPConnection(os.environ['EMBEDDING_SERVER'], os.environ['EMBEDDING_PORT'])
    headers = {"Content-type": "application/x-www-form-urlencoded",
            "Accept": "text/json"}
    conn.request("POST", "/embed", params, headers)
    response = conn.getresponse()
    data = response.read()
    return json.loads(data)['embedding']

  def store(self, image_path, label):
    try:
      i = self.class_names.index(label)
    except ValueError:
      i = len(self.class_names)
      print("First time seeing: "+label)
      self.class_names.append(label)
    print("%s is at %d" % (label, i))

    self.labels.append(i)
    images = facenet.load_data([image_path], False, False, self.image_size)
    res = self.request_embedding(images)
    self.emb_array = np.append(self.emb_array, res, axis=0)
    self.fit()

  def train_on_dataset(self, dataset):
    batch_size = 90
    embedding_size = 128
    paths, self.labels = facenet.get_image_paths_and_labels(dataset)
    nrof_images = len(paths)
    print("Training on %i images" % nrof_images)
    nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / batch_size))
    self.emb_array = np.zeros((nrof_images, embedding_size))
    for i in range(nrof_batches_per_epoch):
      start_index = i * batch_size
      end_index = min((i + 1) * batch_size, nrof_images)
      paths_batch = paths[start_index:end_index]
      images = facenet.load_data(paths_batch, False, False, self.image_size)
      self.emb_array[start_index:end_index, :] = self.request_embedding(images)
    self._set_model()
    self.class_names = [cls.name.replace('_', ' ') for cls in dataset]
    self.fit()

  def _set_model(self):
    # self.model = SVC(kernel='linear', probability=True)
    self.model = linear_model.SGDClassifier(loss='log')


  def train(self, data_dir):
    print('Training classifier model from path "%s"' % data_dir)
    dataset = facenet.get_dataset(data_dir)
    self.train_on_dataset(dataset)

  def fit(self):
    print("Fitting %i samples" % len(self.emb_array))
    self.model.fit(self.emb_array, self.labels)
