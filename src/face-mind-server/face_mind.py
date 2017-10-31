import os
import pickle

import numpy as np
from sklearn import linear_model


class FaceMind:
  def __init__(self):
    self.reset()

  def reset(self):
    self._set_model()
    self.emb_array = np.zeros([0,128])
    self.labels = []
    self.class_names = []

  def save_classifier(self, classifier_filename_exp):
    print('Saving classifier model to file "%s"' % classifier_filename_exp)
    with open(classifier_filename_exp, 'wb') as outfile:
      pickle.dump((self.emb_array, self.labels, self.class_names), outfile)

  def load_classifier(self, classifier_filename_exp):

    classifier_filename_exp = os.path.expanduser(classifier_filename_exp)
    print('Loaded classifier model from file "%s"' % classifier_filename_exp)
    with open(classifier_filename_exp, 'rb') as infile:
      (self.emb_array, self.labels, self.class_names) = pickle.load(infile)
    self._set_model()
    self.fit()

  def classify(self, embeddings):
    predictions = self.model.predict_proba(embeddings)
    best_class_indices = np.argmax(predictions, axis=1)
    best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
    return map(lambda i: [self.class_names[best_class_indices[i]], best_class_probabilities[i]], range(len(best_class_indices)))

  def store(self, embedding, label):
    try:
      i = self.class_names.index(label)
    except ValueError:
      i = len(self.class_names)
      self.class_names.append(label)

    self.labels.append(i)
    self.emb_array = np.append(self.emb_array, [embedding], axis=0)

  def _set_model(self):
    self.model = linear_model.SGDClassifier(loss='log')

  def train(self):
    self.fit()

  def fit(self):
    print("Fitting %i samples" % len(self.emb_array))
    self.model.fit(self.emb_array, self.labels)
