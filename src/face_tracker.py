from sklearn.cluster import Birch
import numpy as np
class FaceTracker:
  """A simple example class"""
  def __init__(self):
    self.model = Birch(branching_factor=50, n_clusters=None, threshold=0.7,compute_labels=True)
    self.images = {}

  def track(self, image, emb):
    print("tracking %s" % image)
    self.model.partial_fit(np.array([emb]))
    assigned_label = self.model.labels_[0]
    if assigned_label not in  self.images:
      self.images[assigned_label] = []
    self.images[assigned_label].append(image)

  def get_history(self, emb):
    pred = self.model.predict(np.array([emb]))[0]
    return self.images[pred]
