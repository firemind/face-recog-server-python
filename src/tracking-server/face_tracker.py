from sklearn.cluster import Birch
import numpy as np
class FaceTracker:
  def __init__(self):
    self._set_model();

  def _set_model(self):
    self.model = Birch(branching_factor=50, n_clusters=None, threshold=0.8,compute_labels=True, copy=False)
    self.images = {}

  def track(self, image, emb):
    self.model.partial_fit(np.array([emb]))
    assigned_label = self.model.labels_[0]
    if assigned_label not in  self.images:
      self.images[assigned_label] = []
    self.images[assigned_label].append(image)
    return assigned_label

  def history_by_label(self, label):
    return self.images[label]

  def history_by_emb(self, emb):
    pred = self.model.predict(np.array([emb]))[0]
    return self.images[pred]

  def labels(self):
    return self.images.keys()

  def reset(self):
    self._set_model();
