import matplotlib.pyplot as plt
import pickle
from sklearn.cluster import Birch
import numpy as np
from sklearn import metrics
from sklearn.metrics import pairwise_distances

# Assumes embeddings are already calculated and saved in this pickle file
classifier_filename_exp = "./svm-model/model.pkl"

with open(classifier_filename_exp, 'rb') as infile:
  (emb_array, labels, class_names) = pickle.load(infile)

print(len(emb_array))
print(len(labels))
print(len(class_names))

test_sizes = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
scores = []
np.random.seed(seed=666)
for test_size in test_sizes:
  if test_size > len(emb_array):
    print('Not enough samples "%s"' % len(emb_array))
  else:
    model = Birch(branching_factor=50, n_clusters=None, threshold=0.7,compute_labels=True, copy=False)
    indices = np.random.random_integers(0, len(emb_array)-1, test_size-1)
    labels = []
    X = []
    for i in indices:
      emb = emb_array[i]
      X.append(emb)
      model.partial_fit(np.array([emb]))
      labels.append(model.labels_[0])

    score = metrics.silhouette_score(X, labels, metric='euclidean')
    scores.append(score)


red_patch = mpatches.Patch(color='green', label='FaceAPI')
plt.legend(handles=[red_patch, blue_patch,green_patch])
plt.plot(test_sizes, scores, 'ro')
plt.show()
