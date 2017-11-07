import matplotlib.pyplot as plt
import pickle
from sklearn.cluster import Birch
import numpy as np
from sklearn import metrics
import matplotlib.patches as mpatches
from sklearn.metrics import pairwise_distances

# Assumes embeddings are already calculated and saved in this pickle file
classifier_filename_exp = "./svm-model/model.pkl"

with open(classifier_filename_exp, 'rb') as infile:
  (emb_array, labels, class_names) = pickle.load(infile)

print(len(emb_array))
print(len(labels))
print(len(class_names))

test_sizes = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
threshold = 0.9
scores = []
class_count_diffs = []
np.random.seed(seed=666)
for test_size in test_sizes:
  if test_size > len(emb_array):
    print('Not enough samples "%s"' % len(emb_array))
  else:
    model = Birch(branching_factor=50, n_clusters=None, threshold=threshold, compute_labels=True, copy=False)
    indices = np.random.random_integers(0, len(emb_array)-1, test_size-1)
    pred_labels = []
    real_labels = []
    X = []
    for i in indices:
      emb = emb_array[i]
      X.append(emb)
      real_labels.append(labels[i])
      model.partial_fit(np.array([emb]))
      pred_labels.append(model.labels_[0])

    real_count = len(set(real_labels))
    pred_count = len(set(pred_labels))
    # count_diff = abs(pred_count - real_count)
    #class_count_diffs.append(1 - float(count_diff) / (pred_count+real_count))
    # class_count_diffs.append(1 - float(count_diff) / (pred_count+real_count))
    sim = (float(pred_count) / float(real_count))
    if sim > 1:
      sim = 1/ sim
    class_count_diffs.append(sim)
    score = metrics.silhouette_score(X, pred_labels, metric='euclidean')
    scores.append(score)


red_patch = mpatches.Patch(color='red', label='silhouette-coefficient')
blue_patch = mpatches.Patch(color='blue', label='Class Diff')
plt.legend(handles=[red_patch, blue_patch])
plt.plot(test_sizes, scores, 'ro', test_sizes, class_count_diffs, 'bo')
plt.show()
