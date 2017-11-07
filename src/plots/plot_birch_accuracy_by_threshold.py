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

#thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98]
thresholds = [0.8, 0.81, 0.82, 0.83, 0.84,0.85,0.86,0.87, 0.88, 0.89, 0.9, 0.91, 0.92, 0.93, 0.94,0.95,0.96,0.97, 0.98]
test_size = 500
scores = []
class_count_diffs = []
np.random.seed(seed=666)
for threshold in thresholds:
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
    count_diff = abs(pred_count - real_count)
    class_count_diffs.append(1 - float(count_diff) / (pred_count+real_count))
    score = metrics.silhouette_score(X, pred_labels, metric='euclidean')
    scores.append(score)

fig = plt.figure()
ax = fig.add_subplot(111)
ymax = max(class_count_diffs)
xpos = class_count_diffs.index(ymax)
xmax = thresholds[xpos]

ax.annotate('local max', xy=(xmax, ymax), xytext=(xmax, ymax+0.2),
            arrowprops=dict(facecolor='black', shrink=0.05),
            )
ymax2 = max(scores)
xpos2 = scores.index(ymax2)
xmax2 = thresholds[xpos2]

ax.annotate('local max', xy=(xmax2, ymax2), xytext=(xmax2, ymax2+0.2),
            arrowprops=dict(facecolor='black', shrink=0.05),
            )
red_patch = mpatches.Patch(color='red', label='silhouette-coefficient')
blue_patch = mpatches.Patch(color='blue', label='Class Diff')
plt.legend(handles=[red_patch, blue_patch])
plt.plot(thresholds, scores, 'ro', thresholds, class_count_diffs, 'bo')
plt.show()
