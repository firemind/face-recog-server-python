import matplotlib.pyplot as plt
import pickle
from sklearn.svm import SVC
import random
from sklearn.cluster import Birch
import numpy as np

classifier_filename_exp = "./svm-model/model.pkl"

model = SVC(kernel='linear', probability=True)
with open(classifier_filename_exp, 'rb') as infile:
  (emb_array, labels, class_names) = pickle.load(infile)

print(len(emb_array))
print(len(labels))
print(len(class_names))

test_sizes = [10, 50, 100, 250, 500, 750, 1000, 2000, 3000, 4000, 5000, 6000]
results = []
for test_size in test_sizes:
  if test_size > len(emb_array):
    print('Not enough samples "%s"' % len(emb_array))
  else:
    model = Birch(branching_factor=50, n_clusters=None, threshold=0.7,compute_labels=True, copy=False)
    indices = random.sample(range(0, len(emb_array)), test_size-1)
    results = []
    for i in indices:
      emb = emb_array[i]
      res = model.partial_fit(np.array([emb]))
      results.append(res)
    np.sum(results == labels)
    (a == b).sum()


plt.plot(test_sizes, results, 'ro')
plt.show()
