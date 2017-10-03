from Tkinter import *
import matplotlib.pyplot as plt
import pickle
from sklearn.svm import SVC
import random
import time

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
    indices = random.sample(range(0, len(emb_array)), test_size-1)
    test_emb_array = []
    test_labels = []
    for i in indices:
      test_emb_array.append(emb_array[i])
      test_labels.append(labels[i])
    start = time.time()
    model.fit(test_emb_array, test_labels)
    results.append(time.time()-start)


plt.plot(test_sizes, results, 'ro')
plt.show()
