import facenet
import align.detect_face
import numpy as np
from scipy import misc
import os
import pickle
import tensorflow as tf
from sklearn.svm import SVC
import math

class FaceMind:
  """A simple example class"""
  i = 12345
  def __init__(self, sess):
    with sess.as_default():
      self.sess = sess
      self.pnet, self.rnet, self.onet = align.detect_face.create_mtcnn(sess, None)
      self.margin = 44
      self.image_size = 160

  def load_model(self, model):

    # Load the model
    print('Loading feature extraction model')
    facenet.load_model(model)
    self.images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
    self.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
    self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

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
    self.model = SVC(kernel='linear', probability=True)
    self.fit()

  def classify(self, image_path):
    return self.classify_all([image_path])[0]

  def classify_all(self, image_paths):
    # Get input and output tensors
    images = facenet.load_data(image_paths, False, False, self.image_size)
    feed_dict = {self.images_placeholder: images, self.phase_train_placeholder: False}
    emb_array = self.sess.run(self.embeddings, feed_dict=feed_dict)
    predictions = self.model.predict_proba(emb_array)
    best_class_indices = np.argmax(predictions, axis=1)
    best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
    return map(lambda i: [self.class_names[best_class_indices[i]], best_class_probabilities[i]], range(len(best_class_indices)))

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
    feed_dict = {self.images_placeholder: images, self.phase_train_placeholder: False}
    res = self.sess.run(self.embeddings, feed_dict=feed_dict)
    self.emb_array = np.append(self.emb_array, res, axis=0)
    self.fit()

  def train_on_dataset(self, dataset):
    batch_size = 90
    embedding_size = self.embeddings.get_shape()[1]

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
      feed_dict = {self.images_placeholder: images, self.phase_train_placeholder: False}
      self.emb_array[start_index:end_index, :] = self.sess.run(self.embeddings, feed_dict=feed_dict)
    self.model = SVC(kernel='linear', probability=True)
    self.class_names = [cls.name.replace('_', ' ') for cls in dataset]
    self.fit()

  def train(self, data_dir):
    print('Training classifier model from path "%s"' % data_dir)
    dataset = facenet.get_dataset(data_dir)
    self.train_on_dataset(dataset)

  def fit(self):
    print("Fitting %i samples" % len(self.emb_array))
    self.model.fit(self.emb_array, self.labels)

  def align(self, img):
    print("doing alignment")
    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    nrof_successfully_aligned = 0

    if img.ndim < 2:
      print('Unable to align')
      return img
    if img.ndim == 2:
      img = facenet.to_rgb(img)
    img = img[:, :, 0:3]

    bounding_boxes, _ = align.detect_face.detect_face(img, minsize, self.pnet, self.rnet, self.onet, threshold,
                                                      factor)
    nrof_faces = bounding_boxes.shape[0]
    if nrof_faces > 0:
      det = bounding_boxes[:, 0:4]
      img_size = np.asarray(img.shape)[0:2]
      if nrof_faces > 1:
        bounding_box_size = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
        img_center = img_size / 2
        offsets = np.vstack([(det[:, 0] + det[:, 2]) / 2 - img_center[1],
                             (det[:, 1] + det[:, 3]) / 2 - img_center[0]])
        offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
        index = np.argmax(
          bounding_box_size - offset_dist_squared * 2.0)  # some extra weight on the centering
        det = det[index, :]
      det = np.squeeze(det)
      bb = np.zeros(4, dtype=np.int32)
      bb[0] = np.maximum(det[0] - self.margin / 2, 0)
      bb[1] = np.maximum(det[1] - self.margin / 2, 0)
      bb[2] = np.minimum(det[2] + self.margin / 2, img_size[1])
      bb[3] = np.minimum(det[3] + self.margin / 2, img_size[0])
      cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
      scaled = misc.imresize(cropped, (self.image_size, self.image_size), interp='bilinear')
      nrof_successfully_aligned += 1
      return scaled
      # text_file.write('%s %d %d %d %d\n' % (output_filename, bb[0], bb[1], bb[2], bb[3]))
    else:
      print('Unable to align')
      return img
