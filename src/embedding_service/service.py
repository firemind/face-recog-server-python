import facenet
import tensorflow as tf
import time

class Service:
  """A simple example class"""
  def __init__(self, sess):
    with sess.as_default():
      self.sess = sess

  def load_model(self, model):
    # Load the model
    print('Loading feature extraction model')
    facenet.load_model(model)
    self.images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
    self.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
    self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

  def embed(self, image):
    return self.embed_all([image])[0]

  def embed_all(self, images):
    start = time.time()
    feed_dict = {self.images_placeholder: images, self.phase_train_placeholder: False}
    emb_array = self.sess.run(self.embeddings, feed_dict=feed_dict)
    print(time.time() - start)
    return emb_array

