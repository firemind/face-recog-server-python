from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from flask import Flask, request, redirect, url_for, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
import argparse
import facenet
import os
import random
import sys
import math
import pickle
import align.detect_face
from sklearn.svm import SVC
from scipy import misc

IMAGE_FOLDER = "/images"
TMP_FOLDER = '/tmp/flask-uploads'
STORE_FOLDER = '/store'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])


def allowed_file(filename):
  return '.' in filename and \
         filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


app = Flask(__name__)
for d in [IMAGE_FOLDER,TMP_FOLDER,STORE_FOLDER]:
  if not os.path.isdir(d):
    os.makedirs(d)

tf.Graph().as_default()
sess = tf.Session()
model = None
embeddings = None
class_names = None

with sess.as_default():
  pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)


def do_align(img):
  minsize = 20  # minimum size of face
  threshold = [0.6, 0.7, 0.7]  # three steps's threshold
  factor = 0.709  # scale factor


  nrof_images_total = 0
  nrof_successfully_aligned = 0

  if img.ndim < 2:
    print('Unable to align')
    return img
  if img.ndim == 2:
    img = facenet.to_rgb(img)
  img = img[:, :, 0:3]

  bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold,
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
    bb[0] = np.maximum(det[0] - args.margin / 2, 0)
    bb[1] = np.maximum(det[1] - args.margin / 2, 0)
    bb[2] = np.minimum(det[2] + args.margin / 2, img_size[1])
    bb[3] = np.minimum(det[3] + args.margin / 2, img_size[0])
    cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
    scaled = misc.imresize(cropped, (args.image_size, args.image_size), interp='bilinear')
    nrof_successfully_aligned += 1
    return scaled
    # text_file.write('%s %d %d %d %d\n' % (output_filename, bb[0], bb[1], bb[2], bb[3]))
  else:
    print('Unable to align')
    return img


@app.route("/classify", methods=['POST'])
def classify():
  # check if the post request has the file part
  if 'image' not in request.files:
    flash('No file part')
    return redirect(request.url)
  file = request.files['image']
  # if user does not select file, browser also
  # submit a empty part without filename
  if file.filename == '':
    flash('No selected file')
    return redirect(request.url)
  if file and allowed_file(file.filename):
    ext = secure_filename(file.filename).rsplit('.', 1)[1].lower()
    random_key = np.random.randint(0, high=99999)
    image_path = os.path.join(TMP_FOLDER, ("%05d-classify." % random_key)+ext)
    # file.save(image_path)
    image = do_align(misc.imread(file))
    misc.imsave(image_path, image)
    # Get input and output tensors
    images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
    embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
    phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
    embedding_size = embeddings.get_shape()[1]
    images = facenet.load_data([image_path], False, False, args.image_size)
    feed_dict = {images_placeholder: images, phase_train_placeholder: False}
    emb_array = sess.run(embeddings, feed_dict=feed_dict)
    predictions = model.predict_proba(emb_array)
    best_class_indices = np.argmax(predictions, axis=1)
    best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]

    for i in range(len(best_class_indices)):
      if os.path.isdir(IMAGE_FOLDER):
        class_dir = os.path.join(IMAGE_FOLDER, class_names[best_class_indices[i]])
        if os.path.isdir(class_dir):
          image_path="/images/"+class_names[best_class_indices[i]]+"/"+random.choice(os.listdir(class_dir)) 
      else:
        image_path=""

      return jsonify(
                  label= class_names[best_class_indices[i]], 
                  score= best_class_probabilities[i],
                  image= image_path
                  )

@app.route("/store", methods=['POST'])
def store():
  if 'image' not in request.files:
    flash('No file part')
    return redirect(request.url)
  file = request.files['image']
  label = request.form.get('label')
  if file.filename == '':
    flash('No selected file')
    return redirect(request.url)
  if file and allowed_file(file.filename):
    ext = secure_filename(file.filename).rsplit('.', 1)[1].lower()
    random_key = np.random.randint(0, high=99999)
    file_name = ("%05d-%s.%s" % (random_key, label, ext))
    base_path = os.path.join(STORE_FOLDER, secure_filename(label))
    if not os.path.isdir(base_path):
      os.makedirs(base_path)
    image_path = os.path.join(base_path, secure_filename(file_name))
    file.save(image_path)
    return jsonify(image= image_path)

@app.route('/images/<path:path>')
def send_image(path):
  return send_from_directory('/images', path)

def main():
  global model
  global embeddings
  global class_names
  global sess
  with sess as sess:
    np.random.seed(seed=args.seed)

    # Load the model
    print('Loading feature extraction model')
    facenet.load_model(args.model)

    classifier_filename_exp = os.path.expanduser(args.classifier_filename)

    # Classify images
    print('Testing classifier')
    with open(classifier_filename_exp, 'rb') as infile:
      (model, class_names) = pickle.load(infile)

    print('Loaded classifier model from file "%s"' % classifier_filename_exp)

    app.run(host="0.0.0.0")


def parse_arguments(argv):
  parser = argparse.ArgumentParser()

  parser.add_argument('--model', type=str,
                      help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file',
                      default="/base-model")
  parser.add_argument('--classifier_filename',
                      help='Classifier model file name as a pickle (.pkl) file. ' +
                           'For training this is the output and for classification this is an input.',
                      default="/svm-model/model.pkl")
  parser.add_argument('--batch_size', type=int,
                      help='Number of images to process in a batch.', default=90)
  parser.add_argument('--image_size', type=int,
                      help='Image size (height, width) in pixels.', default=160)
  parser.add_argument('--seed', type=int,
                      help='Random seed.', default=666)
  parser.add_argument('--min_nrof_images_per_class', type=int,
                      help='Only include classes with at least this number of images in the dataset', default=20)
  parser.add_argument('--nrof_train_images_per_class', type=int,
                      help='Use this number of images from each class for training and the rest for testing',
                      default=10)
  parser.add_argument('--margin', type=int,
                      help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)

  return parser.parse_args(argv)


if __name__ == '__main__':
  args = parse_arguments(sys.argv[1:])
  main()
