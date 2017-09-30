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
from face_mind import FaceMind
import align.detect_face
from sklearn.svm import SVC
from scipy import misc
import base64

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

face_mind = FaceMind(sess)

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
    image = misc.imread(file)
    if request.form.get("align") == "true":
      image = face_mind.align(image)

    misc.imsave(image_path, image)

    label, score = face_mind.classify(image_path)
    if os.path.isdir(IMAGE_FOLDER):
      class_dir = os.path.join(IMAGE_FOLDER, label)
      if os.path.isdir(class_dir):
        path = base64.b64encode(label+"/"+random.choice(os.listdir(class_dir)))
        image_path="/images/"+ path
    else:
      image_path=""

    return jsonify(
                label= label,
                score= score,
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
    # todo secure_filename removes umlauts. maybe there is a way to preserve them?
    ext = secure_filename(file.filename).rsplit('.', 1)[1].lower()
    random_key = np.random.randint(0, high=99999)
    file_name = ("%05d-%s.%s" % (random_key, label, ext))
    base_path = os.path.join(STORE_FOLDER, secure_filename(label))
    if not os.path.isdir(base_path):
      os.makedirs(base_path)
    image_path = os.path.join(base_path, secure_filename(file_name))
    file.save(image_path)
    face_mind.store(image_path, secure_filename(label))
    return jsonify(image= image_path)

@app.route('/images/<path:path>')
def send_image(path):
  path = base64.b64decode(path)
  print(path)
  return send_from_directory('/images', path)

def main():
  global sess
  with sess as sess:
    np.random.seed(seed=args.seed)

    face_mind.load_model(args.model)
    # face_mind.load_classifier(args.classifier_filename)
    face_mind.train(args.data_dir)

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
  parser.add_argument('--seed', type=int,
                      help='Random seed.', default=666)
  parser.add_argument('--min_nrof_images_per_class', type=int,
                      help='Only include classes with at least this number of images in the dataset', default=20)
  parser.add_argument('--nrof_train_images_per_class', type=int,
                      help='Use this number of images from each class for training and the rest for testing',
                      default=10)
  parser.add_argument('--data_dir', type=str,
                      help='Path to the data directory containing aligned LFW face patches.', default="/face-data/")

  return parser.parse_args(argv)


if __name__ == '__main__':
  args = parse_arguments(sys.argv[1:])
  main()
