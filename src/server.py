from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from flask import Flask,  request, redirect, url_for
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
import argparse
import facenet
import os
import sys
import math
import pickle
from sklearn.svm import SVC

UPLOAD_FOLDER = '/tmp/flask-uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
  return '.' in filename and \
    filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

app = Flask(__name__)
if not os.path.isdir(UPLOAD_FOLDER):
  os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

tf.Graph().as_default()
sess = tf.Session()
model =None
embeddings=None
class_names=None
image_size=None

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
    filename = secure_filename(file.filename)
    image_path=os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(image_path)

    # Get input and output tensors
    images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
    embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
    phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
    embedding_size = embeddings.get_shape()[1]
    images = facenet.load_data([image_path], False, False, image_size)
    feed_dict = { images_placeholder:images, phase_train_placeholder:False }
    emb_array = sess.run(embeddings, feed_dict=feed_dict)
    predictions = model.predict_proba(emb_array)
    best_class_indices = np.argmax(predictions, axis=1)
    best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]

    for i in range(len(best_class_indices)):
      return('%4d  %s: %.3f' % (i, class_names[best_class_indices[i]], best_class_probabilities[i]))

def main(args):
  global model
  global embeddings
  global class_names
  global sess
  global image_size
  image_size = args.image_size
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
	    help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file', default="/base-model")
    parser.add_argument('--classifier_filename', 
	    help='Classifier model file name as a pickle (.pkl) file. ' + 
	    'For training this is the output and for classification this is an input.', default="/facenet/model.pb")
    parser.add_argument('--batch_size', type=int,
	    help='Number of images to process in a batch.', default=90)
    parser.add_argument('--image_size', type=int,
	    help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--seed', type=int,
	    help='Random seed.', default=666)
    parser.add_argument('--min_nrof_images_per_class', type=int,
	    help='Only include classes with at least this number of images in the dataset', default=20)
    parser.add_argument('--nrof_train_images_per_class', type=int,
	    help='Use this number of images from each class for training and the rest for testing', default=10)

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
