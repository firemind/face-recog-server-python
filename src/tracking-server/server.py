from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from flask import Flask, request, request, url_for, jsonify, send_from_directory, render_template
import tensorflow as tf
from werkzeug.utils import secure_filename
import argparse
import sys
from face_tracker import FaceTracker
from face_mind import FaceMind
import json
from scipy import misc
import numpy as np
import os

app = Flask(__name__)


tf.Graph().as_default()
sess = tf.Session()

face_tracker = FaceTracker();
face_mind = FaceMind();

STORE_FOLDER = '/store'
for d in [STORE_FOLDER]:
  if not os.path.isdir(d):
    os.makedirs(d)

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
def allowed_file(filename):
  return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/track", methods=['POST'])
def track():

  data = json.loads(request.form.get('data'))
  file = request.files['image']
  if file and allowed_file(file.filename):
    print(file.filename)
    # todo secure_filename removes umlauts. maybe there is a way to preserve them?
    ext = secure_filename(file.filename).rsplit('.', 1)[1].lower()
    random_key = np.random.randint(0, high=9999999999)
    file_name = ("%010d.%s" % (random_key, ext))
    base_path = os.path.join(STORE_FOLDER)
    if not os.path.isdir(base_path):
      os.makedirs(base_path)
    image_path = os.path.join(base_path, secure_filename(file_name))
    image = misc.imread(file)
    misc.imsave(image_path, image)
    if 'embedding' in data:
      emb = data['embedding']
    else:
      emb = face_mind.request_embedding([image_path])[0]
    label = face_tracker.track(image_path, emb)
    print(image_path)
    return jsonify(
                label=label,
                history_url="/history/"+str(label)
                )

@app.route("/history/<int:label>", methods=['GET'])
def history(label):
  return render_template('history.html', entries=map(lambda x: os.path.basename(x), face_tracker.history_by_label(label)), label=label)

@app.route('/images/<path:path>')
def send_image(path):
  return send_from_directory(STORE_FOLDER, path)

def main():
  global sess
  with sess as sess:

    app.run(host="0.0.0.0")


def parse_arguments(argv):
  parser = argparse.ArgumentParser()

  parser.add_argument('--model', type=str,
                      help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file',
                      default="/base-model")

  return parser.parse_args(argv)


if __name__ == '__main__':
  args = parse_arguments(sys.argv[1:])
  main()
