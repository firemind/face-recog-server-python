from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from flask import Flask, request, request, url_for, jsonify, send_from_directory, render_template
import argparse
import sys
from face_tracker import FaceTracker
from face_mind import FaceMind
import os

app = Flask(__name__)


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
  data = request.get_json()
  emb = data['embedding']
  id = data['id']
  label = face_tracker.track(id, emb)
  return jsonify(
              label=label
              )

@app.route("/history/<int:label>", methods=['GET'])
def history(label):
  return render_template('history.html', entries=map(lambda x: os.path.basename(x), face_tracker.history_by_label(label)), label=label)

@app.route('/images/<path:path>')
def send_image(path):
  return send_from_directory(STORE_FOLDER, path)

@app.route('/')
def index_labels():
  return render_template('index.html', labels=map(lambda label: [label, os.path.basename(face_tracker.history_by_label(label)[-1])], face_tracker.labels()))

def main():

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
