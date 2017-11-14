from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from flask import Flask, request, request, url_for, jsonify, send_from_directory, render_template
import argparse
import sys
from face_tracker import FaceTracker

app = Flask(__name__)

face_tracker = FaceTracker();


@app.route("/track", methods=['POST'])
def track():
  data = request.get_json()
  emb = data['embedding']
  id = data['id']
  label = face_tracker.track(id, emb)
  return jsonify(
              label=label
              )

@app.route("/reset", methods=['POST'])
def reset():
  face_tracker.reset()
  return "done"

def main():
  app.run(host="0.0.0.0")


def parse_arguments(argv):
  parser = argparse.ArgumentParser()

  return parser.parse_args(argv)


if __name__ == '__main__':
  args = parse_arguments(sys.argv[1:])
  main()
