from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from flask import Flask, request, jsonify
import tensorflow as tf
import argparse
import sys
import facenet
from service import Service

app = Flask(__name__)


tf.Graph().as_default()
sess = tf.Session()

service = Service(sess)

@app.route("/embed", methods=['POST'])
def embed():
    uploaded_files = request.files.getlist("images")
    images = facenet.load_data(map(lambda x: x.stream, uploaded_files), False, False, 160)

    embedding = service.embed_all(images)

    return jsonify(
                embedding= embedding.tolist(),
                )

def main():
  global sess
  with sess as sess:

    service.load_model(args.model)

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
