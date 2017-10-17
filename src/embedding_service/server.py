from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from flask import Flask, request, redirect, url_for, jsonify, send_from_directory, render_template
import tensorflow as tf
import argparse
import sys
from src.embedding_service.service import Service

app = Flask(__name__)

tf.Graph().as_default()
sess = tf.Session()

service = Service(sess)

@app.route("/embed", methods=['POST'])
def embed():
    image = request.form.get("image")

    embedding = service.embed(image)

    return jsonify(
                embedding= embedding,
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
