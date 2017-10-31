from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from flask import Flask, request, request, url_for, jsonify, send_from_directory, render_template
from face_mind import FaceMind

app = Flask(__name__)

face_mind = FaceMind();

@app.route("/store", methods=['POST'])
def store():
  data = request.get_json()
  emb = data['embedding']
  label = data['label']
  face_mind.store(emb, label)
  return jsonify(
              label=label
              )

@app.route("/classify", methods=['POST'])
def classify():
  data = request.get_json()
  emb = data['embeddings']
  labels = face_mind.classify(emb)
  return jsonify(
              result=labels
              )

@app.route("/train", methods=['POST'])
def train():
  face_mind.train()
  return "done"

@app.route("/reset", methods=['POST'])
def reset():
  face_mind.reset()
  return "done"

def main():
  app.run(host="0.0.0.0")

if __name__ == '__main__':
  main()
