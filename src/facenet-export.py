"""An example of how to use your own dataset to train a classifier that recognizes people.
"""
# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import utils

import tensorflow as tf
import numpy as np
import argparse
import facenet
import os
import sys
import math
import pickle
from sklearn.svm import SVC
from tensorflow.contrib.session_bundle import exporter
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def main(args):
  
    with tf.Graph().as_default():
      
        with tf.Session() as sess:
            
            np.random.seed(seed=args.seed)

            #print('Number of classes: %d' % len(dataset))
            #print('Number of images: %d' % len(paths))
            dataset = facenet.get_dataset(args.train_dir)
            # Load the model
            #print('Loading feature extraction model')
            facenet.load_model(args.model)
            paths, labels = facenet.get_image_paths_and_labels(dataset)
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

            feature_configs = {
                'x': tf.FixedLenFeature(shape=[num_inputs], dtype=tf.int64),
            }
            tf_example = tf.parse_example(serialized_tf_example, feature_configs)

            # Run forward pass to calculate embeddings

            print('Calculating features for images')
            nrof_images = len(paths)
            nrof_batches_per_epoch = int(math.ceil(1.0*nrof_images / args.batch_size))
            emb_array = np.zeros((nrof_images, embedding_size))

            for i in range(nrof_batches_per_epoch):
                start_index = i*args.batch_size
                end_index = min((i+1)*args.batch_size, nrof_images)
                paths_batch = paths[start_index:end_index]
                images = facenet.load_data(paths_batch, False, False, args.image_size)
                feed_dict = { images_placeholder:images, phase_train_placeholder:False }
                emb_array[start_index:end_index,:] = sess.run(embeddings, feed_dict=feed_dict)

            classifier_filename_exp = os.path.expanduser(args.classifier_filename)

            # Train classifier
            print('Training classifier')
            model = SVC(kernel='linear', probability=True)
            model.fit(emb_array, labels)

            # Create a list of class names
            class_names = [cls.name.replace('_', ' ') for cls in dataset]

            # Saving classifier model
            with open(classifier_filename_exp, 'wb') as outfile:
                pickle.dump((model, class_names), outfile)
            print('Saved classifier model to file "%s"' % classifier_filename_exp)

            #print('Loaded classifier model from file "%s"' % classifier_filename_exp)

            predictions = model.predict_proba(emb_array);


            print('Exporting trained model to %s' % args.export_path)
            builder = saved_model_builder.SavedModelBuilder(args.export_path)
            best_class_probabilities, best_class_indices = tf.nn.top_k(predictions, len(class_names))

            table = tf.contrib.lookup.index_to_string_table_from_tensor(mapping=class_names, default_value="UNKNOWN")
            prediction_classes = table.lookup(tf.to_int64(best_class_indices))
            classification_inputs = utils.build_tensor_info(images_placeholder)
            classification_outputs_classes = utils.build_tensor_info(prediction_classes)
            classification_outputs_scores = utils.build_tensor_info(best_class_probabilities)

            sess.run(tf.global_variables_initializer())

            classification_signature = (
                tf.saved_model.signature_def_utils.build_signature_def(
                    inputs={
                        tf.saved_model.signature_constants.CLASSIFY_INPUTS:
                            classification_inputs
                    },
                    outputs={
                        tf.saved_model.signature_constants.CLASSIFY_OUTPUT_CLASSES:
                            classification_outputs_classes,
                        tf.saved_model.signature_constants.CLASSIFY_OUTPUT_SCORES:
                            classification_outputs_scores
                    },
                    method_name=tf.saved_model.signature_constants.CLASSIFY_METHOD_NAME))

            legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')

            builder.add_meta_graph_and_variables(
                sess, [tag_constants.SERVING],
                signature_def_map={
                    signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                        classification_signature,
                },
                legacy_init_op=legacy_init_op)

            builder.save()

            print('Done exporting!')

            #accuracy = np.mean(np.equal(best_class_indices, labels))
            #print('Accuracy: %.3f' % accuracy)
                
            
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('train_dir', type=str,
        help='Path to the data directory containing aligned LFW face patches.')
    parser.add_argument('model', type=str, 
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('export_path', type=str,
        help='path to save exported model to')
    parser.add_argument('classifier_filename', 
        help='Classifier model file name as a pickle (.pkl) file. ' + 
        'For training this is the output and for classification this is an input.')
    parser.add_argument('--batch_size', type=int,
        help='Number of images to process in a batch.', default=90)
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--seed', type=int,
        help='Random seed.', default=666)
    
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
