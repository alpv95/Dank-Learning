# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Generate captions for images using default beam search parameters."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
import numpy as np
from PIL import Image
from alexnet import AlexNet

import tensorflow as tf

import configuration
import inference_wrapper
import sys
sys.path.insert(0, '/data/alpv95/MemeProject/im2txt/inference_utils')
import caption_generator
import vocabulary

current_dir = os.getcwd()
image_dir = os.path.join(current_dir, 'memes')

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("checkpoint_path", "",
                       "Model checkpoint file or directory containing a "
                       "model checkpoint file.")
tf.flags.DEFINE_string("vocab_file", "", "Text file containing the vocabulary.")
tf.flags.DEFINE_string("input_files", "",
                       "File pattern or comma-separated list of file patterns "
                       "of image files.")

tf.logging.set_verbosity(tf.logging.INFO)


def main(_):
  #convert jpg image(s) into iamge representations using alexnet:
  filenames = [os.path.join(image_dir, f) for f in ['overly-attached-girlfriend.jpg','high-expectations-asian-father.jpg','foul-bachelor-frog.jpg','stoner-stanley.jpg','y-u-no.jpg','willy-wonka.jpg','futurama-fry.jpg','success-kid.jpg','one-does-not-simply.jpg','bad-luck-brian.jpg','first-world-problems.jpg','philosoraptor.jpg','what-if-i-told-you.jpg','TutorPP.jpg']]
  print(filenames)
  tf.logging.info("Running caption generation on %d files matching %s",
                  len(filenames), FLAGS.input_files)
  #mean of imagenet dataset in BGR
  imagenet_mean = np.array([104., 117., 124.], dtype=np.float32)

  #placeholder for input and dropout rate
  x_Alex = tf.placeholder(tf.float32, [1, 227, 227, 3])
  keep_prob_Alex = tf.placeholder(tf.float32)

  #create model with default config ( == no skip_layer and 1000 units in the last layer)
  modelAlex = AlexNet(x_Alex, keep_prob_Alex, 1000,[],['fc7','fc8'],512) #maybe need to put fc8 in skip_layers

  #define activation of last layer as score
  score = modelAlex.fc6

  meme_embeddings= []
  with tf.Session() as sess:

    # Initialize all variables
    sess.run(tf.global_variables_initializer())

    # Load the pretrained weights into the model
    modelAlex.load_initial_weights(sess)

    for i,meme in enumerate(filenames):
        img = Image.open(meme)
        try:
            img.thumbnail((227, 227), Image.ANTIALIAS)
            #img = img.resize((227,227))
            #use img.thumbnail for square images, img.resize for non square
            assert np.shape(img) == (227, 227, 3)
        except AssertionError:
            img = img.resize((227,227))
            print('sizing error')

        # Subtract the ImageNet mean
        img = img - imagenet_mean #should probably change this

        # Reshape as needed to feed into model
        img = img.reshape((1,227,227,3))

        meme_vector = sess.run(score, feed_dict={x_Alex: img, keep_prob_Alex: 1}) #[1,4096]
	print(meme_vector)
        meme_vector = np.reshape(meme_vector,[4096])
        assert np.shape(meme_vector) == (4096,)
	print(np.shape(meme_vector))

        #now have np embeddings to feed for inference
        meme_embeddings.append(meme_vector)

  print(len(meme_embeddings))
  print(meme_embeddings)
  # Build the inference graph.
  g = tf.Graph()
  with g.as_default():
    model = inference_wrapper.InferenceWrapper()
    restore_fn = model.build_graph_from_config(configuration.ModelConfig(),
                                               FLAGS.checkpoint_path)
  g.finalize()

  # Create the vocabulary.
  vocab = vocabulary.Vocabulary(FLAGS.vocab_file)

  #filenames = []
  #for file_pattern in FLAGS.input_files.split(","):
    #filenames.extend(tf.gfile.Glob(file_pattern))
  #tf.logging.info("Running caption generation on %d files matching %s",
                  #len(filenames), FLAGS.input_files)
  print('ok')
  with tf.Session(graph=g) as sess:
    # Load the model from checkpoint.
    restore_fn(sess)
    
    # Prepare the caption generator. Here we are implicitly using the default
    # beam search parameters. See caption_generator.py for a description of the
    # available beam search parameters.
    generator = caption_generator.CaptionGenerator(model, vocab)
    print('yes')
    for i,meme in enumerate(meme_embeddings):
      print('its working')
      #with tf.gfile.GFile(filename, "rb") as f:
        #image = f.read()
      captions = generator.beam_search(sess, meme)
      print("Captions for image %s:" % os.path.basename(filenames[i]))
      for i, caption in enumerate(captions):
        # Ignore begin and end words.
        sentence = [vocab.id_to_word(w) for w in caption.sentence[1:-1]]
        sentence = " ".join(sentence)
        print("  %d) %s (p=%f)" % (i, sentence, math.exp(caption.logprob)))


if __name__ == "__main__":
  tf.app.run()
