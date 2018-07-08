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
"""Class for generating captions from an image-to-text model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import random
import heapq
import math


import numpy as np


class Caption(object):
  """Represents a complete or partial caption."""

  def __init__(self, sentence, state, logprob, score, metadata=None):
    """Initializes the Caption.

    Args:
      sentence: List of word ids in the caption.
      state: Model state after generating the previous word.
      logprob: Log-probability of the caption.
      score: Score of the caption.
      metadata: Optional metadata associated with the partial sentence. If not
        None, a list of strings with the same length as 'sentence'.
    """
    self.sentence = sentence
    self.state = state
    self.logprob = logprob
    self.score = score
    self.metadata = metadata

  def __cmp__(self, other):
    """Compares Captions by score."""
    assert isinstance(other, Caption)
    if self.score == other.score:
      return 0
    elif self.score < other.score:
      return -1
    else:
      return 1
  
  # For Python 3 compatibility (__cmp__ is deprecated).
  def __lt__(self, other):
    assert isinstance(other, Caption)
    return self.score < other.score
  
  # Also for Python 3 compatibility.
  def __eq__(self, other):
    assert isinstance(other, Caption)
    return self.score == other.score


class TopN(object):
  """Maintains the top n elements of an incrementally provided set."""

  def __init__(self, n):
    self._n = n
    self._data = []

  def size(self):
    assert self._data is not None
    return len(self._data)

  def push(self, x):
    """Pushes a new element."""
    assert self._data is not None
    if len(self._data) < self._n:
      heapq.heappush(self._data, x)
    else:
      heapq.heappushpop(self._data, x)

  def extract(self, sort=False):
    """Extracts all elements from the TopN. This is a destructive operation.

    The only method that can be called immediately after extract() is reset().

    Args:
      sort: Whether to return the elements in descending sorted order.

    Returns:
      A list of data; the top n elements provided to the set.
    """
    assert self._data is not None
    data = self._data
    self._data = None
    if sort:
      data.sort(reverse=True)
    return data

  def reset(self):
    """Returns the TopN to an empty state."""
    self._data = []


class CaptionGenerator(object):
  """Class to generate captions from an image-to-text model."""

  def __init__(self,
               model,
               vocab,
               beam_size=2,
	             temperature=1.2,
               max_caption_length=20,
               length_normalization_factor=0.2):
    """Initializes the generator.

    Args:
      model: Object encapsulating a trained image-to-text model. Must have
        methods feed_image() and inference_step(). For example, an instance of
        InferenceWrapperBase.
      vocab: A Vocabulary object.
      beam_size: Beam size to use when generating captions.
      max_caption_length: The maximum caption length before stopping the search.
      length_normalization_factor: If != 0, a number x such that captions are
        scored by logprob/length^x, rather than logprob. This changes the
        relative scores of captions depending on their lengths. For example, if
        x > 0 then longer captions will be favored.
    """
    self.vocab = vocab
    self.model = model

    self.beam_size = beam_size
    self.max_caption_length = max_caption_length
    self.length_normalization_factor = length_normalization_factor
    self.temperature = temperature

  def random_pick(self, w_and_p, num_picks, temp):
    picks = []

    total_p = 0
    for _,p in w_and_p:
       total_p += p
    total_temp = sum([(p/total_p)**(1/temp) for _,p in w_and_p])
    temp_probs = [(p/total_p)**(1/temp)/total_temp  for _,p in w_and_p]
    eps = 0.0001
    assert abs(sum(temp_probs)-1.0) < eps
    w_and_p = [(wp[0],temp_probs[j]) for j,wp in enumerate(w_and_p)]
    
    for i in range(num_picks):
      x = random.uniform(0, 1)
      cumulative_probability = 0.0
      index = -1
      for item, item_probability in zip(w_and_p, temp_probs):
         cumulative_probability += item_probability
         index += 1
         if x < cumulative_probability: break
    
      del w_and_p[index]; del temp_probs[index]
      if i < num_picks-1:
        total_probs = sum(temp_probs)
        temp_probs = [p/total_probs for p in temp_probs]
      picks.append(item)
    return picks


  def beam_search(self, sess, encoded_image):
    """Runs beam search caption generation on a single image.

    Args:
      sess: TensorFlow Session object.
      encoded_image: An encoded image string.

    Returns:
      A list of Caption sorted by descending score.
    """
    # Feed in the image to get the initial state.
    initial_state = self.model.feed_image(sess, encoded_image)

    initial_beam = Caption(
        sentence=[self.vocab.start_id],
        state=initial_state[0],
        logprob=0.0,
        score=0.0,
        metadata=[""])
    partial_captions = TopN(self.beam_size)
    for i in range(self.beam_size):
      partial_captions.push(initial_beam)
    complete_captions = TopN(self.beam_size)

    # Run beam search.
    for _ in range(self.max_caption_length - 1):
      partial_captions_list = partial_captions.extract()
      partial_captions.reset()
      input_feed = np.array([c.sentence[-1] for c in partial_captions_list])
      state_feed = np.array([c.state for c in partial_captions_list])

      softmax, new_states, metadata = self.model.inference_step(sess,
                                                                input_feed,
                                                                np.expand_dims(state_feed,axis=0))

      for i, partial_caption in enumerate(partial_captions_list):
        # print(softmax)
        word_probabilities = softmax[i][0]
        word_probabilities[self.vocab.unk_id] = 0 #set UNK prob to zero
        #word_probabilities[self.vocab.end_id] *= 0.5 #try lowering probability of sentence ending, longer captions
        state = np.squeeze(new_states)[i]
        # For this partial caption, get the beam_size most probable next words.
        words_and_probs = list(enumerate(word_probabilities))
        words_and_probs.sort(key=lambda x: -x[1])
        #Introducing temperature into beam search: instead of taking the top 5, choose 5 from probability distribution
        words_and_probs = self.random_pick(words_and_probs[:100],self.beam_size,self.temperature)
        #words_and_probs = words_and_probs[0:self.beam_size]
        
        # Each next word gives a new partial caption.
        for w, p in words_and_probs:
          if p < 1e-12:
            continue  # Avoid log(0).
          sentence = partial_caption.sentence + [w]
          logprob = partial_caption.logprob + math.log(p)
          score = logprob
          if metadata:
            metadata_list = partial_caption.metadata + [metadata[i]]
          else:
            metadata_list = None
          if w == self.vocab.end_id:
            if self.length_normalization_factor > 0:
              score /= len(sentence)**self.length_normalization_factor
            beam = Caption(sentence, state, logprob, score, metadata_list)
            complete_captions.push(beam)
          else:
            beam = Caption(sentence, state, logprob, score, metadata_list)
            partial_captions.push(beam)
      if partial_captions.size() == 0:
        # We have run out of partial candidates; happens when beam_size = 1.
        break

    # If we have no complete captions then fall back to the partial captions.
    # But never output a mixture of complete and partial captions because a
    # partial caption could have a higher score than all the complete captions.
    if not complete_captions.size():
      complete_captions = partial_captions

    return complete_captions.extract(sort=True)
