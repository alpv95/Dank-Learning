import os
import numpy as np
from PIL import Image
import tensorflow as tf
from alexnet import AlexNet
from random import shuffle



#mean of imagenet dataset in BGR
imagenet_mean = np.array([104., 117., 124.], dtype=np.float32)

current_dir = os.getcwd()
image_dir = os.path.join(current_dir, 'memes')
#image_dir = current_dir

#placeholder for input and dropout rate
x = tf.placeholder(tf.float32, [1, 227, 227, 3])
keep_prob = tf.placeholder(tf.float32)

#create model with default config ( == no skip_layer and 1000 units in the last layer)
model = AlexNet(x, keep_prob, 1000,[],['fc7','fc8'],512) #maybe need to put fc8 in skip_layers

#define activation of last layer as score
score = model.fc6

img_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('jpg')]
with open('captions.txt','r') as f:
    captions = f.readlines()
captions = list(set(captions))
captions = [s.lower() for s in captions]
data_memes = []
data_captions = []

#Doing everything in one script: (the fc6 vectors are quite sparse), will have to change this up to not get repeats
with tf.Session() as sess:

    # Initialize all variables
    sess.run(tf.global_variables_initializer())

    # Load the pretrained weights into the model
    model.load_initial_weights(sess)

    for i,meme in enumerate(img_files):
        meme_name = meme.replace('/Users/ALP/Desktop/Stanford/CS224n/MemeProject/memes/','')
        meme_name = meme_name.replace('.jpg','').lower()
        meme_name = meme_name.replace('-',' ')
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

        meme_vector = sess.run(score, feed_dict={x: img, keep_prob: 1}) #[1,4096]
        meme_vector = np.reshape(meme_vector,[4096])
        assert np.shape(meme_vector) == (4096,)
        match = [s.split('-',1)[-1].lstrip() for s in captions if meme_name in s]

        #now save in tfrecords format, or prepare for that action
        meme_vectors = [meme_vector for cap in match]
        assert len(meme_vectors) == len(match)
        data_memes.extend(meme_vectors)
        data_captions.extend(match)

        if i % 100 == 0:
            print(i,len(data_memes),len(data_captions))

#deleting bad examples from data
deleters = []
for i,ting in enumerate(data_captions):
    if ting == '':
        deleters.append(i)
for i,ting in enumerate(deleters):
    del data_captions[ting-i]
    del data_memes[ting-i]

#splitting into list of lists of words
import re
word_captions = []
for capt in data_captions:
    words = re.findall(r'[\w]+|[.,!?;><(){}%$#£@-_+=|\/~`^&*]', capt)
    word_captions.append(words)

#create Vocabulary
from collections import Counter
print("Creating vocabulary.")
counter = Counter()
for c in word_captions:
    counter.update(c)
print("Total words:", len(counter))

# Filter uncommon words and sort by descending count.
word_counts = [x for x in counter.items() if x[1] >= 3]
word_counts.sort(key=lambda x: x[1], reverse=True)
print("Words in vocabulary:", len(word_counts))

# Create the vocabulary dictionary.
reverse_vocab = [x[0] for x in word_counts]
#unk_id = len(reverse_vocab)
vocab_dict = dict([(x, y) for (y, x) in enumerate(reverse_vocab)])

#LOAD PRE TRAINED GLOVE VECTORS and get tokenizer
EMBEDDING_DIMENSION=300 # Available dimensions for 6B data is 50, 100, 200, 300
data_directory = '~/Desktop/Stanford/CS224n/MemeProject'

PAD_TOKEN = 0

word2idx = { 'PAD': PAD_TOKEN } # dict so we can lookup indices for tokenising our text later from string to sequence of integers
weights = []
index_counter = 0

with open('glove.42B.300d.txt','r') as file:
    for index, line in enumerate(file):
        values = line.split() # Word and weights separated by space
        word = values[0] # Word is first symbol on each line
        if word in vocab_dict:
            index_counter += 1
            word_weights = np.asarray(values[1:], dtype=np.float32) # Remainder of line is weights for word
            word2idx[word] = index_counter # PAD is our zeroth index so shift by one
            weights.append(word_weights)
        if index % 20000 == 0:
            print(index)
        if index + 1 == 1500000:
            # Limit vocabulary to top 40k terms
            break

EMBEDDING_DIMENSION = len(weights[0])
# Insert the PAD weights at index 0 now we know the embedding dimension
weights.insert(0, np.random.randn(EMBEDDING_DIMENSION))

# Append unknown and pad to end of vocab and initialize as random #maybe include start and end token here
UNKNOWN_TOKEN=len(weights)
word2idx['UNK'] = UNKNOWN_TOKEN
word2idx['<S>'] = UNKNOWN_TOKEN + 1
word2idx['</S>'] = UNKNOWN_TOKEN + 2
weights.append(np.random.randn(EMBEDDING_DIMENSION))
weights.append(np.random.randn(EMBEDDING_DIMENSION))
weights.append(np.random.randn(EMBEDDING_DIMENSION))

# Construct our final vocab
weights = np.asarray(weights, dtype=np.float32)

VOCAB_SIZE=weights.shape[0]

#Save Vocabulary
with tf.gfile.FastGFile('vocab.txt', "w") as f:
    f.write("\n".join(["%s %d" % (w, c) for w, c in word2idx.iteritems()]))
print("Wrote vocabulary file:", 'vocab.txt')

#save embedding matrix
#np.savetxt('embedding_matrix2',weights)
'''
#Tokenize all the captions
import re
token_captions = []
for capt in data_captions:
    token_caption = []
    token_caption.append(word2idx['<S>'])
    words = re.findall(r"[\w']+|[.,!?;'><(){}%$#£@-_+=|\/~`^&*]", capt)
    for word in words:
        try:
            token = word2idx[word]
        except KeyError:
            token = word2idx['UNK']
        token_caption.append(token)
    token_caption.append(word2idx['</S>'])
    token_captions.append(token_caption)

#potentially another filering step
deleters = []
for i,ting in enumerate(token_captions):
    if len(ting) == 2:
        deleters.append(i)
for i,ting in enumerate(deleters):
    del data_captions[ting-i]
    del data_memes[ting-i]
    del token_captions[ting-i]

#shuffle data
c = list(zip(data_memes, token_captions))
shuffle(c)
memes_shuffled, captions_shuffled = zip(*c)

def _int64_feature(value):
    """Wrapper for inserting an int64 Feature into a SequenceExample proto."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    """Wrapper for inserting a bytes Feature into a SequenceExample proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature_list(values):
    """Wrapper for inserting an int64 FeatureList into a SequenceExample proto."""
    return tf.train.FeatureList(feature=[_int64_feature(v) for v in values])

#Tranform meme embeddings into integers for easy conversion to tfrecords file
memes_shuffled_int = []
for i,meme in enumerate(memes_shuffled):
    memes_shuffled_int.append(np.int_(meme*1000000000))
print(memes_shuffled_int[0][:100])

#write tfrecords file as joint sequence of images embeddings and captions
import sys
train_filename = 'train.tfrecords4'  # address to save the TFRecords file
# open the TFRecords file
writer = tf.python_io.TFRecordWriter(train_filename)
for i in range(len(memes_shuffled_int)):
    if not i % 20000:
        print 'Train data: {}/{}'.format(i, len(memes_shuffled_int))
        sys.stdout.flush()
    context = tf.train.Features(feature={
          "train/meme": _bytes_feature(memes_shuffled_int[i].tostring()),  #this is the part that needs to be a float save
      })
    feature_lists = tf.train.FeatureLists(feature_list={
          "train/captions": _int64_feature_list(captions_shuffled[i])
      })
    sequence_example = tf.train.SequenceExample(
          context=context, feature_lists=feature_lists)

    writer.write(sequence_example.SerializeToString())

writer.close()
sys.stdout.flush()
'''