# Dank Learning: Generating Memes Using Deep Neural Networks

![Alt text](Picture1.png?raw=true "Title")
![Alt text](Picture2.png?raw=true "Title")
![Alt text](Picture3.png?raw=true "Title")
![Alt text](Picture4.png?raw=true "Title")

## Overview

This is the code for the paper "[Dank Learning: Generating Memes Using Deep Neural Networks]"(https://arxiv.org/abs/1806.04510).

### Abstract

We introduce a novel meme generation system, which given any image can produce a humorous and relevant caption. Furthermore, the system can be conditioned on not only an image but also a user-defined label relating to the meme template, giving a handle to the user on meme content. The system uses a pretrained Inception-v3 network to return an image embedding which is passed to an attention-based deep-layer LSTM model producing the caption - inspired by the widely recognised Show and Tell Model. We implement a modified beam search to encourage diversity in the captions. We evaluate the quality of our model using perplexity and human assessment on both the quality of memes generated and whether they can be differentiated from real ones. Our model produces original memes that cannot on the whole be differentiated from real ones.

### Apple App available! Try the meme generator out for yourself, again see https://danklearning.com/

## Installation

Clone the repo

`git clone git@github.com:alpv95/MemeProject`

Now cd in

```bash
cd danklearning
```

Create a virtual environment and activate it

```bash
python -m virtualenv venv
source venv/bin/activate
```

Now install dependencies

```bash
pip install tensorflow Pillow jupyter
```

We've forked our own tf-coreml, which you can install in the virtual environment by doing the following:

```bash
cd tf-coreml/
pip install -e .
cd ..
```

Copy in the big files from wherever you got them (Google Drive)

```
cp [path]/REAL_EMBEDDING_MATRIX im2txt/REAL_EMBEDDING_MATRIX
cp -r [path]/trainlogIncNEW im2txt/trainlogIncNew
```

Now you should be all set up.

# Running

Get Jupyter notebooks fired up

```bash
jupyter notebook
```

Navigate in the browser that launched to `im2txt/model_conversion_debug.ipynb` and follow the steps in the notebook.