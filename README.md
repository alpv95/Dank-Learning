# Dank Learning: Generating Memes with Machine Learning

## Overview

This is the code for the paper "[Dank Learning: Generating Memes Using Deep Neural Networks]"(https://arxiv.org/abs/1806.04510).

![Alt text](Picture1.png?raw=true "Title")
![Alt text](Picture2.png?raw=true "Title")
![Alt text](Picture3.png?raw=true "Title")
![Alt text](Picture4.png?raw=true "Title")

## Abstract

We introduce a novel meme generation system, which given any image can produce a humorous and relevant caption. Furthermore, the system can be conditioned on not only an image but also a user-defined label relating to the meme template, giving a handle to the user on meme content. The system uses a pretrained Inception-v3 network to return an image embedding which is passed to an attention-based deep-layer LSTM model producing the caption - inspired by the widely recognised Show and Tell Model. We implement a modified beam search to encourage diversity in the captions. We evaluate the quality of our model using perplexity and human assessment on both the quality of memes generated and whether they can be differentiated from real ones. Our model produces original memes that cannot on the whole be differentiated from real ones.
