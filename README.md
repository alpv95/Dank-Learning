# Dank Learning: Generating Memes with Machine Learning
See https://danklearning.com/ for more information.
## Apple App available! Try the meme generator out for yourself, again see https://danklearning.com/
Paper found at https://arxiv.org/abs/1806.04510 or https://web.stanford.edu/class/cs224n/reports/6909159.pdf


![Alt text](Picture1.png?raw=true "Title")
![Alt text](Picture2.png?raw=true "Title")
![Alt text](Picture3.png?raw=true "Title")
![Alt text](Picture4.png?raw=true "Title")
# Installation

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
