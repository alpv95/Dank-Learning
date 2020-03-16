# Installation

Clone the repo

`git clone git@github.com:freedmand/danklearning`

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

Now you should be all set up. Sick.

# Running

Get Jupyter notebooks fired up

```bash
jupyter notebook
```

Navigate in the browser that launched to `im2txt/model_conversion_debug.ipynb` and follow the steps in the notebook.