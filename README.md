# Installation

Clone the repo

`git clone git@github.com:freedmand/danklearning`

Now cd in

`cd danklearning`

Create a virtual environment and activate it

```bash
python -m virtualenv venv
source venv/bin/activate
```

Now install dependencies

```bash
pip install tensorflow Pillow jupyter
git clone git@github.com:tf-coreml/tf-coreml.git
cd tf-coreml/
pip install -e .
cd ..
```

Copy in the big files from wherever you got them (Google Drive)

```
cp [path]/REAL_EMBEDDING_MATRIX im2txt/REAL_EMBEDDING_MATRIX
cp -r [path]/trainlogIncNEW im2txt/trainlogIncNew
```