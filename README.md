# Simple Driving Model

## Dependencies

The main dependencies for this project are as follows:
```python
  - python=3.8.*
  - pip
  - matplotlib
  - numpy
  - matplotlib
  - pytorch
  - torchvision
  - captum=0.5.0=0
  - scikit-learn
  - scipy
```
These should all be installable via `pip` or `conda`.

It is recommended to use `conda` as a Python package manager. 

Then you can simply execute
```bash
conda env create -f environment.yml
conda activate model
```


## How to use?

```bash

# generate some example outputs to results/
# needs a file to parse, can use any of the .txt's in data/
python example.py --file data/jacob21.txt

# load the cached model rather than training from scratch
python model.py --load True --eval True

# train the model from scratch (>1h total runtime)
python mode.py --eval True # this will take a very long time
```