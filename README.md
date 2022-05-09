# Simple Driving Model

## Dependencies

The main dependencies for this project are as follows:
```python
  - python=3.8.*
  - pip
  - matplotlib
  - numpy
  - matplotlib
  - pytorch==1.11.0 (cpu)
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


## Visualize the data

To parse and visualize the raw data held in `data/` you can run the `example.py` script as follows (pointing to one of the data (`.txt`) files). This will parse the recording and gather the data, then visualize several plots from the data in a new `results/` directory.

```bash
# generate some example outputs to results/
# needs a file to parse, can use any of the .txt's in data/
python example.py --file data/jacob21.txt
```

## Reproduce my results

To reproduce my results from the paper, after downloading the repository and installing the prerequisites, you should be able to simply run `python model.py --load` with the `--load` flag to used the saved model checkpoints I have saved for you. This will output all the results in the `results.model` directory as `.png`'s.

```bash
# load the cached model rather than training from scratch
python model.py --load
```

## Train from scratch (the long way)

To train the model from nothing (randomized weights), feel free to omit the `--load` flag which will tell the program to discard the saved checkpoints and train everything from 0. Note that this will take a very long time (for reference, it takes my 12 core desktop >1h in total) and the results may be different depending on which version and architecture of `pytorch` you have installed. I found that `pytorch 1.11.0` on MacOS still yielded slightly different results than `pytorch 1.11.0` on Linux, even though everything was seeded the same. 

```bash
# train the model from scratch (>1h total runtime)
python model.py # this will take a very long time
```