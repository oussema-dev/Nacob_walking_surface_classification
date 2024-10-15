# Walking surface classification

## Code support to a study using the Nacob dataset. More details are available on the published [paper](https://doi.org/10.1038/s41597-024-03683-5)

## Pre-requisites

cd to the root folder and run the following commands:
- `conda create -n nacob python=3.10 -y`
- `conda activate nacob`
- `conda install --y --file conda_requirements.txt`
- `pip install -r pip_requirements.txt`.

Download the raw data found in the [release](https://github.com/oussema-dev/Nacob_walking_surface_classification/releases/tag/1.0.0) and put it in the root folder

## Running the provided python scripts

To run a specific python script, use the command `python script.py` where `script.py` is the specific script to run