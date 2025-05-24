#!/bin/bash
# Commands to setup a new virtual environment and install all the necessary packages

set -e

pip install --upgrade pip

python3.12 -m venv fmri
source fmri/bin/activate

pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git --no-deps
