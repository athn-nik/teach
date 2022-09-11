#!/usr/bin/env bash

echo "Creating virtual environment"
python3.9 -m venv teach-env
echo "Activating virtual environment"

source $PWD/teach-env/bin/activate
$PWD/teach-env/bin/pip install --upgrade pip setuptools

$PWD/teach-env/bin/pip install torch==1.11.0 torchmetrics==0.7.2 torchvision==0.12.0 numpy==1.22.3
$PWD/teach-env/bin/pip install -r requirements.txt