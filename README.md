# TEACH
Official PyTorch implementation of the paper "TEACH: Temporal Action Compositions for 3D Humans" 
# TEACH: Temporal Action Compositions for 3D Humans [3DV-2022]
[![report](https://img.shields.io/badge/arxiv-report-red)](https://arxiv.org/abs/1912.05656) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1dFfwxZ52MN86FA6uFNypMEdFShd2euQA) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/vibe-video-inference-for-human-body-pose-and/3d-human-pose-estimation-on-3dpw)](https://paperswithcode.com/sota/3d-human-pose-estimation-on-3dpw?p=vibe-video-inference-for-human-body-pose-and)

Check our YouTube videos below for more details.

| Paper Video                                                                                                | Qualitative Results                                                                                                |
|------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| [![PaperVideo](https://img.youtube.com/vi/rIr-nX63dUA/0.jpg)](https://www.youtube.com/watch?v=rIr-nX63dUA) |



## Features


This implementation:
-
-

## Updates

- 05/01/2021: Windows installation tutorial is added thanks to amazing [@carlosedubarreto](https://github.com/carlosedubarreto)
- 06/10/2020: Support OneEuroFilter smoothing.
- 14/09/2020: FBX/glTF conversion script is released.

## Getting Started
VIBE has been implemented and tested on Ubuntu 18.04 with python >= 3.7. It supports both GPU and CPU inference.
If you don't have a suitable device, try running our Colab demo. 

Clone the repo:
```bash
git clone https://github.com/athn-nik/teach.git
```

Install the requirements using `virtualenv` or `conda`:
```bash
# pip
source scripts/install_pip.sh
```

## Running the Demo

We have prepared a nice demo code to run VIBE on arbitrary videos. 
First, you need download the required data(i.e our trained model and SMPL model parameters). To do this you can just run:

```bash
source scripts/prepare_data.sh
```

Then, running the demo is as simple as:

```bash

# Run on your description
python demo.py 
```

Refer to [`doc/demo.md`](doc/demo.md) for more details about the demo code.

## Google Colab
If you do not have a suitable environment to run this project then you could give Google Colab a try. 
It allows you to run the project in the cloud, free of charge. You may try our Colab demo using the notebook we have prepared: 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1dFfwxZ52MN86FA6uFNypMEdFShd2euQA)


## Training
Run the commands below to start training:

```shell script
source scripts/prepare_training_data.sh
python train.py --cfg configs/config.yaml
```

Note that the training datasets should be downloaded and prepared before running data processing script.
Please see [`doc/train.md`](doc/train.md) for details on how to prepare them.
 
## Evaluation

See [`doc/eval.md`](doc/eval.md) to reproduce the results in this table or 
evaluate a pretrained model.

## Citation

```bibtex
@inproceedings{athanasiou2022teach,
  title={TEACH: Temporal Action Compositions for 3D Humans},
  author={Athanasiou, Nikos and Petrovich, Mathis and Black, Michael J. and Varol, Gul},
  booktitle = {International Conference on 3D Vision (3DV)},
  month = {September},
  year = {2022}
}
```

## License
This code is available for **non-commercial scientific research purposes** as defined in the [LICENSE file](LICENSE). By downloading and using this code you agree to the terms in the [LICENSE](LICENSE). Third-party datasets and software are subject to their respective licenses.


## References
We indicate if a function or script is borrowed externally inside each file. Here are some great resources we 
benefit:

- Pretrained HMR and some functions are borrowed from [SPIN](https://github.com/Mathux/TEMOS).
- SMPL models and layer is from [SMPL-X model](https://github.com/vchoutas/smplx).
