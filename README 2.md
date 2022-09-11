# TEmporal Action Compositions for 3D Humans
```python interact_teach.py folder=/path/to/experiment output=/path/to/sample.npy texts='[step on the left, look right, wave with right hand]' durs='[1.5, 1.5, 1.5]'```
## Environment
Create the environment:

`python3.9 -m venv ~/.venvs/teach`

Activate it

`source ~/.venvs/teach/bin/activate`

Make sure `setuptools` and `pip` are the latest.

`pip install --upgrade pip setuptools`

 Install the packages:

`pip install -r requirements.txt`

Do this:

```shell
cd deps/
git lfs install
git clone https://huggingface.co/distilbert-base-uncased
cd ..
```
CUDA 10.2, Pytorch 1.11.0

## Data
Download the data from [AMASS website](amass.is.tue.mpg.de).

```shell
python divotion/dataset/process_amass.py --input-path /your path --output-path /out/path --model-type smplh --use-betas
```

Download the data from [BABEL website](babel.is.tue.mpg.de)[GET IT FROM ME]:

```shell
python divotion/dataset/add_babel_labels.py --input-path /is/cluster/nathanasiou/data/amass/processed_amass_smplh_wshape_30fps --out-path /is/cluster/nathanasiou/data/babel/babel-smplh30fps-gender --babel-path /is/cluster/nathanasiou/data/babel/babel_v2.1/
```

Softlink the data or copy them based on where you have them. You should have a data folder with the structure:
```
|-- amass
|   |-- processed_amass_smplh_wshape_30fps
|-- babel
|   |-- babel-smplh30fps-gender
|   |-- babel_v2.1
|-- smpl_models
|   |-- markers_mosh
|   |-- README.md
|   |-- smpl
|   |-- smplh
|   `-- smplx
```

Be careful not to push any data! To softlink your data, do:

`ln -s /is/cluster/nathanasiou/data`

## Training
To start training after activating your environment. Do:

`python train.py experiment=baseline logger=none`

Explore `configs/train.yaml` to change some basic things like where you want
your output stored, which data you want to choose if you want to do a small
experiment on a subset of the data etc.

## GENERATE SAMPLES
For sampling do.

`python sample_seq.py folder=/path/to/experiment align=full slerp_ws=8`

In general it is: `folder_our/<project>/<dataname_config>/<experimet>/<run_id>`

The folder should point to the output folder you 've chosen in `train.yaml` for out-of-the-box sampling. This will save joint positions in `.npy` files.

## EVALUATE
After sampling, to get numbers you can do:

`python eval.py folder=/is/cluster/work/nathanasiou/experiments/teach/babel-amass/babel-full/rot-5secs-full-sched-lr/ number_of_samples=3 fact=1.1 `

You need to point to the folder of the experiment only!

To submit a single experiment to cluster:
`python cluster/single_run.py --expname babel-default --run-id first-try-full --extras data=babel-amass`

Please follow this in `train.yaml`:

## BLENDER RENDERING
To render with blender an example is(BABEL):

`blender --background --python render_video.py -- folder=/is/cluster/nathanasiou/experimentals/teach/babel-amass/baseline/o234tnul/samples_skin/test/CMU/CMU/28/28_11_poses.npz-8593_allground_objs fast=false high_res=true`

To render with blender an example is(KIT):

`blender --background --python render_video.py -- folder=/is/cluster/nathanasiou/experimentals/teach/kit-mmm-xyz/baseline/2awlfcm9/samples/test fast=false high_res=true`


### Global configurations shared between different modules

- `experiment: the experiment name overall`

- `run_id: specific info about the current run` (wandb name)

## FOR CLUSTER SINGLE EXPERIMENT TRAINING:

`python cluster/single_run.py --expname babel-default --run-id debugging --extras data=babel-amass data.batch_size=123 --mode train`

## OR SAMPLING:

`python cluster/single_run.py --folder folder/to/experiment --mode sample`

## FAST RENDERING OF RESULTS 

For fast rendering(less than 30" / per video):

`python render_video_fast.py dim=2 folder=/ps/scratch/nathanasiou/oracle_experiments/kit-temos-version/1v6vh9w2/path/to/npys/files`

and check the durs key in the configuration to add durations. You can do `durs='[<dur1_in_secs>, <dur2_in_secs>, ...]'`. The video outputs will be saved in the absolute output directory.

Your experiments will be always structured like this(see `train.yaml`):
`<project>/<dataname>/<experiment>/<run_id>`


## BLENDER SETUP
`./python3.10 -m ensurepip`
`/is/cluster/work/nathanasiou/blender/blender-3.1.2-linux-x64/3.1/python/bin » ./python3.10 -m pip install moviepy`  
`blender or cluster_blender --background --python render.py -- npy=path/to/file.npy`
`python cluster/single_run.py --folder /some/folder/path/or/file --mode render --bid 100`
