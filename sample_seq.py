# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2020 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import logging

import hydra
import os
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from teach.data.babel import plot_timeline
from teach.data.tools.collate import collate_length_and_text
import teach.launch.prepare
from teach.render.mesh_viz import visualize_meshes
from teach.render.video import save_video_samples, stack_vids
import torch
from teach.utils.inference import test_set_seqs_nowalk, test_set_seqs_walk
from teach.utils.file_io import read_json
labels = read_json('deps/inference/labels.json')

logger = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="sample_seq")
def _sample(cfg: DictConfig):
    return sample(cfg)

def get_path(sample_path: Path, split: str, onesample: bool, mean: bool, fact: float):
    extra_str = ("_mean" if mean else "") if onesample else "_multi"
    fact_str = "" if fact == 1 else f"{fact}_"
    path = sample_path / f"{fact_str}{split}{extra_str}"
    return path


def cfg_mean_nsamples_resolution(cfg):
    if cfg.mean and cfg.number_of_samples > 1:
        logger.error("All the samples will be the mean.. cfg.number_of_samples=1 will be forced.")
        cfg.number_of_samples = 1

    return cfg.number_of_samples == 1

def sample(newcfg: DictConfig) -> None:
    # Load last config

    output_dir = Path(hydra.utils.to_absolute_path(newcfg.folder))
    last_ckpt_path = newcfg.last_ckpt_path
    # use this path for oracle experiment on kit-xyz
    # '/is/cluster/work/nathanasiou/experiments/temos-original/last.ckpt'
    # Load previous config
    prevcfg = OmegaConf.load(output_dir / ".hydra/config.yaml")
    # use this path for oracle experiment on kit-xyz
    # OmegaConf.load('/is/cluster/work/nathanasiou/experiments/temos-original/config.yaml')

    # Overload it
    cfg = OmegaConf.merge(prevcfg, newcfg)

    fact_str = "" if cfg.fact == 1 else f"{cfg.fact}_"
    onesample = cfg_mean_nsamples_resolution(cfg)
    if cfg.mean and cfg.number_of_samples > 1:
        logger.error("All the samples will be the mean.. cfg.number_of_samples=1 will be forced.")
        cfg.number_of_samples = 1

    # Get back run id
    run_id = Path(output_dir).name
    logger.info("Sample script. The outputs will be stored in:")
    eval_pair = '_pairs' if cfg.evaluate_pairs else ''
    slerp_ws = '_slerp' if cfg.slerp_ws is not None else '_no-slerp'
    align = '_aligned' if cfg.align == 'full' else '_unaligned'
    if cfg.ckpt_name != 'last.ckpt':
        logger.info("Evaluating on a differnt than last checkpoint")
        ckpt_fd = f'checkpoint-{cfg.ckpt_name}'
    else:
        ckpt_fd = 'checkpoint-last'

    if cfg.jointstype == 'vertices' :
        storage = output_dir / f'samples_vertices{slerp_ws}{align}{eval_pair}/{ckpt_fd}'
    else:
        storage = output_dir / f'samples{slerp_ws}{align}{eval_pair}/{ckpt_fd}'
    storage.mkdir(exist_ok=True,  parents=True)

    path = get_path(storage, cfg.split if cfg.set is None else cfg.set,
                    onesample, cfg.mean, cfg.fact)

    path.mkdir(exist_ok=True, parents=True)

    logger.info(f"{path}")

    import pytorch_lightning as pl
    import numpy as np
    from hydra.utils import instantiate
    pl.seed_everything(cfg.seed)

    logger.info("Loading data module")
    # only pair evaluation to be fair
    if cfg.data.dtype in ['pairs', 'pairs_only']:
        # if cfg.evaluate_pairs:
        #     cfg.data.dtype = 'separate_pairs'
        # else:
        cfg.data.dtype = 'separate_pairs'


    # single motion model + slerp baseline
    if cfg.model.modelname == 'temos' and cfg.evaluate_pairs:
        cfg.data.dtype = 'separate_pairs'

    if cfg.model.modelname == 'temos' and cfg.evaluate_pairs and 'pairs' in cfg.data.dtype:
        cfg.data.dtype = 'separate_pairs'

    if cfg.model.modelname == 'temos' and cfg.evaluate_pairs:
        cfg.data.dtype = 'separate_pairs'

    if cfg.set == 'submission':
        cfg.split = 'test'

    data_module = instantiate(cfg.data)
    logger.info(f"Data module '{cfg.data.dataname}' loaded")

    from teach.data.tools.collate import collate_datastruct_and_text, collate_pairs_and_text

    if cfg.set == 'submission':
        dataset_test = getattr(data_module, "test_dataset")
        dataset_val = getattr(data_module, "val_dataset")
    else:
        dataset = getattr(data_module, f"{cfg.split}_dataset")

    seqids = test_set_seqs_walk + test_set_seqs_nowalk
    from tqdm import tqdm

    logger.info("Loading model")
    # Instantiate all modules specified in the configs
    model = instantiate(cfg.model,
                        nfeats=data_module.nfeats,
                        logger_name="none",
                        nvids_to_save=None,
                        _recursive_=False)

    logger.info(f"Model '{cfg.model.modelname}' loaded")

    # Load the last checkpoint
    model = model.load_from_checkpoint(last_ckpt_path)
    model.eval()
    align_full_bodies = True if cfg.align == 'full' else False
    align_trans = True if cfg.align == 'trans' else False

    logger.info("Model weights restored")
    model.sample_mean = cfg.mean
    model.fact = cfg.fact

    if not model.hparams.vae and cfg.number_of_samples > 1:
        raise TypeError("Cannot get more than 1 sample if it is not a VAE.")

    trainer = pl.Trainer(**OmegaConf.to_container(cfg.trainer, resolve=True))

    logger.info("Trainer initialized")
    
    model.transforms.rots2joints.jointstype = cfg.jointstype

    if cfg.set == 'submission':
        from teach.utils.inference import pairs_figures
        keyids = pairs_figures 
    else:
        keyids = dataset.keyids

    ommited = 0
    with torch.no_grad():
        for keyid in (pbar := tqdm(keyids)):
            pbar.set_description(f"Processing {keyid}")

            if cfg.set == 'submission':
                # for submission time renderings
                if keyid in dataset_val.keyids:
                    one_data = dataset_val.load_keyid(keyid, mode='inference')
                elif keyid in dataset_test.keyids:
                    one_data = dataset_test.load_keyid(keyid, mode='inference')
            else:
                one_data = dataset.load_keyid(keyid, mode='inference')
            # buggy gt
            if one_data['length_0'] == 1 or one_data['length_1'] == 1 :
                logger.info(f'Omitted {keyid}')
                ommited += 1
                continue
            if not cfg.naive:
                batch = collate_pairs_and_text([one_data])
                cur_lens = [batch['length_0'][0], batch['length_1'][0] + batch['length_transition'][0]]
                cur_texts = [batch['text_0'][0], batch['text_1'][0]]
            else:
                batch = collate_length_and_text([one_data])
                cur_texts = [f"{batch['text_0'][0]}, {batch['text_1'][0]}"]
                cur_lens = [batch['length_0'][0]+ batch['length_1'][0] + batch['length_transition'][0]]
                ds_sample = {'text': cur_texts,
                             'length': cur_lens }

            # batch_size = 1 for reproductability
            for index in range(cfg.number_of_samples):
                # fix the seed
                pl.seed_everything(index)
                from teach.transforms.smpl import RotTransDatastruct
                if cfg.naive:
                    motion = model(ds_sample)[0]
                else:
                    motion = model.forward_seq(cur_texts, cur_lens,
                                            align_full_bodies=align_full_bodies,
                                            align_only_trans=align_trans,
                                            slerp_window_size=cfg.slerp_ws)

                # only visuals in this branch
                # if cfg.jointstype == "vertices": # defaults
                if cfg.number_of_samples > 1:
                    npypath = path / f"{keyid}_{index}.npy"
                else:
                    npypath = path / f"{keyid}.npy"
                    np.save(npypath, {'motion': motion.numpy(), 'text': cur_texts, 'lengths': cur_lens} )

    logger.info("All the sampling are done")

    logger.info(f"All the sampling are done. You can find them here:\n{path}")


if __name__ == '__main__':
    _sample()
