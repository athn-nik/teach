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
import teach.launch.prepare
from teach.render.mesh_viz import visualize_meshes
from teach.render.video import save_video_samples, stack_vids
import torch
logger = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="compute_td")
def _sample(cfg: DictConfig):
    return sample(cfg)


def sample(newcfg: DictConfig) -> None:
    logger.info("Compute distance script")
    output_dir = Path(hydra.utils.to_absolute_path(newcfg.folder))
    last_ckpt_path = newcfg.last_ckpt_path
    prevcfg = OmegaConf.load(output_dir / ".hydra/config.yaml")
    cfg = OmegaConf.merge(prevcfg, newcfg)

    logger.info("Loading packages")
    import pytorch_lightning as pl
    import numpy as np
    from hydra.utils import instantiate
    import torch
    pl.seed_everything(cfg.seed)
    logger.info("Loading data module")

    cfg.data.dtype = 'separate_pairs'
    data_module = instantiate(cfg.data)
    logger.info(f"Data module '{cfg.data.dataname}' loaded")

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
    logger.info("Model weights restored")
    model.transforms.rots2joints.jointstype = cfg.jointstype
    model.eval()
    logger.info(f"Put in eval mode and will produce {model.transforms.rots2joints.jointstype}")

    if cfg.jointstype == "vertices":
        return_type = "vertices"
    else:
        return_type = "joints"

    # test a dummy example
    with torch.no_grad():
        if cfg.model.modelname == 'temos' and cfg.naive:
            mjoints = model.forward_seq(["walk, jump"], [30+20], return_type=return_type)
        else:
            mjoints = model.forward_seq(["walk", "jump"], [30, 20],
                                        align_full_bodies=cfg.align_full_bodies,
                                        align_only_trans=cfg.align_only_trans,
                                        slerp_window_size=cfg.slerp_window_size,
                                        return_type=return_type)

    if cfg.align_full_bodies:
        option_text = "with aligning on rotation and translation"
    elif cfg.align_only_trans:
        option_text = "with aligning on translation only"
    else:
        option_text = "without any alignement"
    if cfg.slerp_window_size is None:
        option_text += " without slerp"
    else:
        option_text += f" with slerp, with a window size of {cfg.slerp_window_size}"

    logger.info(f"Computing distance on {output_dir} model {option_text}")
    logger.info(f"Computing distance on {cfg.model.modelname} model {option_text}")
    dataset = getattr(data_module, f"{cfg.split}_dataset")

    from teach.data.sampling import upsample
    from tqdm import tqdm
    ommited = 0
    transition_distance = 0
    nframes_for_computing_distance = 0

    # remove printing for changing the seed
    logging.getLogger('pytorch_lightning.utilities.seed').setLevel(logging.WARNING)
    with torch.no_grad():
        for keyid in (pbar := tqdm(dataset.keyids)):
            pbar.set_description(f"Processing {keyid}")
            one_data = dataset.load_keyid(keyid, mode='inference')

            # dataset.dtype == 'separate_pairs'
            if one_data['length_0'] == 1 or one_data['length_1'] == 1 :
                logger.info(f'Omitted {keyid}')
                ommited += 1
                continue

            a1 = one_data['text_0']
            a2 = one_data['text_1']
            l1 = one_data['length_0']
            l2 = one_data['length_1']
            l_trans = one_data['length_transition']

            # fix the seed
            pl.seed_everything(0)
            from teach.transforms.smpl import RotTransDatastruct

            if cfg.model.modelname == 'temos' and cfg.naive:
                # extra parametrs does not matter as we produce one motion only
                mjoints = model.forward_seq([f'{a1}, {a2}'], [l1+l2+l_trans], return_type=return_type)
            else:
                # + params etc TODO check
                mjoints = model.forward_seq([a1, a2], [l1, l2+l_trans],
                                            align_full_bodies=cfg.align_full_bodies,
                                            align_only_trans=cfg.align_only_trans,
                                            slerp_window_size=cfg.slerp_window_size,
                                            return_type=return_type)

            # check the transition frame distance
            # for temos naive, the transition distance should be the lowest
            transition_distance += torch.linalg.norm(mjoints[l1]-mjoints[l1-1], dim=-1).mean()
            nframes_for_computing_distance += 1
    mean_trans_dist = transition_distance / nframes_for_computing_distance
    logger.info(f"Transition distance is: {mean_trans_dist*1000}, {mean_trans_dist}")
    logger.info(f'Number of buggy groundtruth: {ommited}')


if __name__ == '__main__':
    _sample()
