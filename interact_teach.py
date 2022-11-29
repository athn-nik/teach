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
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
import teach.launch.prepare  # noqa
from hydra.utils import get_original_cwd
from teach.render.video import save_video_samples
from teach.render.mesh_viz import visualize_meshes
import torch
from tqdm import tqdm
import pytorch_lightning as pl

logger = logging.getLogger(__name__)

@hydra.main(config_path="configs", config_name="interact_teach")
def _interact(cfg: DictConfig):
    return interact(cfg)


def interact(newcfg: DictConfig) -> None:
    # Verify arguments
    for arg in ['folder', 'texts', 'output','durs']:
        newcfg[arg]
    # Load last config
    output_dir = Path(hydra.utils.to_absolute_path(newcfg.folder))
    last_ckpt_path = newcfg.last_ckpt_path
    prevcfg = OmegaConf.load(output_dir / ".hydra/config.yaml")
    # Overload it
    cfg = OmegaConf.merge(prevcfg, newcfg)

    import pytorch_lightning as pl
    import numpy as np
    from hydra.utils import instantiate
    pl.seed_everything(cfg.seed)

    logger.info("Loading model")
    # Instantiate all modules specified in the configs
    
    model = instantiate(cfg.model,
                        nfeats=135,
                        logger_name="none",
                        nvids_to_save=None,
                        _recursive_=False)
    logger.info(f"Model '{cfg.model.modelname}' loaded")

    # Load the last checkpoint
    last_ckpt_path = output_dir / "checkpoints" / "last.ckpt"
    model = model.load_from_checkpoint(last_ckpt_path)
    output_type = cfg.repr_type
    if output_type != 'smpl':
        model.transforms.rots2joints.jointstype = output_type

    logger.info("Model weights restored")

    durations = cfg.durs
    texts = cfg.texts
    lengths = [int(d*cfg.train_fps) for d in durations]
    assert len(lengths) == len(texts)
    model.eval()
    from teach.transforms.smpl import RotTransDatastruct
    with torch.no_grad():
        for index in range(cfg.samples):
            pl.seed_everything(index)

            motion = model.forward_seq(texts, lengths,
                                       align_full_bodies=True,
                                       slerp_window_size=cfg.slerp_ws,
                                       return_type=output_type)
            outd = Path(cfg.output).absolute()

            if output_type == 'smpl':

                np.save(f'{str(outd)}_sample-{index}.npy',
                        {'vertices': motion['vertices'].numpy(),
                         'rots': motion['rots'].numpy(),
                         'transl': motion['transl'].numpy(),
                         'text': texts,
                         'lengths': lengths} 
                        )
                motion = motion['vertices'].numpy()
            else:
                np.save(f'{str(outd)}_sample-{index}.npy',
                        {'motion': motion.numpy(), 'text': texts, 'lengths': lengths} )

            vid_ = visualize_meshes(motion.numpy())
            save_video_samples(vid_, f'{str(outd)}_sample-{index}.mp4', texts, fps=30)


if __name__ == '__main__':
    _interact()
