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
from multiprocessing.spawn import prepare
from re import I
import hydra
from omegaconf import DictConfig, OmegaConf
import teach.launch.prepare  # noqa
from teach.launch.prepare import get_last_checkpoint
from hydra.utils import to_absolute_path
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

@hydra.main(config_path="configs", config_name="train")
def _train(cfg: DictConfig):
    ckpt_ft = None
    if cfg.resume is not None:
        experiment, run_id = cfg.resume.split('/')[-2:]
        ckpt_ft = get_last_checkpoint(cfg.resume)
        cfg = OmegaConf.load(f'{cfg.resume}/.hydra/config.yaml')

        cfg.experiment = experiment
        cfg.run_id = run_id
        # this only works if you put the experiments in the same place
        # and then you change experiment and run_id also
        # not bad not good solution
        def_work_dir = '/'.join(cfg.path.working_dir.split('/')[:-2])
        cfg.path.working_dir = str(Path(def_work_dir) / experiment / run_id)

    cfg.trainer.enable_progress_bar = True
    return train(cfg, ckpt_ft)
 

def train(cfg: DictConfig, ckpt_ft: Optional[str] = None) -> None:
    logger.info("Training script. The outputs will be stored in:")
    working_dir = cfg.path.working_dir
    logger.info(f"The working directory is:{to_absolute_path(working_dir)}")
    logger.info("Loading libraries")
    import torch
    import pytorch_lightning as pl
    from hydra.utils import instantiate
    from teach.logger import instantiate_logger
    logger.info("Libraries loaded")

    logger.info(f"Set the seed to {cfg.seed}")
    pl.seed_everything(cfg.seed)
    logger.info("Loading data module")
    data_module = instantiate(cfg.data)
    logger.info(f"Data module '{cfg.data.dataname}' loaded")

    logger.info("Loading model")
    model = instantiate(cfg.model,
                        nfeats=data_module.nfeats,
                        _recursive_=False)
    logger.info(f"Model '{cfg.model.modelname}' loaded")
    logger.info("Loading logger")
    train_logger = instantiate_logger(cfg)
    # train_logger.begin(cfg.path.code_dir, cfg.logger.project, cfg.run_id)
    logger.info(f"Logger '{train_logger.lname}' ready")
    logger.info("Loading callbacks")

    # if not cfg.model.modelname == 'metatemos2' or cfg.model.losses.loss_on_transition:
    metric_monitor = {
        "Train_jf": "recons/text2jfeats/train",
        "Val_jf": "recons/text2jfeats/val",
        "Train_rf": "recons/text2rfeats/train",
        "Val_rf": "recons/text2rfeats/val",
        "APE root": "Metrics/APE_root",
        "APE mean pose": "Metrics/APE_mean_pose",
        "AVE root": "Metrics/AVE_root",
        "AVE mean pose": "Metrics/AVE_mean_pose"
    }
    # else:
    #     metric_monitor = {
    #         "Train_jf_0": "recons/text2jfeats_0/train",
    #         "Train_jf_1": "recons/text2jfeats_1/train",
    #         "Val_jf_0": "recons/text2jfeats_0/val",
    #         "Val_jf_1": "recons/text2jfeats_1/val",
    #         "Train_rf_0": "recons/text2rfeats_0/train",
    #         "Train_rf_1": "recons/text2rfeats_1/train",
    #         "Val_rf_0": "recons/text2rfeats_0/val",
    #         "Val_rf_1": "recons/text2rfeats_1/val",
    #         "APE root 0": "Metrics/APE_root_0",
    #         "APE root 1": "Metrics/APE_root_1",
    #         "APE mean pose 0 ": "Metrics/APE_mean_pose_0",
    #         "APE mean pose 1 ": "Metrics/APE_mean_pose_1",
    #         "AVE root 0": "Metrics/AVE_root_0",
    #         "AVE root 1": "Metrics/AVE_root_1",
    #         "AVE mean pose 0": "Metrics/AVE_mean_pose_0",
    #         "AVE mean pose 1": "Metrics/AVE_mean_pose_1"
    #     }

    callbacks = [
        instantiate(cfg.callback.progress, metric_monitor=metric_monitor),
        instantiate(cfg.callback.latest_ckpt),
        # instantiate(cfg.callback.best_ckpt, monitor="Metrics/APE_root_0"),
        # instantiate(cfg.callback.render),
        # added for LR monitoring --> does not work :(
        # instantiate(cfg.callback.lr_logging)
    ]

    logger.info("Callbacks initialized")

    logger.info("Loading trainer")
    trainer = pl.Trainer(
        **OmegaConf.to_container(cfg.trainer, resolve=True),
        logger=train_logger,
        callbacks=callbacks,
    )
    logger.info("Trainer initialized")

    logger.info("Fitting the model..")
    trainer.fit(model, datamodule=data_module, ckpt_path=ckpt_ft)
    logger.info("Fitting done")

    checkpoint_folder = trainer.checkpoint_callback.dirpath
    logger.info(f"The checkpoints are stored in {checkpoint_folder}")

    # train_logger.end(checkpoint_folder)
    logger.info(f"Training done. Reminder, the outputs are stored in:\n{working_dir}")


if __name__ == '__main__':
    _train()
