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

from typing import Dict, Optional
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers import WandbLogger as _pl_WandbLogger
import os
from pathlib import Path
import time

# Fix the step logging
class WandbLogger(_pl_WandbLogger):
    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
 
        assert rank_zero_only.rank == 0, "experiment tried to log from global_rank != 0"
        metrics = _add_prefix(metrics, self._prefix, self.LOGGER_JOIN_CHAR)

        # if 'epoch' not in metrics:
        #     self.experiment.log({**metrics, "trainer/global_step": step},
        #                             step=step)
        # else:
        wandb_step = int(metrics["epoch"])
                
        if step is not None:
            self.experiment.log({**metrics, "trainer/global_step": step},
                                step=wandb_step)
        else:
            self.experiment.log(metrics, step=wandb_step)

    @property
    def name(self) -> Optional[str]:
        """ Override the method because model checkpointing define the path before
        the initialization, and in offline mode you can't get the good path
        """
        # don't create an experiment if we don't have one
        # return self._experiment.project_name() if self._experiment else self._name
        return self._wandb_init["project"]

    def symlink_checkpoint(self, code_dir, project, run_id):
        #  this is the hydra run dir!! see train.yaml
        local_project_dir = Path("wandb") / project
        local_project_dir.mkdir(parents=True, exist_ok=True)
        
        # ... but code_dir is the current dir see path.yaml
        Path(code_dir) / project / run_id
        os.symlink(Path(code_dir) / "wandb" / project / run_id,
                   local_project_dir / f'{run_id}_{time.strftime("%Y%m%d%H%M%S")}')
        # # Creating a another symlink for easy access
        os.symlink(Path(code_dir) / "wandb" / project / run_id / "checkpoints",
                   Path("checkpoints"))
        # if it exists an error is spawned which makes sense, but ...
        
    def symlink_run(self, checkpoint_folder: str):

        code_dir = checkpoint_folder.split("wandb/")[0]
        # # local run
        local_wandb = Path("wandb/wandb")
        local_wandb.mkdir(parents=True, exist_ok=True)
        offline_run = self.experiment.dir.split("wandb/wandb/")[1].split("/files")[0]
        # # Create the symlink
        os.symlink(Path(code_dir) / "wandb/wandb" / offline_run, local_wandb / offline_run)

    def begin(self, code_dir, project, run_id):
        self.symlink_checkpoint(code_dir, project, run_id)

    def end(self, checkpoint_folder):
        self.symlink_run(checkpoint_folder)
