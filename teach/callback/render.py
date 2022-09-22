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
from pathlib import Path
from tkinter import font

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback
from teach.render import visualize_meshes, render_animation
from teach.tools.easyconvert import matrix_to
from teach.utils.file_io import read_json
from hydra.utils import get_original_cwd

logger = logging.getLogger(__name__)
plt_logger = logging.getLogger("matplotlib.animation")
plt_logger.setLevel(logging.WARNING)


# Can be run asynchronously 
def render_and_save(args, vid_format="mp4"):
    jts_or_vts, name, index, split, folder, fps, description, current_epoch = args
    fig_number = str(index).zfill(2)
    filename = f"{name}_{split}_{fig_number}.{vid_format}"
    output = folder / filename
    output = str(output.absolute())
    # Render
    if jts_or_vts.shape[1] > 25:
        output = visualize_meshes(jts_or_vts)
    else:
        render_animation(jts_or_vts, output=output, title=description, fps=30, 
                         figsize=(480/96, 480/96), fontsize=5)

    return output, fig_number, name, description


def log_to_none(path: str, log_name: str, fps: float,
                 global_step: int, train_logger,
                 vid_format, **kwargs):
    return


def log_to_wandb(path: str, log_name: str, caption: str, fps: float,
                 global_step: int, train_logger,
                 vid_format, **kwargs):
    import wandb
    train_logger.log({log_name: wandb.Video(path,
                                            fps=int(fps),
                                            format=vid_format, caption=caption),
                     "epoch": global_step}, step=global_step, commit=False)


def log_to_tensorboard(path: str, log_name: str, caption: str, fps: float,
                       global_step: int, train_logger,
                       vid_format, **kwargs):
    if vid_format == "gif":
        # Need to first load the gif by hand
        from PIL import Image, ImageSequence
        import numpy as np
        import torch
        # Load
        gif = Image.open(path)
        seq = np.array([np.array(frame.convert("RGB"))
                        for frame in ImageSequence.Iterator(gif)])
        vid = torch.tensor(seq)[None].permute(0, 1, 4, 2, 3)
    elif vid_format == "mp4":
        a = 1
        import ipdb
        ipdb.set_trace()

    # Logger name
    train_logger.add_video(log_name,
                           vid, fps=fps,
                           global_step=global_step)


class RenderCallback(Callback):
    def __init__(self, bm_path: str = None,
                 path: str = "visuals",
                 logger_type: str = "wandb",
                 save_last: bool = True,
                 vid_format: str = "mp4",
                 every_n_epochs: int = 20,
                 num_workers: int = 0,
                 nvids_to_save: int = 5,
                 fps: float = 30.0,
                 modelname = 'teach') -> None:

        if logger_type == "wandb":
            self.log_to_logger = log_to_wandb
        elif logger_type == "tensorboard":
            self.log_to_logger = log_to_tensorboard
        elif logger_type == "none":
            self.log_to_logger = log_to_none
        else:
            raise NotImplementedError("This logger is unknown, please use tensorboard or wandb.")

        self.logger_type = logger_type
        self.path = Path(path)
        self.path.mkdir(exist_ok=True)

        self.fps = fps
        self.nvids = nvids_to_save
        self.every_n_epochs = every_n_epochs
        self.num_workers = num_workers
        self.vid_format = vid_format
        self.save_last = save_last
        self.model = modelname
        if bm_path is not None:
            self.body_model_path = Path(bm_path) / 'smpl_models'

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule,
                           **kwargs) -> None:
        return self.call_renderer("train", trainer, pl_module)

    def on_validation_epoch_end(self, trainer: Trainer,
                                pl_module: LightningModule) -> None:
        return self.call_renderer("val", trainer, pl_module)

    def on_test_epoch_end(self, trainer: Trainer,
                          pl_module: LightningModule) -> None:
        return self.call_renderer("test", trainer, pl_module)

    def call_renderer(self, split: str, trainer: Trainer,
                      pl_module: LightningModule) -> None:
        if trainer.sanity_checking:
            return

        if self.nvids is None or self.nvids == 0:
            return

        # mid-epoch starting for finetuning
        # if pl_module.store_examples[split] is None:
        #     return
        
        logger.debug(f"Render {split} samples and log to {self.logger_type}")

        # Don't log epoch 0
        if trainer.current_epoch == 0 or trainer.current_epoch % self.every_n_epochs != 0:
            # Log last one (return = don't log, if it is not the last one)
            if trainer.current_epoch != (trainer.max_epochs - 1):
                return
            # Don't log last one if we don't want it
            elif not self.save_last:
                return
        # Extract the stored data
        # store_examples = pl_module.store_examples[split]
        text = []
        # ref_joints_or_verts = store_examples["ref"]
        from_text_joints_or_verts = []
        # from_motion_joints_or_verts = store_examples["from_motion"]

        # Prepare the folder
        folder = "epoch_" + str(trainer.current_epoch).zfill(3)
        folder = self.path / folder
        folder.mkdir(exist_ok=True)

        # Render + log
        # nvids = min(self.nvids, len(ref_joints_or_verts))
        render_list_train = ['1574-1', '7286-0', '6001-0', '4224-2', '3415-0', '2634-0', '2424-1', '4550-0']
        render_list_val = ['2307-1', '6078-0', '5210-0', '12255-0', '11346-2', '11671-1', '443-8', '3290-3', '2014-0', '973-12']
        pl_module.eval()
        for set_name, keyids in zip(['train', 'val'], [render_list_train, render_list_val]):
            for keyid in keyids:
                
                one_data = labels_dict[set_name][keyid]
                cur_lens = [one_data['len'][0], one_data['len'][1] + one_data['len'][2]]
                cur_texts = [one_data['text'][0], one_data['text'][1]]
                if cur_lens[0] < 15:
                    cur_lens[0] += 15 - cur_lens[0]

                verts = pl_module.forward_seq(cur_texts, cur_lens, align_full_bodies=False, return_type='joints')
 
                from_text_joints_or_verts.append(verts.detach().cpu().numpy())
                acts = ', '.join(cur_texts)
                acts = f'{acts}_{set_name}'
                text.append(acts)
        import multiprocessing
        nvids = len(from_text_joints_or_verts)
        list_of_logs = []
        num_workers = min(self.num_workers, 3 * nvids)
        with multiprocessing.Pool(num_workers) as pool:
            iterable = ((joints[index], name, index, split,
                            folder, self.fps, description,
                            trainer.current_epoch)
                        for joints, name in zip([from_text_joints_or_verts],
                                                ["text"])
                        for index, description in zip(range(nvids), text))
            for output, fig_number, name, desc in pool.imap_unordered(render_and_save, iterable):
                set = desc.split('_')[-1]
                log_name = f"visuals_actions/{set}_{fig_number}"
                list_of_logs.append((output, log_name, desc))

                train_logger = pl_module.logger.experiment

                # self.log_to_logger(path=output, log_name=log_name, caption=desc,
                #                    fps=self.fps, global_step=trainer.current_epoch,
                #                    train_logger=train_logger, vid_format=self.vid_format)
        for vid_path, panel_name, text_desc in list_of_logs: 
            log_name_start, split_id = panel_name.split('/')
            split, id = split_id.split('_') 
            log_name = f'{log_name_start}_{split}/{id}'
            self.log_to_logger(path=vid_path, log_name=log_name, caption=text_desc,
                                fps=self.fps, global_step=trainer.current_epoch,
                                train_logger=train_logger, vid_format=self.vid_format)

        defaults = ['walk', 'walk forwards', 'walk backwards', 'punch', 'kick', 'jumping jacks']
        durs = [30, 45, 60, 90, 120, 150]
        from_text_joints_or_verts = []
        text = []
        for t in defaults:
            for d in durs:
                cur_lens = [d]
                cur_texts = [t]
                verts = pl_module.forward_seq(cur_texts, cur_lens, align_full_bodies=False, return_type='joints')
                
                from_text_joints_or_verts.append(verts.detach().cpu().numpy())
                text.append(f'{t}_{d/30}')
        nvids = len(from_text_joints_or_verts)
        list_of_logs = []
        num_workers = min(self.num_workers, 3 * nvids)
        with multiprocessing.Pool(num_workers) as pool:
            iterable = ((joints[index], name, index, split,
                            folder, self.fps, description,
                            trainer.current_epoch)
                        for joints, name in zip([from_text_joints_or_verts],
                                                ["text"])
                        for index, description in zip(range(nvids), text))
            for output, fig_number, name, desc in pool.imap_unordered(render_and_save, iterable):
                log_name = f"simple_actions/{name}_{fig_number}"
                train_logger = pl_module.logger.experiment
                list_of_logs.append((output, log_name, desc))
                # self.log_to_logger(path=output, log_name=log_name, caption=desc,
                #                     fps=self.fps, global_step=trainer.current_epoch,
                #                     train_logger=train_logger, vid_format=self.vid_format)
        for vid_path, panel_name, text_dur in list_of_logs: 
            variant = panel_name.split('/')[0]
            action, dur = text_dur.split('_')
            log_name = f'{variant}/{action}_{dur}'
            self.log_to_logger(path=vid_path, log_name=log_name, caption=text_dur,
                                fps=self.fps, global_step=trainer.current_epoch,
                                train_logger=train_logger, vid_format=self.vid_format)
            
            # ###
            # groundtruth_motion = dataset.load_keyid(keyid, mode='inference')['datastruct']
            # ds = model.transforms.Datastruct(features=groundtruth_motion) 
            # gt_ds = ds.rots
            # gt_rots, gt_trans = gt_ds.rots, gt_ds.trans
            # from temos.transforms.smpl import RotTransDatastruct

            # final_datastruct = model.transforms.Datastruct(rots_=RotTransDatastruct(rots=gt_rots, trans=gt_trans))

            # gt_verts = final_datastruct.vertices
            ####

        # if self.num_workers == 0:
        # for jts_or_vts, name in zip([ref_joints_or_verts, from_text_joints_or_verts,
        #                           from_motion_joints_or_verts],
        #                         ['ref', 'from_text', 'from_motion']):
        #     for index, description in zip(range(nvids), text):
        #         output, fig_number, name, text_d = render_and_save((
        #                                                     jts_or_vts[index],
        #                                                     name, index,
        #                                                     split, folder,
        #                                                     self.fps,
        #                                                     description,
        #                                                     trainer.current_epoch))
        #         log_name = f"{split}_{fig_number}/{name}"
                
        #         train_logger = pl_module.logger.experiment if self.logger_type is not None else None
        #         self.log_to_logger(path=output, log_name=log_name, 
        #                            caption=text_d,
        #                            fps=self.fps,
        #                            global_step=trainer.current_epoch,
        #                            train_logger=train_logger,
        #                            vid_format=self.vid_format)
        # multiprocess does not work with pyrender probably it stucks forever
        # import multiprocessing
        # num_workers = min(self.num_workers, 6 * nvids)
        # breakpoint()
        # with multiprocessing.Pool(num_workers) as pool:
        #     iterable = ((self.body_model_path,
        #                     hm_repr[0][index], hm_repr[1][index], 
        #                     name, index, split,
        #                     folder, self.fps, description,
        #                     trainer.current_epoch)
        #                     for hm_repr, name in zip([ref_joints_or_verts, 
        #                                     from_text_joints_or_verts],
        #                             ['ref', 'text_r'])
        #                 for index, description in zip(range(nvids), text))
        #     for output, fig_number, name, text_d in pool.imap_unordered(render_and_save, iterable):
        #         log_name = f"{split}_{fig_number}/{name}/"
        #         train_logger = pl_module.logger.experiment
        #         self.log_to_logger(path=output, log_name=log_name,
        #                             caption=text_d,
        #                             fps=self.fps,
        #                             global_step=trainer.current_epoch,
        #                             train_logger=train_logger,
        #                             vid_format=self.vid_format)
 
