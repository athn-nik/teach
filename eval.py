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
from statistics import mode
import yaml
import hydra
import os
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import teach.launch.prepare  # noqa
from tqdm import tqdm
logger = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="eval")
def _eval(cfg: DictConfig):
    return eval(cfg)


def regroup_metrics(metrics):
    from teach.info.joints import mmm_joints
    pose_names = mmm_joints[1:]
    dico = {key: val.numpy() for key, val in metrics.items()}

    APE_pose = dico.pop("APE_pose")
    APE_joints = dico.pop("APE_joints")

    for name, ape in zip(pose_names, APE_pose):
        dico[f"APE_pose_{name}"] = ape

    for name, ape in zip(mmm_joints, APE_joints):
        dico[f"APE_joints_{name}"] = ape

    AVE_pose = dico.pop("AVE_pose")
    AVE_joints = dico.pop("AVE_joints")

    for name, ave in zip(pose_names, AVE_pose):
        dico[f"AVE_pose_{name}"] = ave

    for name, ape in zip(mmm_joints, AVE_joints):
        dico[f"AVE_joints_{name}"] = ave

    return dico

def sanitize(dico):
    dico = {key: "{:.5f}".format(float(val)) for key, val in dico.items()}
    return dico

def get_samples_folder(path, eval_pairs, slerp, ckpt, align, * , jointstype):
    if jointstype == "vertices":
        raise ValueError("No evaluation for vertices, sample the joints instead.")

    output_dir = Path(hydra.utils.to_absolute_path(path))
    candidates = [x for x in os.listdir(output_dir) if "samples" in x]
    if not candidates:
        raise ValueError("There is no samples for this model.")

    # amass = False
    # for candidate in candidates:
    #     amass = amass or ("amass" in candidate)

    # if amass:
    #     samples_path = output_dir / f"amass_samples_{jointstype}"
    #     if not samples_path.exists():
    #         jointstype = "mmm"
    #         samples_path = output_dir / f"amass_samples_mmm"
    #         if not samples_path.exists():
    #             raise ValueError("You must specify a correct jointstype.")
    #         logger.info(f"Samples from {jointstype} not found, take mmm instead.")
    # else:
    use_slerp = '_slerp' if slerp else '_no-slerp'
    align = '_aligned' if align else '_unaligned'# == 'full' else '_unaligned'
    if eval_pairs:
        samples_path = output_dir / f"samples{use_slerp}{align}_pairs" / f'checkpoint-{ckpt}'
    else:
        samples_path = output_dir / f"samples{align}" / f'checkpoint-{ckpt}'

    return samples_path, jointstype

def get_metric_paths(sample_path: Path, eval_pairs: bool, split: str, onesample: bool, mean: bool, fact: float):
    extra_str = ("_mean" if mean else "") if onesample else "_multi"
    fact_str = "" if fact == 1 else f"{fact}_"
    metric_str = "babel_metrics" if not eval_pairs else 'babel_metrics_pairs'# if amass else "metrics"

    if onesample:
        file_path = f"{fact_str}{metric_str}_{split}{extra_str}"
        save_path = sample_path / file_path
        return save_path
    else:
        file_path = f"{fact_str}{metric_str}_{split}_multi"
        avg_path = sample_path / (file_path + "_avg")
        best_path = sample_path / (file_path + "_best")
        return avg_path, best_path


def save_metric(path, metrics):
    strings = yaml.dump(metrics, indent=4, sort_keys=False)
    with open(path, "w") as f:
        f.write(strings)



def foot_skate(jts_seq):
    import numpy as np
    ## get vertices at the feetbottom
    from teach.info.joints import mmm_joints_info
    feetidx = mmm_joints_info['feet']
    feet_jts = jts_seq[:,feetidx,:]
    verts_feet_horizon_vel = np.linalg.norm(feet_jts[1:, :, :-1]-feet_jts[:-1,:, :-1], axis=-1)[14:]
    verts_feet_height = jts_seq[15:,feetidx,-1]
    thresh_height = 5e-2
    thresh_vel = 5e-3
    skating = (verts_feet_horizon_vel>thresh_vel)*(np.abs(verts_feet_height)<thresh_height)
    skating = np.sum(np.logical_and(skating[:,0], skating[:,1])) /45

    return skating

def eval(cfg: DictConfig) -> None:
    logger.info(f"Evaluation script.")
    # Load last config
    output_dir = Path(hydra.utils.to_absolute_path(cfg.folder))
    # Load previous config
    prevcfg = OmegaConf.load(output_dir / ".hydra/config.yaml")
    # Overload it
    cfg = OmegaConf.merge(prevcfg, cfg)

    from sample import cfg_mean_nsamples_resolution, get_path
    onesample = cfg_mean_nsamples_resolution(cfg)
    model_samples, jointstype = get_samples_folder(cfg.folder, cfg.eval_pairs,
                                                   cfg.slerp, cfg.checkpoint,
                                                   cfg.align,
                                                   jointstype=cfg.jointstype)
    split = cfg.split

    path = get_path(model_samples, cfg.split, onesample, cfg.mean, cfg.fact)
    file_path = f"babel_metrics_{split}" # if amass else f"metrics_{split}"

    save_paths = get_metric_paths(model_samples, cfg.eval_pairs, 
                                  cfg.split, onesample, cfg.mean, cfg.fact)
    if onesample:
        save_path = save_paths
        logger.info(f"The outputs will be stored in: {save_path}")
    else:
        avg_path, best_path = save_paths
        logger.info(f"The outputs will be stored in: {avg_path} and {best_path}")

    logger.info("Loading the libraries")
    import numpy as np
    import torch
    import json
    from hydra.utils import instantiate
    from teach.model.metrics import ComputeMetrics, ComputeMetricsBest
    logger.info("Libraries loaded")

    from teach.data.tools.smpl import smpl_data_to_matrix_and_trans
    rots2joints = instantiate(cfg.rots2joints, jointstype=jointstype)

    # If mmmns, it is smpl scale, so it is already in meters
    force_in_meter = cfg.jointstype != "mmmns"
    if onesample:
        CMetrics = ComputeMetrics(force_in_meter=force_in_meter)
    else:
        CMetrics_best = ComputeMetricsBest(force_in_meter=force_in_meter)
        CMetrics_avg = [ComputeMetrics(force_in_meter=force_in_meter) for index in range(cfg.number_of_samples)]

    logger.info(f"Computing the {split} metrics")
    # keep infos for computing
    all_infos = []
    logger.info("Loading data module")
    if cfg.data.dtype in ['pairs', 'pairs_only', 'separate_pairs'] or cfg.eval_pairs:
        cfg.data.dtype = 'separate_pairs'
    data_module = instantiate(cfg.data)
    logger.info(f"Data module '{cfg.data.dataname}' loaded")
    dataset = getattr(data_module, f"{cfg.split}_dataset")

    # test_dataset = data_module.test_dataset
    import torch
    with torch.no_grad():
        # import random
        # nrs = 100
        # rd_samps = random.choices(dataset._split_index, k=nrs)
        for keyid in tqdm(dataset._split_index):
            if keyid not in dataset._split_index:
                print(f"{keyid} not found..")
                continue
            ref_joints = dataset.load_keyid(keyid, mode='inference')['datastruct']
            # it is already in this form check babel.py to see why       
            # ref_smpl_data = smpl_data_to_matrix_and_trans(ref_smpl_data, nohands=True)
            ref_joints = rots2joints(ref_joints)
            if not onesample:
                model_joints_all = []
                ref_joints_all = []
                length_all = []
            for index in range(cfg.number_of_samples):
                # Load model joints
                seq_id = "" if onesample else f"_{index}"
                try:
                    model_joints = np.load(path / f"{keyid}{seq_id}.npy",
                                           allow_pickle=True).item()['motion']
                except:
                    print( f"{keyid}{seq_id}.npy not found")                
                    continue
                model_joints = torch.from_numpy(model_joints).float()
                # Take the common lengths to facilitate the computation
                length = min(len(model_joints), len(ref_joints))
                if onesample:
                    # Compute part of the metrics
                    CMetrics.update(model_joints[None], ref_joints[None], [length])
                else:
                    CMetrics_avg[index].update(model_joints[None], ref_joints[None], [length])
                    # keep them all to compute the best one
                    model_joints_all.append(model_joints[None])
                    ref_joints_all.append(ref_joints[None])
                    length_all.append([length])

            if not onesample:
                CMetrics_best.update(model_joints_all, ref_joints_all, length_all)
    if onesample:
        metrics = sanitize(regroup_metrics(CMetrics.compute()))
        logger.info(f"All done, saving at {save_path}")
        save_metric(save_path, metrics)
        logger.info("Done.")

        for key in ["APE_root", "AVE_root"]:
            logger.info(f"{key}: {metrics[key]}")
    else:
        # best metrics
        best_metrics = sanitize(regroup_metrics(CMetrics_best.compute()))

        avgs = []
        for index in range(cfg.number_of_samples):
            avgs.append(regroup_metrics(CMetrics_avg[index].compute()))

        # avg metrics
        avg_metrics = sanitize({key: np.mean([avg[key] for avg in avgs]) for key in avgs[0].keys()})

        logger.info(f"All done, saving at {best_path} and {avg_path}")
        save_metric(avg_path, avg_metrics)
        save_metric(best_path, best_metrics)
        logger.info("Done.")

        for name, metrics in [("avg", avg_metrics), ("best", best_metrics)]:
            logger.info(f"{name}")
            for key in ["APE_root", "AVE_root"]:
                logger.info(f"  {key}: {metrics[key]}")

if __name__ == '__main__':
    _eval()
