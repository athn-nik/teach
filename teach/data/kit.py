import json
import os
from glob import glob
from typing import Dict, Optional
import logging

import numpy as np
import pandas
import torch
from torch import nn
from torch.utils.data import Dataset
from tqdm import tqdm
from pathlib import Path

from temos.tools.easyconvert import matrix_to, axis_angle_to
from temos.transforms import Transform

from .base import BASEDataModule

logger = logging.getLogger(__name__)

SPLITS = ["train", "val", "test", "all", "subset"]


def get_split_keyids(path: str, split: str):
    assert split in SPLITS
    filepath = Path(path) / split
    with filepath.open("r") as file_split:
        keyids = file_split.readlines()

    keyids = [key.strip() for key in keyids]
    return keyids


class KITDataModule(BASEDataModule):
    def __init__(self, data_dir: str = "",
                 batch_size: int = 32,
                 num_workers: int = 16,
                 **kwargs):
        super().__init__(batch_size=batch_size,
                         num_workers=num_workers)
        self.save_hyperparameters(logger=False)
        self.Dataset = KIT

        sample_overrides = {"split": "train", "tiny": True,
                            "progress_bar": False}
        self._sample_set = self.get_sample_set(overrides=sample_overrides)

        # Get additional info of the dataset
        self.nfeats = self._sample_set.nfeats
        self.transforms = self._sample_set.transforms


class KIT(Dataset):
    dataname = "KIT Motion-Language"

    def __init__(self, datapath: str,
                 splitpath: str,
                 transforms: Transform,
                 split: str = "train",
                 transforms_xyz: Optional[Transform] = None,
                 transforms_smpl: Optional[Transform] = None,
                 correspondance_path: str = None,
                 amass_path: str = None,
                 smplh_path: str = None,
                 sampler=None,
                 framerate: float = 12.5,
                 progress_bar: bool = True,
                 pick_one_text: bool = True,
                 load_canonicalized_text=False,
                 load_amass_data=False,
                 load_with_rot=False,
                 load_actions=False,
                 downsample=True,
                 tiny: bool = False, **kwargs):

        self.split = split
        self.load_actions = load_actions
        self.load_canonicalized_text = load_canonicalized_text
        self.load_amass_data = load_amass_data
        self.load_with_rot = load_with_rot
        self.downsample = downsample

        if load_amass_data and not self.load_with_rot:
            self.transforms_xyz = transforms_xyz
            self.transforms_smpl = transforms_smpl
            self.transforms = transforms_xyz
        else:
            self.transforms = transforms

        if self.split not in SPLITS:
            raise ValueError(f"{self.split} is not a valid split")

        self.sampler = sampler
        self.pick_one_text = pick_one_text

        super().__init__()
            # meta_paths = sorted(glob(f"{datapath}/*_meta.json"))
        keyids = get_split_keyids(path=splitpath, split=split)

        motion_data = {}
        texts_data = {}
        durations = {}

        if load_canonicalized_text:
            text_canonicalized_data = {}

        if load_amass_data:
            with open(correspondance_path) as correspondance_path_file:
                kitml_correspondances = json.load(correspondance_path_file)

        if progress_bar:
            enumerator = enumerate(tqdm(keyids, f"Loading KIT {split}"))
        else:
            enumerator = enumerate(keyids)

        if tiny:
            maxdata = 2
        else:
            maxdata = np.inf

        datapath = Path(datapath)
        num_bad = 0

        bad_smpl = 0
        good_smpl = 0

        if load_actions:
            self.action_datas = {}

        for i, keyid in enumerator:
            if len(motion_data) >= maxdata:
                break

            # if split == "test" and keyid != "01284":  # 00073 circle
            #     continue

            metapath = datapath / (keyid + "_meta.json")
            metadata = json.load(metapath.open())

            if metadata["nb_annotations"] == 0:
                logger.error(f"{keyid} has no annotations")
                continue

            annpath = datapath / (keyid + "_annotations.json")
            anndata = json.load(annpath.open())

            assert len(anndata) == metadata["nb_annotations"]

            if load_actions:
                action_path = datapath / (keyid + "_actions.json")
                if not os.path.exists(action_path):
                    # bad sequence, or not processed
                    print(action_path)
                    continue
                action_data = json.load(action_path.open())

                if tiny:
                    # make sure to get a batch with more than one actions
                    if len(motion_data) == 1:
                        # each annotations should have more than 1 action
                        try:
                            for x in action_data:
                                if len(x) < 2:
                                    raise ValueError
                        except ValueError:
                            continue

            # read better text
            if load_canonicalized_text:
                ann_can_path = datapath / (keyid + "_annotations_canonicalized.json")

                if not os.path.exists(ann_can_path):
                    # bad sequence, or not processed
                    num_bad += 1
                    continue

                ann_canonicalized_data = json.load(ann_can_path.open())

            from temos.data.sampling import subsample
            # read smpl params
            if load_amass_data:
                identifier = kitml_correspondances[keyid]["identifier"]
                smpl_keyid_path = kitml_correspondances[keyid]["path"]

                if identifier == "kit":
                    smpl_datapath = Path(amass_path) / "KIT" / "KIT" / smpl_keyid_path
                elif identifier == "cmu":
                    smpl_datapath = Path(amass_path) / "CMU" / "CMU" / smpl_keyid_path

                if not os.path.exists(smpl_datapath):
                    # try with EKUT folder instead
                    smpl_datapath = Path(amass_path) / "EKUT" / "EKUT" / smpl_keyid_path

                    # still bad: todo find good one
                    if not os.path.exists(smpl_datapath):
                        bad_smpl += 1
                        continue
                    # import ipdb;
                    # ipdb.set_trace()

                # load amass data
                try:
                    smpl_data = np.load(smpl_datapath)
                    good_smpl += 1
                except FileNotFoundError:
                    print(smpl_datapath)
                    import ipdb; ipdb.set_trace()  # noqa
                    pass

                smpl_data = {x: smpl_data[x] for x in smpl_data.files}

                nframes_total = len(smpl_data["poses"])
                last_framerate = smpl_data["mocap_framerate"].item()

                if self.downsample:
                    frames = subsample(nframes_total, last_framerate=last_framerate, new_framerate=framerate)
                else:
                    frames = np.arange(nframes_total)
                duration = len(frames)

                # subsample
                smpl_data = {"poses": torch.from_numpy(smpl_data["poses"][frames]).float(),
                             "trans": torch.from_numpy(smpl_data["trans"][frames]).float()}
                from temos.data.tools.smpl import smpl_data_to_matrix_and_trans
                smpl_data = smpl_data_to_matrix_and_trans(smpl_data, nohands=True)
            else:
                xyzpath = datapath / (keyid + "_fke.csv")
                xyzdata = pandas.read_csv(xyzpath, index_col=0)
                joints = np.array(xyzdata).reshape(-1, 21, 3)

                nframes_total = len(joints)
                last_framerate = 100

                if self.downsample:
                    frames = subsample(nframes_total, last_framerate=last_framerate, new_framerate=framerate)
                else:
                    frames = np.arange(nframes_total)

                duration = len(frames)
                # subsample
                joints = torch.from_numpy(joints[frames]).float()

            if split != "test" and not tiny:
                # Accept or not the sample, based on the duration
                if not self.sampler.accept(duration):
                    num_bad += 1
                    continue

            if load_amass_data and load_with_rot:
                features = self.transforms.rots2rfeats(smpl_data)
            elif load_amass_data and not load_with_rot:
                joints = self.transforms_smpl.rots2joints(smpl_data)
                features = self.transforms_xyz.joints2jfeats(joints)
            else:
                features = self.transforms.joints2jfeats(joints)

            motion_data[keyid] = features
            texts_data[keyid] = anndata

            if load_canonicalized_text:
                text_canonicalized_data[keyid] = ann_canonicalized_data

            if load_actions:
                self.action_datas[keyid] = action_data

            durations[keyid] = duration

        if load_amass_data and not tiny:
            percentage = 100 * bad_smpl / (bad_smpl + good_smpl)
            logger.info(f"There are {percentage:.4}% of the motion which are not found. (AMASS)")

        if split != "test" and not tiny:
            total = len(motion_data)
            percentage = 100 * num_bad / (total+num_bad)
            logger.info(f"There are {percentage:.4}% of the sequence which are rejected by the sampler.")

        self.motion_data = motion_data
        self.texts_data = texts_data
        # v = []
        # import spacy
        # nlp = spacy.load("en_core_web_trf")
        # doc =  [ v[0] for v in texts_data.values() ]
        # for dc in doc:            
        #     for token in nlp(dc):
        #         if token.pos_ == 'VERB':
        #             v.append(token.text)
        # from nltk.stem import PorterStemmer
        # ps =PorterStemmer()
        # v = [ps.stem(w) for w in v]

        # breakpoint()

        if load_canonicalized_text:
            self.text_canonicalized_data = text_canonicalized_data
        # if not tiny:
        #     import ipdb; ipdb.set_trace()  # noqa
        self._split_index = list(motion_data.keys())
        self._num_frames_in_sequence = durations
        self.nfeats = len(self[0]["datastruct"].features[0])

    def _load_datastruct(self, keyid, frame_ix=None):
        features = self.motion_data[keyid]
        datastruct = self.transforms.Datastruct(features=features)
        return datastruct

    def _load_text(self, keyid):
        if self.load_canonicalized_text:
            sequences = self.text_canonicalized_data[keyid]
        else:
            sequences = self.texts_data[keyid]
        if not self.pick_one_text:
            return sequences
        n = len(sequences)
        if self.split != "test":
            index = np.random.randint(n)
        else:
            index = 0
        text = sequences[index]
        return text

    def _load_actions(self, keyid):
        actions_all = self.action_datas[keyid]
        if not self.pick_one_text:
            return actions_all
        n = len(actions_all)
        if self.split != "test":
            index = np.random.randint(n)
        else:
            index = 0
        actions = actions_all[index]
        return actions

    def load_keyid(self, keyid):
        num_frames = self._num_frames_in_sequence[keyid]
        frame_ix = self.sampler(num_frames)

        datastruct = self._load_datastruct(keyid, frame_ix)
        text = self._load_text(keyid)
        element = {"datastruct": datastruct, "text": text,
                   "length": len(datastruct), "keyid": keyid}

        if self.load_actions:
            element["actions"] = self._load_actions(keyid)

        return element

    def __getitem__(self, index):
        keyid = self._split_index[index]
        return self.load_keyid(keyid)

    def __len__(self):
        return len(self._split_index)

    def __repr__(self):
        return f"{self.dataname} dataset: ({len(self)}, _, ..)"
