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

import smplx
from pathlib import Path

marker_dict = {'C7': 3470, 'CLAV': 3171, 'LANK': 3327, 'LASI': 857, 'LBAK': 1812, 
               'LBCEP': 628, 'LBHD': 182, 'LBUM': 3116, 'LBUST': 3040, 'LCHEECK': 239,
               'LELB': 1666, 'LELBIN': 1725, 'LFHD': 0, 'LFIN': 2174, 'LFRM': 1568,
               'LFTHIIN': 1368, 'LHEE': 3387, 'LIWR': 2112, 'LKNE': 1053, 'LKNI': 1058,
               'LMT1': 3336, 'LMT5': 3346, 'LNWST': 1323, 'LOWR': 2108, 'LPSI': 3122,
               'LRSTBEEF': 3314, 'LSCAP': 1252, 'LSHN': 1082, 'LSHO': 1861, 'LTHI': 1454,
               'LTHILO': 850, 'LTHMB': 2224, 'LTOE': 3233, 'MBLLY': 1769, 'RANK': 6728,
               'RASI': 4343, 'RBAK': 5273, 'RBCEP': 4116, 'RBHD': 3694, 'RBSH': 6399,
               'RBUM': 6540, 'RBUST': 6488, 'RCHEECK': 3749, 'RELB': 5135, 'RELBIN': 5194,
               'RFHD': 3512, 'RFIN': 5635, 'RFRM2': 5210, 'RFTHI': 4360, 'RFTHIIN': 4841,
               'RHEE': 6786, 'RIWR': 5573, 'RKNE': 4538, 'RKNI': 4544, 'RMT1': 6736,
               'RMT5': 6747, 'RNWST': 4804, 'ROWR': 5568, 'RPSI': 6544, 'RRSTBEEF': 6682,
               'RSHO': 5322, 'RTHI': 4927, 'RTHMB': 5686, 'RTIB': 4598, 'RTOE': 6633,
               'STRN': 3506, 'T8': 3508}

markerset_ssm67_smplh = [3470, 3171, 3327, 857, 1812, 628, 182, 3116, 3040, 239,
                         1666, 1725, 0, 2174, 1568, 1368, 3387, 2112, 1053, 1058,
                         3336, 3346, 1323, 2108, 3122, 3314, 1252, 1082, 1861, 1454,
                         850, 2224, 3233, 1769, 6728, 4343, 5273, 4116, 3694, 6399,
                         6540, 6488, 3749, 5135, 5194, 3512, 5635, 5210, 4360, 4841,
                         6786, 5573, 4538, 4544, 6736, 6747, 4804, 5568, 6544, 6682,
                         5322, 4927, 5686, 4598, 6633, 3506, 3508]


marker2bodypart = {
    "head_ids": [12, 45, 9, 42, 6, 38],
    "mid_body_ids": [56, 35, 58, 24, 22, 0, 4, 36, 26, 1, 65, 33, 41, 8, 66, 35, 3, 4, 39],
    "left_hand_ids": [10, 11, 14, 31, 13, 17, 23, 28, 27],
    "right_hand_ids": [60, 43, 44, 47, 62, 46, 51, 57],
    "left_foot_ids": [29, 30, 18, 19, 7, 2, 15],
    "right_foot_ids": [61, 52, 53, 40, 34, 49, 40],
    "left_toe_ids": [32, 25, 20, 21, 16],
    "right_toe_ids": [54, 55, 59, 64, 50, 55],
}

bodypart2color = {
    "head_ids": 'cyan',
    "mid_body_ids": 'blue',
    "left_hand_ids": 'red',
    "right_hand_ids": 'green',
    "left_foot_ids": 'grey',
    "right_foot_ids": 'black',
    "left_toe_ids": 'yellow',
    "right_toe_ids": 'magenta',
    "special": 'light_grey'
}


colors = {
        "blue": [0, 0, 255, 1],
        "cyan": [0, 255, 255, 1],
        "green": [0, 128, 0, 1],
        "yellow": [255, 255, 0, 1],
        "red": [255, 0, 0, 1],
        "grey": [77, 77, 77, 1],
        "black": [0, 0, 0, 1],
        "white": [255, 255, 255, 1],
        "transparent": [255, 255, 255, 0],
        "magenta": [197, 27, 125, 1],
        'pink': [197, 140, 133, 1],
        "light_grey": [217, 217, 217, 255],
        'yellow_pale': [226, 215, 132, 1],
        }


colors_rgb = {
        "blue": [0, 0, 255/255],
        "cyan": [0, 128/255, 255/255],
        "green": [0, 255/255, 0],
        "yellow": [255/255, 255/255, 0],
        "red": [255/255, 0, 0],
        "grey": [77/255, 77/255, 77/255],
        "black": [0, 0, 0],
        "white": [255/255, 255/255, 255/255],
        "magenta": [197/255, 27, 125/255],
        'pink': [197/255, 140/255, 133/255],
        "light_grey": [217/255, 217/255, 217/255],
        }


def get_body_model(path, model_type, gender, batch_size, device='cpu', ext='pkl'):
    '''
    type: smpl, smplx smplh and others. Refer to smplx tutorial
    gender: male, female, neutral
    batch_size: an positive integar
    '''
    mtype = model_type.upper()
    if gender != 'neutral':
        if not isinstance(gender, str):
            gender = str(gender.astype(str)).upper()
        else:
            gender = gender.upper()
    else:
        gender = gender.upper()
    body_model_path = Path(path) / model_type / f'{mtype}_{gender}.{ext}'

    body_model = smplx.create(body_model_path, model_type=type,
                              gender=gender, ext=ext,
                              use_pca=False,
                              num_pca_comps=12,
                              create_global_orient=True,
                              create_body_pose=True,
                              create_betas=True,
                              create_left_hand_pose=False,
                              create_right_hand_pose=False,
                              create_expression=True,
                              create_jaw_pose=True,
                              create_leye_pose=True,
                              create_reye_pose=True,
                              create_transl=True,
                              batch_size=batch_size)
    if device == 'cuda':
        return body_model.cuda()
    else:
        return body_model

def c2rgba(c):
    if len(c) == 3:
        c.append(1)
    c = [c_i/255 for c_i in c[:3]]

    return c
