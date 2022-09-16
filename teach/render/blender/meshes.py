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

import numpy as np

from .materials import colored_material

import matplotlib
# green
# GT_SMPL = colored_material(0.009, 0.214, 0.029)
GT_SMPL = colored_material(0.035, 0.415, 0.122)
COLORMAPS = [matplotlib.cm.get_cmap('Blues'), matplotlib.cm.get_cmap('Greys'), matplotlib.cm.get_cmap('Purples'), matplotlib.cm.get_cmap('Greens'),
             matplotlib.cm.get_cmap('Reds')]
# blue
# GEN_SMPL = colored_material(0.022, 0.129, 0.439)
# Blues => cmap(0.87)
GEN_SMPL = colored_material(0.035, 0.322, 0.615)

def prepare_meshes(data, canonicalize=True, always_on_floor=False):
    if canonicalize:
        print("No canonicalization for now")

    # fix axis
    data[..., 1] = - data[..., 1]
    data[..., 0] = - data[..., 0]

    # Remove the floor
    data[..., 2] -= data[..., 2].min()

    # Put all the body on the floor
    if always_on_floor:
        data[..., 2] -= data[..., 2].min(1)[:, None]

    return data


class Meshes:
    def __init__(self, data, *, gt, mode, faces_path, canonicalize, always_on_floor, action_id=0, lengths=None, **kwargs):
        # data = prepare_meshes(data, canonicalize=canonicalize, always_on_floor=always_on_floor)

        self.faces = np.load(faces_path)
        self.data = data
        self.mode = mode

        self.N = len(data)
        self.trajectory = data[:, :, [0, 1]].mean(1)
        self.lengths = lengths

        self.action_id = action_id
        if lengths is None:
            if gt:
                self.mat = GT_SMPL
            else:
                self.colormap = COLORMAPS[action_id%len(COLORMAPS)]
                # self.mat = colored_material(*matplotlib.cm.get_cmap('Dark2')(0))
        else:
            if mode == 'sequence':
                self.colormap = COLORMAPS[action_id%len(COLORMAPS)]
            elif mode == 'video':
                self.mat = matplotlib.cm.get_cmap('Dark2')(action_id)
            elif mode == 'frame':
                self.mat = matplotlib.cm.get_cmap('Dark2')(action_id)
        self.temp_data = data.copy()
        self.last_idx = None

    def get_sequence_mat(self, frac):
        # cmap = matplotlib.cm.get_cmap('Blues')
        if self.mode == 'sequence':
            cmap = self.colormap
            begin = 0.70
            end = 0.90
            rgbcolor = cmap(begin + (end-begin)*frac)
            mat = colored_material(*rgbcolor)
        elif self.mode == 'video':
            rgbcolor = self.mat
            mat = colored_material(*rgbcolor)
        elif self.mode == 'frame':
            rgbcolor = self.mat
            mat = colored_material(*rgbcolor)
        return mat

    def get_root(self, index):
        return self.data[index].mean(0)

    def get_mean_root(self):
        return self.data.mean((0, 1))

    def load_in_blender(self, index, mat):
        vertices = self.data[index]
        faces = self.faces
        name = f"{str(index).zfill(4)}_{self.action_id}"

        from .tools import load_numpy_vertices_into_blender
        load_numpy_vertices_into_blender(vertices, faces, name, mat)

        return name

    def __len__(self):
        return self.N
