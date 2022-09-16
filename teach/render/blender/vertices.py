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


def prepare_vertices(vertices, canonicalize=True):
    data = vertices
    # Swap axis (gravity=Z instead of Y)
    # data = data[..., [2, 0, 1]]

    # Make left/right correct
    # data[..., [1]] = -data[..., [1]]

    # Center the first root to the first frame
    data -= data[[0], [0], :]

    # Remove the floor
    data[..., 2] -= np.min(data[..., 2])
    return data