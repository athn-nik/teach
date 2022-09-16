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

def get_frameidx(*, mode, nframes, exact_frame, frames_to_keep,
                 lengths=None, return_lists=True):
    if mode == "sequence":
        if lengths is not None:
            frameidx = []
            cumlen = np.cumsum(lengths)
            for i, cum_i in enumerate(cumlen):
                if i == 0 :
                    frameidx_i = np.linspace(0, cum_i - 1, frames_to_keep)
                else:
                    frameidx_i = np.linspace(cumlen[i-1] + 1, cum_i - 1, frames_to_keep)
                frameidx_i = np.round(frameidx_i).astype(int)
                frameidx_i = list(frameidx_i)

                if return_lists:
                    frameidx.append(frameidx_i)
                else:
                    frameidx.extend(frameidx_i)
        else:
            frameidx_t = np.linspace(0, nframes-1, frames_to_keep)
            frameidx_t = np.round(frameidx_t).astype(int)
            frameidx_t = list(frameidx_t)
            frameidx = [frameidx_t]
        # exit()
    elif mode == "frame":
        index_frame = int(exact_frame*nframes)
        frameidx = [index_frame]
    elif mode == "video":
        frameidx = []
        cumlen = np.cumsum(lengths)
        for i, cum_i in enumerate(cumlen):
            if i == 0 :
                frameidx_i = list(range(0, cum_i))
            else:
                frameidx_i = list(range(cumlen[i-1], cum_i))
            
            if return_lists:
                frameidx.append(frameidx_i)
            else:
                frameidx.extend(frameidx_i)

    return frameidx
