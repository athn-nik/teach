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
import torch
import trimesh
import pyrender
import math
import torch.nn.functional as F
from teach.utils.mesh_utils import MeshViewer
from teach.utils.smpl_body_utils import colors,marker2bodypart,bodypart2color
from teach.utils.smpl_body_utils import c2rgba, get_body_model
from hydra.utils import get_original_cwd
from teach.render.video import save_video_samples

import os 
import cv2
from PIL import Image
import subprocess

def visualize_meshes(vertices, pcd=None, multi_col=None,
                     multi_angle=False, h=480, w=480, bg_color='white',
                     save_path=None, fig_label=None):
    """[summary]

    Args:
        rec (torch.tensor): [description]
        inp (torch.tensor, optional): [description]. Defaults to None.
        multi_angle (bool, optional): Whether to use different angles. Defaults to False.

    Returns:
        np.array :   Shape of output (view_angles, seqlen, 3, im_width, im_height, )
 
    """
    import os
    os.environ['PYOPENGL_PLATFORM'] = 'egl'

    im_height = h
    im_width = w
    vis_mar = True if pcd is not None else False

    # if multi_angle:
    #     view_angles = [0, 180, 90, -90]
    # else:
    #     view_angles = [0]
    seqlen = vertices.shape[0]
    kfs_viz = np.zeros((seqlen))
    if multi_col is not None:
        # assert len(set(multi_col)) == len(multi_col)
        multi_col = multi_col.detach().cpu().numpy()
        for i, kf_ids in enumerate(multi_col):
            kfs_viz[i, :].put(np.round_(kf_ids).astype(int), 1, mode='clip')
    

    if isinstance(pcd, np.ndarray):
        pcd = torch.from_numpy(pcd)

    if vis_mar:
        pcd = pcd.reshape(seqlen, 67, 3).to('cpu')
        if len(pcd.shape) == 3:
            pcd = pcd.unsqueeze(0)
    mesh_rec = vertices
    smpl = get_body_model(path=f'{get_original_cwd()}/data/smpl_models',
                          model_type='smpl', gender='neutral',
                          batch_size=1, device='cpu')
    
    mv = MeshViewer(width=im_width, height=im_height,
                    add_ground_plane=False, use_offscreen=True,
                    bg_color=bg_color)
                #ground_height=(mesh_rec.detach().cpu().numpy()[0, 0, 6633, 2] - 0.01))
    # ground plane has a bug if we want batch_size to work
    mv.render_wireframe = False
    video = np.zeros([seqlen, im_width, im_height, 3])
    for i in range(seqlen):
        Rx = trimesh.transformations.rotation_matrix(math.radians(-90), [1, 0, 0])
        Ry = trimesh.transformations.rotation_matrix(math.radians(90), [0, 1, 0])

        if vis_mar:
            m_pcd = trimesh.points.PointCloud(pcd[i])

        if kfs_viz[i]:
            mesh_color = np.tile(c2rgba(colors['light_grey']), (6890, 1))
        else:
            mesh_color = np.tile(c2rgba(colors['yellow_pale']), (6890, 1))
        m_rec = trimesh.Trimesh(vertices=mesh_rec[i],
                                faces=smpl.faces,
                                vertex_colors=mesh_color)

        m_rec.apply_transform(Rx)
        m_rec.apply_transform(Ry)

        all_meshes = []
        if vis_mar:
            m_pcd.apply_transform(Rx)
            m_pcd.apply_transform(Ry)

            m_pcd = np.array(m_pcd.vertices)
            # after trnaofrming poincloud visualize for each body part separately
            pcd_bodyparts = dict()
            for bp, ids in marker2bodypart.items():
                points = m_pcd[ids]
                tfs = np.tile(np.eye(4), (points.shape[0], 1, 1))
                tfs[:, :3, 3] = points
                col_sp = trimesh.creation.uv_sphere(radius=0.01)

                # debug markers, maybe delete it
                # if bp == 'special':
                #     col_sp = trimesh.creation.uv_sphere(radius=0.03)

                if kfs_viz[i]:
                    col_sp.visual.vertex_colors = c2rgba(colors["black"])
                else:
                    col_sp.visual.vertex_colors = c2rgba(colors[bodypart2color[bp]])

                pcd_bodyparts[bp] = pyrender.Mesh.from_trimesh(col_sp, poses=tfs)

            for bp, m_bp in pcd_bodyparts.items():
                all_meshes.append(m_bp)


        all_meshes.append(m_rec)
        mv.set_meshes(all_meshes, group_name='static')
        video[i] = mv.render()

    # if save_path is not None:
    #     save_video_samples(np.transpose(np.squeeze(video),
    #                                     (0, 3, 1, 2)).astype(np.uint8), 
    #                                     save_path, write_gif=True)
    #     return
    if multi_angle:
        return np.transpose(video, (0, 1, 4, 2, 3)).astype(np.uint8)

    if save_path is not None:
        return save_video_samples(np.transpose(np.squeeze(video),
                                               (0, 3, 1, 2)).astype(np.uint8),
                                  save_path, write_gif=False)
    return np.transpose(np.squeeze(video), (0, 3, 1, 2)).astype(np.uint8)


# def save_video_samples(vid_array, video_path, write_images=[], write_gif=True):

#     if write_images:
#         os.makedirs(video_path, exist_ok=True)
#         for fid, img in enumerate(vid_array[write_images]):
#             img_ar = np.uint8(np.transpose(img, (1, 2, 0)))
#             img_f = Image.fromarray(img_ar)

#             img_f.save(f'{video_path}/{fid}.png')
#     video_folder_path = '/'.join(video_path.strip().split('/')[:-1])
#     os.makedirs(video_folder_path, exist_ok=True)

#     vid_path = f'{video_path}.mp4'
#     gif_path = f'{video_path}.gif'
#     # vid_array is (B, T, 3, W, H)
#     w, h = vid_array.shape[-2], vid_array.shape[-1]
#     fps = 10
#     label = os.path.basename(video_path)
#     print(label)
#     if len(vid_array.shape) > 4:
#         duration = vid_array.shape[1] // fps
#         no_vids = vid_array.shape[0]
#         # (B, T, 3, W, H) -> (T, B, W, H, 3)
#         vid_array = np.transpose(vid_array,(1, 0, 3, 4, 2))

#     else:
#         duration = vid_array.shape[0] // fps
#         no_vids = 1
        
#     out = cv2.VideoWriter(vid_path, cv2.VideoWriter_fourcc(*'mp4v'),
#                           fps, (no_vids*w, h)
#                           )
    
#     for img in vid_array:
#         if len(vid_array.shape) > 4:
#             img = np.uint8(np.hstack(img[:]))
#             img_ar = img.copy()
#             # img_ar = np.uint8(np.transpose(img.copy(), (1, 2, 0)))
            
#         else:
#             img_ar = np.uint8(np.transpose(img, (1, 2, 0)))
#         cv2.putText(img_ar, f'{label}_gt_sample', (15,25),
#                     cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0,0,0), 1, cv2.LINE_AA)

#         for i in range(1, no_vids+1):
#             cv2.putText(img_ar, f'gen_{i}', (w*i + 10, 15),
#                         cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0,0,0), 1, cv2.LINE_AA)

#         out.write(img_ar)
#     out.release()
#     if write_gif:
#         cmd = ['ffmpeg', '-loglevel', 'quiet', '-y', '-i', vid_path, gif_path]
#         subprocess.call(cmd)
#     # assert retcode == 0, f"Command {' '.join(cmd)} failed, in function write_video_cmd."
#     return video_path

