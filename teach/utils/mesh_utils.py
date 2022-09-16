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

import pyrender
import trimesh
import numpy as np
import sys
import cv2
from teach.utils.smpl_body_utils import colors

def get_checkerboard_plane(scale, loc, plane_width=4, num_boxes=9, center=True):
    
    ground_color = colors['light_grey']
    meshes = []
    
    ground = trimesh.primitives.Box(center=loc, extents=scale)

    # if center:
    #     c = c[0]+(pw/2)-(plane_width/2), c[1]+(pw/2)-(plane_width/2)
    # # trans = trimesh.transformations.scale_and_translate(scale=1, translate=[c[0], c[1], 0])
    ground.apply_translation([loc[0], loc[1], 0])
    # # ground.apply_transform(trimesh.transformations.rotation_matrix(np.rad2deg(-120), direction=[1,0,0]))
    # # ground.visual.face_colors = black if ((i+j) % 2) == 0 else white
    ground.visual.face_colors = ground_color
    meshes.append(ground)

    return meshes


class MeshViewer(object):

    def __init__(self, width=1200, height=800, add_ground_plane=False, 
                 ground_loc=None, ground_scale=None, use_offscreen=True,
                 bg_color='white'):
        super().__init__()

        self.use_offscreen = use_offscreen
        self.render_wireframe = False

        self.mat_constructor = pyrender.MetallicRoughnessMaterial
        self.trimesh_to_pymesh = pyrender.Mesh.from_trimesh

        self.scene = pyrender.Scene(bg_color=colors[bg_color],
                                    ambient_light=(0.3, 0.3, 0.3))

        pc = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=float(width) / height)
        camera_pose = np.eye(4)
        camera_pose[:3, 3] = np.array([0, 1.75, 3.25])
        self.camera_node = self.scene.add(pc, pose=camera_pose, name='pc-camera')

        self.figsize = (width, height)

        if add_ground_plane:
            ground_mesh = pyrender.Mesh.from_trimesh(get_checkerboard_plane(
                                                     ground_scale, ground_loc),
                                                     smooth=False)
            pose = trimesh.transformations.rotation_matrix(np.radians(90), [1, 0, 0])
            pose[:3, 3] = [0, -1, 0]
            self.scene.add(ground_mesh, pose=pose, name='ground_plane')

        if self.use_offscreen:
            self.viewer = pyrender.OffscreenRenderer(*self.figsize)
            self.use_raymond_lighting(5.)
        else:
            self.viewer = pyrender.Viewer(self.scene, use_raymond_lighting=True, viewport_size=self.figsize, cull_faces=False, run_in_thread=True)

    def set_background_color(self, color=colors['white']):
        self.scene.bg_color = color

    def update_camera_pose(self, camera_pose):
        self.scene.set_pose(self.camera_node, pose=camera_pose)

    def close_viewer(self):
        if self.viewer.is_active:
            self.viewer.close_external()

    def set_cam_trans(self, trans= [0, 0, 3.0]):
        if isinstance(trans, list): trans = np.array(trans)
        trans[2] += 2.8
        camera_pose = np.eye(4)
        camera_pose[:3, 3] += trans
        self.scene.set_pose(self.camera_node, pose=camera_pose)

    def set_meshes(self, meshes, group_name='static', poses=[]):
        for node in self.scene.get_nodes():
            if node.name is not None and '%s-mesh'%group_name in node.name:
                self.scene.remove_node(node)

        if len(poses) < 1:
            for mid, mesh in enumerate(meshes):
                if isinstance(mesh, trimesh.Trimesh):
                    mesh = pyrender.Mesh.from_trimesh(mesh)
                self.scene.add(mesh, '%s-mesh-%2d'%(group_name, mid))
                body_trans = np.array(mesh.centroid)
                self.set_cam_trans(trans=body_trans)

        else:
            for mid, iter_value in enumerate(zip(meshes, poses)):
                mesh, pose = iter_value
                if isinstance(mesh, trimesh.Trimesh):
                    mesh = pyrender.Mesh.from_trimesh(mesh)
                self.scene.add(mesh, '%s-mesh-%2d'%(group_name, mid), pose)

    def set_static_meshes(self, meshes, poses=[]): self.set_meshes(meshes, group_name='static', poses=poses)
    def set_dynamic_meshes(self, meshes, poses=[]): self.set_meshes(meshes, group_name='dynamic', poses=poses)

    def _add_raymond_light(self):
        from pyrender.light import DirectionalLight
        from pyrender.node import Node

        thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
        phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])

        nodes = []

        for phi, theta in zip(phis, thetas):
            xp = np.sin(theta) * np.cos(phi)
            yp = np.sin(theta) * np.sin(phi)
            zp = np.cos(theta)

            z = np.array([xp, yp, zp])
            z = z / np.linalg.norm(z)
            x = np.array([-z[1], z[0], 0.0])
            if np.linalg.norm(x) == 0:
                x = np.array([1.0, 0.0, 0.0])
            x = x / np.linalg.norm(x)
            y = np.cross(z, x)

            matrix = np.eye(4)
            matrix[:3, :3] = np.c_[x, y, z]
            nodes.append(Node(
                light=DirectionalLight(color=np.ones(3), intensity=1.0),
                matrix=matrix
            ))
        return nodes

    def use_raymond_lighting(self, intensity = 1.0):
        if not self.use_offscreen:
            sys.stderr.write('Interactive viewer already uses raymond lighting!\n')
            return
        for n in self._add_raymond_light():
            n.light.intensity = intensity / 3.0
            if not self.scene.has_node(n):
                self.scene.add_node(n)#, parent_node=pc)

    def render(self, render_wireframe=None, RGBA=False):
        from pyrender.constants import RenderFlags

        flags = RenderFlags.SHADOWS_DIRECTIONAL
        if RGBA: flags |=  RenderFlags.RGBA
        if render_wireframe is not None and render_wireframe==True:
            flags |= RenderFlags.ALL_WIREFRAME
        elif self.render_wireframe:
            flags |= RenderFlags.ALL_WIREFRAME
        color_img, depth_img = self.viewer.render(self.scene, flags=flags)

        return color_img


    def save_snapshot(self, fname):
        if not self.use_offscreen:
            sys.stderr.write('Currently saving snapshots only works with off-screen renderer!\n')
            return
        color_img = self.render()
        cv2.imwrite(fname, color_img)
