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
from teach.tools.geometry import axis_angle_to_quaternion, quaternion_to_axis_angle, matrix_to_quaternion, quaternion_to_matrix



def normalize(x, axis=-1, eps=1e-8):
    """
    Normalizes a tensor over some axis (axes)
    :param x: data tensor
    :param axis: axis(axes) along which to compute the norm
    :param eps: epsilon to prevent numerical instabilities
    :return: The normalized tensor
    """
    lgth = np.sqrt(np.sum(x * x, axis=axis, keepdims=True))
    res = x / (lgth + eps)
    return res


def slerp_poses(last_pose, new_pose, number_of_frames, pose_rep="matrix"):
    # interpolation
    from teach.tools.easyconvert import to_matrix, matrix_to
    last_pose_matrix = to_matrix(pose_rep, last_pose)
    new_pose_matrix = to_matrix(pose_rep, new_pose)

    last_pose_quat = matrix_to("quaternion", last_pose_matrix).numpy()
    new_pose_quat = matrix_to("quaternion", new_pose_matrix).numpy()

    # SLERP Local Quats:
    interp_ws = np.linspace(0.0, 1.0, num=number_of_frames+2, dtype=np.float32)
    inter = np.stack([(quat_normalize(quat_slerp(quat_normalize(last_pose_quat),
                                                 quat_normalize(new_pose_quat),
                                                 w))) for w in interp_ws], axis=0)
    inter_matrix = to_matrix("quaternion", torch.from_numpy(inter))

    inter_poserep = matrix_to(pose_rep, inter_matrix)
    return inter_poserep[1:-1]

def slerp_translation(last_transl, new_transl, number_of_frames):
    alpha = torch.linspace(0, 1, number_of_frames+2)
    # 2 more than needed
    inter_trans = torch.einsum("i,...->i...", 1-alpha, last_transl) + torch.einsum("i,...->i...", alpha, new_transl)
    return inter_trans[1:-1]

# poses are in matrix format
def aligining_bodies(last_pose, last_trans, poses, transl, pose_rep="matrix"):
    from teach.tools.easyconvert import to_matrix, matrix_to

    poses_matrix = to_matrix(pose_rep, poses.clone())
    last_pose_matrix = to_matrix(pose_rep, last_pose.clone())

    global_poses_matrix = poses_matrix[:, 0]
    global_last_pose_matrix = last_pose_matrix[0]

    global_poses_axisangle = matrix_to("axisangle", global_poses_matrix)
    global_last_pose_axis_angle = matrix_to("axisangle", global_last_pose_matrix)

    # Find the cancelation rotation matrix
    # First current pose - last pose
    # already in axis angle?
    rot2d_axisangle = global_poses_axisangle[0].clone()
    rot2d_axisangle[:2] = 0
    rot2d_axisangle[2] -= global_last_pose_axis_angle[2]
    rot2d_matrix = to_matrix("axisangle", rot2d_axisangle)

    # turn with the same amount all the rotations
    turned_global_poses_matrix = torch.einsum("...kj,...kl->...jl", rot2d_matrix, global_poses_matrix)
    turned_global_poses = matrix_to(pose_rep, turned_global_poses_matrix)

    turned_poses = torch.cat((turned_global_poses[:, None], poses[:, 1:]), dim=1)

    # turn the trajectory (not with the gravity axis)
    trajectory = transl[:, :2]
    last_trajectory = last_trans[:2]

    vel_trajectory = torch.diff(trajectory, dim=0)
    vel_trajectory = torch.cat((0 * vel_trajectory[[0], :], vel_trajectory), dim=-2)
    vel_trajectory = torch.einsum("...kj,...lk->...lj", rot2d_matrix[:2, :2], vel_trajectory)
    turned_trajectory = torch.cumsum(vel_trajectory, dim=0)

    # align the trajectory
    aligned_trajectory = turned_trajectory + last_trajectory
    aligned_transl = torch.cat((aligned_trajectory, transl[:, [2]]), dim=1)

    return turned_poses, aligned_transl

def align_interpolate(rotations, translation, s, e, align_rot=True, align_trans=True, interpolate=True):
    # trans1[(i+1):j] = trans[i]
    # trans[j:] += trans[i]-trans[j]
    from teach.tools.interpolation import align_orientations, interpolate_track, linear_interp
    # align orientations (and rotating the translation)
    if align_rot:
        aligned_rots, aligned_transl = align_orientations(s, e, rotations, translation)
    else:
        aligned_rots = rotations
        aligned_transl = translation

    if align_trans:
        aligned_transl[(s+1):e][..., :2] = aligned_transl[s][..., :2]
        aligned_transl[e:][..., :2] += aligned_transl[s][..., :2] - aligned_transl[e][..., :2]

    aligned_transl[(s+1):e][..., 2] = linear_interp(s, e, aligned_transl[..., 2])

    if interpolate:
        # interpolate trans Z
        # interpolate rotations
        slerp_rots = interpolate_track(s, e, aligned_rots)
        # interpolate translation TO REMOVE
        # # lerp_trans = linear_interp(l1-1, l1+l_trans, seq_transl)
        # # assert slerp_trans.shape[0] == l_trans and slerp_trans.shape[0] == l_trans == lerp_trans.shape[0]
        aligned_slerped_rots = torch.vstack((aligned_rots[:(s+1)], slerp_rots, aligned_rots[e:]))
        return aligned_slerped_rots, aligned_transl
    else:
        return aligned_rots, aligned_transl

def linear_interp(s, e, trans):
    begin = trans[s]
    end = trans[e]
    alpha = torch.linspace(0, 1, e-s+1)
    # 2 more than needed
    inter_trans = torch.einsum("i,...->i...", 1-alpha, begin) + torch.einsum("i,...->i...", alpha, end)
    return inter_trans[1:-1]

def align_orientations(s, e, poses_interp, transl):
    # remove the translation
    # trans[(i+1):j] = trans[i]
    # trans[j:] += trans[i]-trans[j]

    # starting from j
    # remove the rotation
    from teach.tools.geometry import matrix_to_axis_angle, axis_angle_to_matrix

    poses_interp = matrix_to_axis_angle(poses_interp)
    # remove the rotation
    global_afterj_axis_angle = poses_interp[e:, 0].clone()
    global_afterj = axis_angle_to_matrix(global_afterj_axis_angle)

    transl = transl.clone()
    # global_afterj_axis_angle = matrix_to_axis_angle(global_afterj)

    # Remove the fist rotation along the vertical axis
    # construct this by extract only the vertical component of the rotation
    rot2d = global_afterj_axis_angle[0]
    # match rotation in z axis
    rot2d[:2] = 0
    rot2d[2] -= poses_interp[s, 0, 2]
    rot2d = axis_angle_to_matrix(rot2d)

    # turn with the same amount all the rotations
    global_afterj = torch.einsum("...kj,...kl->...jl", rot2d, global_afterj)
    global_afterj_axis_angle = matrix_to_axis_angle(global_afterj)

    poses_interp[e:, 0] = global_afterj_axis_angle

    vel_trans_afterj = torch.diff(transl[e:], dim=0)
    vel_trans_afterj = torch.cat((0 * vel_trans_afterj[[0], :], vel_trans_afterj), dim=-2)
    vel_trans_afterj = torch.einsum("...kj,...lk->...lj", rot2d, vel_trans_afterj)

    trans_afterj = torch.cumsum(vel_trans_afterj, dim=0)
    transl[e:] = transl[e] + trans_afterj
    return  axis_angle_to_matrix(poses_interp), transl


def align_trajectory(last_trans, transl):
    trajectory = transl[:, :2]
    last_trajectory = last_trans[:2]

    vel_trajectory = torch.diff(trajectory, dim=0)
    vel_trajectory = torch.cat((0 * vel_trajectory[[0], :], vel_trajectory), dim=-2)

    aligned_trajectory = torch.cumsum(vel_trajectory, dim=0)
    aligned_trajectory = aligned_trajectory + last_trajectory
    aligned_transl = torch.cat((aligned_trajectory, transl[:, [2]]), dim=-1)

    return aligned_transl


def slerp_poses(last_pose, new_pose, number_of_frames, pose_rep="matrix"):
    # interpolation
    from teach.tools.easyconvert import to_matrix, matrix_to
    last_pose_matrix = to_matrix(pose_rep, last_pose)
    new_pose_matrix = to_matrix(pose_rep, new_pose)

    last_pose_quat = matrix_to("quaternion", last_pose_matrix).numpy()
    new_pose_quat = matrix_to("quaternion", new_pose_matrix).numpy()

    # SLERP Local Quats:
    interp_ws = np.linspace(0.0, 1.0, num=number_of_frames+2, dtype=np.float32)
    inter = np.stack([(quat_normalize(quat_slerp(quat_normalize(last_pose_quat),
                                                 quat_normalize(new_pose_quat),
                                                 w))) for w in interp_ws], axis=0)
    inter_matrix = to_matrix("quaternion", torch.from_numpy(inter))

    inter_poserep = matrix_to(pose_rep, inter_matrix)
    return inter_poserep[1:-1]


def slerp_translation(last_transl, new_transl, number_of_frames):
    alpha = torch.linspace(0, 1, number_of_frames+2)
    # 2 more than needed
    inter_trans = torch.einsum("i,...->i...", 1-alpha, last_transl) + torch.einsum("i,...->i...", alpha, new_transl)
    return inter_trans[1:-1]

def interpolate_track(s, e, poses, inrep='matrix', outrep='matrix'):
    # interpolation

    # extract the last good info
    lastgoodinfo = poses[s]

    # extract the first regood info
    newfirstgoodinfo = poses[e]

    if inrep == 'matrix':
        q0 = matrix_to_quaternion(lastgoodinfo.reshape(22, 3, 3))
        q1 = matrix_to_quaternion(newfirstgoodinfo.reshape(22, 3, 3))
    elif inrep =='aa':
        q0 = axis_angle_to_quaternion(lastgoodinfo.reshape(22, 3))
        q1 = axis_angle_to_quaternion(newfirstgoodinfo.reshape(22, 3))

    # SLERP Local Quats:
    interp_ws = np.linspace(0.0, 1.0, num= e - s + 1, dtype=np.float32)
    inter = np.stack([(quat_normalize(quat_slerp(quat_normalize(q0.numpy()),
                                                 quat_normalize(q1.numpy()),
                                                 w))) for w in interp_ws],
                     axis=0)
    if outrep == 'aa':
        res = quaternion_to_axis_angle(torch.from_numpy(inter))
    elif outrep == 'quat':
        res = res
    elif outrep == 'matrix':
        res = quaternion_to_matrix(torch.from_numpy(inter))
    return res[1:-1]


def quat_slerp(x, y, a):
    """
    Performs spherical linear interpolation (SLERP) between x and y, with proportion a
    :param x: quaternion tensor
    :param y: quaternion tensor
    :param a: indicator (between 0 and 1) of completion of the interpolation.
    :return: tensor of interpolation results
    """

    len = np.sum(x * y, axis=-1)

    neg = len < 0.0
    len[neg] = -len[neg]
    y[neg] = -y[neg]

    a = np.zeros_like(x[..., 0]) + a
    amount0 = np.zeros(a.shape)
    amount1 = np.zeros(a.shape)

    linear = (1.0 - len) < 0.01
    omegas = np.arccos(len[~linear])
    sinoms = np.sin(omegas)

    amount0[linear] = 1.0 - a[linear]
    amount0[~linear] = np.sin((1.0 - a[~linear]) * omegas) / sinoms

    amount1[linear] = a[linear]
    amount1[~linear] = np.sin(a[~linear] * omegas) / sinoms
    res = amount0[..., np.newaxis] * x + amount1[..., np.newaxis] * y

    return res

def quat_normalize(x, eps=1e-8):
    """
    Normalizes a quaternion tensor
    :param x: data tensor
    :param eps: epsilon to prevent numerical instabilities
    :return: The normalized quaternions tensor
    """
    res = normalize(x, eps=eps)
    return res
