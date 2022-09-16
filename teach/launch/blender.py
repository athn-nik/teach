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

# Fix blender path
import sys
sys.path.append("/linkhome/rech/genlgm01/uwm78rj/.local/lib/python3.9/site-packages")

import bpy
import os
DIR = os.path.dirname(bpy.data.filepath)
if DIR not in sys.path:
    sys.path.append(DIR)

from argparse import ArgumentParser

# Workaround temorary for cluster vs local
# TODO fix it
import socket
if socket.gethostname() == 'ps018':
    packages_path = '/home/nathanasiou/.local/lib/python3.10/site-packages'
    sys.path.insert(0, packages_path)
    sys.path.append("/home/nathanasiou/.venvs/teach/lib/python3.10/site-packages")
    sys.path.append('/usr/lib/python3/dist-packages')

# Monkey patch argparse such that
# blender / python / hydra parsing works
def parse_args(self, args=None, namespace=None):
    if args is not None:
        return self.parse_args_bak(args=args, namespace=namespace)
    try:
        idx = sys.argv.index("--")
        args = sys.argv[idx+1:]  # the list after '--'
    except ValueError as e:  # '--' not in the list:
        args = []
    return self.parse_args_bak(args=args, namespace=namespace)

setattr(ArgumentParser, 'parse_args_bak', ArgumentParser.parse_args)
setattr(ArgumentParser, 'parse_args', parse_args)
