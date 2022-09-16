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

import os
import sys

if "blender" in sys.argv[0].lower():
    import bpy
    DIR = os.path.dirname(bpy.data.filepath)
    if DIR not in sys.path:
        sys.path.append(DIR)
    import teach.launch.blender

import teach.launch.prepare  # noqa
import logging
import hydra
from pathlib import Path
from omegaconf import DictConfig
import numpy as np

logger = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="render")
def _render_cli(cfg: DictConfig):
    return render_cli(cfg)


# "01626"
good_ones = ["00821", "00596", "02417", "01284", "01892", "01209", "01321", "02795"]

good_ones = ["02795", "01321"]
good_ones = ["00158"]
good_ones = ["01264", "00795"]

good_ones = ["01626"]
good_ones = ["01619"]

# all visuals
good_ones = ["00009", "00158", "00795", "01209", "02417", "02996", "00029", "00596", "00821", "01284", "02795", "00269", "01193", "00034"]

# comparaisons with prior works
good_ones = ["00073", "00821", "01284", "03083", "00848"]


def extend_paths(path):
    if "INDEX" in path:
        paths = [path.replace("INDEX", str(i)) for i in range(10)]
    else:
        paths = [path]

    if "NAME" in path:
        all_paths = []
        for path in paths:
            all_paths.extend([path.replace("NAME", x) for x in good_ones])
        paths = all_paths

    return paths


def render_cli(cfg: DictConfig) -> None:
    from teach.render.blender import render

    import glob
    if (cfg.npy).endswith('.npy'):
        paths = extend_paths(cfg.npy)
    else:
        paths = glob.glob(f'{cfg.npy}/*.npy')
    init = True
    for path in paths:
        try:
            # data = np.load(path)
            data = np.load(path, allow_pickle=True).item()
            motion = data['motion']
            text = data['text']
            if 'lengths' in data:
                lens = data['lengths']
            else:
                lens = None
        except FileNotFoundError:
            logger.info(f"{path} not found")
            continue

        if cfg.mode == "video":
            frames_folder = path.replace(".npy", "_frames")
        else:
            frames_folder = path.replace(".npy", ".png")

        if True:
            out = render(motion, frames_folder,
                         cycle=cfg.cycle, res=cfg.res,
                         canonicalize=cfg.canonicalize,
                         exact_frame=cfg.exact_frame,
                         num=cfg.num, mode=cfg.mode,
                         faces_path=cfg.faces_path,
                         downsample=cfg.downsample,
                         always_on_floor=cfg.always_on_floor,
                         init=init,
                         gt=cfg.gt,
                         lengths=lens,
                         fake_translation=cfg.fake_trans,
                         separate_actions=cfg.separate_actions)

        init = False
        from teach.render.video import Video
        from wand.image import Image 
        from wand.drawing import Drawing 
        from wand.color import Color 
        # check here for colornames 
        # https://imagemagick.org/script/color.php
        from pathlib import Path
        
        if cfg.mode == "video":
            if cfg.downsample:
                video = Video(frames_folder, fps=30)
            else:
                video = Video(frames_folder, fps=30.0)
            # text = text.replace('--', ', ')
            vid_path = path.replace(".npy", ".mp4")
            video.save(out_path=vid_path)
            if cfg.text_vid:
                text = ', '.join(text)

                video.add_text(f'[{text}]')
                file = Path(path)
                new_name = file.stem + '_wtext' + '.mp4'
                video.save(out_path=f'{file.parent}/{new_name}')
                logger.info(vid_path)

        elif cfg.mode == 'sequence':
            if cfg.text_vid:

                if cfg.separate_actions:
                    for i, p in enumerate(out):
                        with Drawing() as draw: 
                            with Image(filename = p) as img: 
                                draw.font = 'Helvetica-Bold'
                                draw.font_size = 30
                                # draw.text_under_color = Color('gray30')
                                # draw.fill_color = Color('white')
                                draw.text_alignment = 'center'
                                draw.text(int(img.width / 2)-5, img.height - 15, f'{text[i]}')
                                draw(img)
                                file = Path(p)
                                new_name = file.stem + '_wtext' + '.png'
                                img.save(filename = f'{file.parent}/{new_name}')

                else:
                    image_path = out[0]
                    txt = '' 
                    print(text)
                    for i, a in enumerate(text):
                        if i == 0:
                            txt += f'[{a}, '
                        elif i == (len(text) - 1):
                            txt += f'{a}]' 
                        else:
                            txt +=  f'{a}, '
                    with Drawing() as draw: 
                        with Image(filename = image_path) as img: 
                            draw.font = 'Helvetica-Bold'
                            draw.font_size = 25
                            draw.text_under_color = Color('snow4')
                            draw.fill_color = Color('white')
                            draw.text_alignment = 'center'

                            draw.text(img.width//2, int(img.height)-15, txt)
                            draw(img)
                            file = Path(image_path)
                            new_name = file.stem + '_wtext' + '.png'
                            img.save(filename = f'{file.parent}/{new_name}')

            logger.info(f"Frame generated at: {out}")


if __name__ == '__main__':
    _render_cli()
