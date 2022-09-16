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

import json
from PIL import Image
import cv2
import subprocess
import numpy as np
import moviepy.editor as mp
import moviepy.video.fx.all as vfx
import os
def to_vtt(frames, fps, acts, fname):
    import datetime
    # str(datetime.timedelta(seconds=666))
    # duration = max(max(frames)) / fps
    
    with open(fname, 'w') as f:
        f.write('WEBVTT\n\n')
        for f_pair, a in zip(frames, acts):
            f_start, f_end = f_pair

            def format_time(input_secs):
                hours , remainder = divmod(input_secs, 3600)
                minutes, seconds = divmod(remainder, 60)
                mins = str(int(minutes)).zfill(2)
                secs = str(round(seconds, 3)).zfill(6)
                hours = str(hours).zfill(2)
                return hours,mins, secs

            hs, ms, ss = format_time(f_start/fps)
            he, me, se = format_time(f_end/fps)

            f.write(f'{hs}:{ms}:{ss} --> {he}:{me}:{se}\n')
            f.write(f'{a}\n\n')

    return fname

def to_srt(frames, fps, acts, fname):
    import datetime
    # str(datetime.timedelta(seconds=666))
    # duration = max(max(frames)) / fps
    ln = 1
    with open(fname, 'w') as f:

        for f_pair, a in zip(frames, acts):
            f_start, f_end = f_pair
            f.write(f'{ln}\n')
            ln += 1
            def format_time(input_secs):
                hours , remainder = divmod(input_secs, 3600)
                minutes, seconds = divmod(remainder, 60)
                mins = str(int(minutes)).zfill(2)
                secs = str(round(seconds, 3)).zfill(6).replace('.', ',')
                hours = str(hours).zfill(2)
                return hours, mins, secs

            hs, ms, ss = format_time(f_start/fps)
            he, me, se = format_time(f_end/fps)

            f.write(f'{hs}:{ms}:{ss} --> {he}:{me}:{se}\n')
            f.write(f'{a}\n')
    # ffmpeg -i finalOutput.mp4 -vf subtitles=subs.srt out.mp4
    return fname

def read_json(p):
    with open(p, 'r') as fp:
        json_contents = json.load(fp)
    return json_contents

def write_json(data, p):
    with open(p, 'w') as fp:
        json.dump(data, fp, indent=2)

# Load npys
def loadnpys(path: str):
    import glob
    dict_of_npys = {}
    import numpy as np
    for p in glob.glob(f'{path}/*.npy'):
        data_sample = np.load(p, allow_pickle=True).item()
        fname = p.split('/')[-1]
        keyid = fname.replace('.npy', '')
        dict_of_npys[keyid] = (data_sample['text'], data_sample['lengths'])
    return dict_of_npys

class Video:
    # Credits to Lucas Ventura
    def __init__(self, frame_path: str, fps: float = 12.5):
        frame_path = str(frame_path)
        self.fps = fps

        self._conf = {"codec": "libx264",
                      "fps": self.fps,
                      "audio_codec": "aac",
                      "temp_audiofile": "temp-audio.m4a",
                      "remove_temp": True}

        self._conf = {"bitrate": "5000k",
                      "fps": self.fps}

        # Load videos
        # video = mp.VideoFileClip(video1_path, audio=False)
        frames = [os.path.join(frame_path, x) for x in sorted(os.listdir(frame_path))]
        video = mp.ImageSequenceClip(frames, fps=fps)
        # in case
        # video1 = video1.fx(vfx.mirror_x)
        self.video = video
        self.duration = video.duration

    def add_text(self, text):
        video_text = mp.TextClip(text,
                                 # font="Amiri-Regular", # 'Amiri - regular',
                                 font='Amiri',
                                 color='white',
                                 method='caption',
                                 align="center",
                                 size=(self.video.w, None),
                                 fontsize=30)
        video_text = video_text.on_color(size=(self.video.w, video_text.h + 5),
                                         color=(0, 0, 0),
                                         col_opacity=0.6)
        # video_text = video_text.set_pos('bottom')
        video_text = video_text.set_pos('top')

        self.video = mp.CompositeVideoClip([self.video, video_text])

    def save(self, out_path):
        out_path = str(out_path)
        self.video.subclip(0, self.duration).write_videofile(out_path, **self._conf)


