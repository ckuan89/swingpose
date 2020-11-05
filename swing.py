################### onset detection ###################

import numpy as np
import librosa
#import matplotlib.pyplot as plt
from pathlib import Path

from scipy.signal import butter,filtfilt

import os
import pandas as pd

from moviepy.tools import subprocess_call
from moviepy.config import get_setting

def butter_highpass(data,cutoff, fs, order=5):
    """
    Design a highpass filter.

    Args:
        - cutoff (float) : the cutoff frequency of the filter.
        - fs     (float) : the sampling rate.
        - order    (int) : order of the filter, by default defined to 5.
    """
    # calculate the Nyquist frequency
    nyq = 0.5 * fs

    # design filter
    high = cutoff / nyq
    b, a = butter(order, high, btype='high', analog=False)

    # returns the filter coefficients: numerator and denominator
    y = filtfilt(b, a, data)
    return y 


def onset_detection(file,cutoff=1000):
    x, sr = librosa.load(str(file))
    x_f=butter_highpass(x,cutoff, sr, order=5)
    o_env = librosa.onset.onset_strength(x_f, sr=sr)
    o_env=o_env/o_env.max()
    times = librosa.frames_to_time(np.arange(len(o_env)), sr=sr)
    #onset_frames = librosa.onset.onset_detect(onset_envelope=o_env, sr=sr)
    onset_frames = librosa.util.peak_pick(o_env, 3, 3, 3, 5, 0.3, 100)
    onset_times = librosa.frames_to_time(onset_frames)
    return onset_times
    
def ffmpeg_extract_subclip(filename, t1, t2, targetname=None):
    """ Makes a new video file playing video file ``filename`` between
    the times ``t1`` and ``t2``. """
    name, ext = os.path.splitext(filename)
    if not targetname:
        T1, T2 = [int(1000*t) for t in [t1, t2]]
        targetname = "%sSUB%d_%d.%s" % (name, T1, T2, ext)

    cmd = [get_setting("FFMPEG_BINARY"),"-y",
           "-ss", "%0.2f"%t1,
           "-i", filename,
           "-t", "%0.2f"%(t2-t1),
           "-vcodec", "copy", "-acodec", "copy", targetname]

    subprocess_call(cmd)

def cut_video(filename,onset_times,path_out,duration=3.):
    (path_out/(filename)).mkdir(exist_ok=True)
    for hit_time in onset_times:
        if round(hit_time)-duration/2 <0:

            ffmpeg_extract_subclip('video/'+filename+'.mp4',
                        0, round(hit_time)+duration/2, targetname=str(path_out)+'/'
                        +filename+'/'+filename+'_'+str(int(hit_time))+".mp4")
        else:

            ffmpeg_extract_subclip('video/'+filename+'.mp4',
                        round(hit_time)-duration/2, round(hit_time)+duration/2, targetname=str(path_out)+'/'
                        +filename+'/'+filename+'_'+str(int(hit_time))+".mp4")


