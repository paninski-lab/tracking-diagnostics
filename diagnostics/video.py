import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import subprocess
from typing import Optional

from diagnostics.utils import load_marker_csv


def get_frames_from_idxs(cap, idxs):
    """Helper function to load video segments.

    Parameters
    ----------
    cap : cv2.VideoCapture object
    idxs : array-like
        frame indices into video

    Returns
    -------
    np.ndarray
        returned frames of shape shape (n_frames, n_channels, ypix, xpix)

    """
    is_contiguous = np.sum(np.diff(idxs)) == (len(idxs) - 1)
    n_frames = len(idxs)
    for fr, i in enumerate(idxs):
        if fr == 0 or not is_contiguous:
            cap.set(1, i)
        ret, frame = cap.read()
        if ret:
            if fr == 0:
                height, width, _ = frame.shape
                frames = np.zeros((n_frames, 1, height, width), dtype='uint8')
            frames[fr, 0, :, :] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            print(
                'warning! reached end of video; returning blank frames for remainder of ' +
                'requested indices')
            break
    return frames


def make_labeled_video(
        save_file, cap, points, labels=None, likelihood_thresh=0.05, max_frames=None,
        markersize=6, framerate=20, height=4):
    """Behavioral video overlaid with markers.

    Parameters
    ----------
    save_file
    cap : cv2.VideoCapture object
    points : list of dicts
        keys of marker names and vals of marker values,
        i.e. `points['paw_l'].shape = (n_t, 3)`
    labels : list of strs
        name for each model in `points`
    likelihood_thresh : float
    max_frames : int or NoneType
    markersize : int
    framerate : float
        framerate of video
    height : float
        height of video in inches

    """

    tmp_dir = os.path.join(os.path.dirname(save_file), 'tmpZzZ')
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    keypoint_names = list(points[0].keys())
    n_frames = np.min([points[0][keypoint_names[0]].shape[0], max_frames])
    frame = get_frames_from_idxs(cap, [0])
    _, _, img_height, img_width, = frame.shape

    h = height
    w = h * (img_width / img_height)
    fig, ax = plt.subplots(1, 1, figsize=(w, h))
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlim([0, img_width])
    ax.set_ylim([img_height, 0])
    plt.subplots_adjust(wspace=0, hspace=0, left=0, bottom=0, right=1, top=1)

    colors = ['g', 'm', 'b']

    txt_kwargs = {
        'fontsize': 14, 'horizontalalignment': 'left',
        'verticalalignment': 'bottom', 'fontname': 'monospace', 'transform': ax.transAxes}

    txt_fr_kwargs = {
        'fontsize': 14, 'color': [1, 1, 1], 'horizontalalignment': 'left',
        'verticalalignment': 'top', 'fontname': 'monospace',
        'bbox': dict(facecolor='k', alpha=0.25, edgecolor='none'),
        'transform': ax.transAxes}

    for n in range(n_frames):

        ax.clear()  # important!! otherwise each frame will plot on top of the last

        if n % 100 == 0:
            print('processing frame %03i/%03i' % (n, n_frames))

        # plot original frame
        frame = get_frames_from_idxs(cap, [n])
        ax.imshow(frame[0, 0], vmin=0, vmax=255, cmap='gray')

        # plot markers
        for p, point_dict in enumerate(points):
            for m, (marker_name, marker_vals) in enumerate(point_dict.items()):
                if marker_vals[n, 2] < likelihood_thresh:
                    continue
                ax.plot(
                    marker_vals[n, 0], marker_vals[n, 1],
                    'o', markersize=markersize, color=colors[p], alpha=0.75)

        # add labels
        if labels is not None:
            for p, label_name in enumerate(labels):
                # plot label string
                ax.text(0.04, 0.04 + p * 0.05, label_name, color=colors[p], **txt_kwargs)

        # add frame number
        im = ax.text(0.02, 0.98, 'frame %i' % n, **txt_fr_kwargs)

        plt.savefig(os.path.join(tmp_dir, 'frame_%06i.jpeg' % n))

    save_video(save_file, tmp_dir, framerate, frame_pattern='frame_%06i.jpeg')


def make_labeled_video_wrapper(
        csvs: list,
        model_names: list,
        video: str,
        save_file: str,
        likelihood_thresh: float = 0.05,
        max_frames: Optional[int] = None,
        markersize: int = 6,
        framerate: float = 20,
        height: float = 4
):
    """

    Parameters
    ----------
    csvs : list
        list of absolute paths to video prediction csvs
    model_names : list
        list of model names for video legend; must have the same number of elements as `csvs`
    video : str
        absolute path to video (.mp4)
    save_file : str
        if absolute path: save labeled video here
        if string that is not absolute path, the labeled video will be saved in the same
        location as the original video with `save_file` appended to the filename

    """
    points = []
    for csv in csvs:
        xs, ys, ls, marker_names = load_marker_csv(csv)
        points_tmp = {}
        for m, marker_name in enumerate(marker_names):
            points_tmp[marker_name] = np.concatenate([
                xs[:, m, None], ys[:, m, None], ls[:, m, None]], axis=1)
        points.append(points_tmp)

    cap = cv2.VideoCapture(video)

    if os.path.isabs(save_file):
        save_file_full = save_file
    else:
        save_file_full = video.replace(".mp4", "%s.mp4" % save_file)

    make_labeled_video(
        save_file_full, cap, points, model_names,
        likelihood_thresh=likelihood_thresh, max_frames=max_frames, markersize=markersize,
        framerate=framerate, height=height
    )


def save_video(save_file, tmp_dir, framerate=20, frame_pattern='frame_%06d.jpeg'):
    """

    Parameters
    ----------
    save_file : str
        absolute path of filename (including extension)
    tmp_dir : str
        temporary directory that stores frames of video; this directory will be deleted
    framerate : float, optional
        framerate of final video
    frame_pattern : str, optional
        string pattern used for naming frames in tmp_dir

    """

    if os.path.exists(save_file):
        os.remove(save_file)

    # make mp4 from images using ffmpeg
    call_str = \
        'ffmpeg -r %f -i %s -c:v libx264 %s' % (
            framerate, os.path.join(tmp_dir, 'frame_%06d.jpeg'), save_file)
    print(call_str)
    subprocess.run(['/bin/bash', '-c', call_str], check=True)

    # delete tmp directory
    shutil.rmtree(tmp_dir)
