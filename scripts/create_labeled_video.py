"""Create a labeled video from csv files.

Users select an arbitrary number of csvs (one per model) from their file system to overlay on a
video

to run from command line:
> python /path/to/create_labeled_video.py

multiple prediction files can be specified from the command line
each must be preceded by "--prediction_files"
these should also be accompanied by a model name, preceded by "--model_names"

> python /path/to/create_labeled_video.py
--prediction_files=/path/to/pred0.csv --model_names=model0
--prediction_files=/path/to/pred1.csv --model_names=model1
--video_file=/path/to/video_file.mp4
--save_file=/path/to/labeled_video_file.mp4

"""

import argparse

from diagnostics.video import make_labeled_video_wrapper


def run():

    args = parser.parse_args()

    # make sure we have at least one input
    assert len(args.prediction_files) > 0
    # make sure we have a model name for each prediction
    assert len(args.prediction_files) == len(args.model_names)

    make_labeled_video_wrapper(
        csvs=args.prediction_files,
        model_names=args.model_names,
        video=args.video_file,
        save_file=args.save_file,
        likelihood_thresh=float(args.likelihood_thresh),
        max_frames=int(args.max_frames),
        markersize=int(args.markersize),
        framerate=float(args.framerate),
        height=float(args.height),
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--prediction_files', action='append', default=[])
    parser.add_argument('--model_names', action='append', default=[])
    parser.add_argument('--video_file')
    parser.add_argument('--save_file')
    parser.add_argument('--likelihood_thresh', default=0.05)
    parser.add_argument('--max_frames', default=500)
    parser.add_argument('--markersize', default=6)
    parser.add_argument('--framerate', default=20)
    parser.add_argument('--height', default=4)

    run()
