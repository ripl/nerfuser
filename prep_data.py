import argparse
import json
import os
import re
import shutil
from collections import defaultdict
from itertools import cycle
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from nerfstudio.process_data.colmap_utils import CameraModel, read_cameras_binary, read_images_binary, run_colmap
from nerfstudio.process_data.hloc_utils import run_hloc

from nerfuser.utils.utils import extract_colmap_pose
from nerfuser.utils.visualizer import Visualizer


def write2json(cam_params, poses, output_dir, name='transforms'):
    out = {cam_param: cam_params[cam_param] for cam_param in ['fl_x', 'fl_y', 'cx', 'cy', 'w', 'h', 'k1', 'k2', 'p1', 'p2']}
    out['camera_model'] = 'OPENCV'
    out['frames'] = [{'file_path': im_name, 'transform_matrix': trans.tolist()} for im_name, trans in poses.items()]
    with open(output_dir / f'{name}.json', 'w') as f:
        json.dump(out, f, indent=4)


ap = argparse.ArgumentParser()
ap.add_argument('--dataset-dir', type=Path, required=True)
ap.add_argument('--vid-ids', nargs='+', required=True)
ap.add_argument('--downsample', default=1, type=int)
ap.add_argument('--demo', action='store_true', help='if True, will not downsample the test split')
ap.add_argument('--flip', action='store_true')
ap.add_argument('--sfm-tool', default='hloc', choices=['colmap', 'hloc'])
ap.add_argument('--extract-images', action='store_true')
ap.add_argument('--run-sfm', action='store_true')
ap.add_argument('--write-json', action='store_true')
ap.add_argument('--vis', action='store_true')
args = ap.parse_args()

sfm_dir = args.dataset_dir / args.sfm_tool

if args.extract_images:
    for vid_id in args.vid_ids:
        output_dir = args.dataset_dir / vid_id
        shutil.rmtree(output_dir, ignore_errors=True)
        os.makedirs(output_dir)
        i = 0
        j = 0
        for f in os.listdir(args.dataset_dir):
            if f.startswith(vid_id) and f.endswith('.MOV'):
                cap = cv2.VideoCapture(str(args.dataset_dir / f))
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if args.demo and vid_id == 'test' or not i % args.downsample:
                        if args.flip:
                            frame = frame[::-1, ::-1]
                        cv2.imwrite(str(output_dir / f'{j:04d}.png'), frame)
                        j += 1
                    i += 1

if args.run_sfm:
    shutil.rmtree(sfm_dir, ignore_errors=True)
    os.makedirs(sfm_dir)
    for vid_id in args.vid_ids:
        for f in os.listdir(args.dataset_dir / vid_id):
            os.symlink(args.dataset_dir / vid_id / f, sfm_dir / f'{vid_id}_{f}')
    run_func = run_colmap if args.sfm_tool == 'colmap' else run_hloc
    run_func(sfm_dir, sfm_dir, CameraModel.OPENCV)

if args.write_json:
    cam = read_cameras_binary(sfm_dir / 'sparse/0/cameras.bin')[1]
    cam_params = {
        'fl_x': cam.params[0],
        'fl_y': cam.params[1],
        'cx': cam.params[2],
        'cy': cam.params[3],
        'w': cam.width,
        'h': cam.height,
        'k1': cam.params[4],
        'k2': cam.params[5],
        'p1': cam.params[6],
        'p2': cam.params[7],
    }
    images = read_images_binary(sfm_dir / 'sparse/0/images.bin')
    print(f'Got {len(images)} poses from {sfm_dir}.')
    pose_dicts = defaultdict(dict)
    for img_data in images.values():
        r = re.fullmatch('(.+)_(.+)', img_data.name)
        vid_id = r[1]
        img_id = r[2]
        pose_dicts[vid_id][img_id] = extract_colmap_pose(img_data)
    for vid_id in args.vid_ids:
        write2json(cam_params, pose_dicts[vid_id], args.dataset_dir / vid_id)


if args.vis:
    vis = Visualizer(show_frame=True)
    colors = cycle(plt.cm.tab10.colors)
    for vid_id in args.vid_ids:
        with open(args.dataset_dir / vid_id / 'transforms.json') as f:
            transforms = json.load(f)
        poses = np.array([frame['transform_matrix'] for frame in transforms['frames']], dtype=np.float32)
        vis.add_trajectory(poses, cam_size=0.1, color=next(colors))
    vis.show()
