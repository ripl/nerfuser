import json
import re
import shutil
from collections import defaultdict
from itertools import cycle
from pathlib import Path
from typing import Literal, Optional

import imageio
import matplotlib.pyplot as plt
import numpy as np
import tyro
from nerfstudio.process_data.colmap_utils import CameraModel, read_cameras_binary, read_images_binary, run_colmap
from nerfstudio.process_data.hloc_utils import run_hloc
from tqdm import tqdm

from nerfuser.utils.utils import extract_colmap_pose, write2json
from nerfuser.utils.visualizer import Visualizer


def main(dataset_dir: Path,
         output_dir: Optional[Path] = None,
         vid_ids: Optional[set[str]] = None,
         vid_exts: Optional[set[str]] = None,
         fps: Optional[float] = None,
         downsample: Optional[int] = None,
         n_frames: Optional[int] = None,
         demo: bool = False,
         flip: bool = False,
         sfm_tool: Literal['hloc', 'colmap'] = 'hloc',
         joint_sfm: bool = True,
         extract_images: bool = False,
         run_sfm: bool = False,
         write_json: bool = False,
         vis: bool = False):
    """prepare datasets of posed-RGB images from videos

    Args:
        dataset_dir: the directory containing source videos
        output_dir: the directory to save the prepared datasets; if None, will use dataset-dir
        vid_ids: the video ids to process; files starting with the same vid-id will be jointly treated; if None, will try all
        vid_exts: the video extensions to process; if None, will try all
        fps: the frame rate at which images are extracted; at most one of fps, downsample and n-frames should be specified; if all None, will use the original frame rate
        downsample: the factor at which videos are downsampled; at most one of fps, downsample and n-frames should be specified; if all None, will not downsample
        n_frames: the number of frames to extract; at most one of fps, downsample and n-frames should be specified; if all None, will use all frames
        demo: if True, will not downsample the test split
        flip: if True, will flip the images
        sfm_tool: the SfM tool to use
        joint_sfm: if True, will run SfM jointly for all extracted images
        extract_images: if True, will extract images from videos
        run_sfm: if True, will run SfM
        write_json: if True, will write the SfM results to json
        vis: if True, will visualize the SfM results
    """
    if output_dir is None:
        output_dir = dataset_dir
    if vid_ids is None:
        vid_ids = set(f.stem for f in dataset_dir.iterdir() if f.is_file())
    assert sum((fps is None, downsample is None, n_frames is None)) >= 2, 'at most one of fps, downsample and n-frames should be specified'

    vids = defaultdict(set)
    for vid_id in vid_ids:
        for f in dataset_dir.iterdir():
            if not f.stem.startswith(vid_id) or (vid_exts and f.suffix not in vid_exts):
                continue
            try:
                assert 'nframes' in imageio.get_reader(f).get_meta_data()
            except Exception:
                continue
            vids[vid_id].add(f)
    sfm_dir = output_dir / sfm_tool

    if extract_images:
        for vid_id in vids:
            cur_output_dir = output_dir / vid_id
            shutil.rmtree(cur_output_dir, ignore_errors=True)
            cur_output_dir.mkdir()
            for f in vids[vid_id]:
                vid = imageio.get_reader(f)
                vid_info = vid.get_meta_data()
                vid_fps = vid_info['fps']
                vid_duration = vid_info['duration']
                vid_n_frames = round(vid_fps * vid_duration)
                if demo and vid_id == 'test':
                    img_ids = np.arange(vid_n_frames)
                if fps is not None:
                    img_ids = np.linspace(0, vid_n_frames - 1, num=round(min(fps, vid_fps) * vid_duration)).round().astype(int)
                elif downsample is not None:
                    img_ids = np.arange(vid_n_frames, step=max(downsample, 1))
                elif n_frames is not None:
                    img_ids = np.linspace(0, vid_n_frames - 1, num=min(n_frames, vid_n_frames)).round().astype(int)
                else:
                    img_ids = np.arange(vid_n_frames)
                i = 0
                for j, img in enumerate(tqdm(vid, total=vid_n_frames, leave=False)):
                    if j == img_ids[i]:
                        if flip:
                            img = img[::-1, ::-1]
                        imageio.v3.imwrite(str(cur_output_dir / f'{i:04d}.png'), img)
                        i += 1
                        if i == len(img_ids):
                            break

    if run_sfm:
        shutil.rmtree(sfm_dir, ignore_errors=True)
        run_func = run_colmap if sfm_tool == 'colmap' else run_hloc
        if joint_sfm:
            sfm_dir.mkdir()
            for vid_id in vids:
                for f in (output_dir / vid_id).iterdir():
                    (sfm_dir / f'{vid_id}_{f.name}').symlink_to(f)
            run_func(sfm_dir, sfm_dir, CameraModel.OPENCV)
        else:
            for vid_id in vids:
                cur_sfm_dir = sfm_dir / vid_id
                cur_sfm_dir.mkdir()
            run_func(output_dir / vid_id, cur_sfm_dir, CameraModel.OPENCV)

    if write_json:
        if joint_sfm:
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
            for vid_id in vids:
                write2json(cam_params, pose_dicts[vid_id], output_dir / vid_id)
        else:
            for vid_id in vids:
                cur_sfm_dir = sfm_dir / vid_id
                cam = read_cameras_binary(cur_sfm_dir / 'sparse/0/cameras.bin')[1]
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
                images = read_images_binary(cur_sfm_dir / 'sparse/0/images.bin')
                print(f'Got {len(images)} poses from {cur_sfm_dir}.')
                pose_dict = {img_data.name: extract_colmap_pose(img_data) for img_data in images.values()}
                write2json(cam_params, pose_dict, output_dir / vid_id)

    if vis:
        if joint_sfm:
            vis = Visualizer(show_frame=True)
            colors = cycle(plt.cm.tab10.colors)
            for vid_id in vids:
                with open(output_dir / vid_id / 'transforms.json') as f:
                    transforms = json.load(f)
                poses = np.array([frame['transform_matrix'] for frame in transforms['frames']], dtype=np.float32)
                vis.add_trajectory(poses, cam_size=0.1, color=next(colors))
            vis.show()
        else:
            for vid_id in vids:
                vis = Visualizer(show_frame=True)
                with open(output_dir / vid_id / 'transforms.json') as f:
                    transforms = json.load(f)
                poses = np.array([frame['transform_matrix'] for frame in transforms['frames']], dtype=np.float32)
                vis.add_trajectory(poses, cam_size=0.1, color=(0, 0.8, 0))
                vis.show()


if __name__ == '__main__':
    tyro.cli(main)
