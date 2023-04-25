import os
import sys

import numpy as np
import open3d as o3d

sys.path.insert(0, os.path.dirname(__file__))
from line_mesh import LineMesh  # noqa: E402


class Visualizer:
    def __init__(self, show_frame=False, pt_size=1) -> None:
        self.o3d_vis = o3d.visualization.Visualizer()
        self.o3d_vis.create_window()
        self.o3d_vis.get_render_option().point_size = pt_size
        if show_frame:
            self.o3d_vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame())

    def add_trajectory(self, *poses, pose_spec=2, pose_type='c2w', K_invs=None, ws=None, hs=None, cam_size=1, line_width=None, color=(0.5, 0.5, 0.5), connect_cams=False):
        """ pose_spec:
                0: x->right, y->front, z->up
                1: x->right, y->down, z->front
                2: x->right, y->up, z->back """
        if len(poses) == 1:
            Rs = [pose[:3, :3] for pose in poses[0]]
            ts = [pose[:3, 3] for pose in poses[0]]
        else:
            Rs, ts = poses
        n_cams = len(Rs)
        if K_invs is None:
            K_invs = np.linalg.inv(np.array([[1000, 0, 400],
                                             [0, 1000, 300],
                                             [0, 0, 1]]))
            ws = 800
            hs = 600
        if np.ndim(K_invs) == 2:
            K_invs = np.broadcast_to(K_invs, (n_cams, 3, 3))
            hs = np.broadcast_to(hs, n_cams)
            ws = np.broadcast_to(ws, n_cams)
        if np.ndim(color) == 1:
            color = np.broadcast_to(color, (n_cams, len(color)))
        pts = np.zeros((n_cams, 4, 3, 1))
        pts[:, 1, 0, 0] = ws
        pts[:, 2, 1, 0] = hs
        pts[:, 3, 0, 0] = ws
        pts[:, 3, 1, 0] = hs
        pts[..., 2, 0] = 1
        points = []
        lines = []
        colors = []
        for i, (R, t) in enumerate(zip(Rs, ts)):
            if pose_type == 'w2c':
                R = R.T
                t = -R @ t
            if pose_spec == 0:
                R = R @ np.array([[1, 0, 0],
                                  [0, 0, 1],
                                  [0, -1, 0]])
            elif pose_spec == 2:
                R = R @ np.array([[1, 0, 0],
                                  [0, -1, 0],
                                  [0, 0, -1]])
            cam_pts = np.vstack((t, (R @ K_invs[i] @ pts[i])[..., 0] * cam_size + t))
            cam_ls = np.array([(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (2, 4), (3, 4)])
            points.extend(cam_pts)
            if connect_cams and len(lines):
                cam_ls = np.vstack((cam_ls, [-5, 0]))
            lines.extend(i * 5 + cam_ls)
            colors.extend([color[i] for _ in range(len(cam_ls))])
        if line_width is None:
            ls = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(points),
                lines=o3d.utility.Vector2iVector(lines),
            )
            ls.colors = o3d.utility.Vector3dVector(colors)
            self.o3d_vis.add_geometry(ls)
        else:
            lm = LineMesh(points, lines, colors, radius=line_width)
            lm.add_line(self.o3d_vis)

    def add_point_cloud(self, pts, color=(0.5, 0, 0.5)):
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        if np.ndim(color) == 1:
            pc.colors = o3d.utility.Vector3dVector(np.broadcast_to(color, (len(pts), len(color))))
        else:
            pc.colors = o3d.utility.Vector3dVector(color)
        self.o3d_vis.add_geometry(pc)

    def show(self):
        self.o3d_vis.run()
