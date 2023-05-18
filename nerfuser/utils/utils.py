import numpy as np
import torch
from nerfstudio.process_data.colmap_utils import qvec2rotmat
from scipy.spatial.transform import Rotation


def ch_pose_spec(T, src, tgt, pose_type='c2w'):
    """ pose_spec:
            0: x->right, y->front, z->up
            1: x->right, y->down, z->front
            2: x->right, y->up, z->back """
    # x-to-0 transforms
    Ts = np.array([np.identity(4),
                   [[1, 0, 0, 0],
                    [0, 0, 1, 0],
                    [0, -1, 0, 0],
                    [0, 0, 0, 1]],
                   [[1, 0, 0, 0],
                    [0, 0, -1, 0],
                    [0, 1, 0, 0],
                    [0, 0, 0, 1]]], dtype=T.dtype)
    s = T.shape[-2:]
    T = complete_trans(T)
    return (T @ np.linalg.inv(Ts[src]) @ Ts[tgt] if pose_type == 'c2w' else np.linalg.inv(Ts[tgt]) @ Ts[src] @ T)[..., :s[0], :s[1]]


def gen_lookat_pose(c, t, u=None, pose_spec=2, pose_type='c2w'):
    """ generates a c2w pose
        c: camera center
        t: target to look at
        u: up vector
        pose_spec: cam frame spec
            0: x->right, y->front, z->up
            1: x->right, y->down, z->front
            2: x->right, y->up, z->back
        we assume world frame spec is 0
        pose_type: one of {'c2w', 'w2c'} """
    if u is None:
        u = np.array([0, 0, 1])
    y = t - c
    y = y / np.linalg.norm(y)
    x = np.cross(y, u)
    x = x / np.linalg.norm(x)
    z = np.cross(x, y)
    R = ch_pose_spec(np.array([x, y, z]).T, 0, pose_spec)
    if pose_type == 'w2c':
        R = R.T
        c = -R @ c
    return np.concatenate((R, c[:, None]), axis=1, dtype=np.float32)


def gen_elliptical_poses(a, b, theta, h, target=np.zeros(3), n=10, pose_spec=2):
    """ generate n poses (c2w) distributed along an ellipse of (a, b, theta) at height h"""
    poses = []
    for alpha in np.linspace(0, np.pi * 2, num=n, endpoint=False):
        x0 = a * np.cos(alpha)
        y0 = b * np.sin(alpha)
        x = x0 * np.cos(theta) - y0 * np.sin(theta)
        y = y0 * np.cos(theta) + x0 * np.sin(theta)
        z = h
        poses.append(gen_lookat_pose(np.array([x, y, z]), target, pose_spec=pose_spec))
    return poses


def gen_circular_poses(r, h, target=np.zeros(3), n=10, pose_spec=2):
    """ generate n poses (c2w) distributed along a circle of radius r at height h"""
    return gen_elliptical_poses(r, r, 0, h, target=target, n=n, pose_spec=pose_spec)


def gen_hemispheric_poses(r, gamma_lo, gamma_hi=None, target=np.zeros(3), m=3, n=10, pose_spec=2):
    if gamma_hi is None:
        gamma_hi = gamma_lo
        gamma_lo = 0
    c2ws = []
    for g in np.linspace(gamma_lo, gamma_hi, num=m):
        c2ws.extend(gen_circular_poses(r * np.cos(g), r * np.sin(g), target=target, n=n, pose_spec=pose_spec))
    return c2ws


def complete_trans(T):
    """ completes T to be [..., 4, 4] """
    s = T.shape
    if s[-2:] == (4, 4):
        return T
    T_comp = np.zeros((*s[:-2], 4, 4), dtype=T.dtype) if isinstance(T, np.ndarray) else torch.zeros(*s[:-2], 4, 4, dtype=T.dtype, device=T.device)
    T_comp[..., :3, :s[-1]] = T
    T_comp[..., 3, 3] = 1
    return T_comp


def decompose_sim3(T):
    """ T: [..., 4, 4] """
    if isinstance(T, np.ndarray):
        G = T.copy()
        s = np.linalg.det(G[..., :3, :3])**(1 / 3)
    else:
        G = T.clone()
        s = torch.linalg.det(G[..., :3, :3])**(1 / 3)
    G[..., :3, :3] /= s[..., None, None]
    return G, s


def compute_trans_diff(T1, T2):
    T = T2 @ np.linalg.inv(T1)
    G, s = decompose_sim3(T)
    R = Rotation.from_matrix(G[:3, :3])
    r = np.linalg.norm(R.as_rotvec()) / np.pi * 180
    t = np.linalg.norm(G[:3, 3])
    s = np.abs(np.log(s))
    return r, t, s


def avg_trans(Ts, s=None, avg_func=np.mean):
    T = avg_func(Ts, axis=0)
    u, s_, vh = np.linalg.svd(T[:3, :3])
    T[:3, :3] = u * (s if s else s_) @ vh
    return T


def extract_colmap_pose(colmap_im):
    rotation = qvec2rotmat(colmap_im.qvec)
    translation = colmap_im.tvec[:, None]
    # w2c of spec 1
    w2c = complete_trans(np.concatenate((rotation, translation), axis=1))
    return ch_pose_spec(np.linalg.inv(w2c), 1, 2).astype(np.float32)


def img_cat(imgs, axis, interval=0, color=255):
    assert axis in [0, 1], 'axis must be either 0 or 1'
    h, w, c = imgs[0].shape
    gap = np.broadcast_to(color, (h, interval, c) if axis else (interval, w, c)).astype(imgs[0].dtype)
    t = [gap] * (len(imgs) * 2 - 1)
    t[::2] = imgs
    return np.concatenate(t, axis=axis)
