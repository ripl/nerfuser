import json

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from nerfstudio.process_data.colmap_utils import qvec2rotmat


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
    def cross(a, b) -> np.ndarray:
        return np.cross(a, b)
    if u is None:
        u = np.array([0, 0, 1])
    y = t - c
    y = y / np.linalg.norm(y)
    x = cross(y, u)
    x = x / np.linalg.norm(x)
    z = cross(x, y)
    R = np.array([x, y, z]).T
    if pose_spec == 1:
        R = R @ np.array([[1, 0, 0],
                         [0, 0, 1],
                          [0, -1, 0]])
    elif pose_spec == 2:
        R = R @ np.array([[1, 0, 0],
                         [0, 0, -1],
                          [0, 1, 0]])
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


def gen_hemispherical_poses(r, gamma_lo, gamma_hi=None, target=np.zeros(3), m=3, n=10, pose_spec=2):
    if gamma_hi is None:
        gamma_hi = gamma_lo
        gamma_lo = 0
    c2ws = []
    for g in np.linspace(gamma_lo, gamma_hi, num=m):
        c2ws.extend(gen_circular_poses(r * np.cos(g), r * np.sin(g), target=target, n=n, pose_spec=pose_spec))
    return c2ws


def complete_transform(T):
    """ completes T to be [..., 4, 4] """
    s = T.shape
    if s[-2:] == (4, 4):
        return T
    if isinstance(T, np.ndarray):
        return np.concatenate((T, np.broadcast_to(np.array([0, 0, 0, 1], dtype=T.dtype), (*s[:-2], 1, 4))), axis=-2)
    return torch.cat((T, torch.tensor([0, 0, 0, 1], dtype=T.dtype, device=T.device).broadcast_to(*s[:-2], 1, -1)), dim=-2)


# def sim3_log_map(T):
#     """ T: nx4x4 """
#     T = T.clone()
#     s = torch.linalg.det(T[:, :3, :3])**(1 / 3)
#     T[:, :3, :3] /= s[:, None, None]
#     v = se3_log_map(T.transpose(1, 2))
#     return torch.cat((v, torch.log(s)[:, None]), dim=1)


# def sim3_exp_map(t):
#     """ t: nx7 """
#     T = se3_exp_map(t[:, :6]).transpose(1, 2)
#     T[:, :3, :3] *= torch.exp(t[:, 6, None, None])
#     return T


def decompose_sim3(T):
    """ T: [..., 4, 4] """
    if isinstance(T, torch.Tensor):
        G = T.clone()
        s = torch.linalg.det(G[..., :3, :3])**(1 / 3)
    else:
        G = T.copy()
        s = np.linalg.det(G[..., :3, :3])**(1 / 3)
    G[:3, :3] /= s[..., None, None]
    return G, s


def compute_trans_diff(T1, T2):
    T = T2 @ np.linalg.inv(T1)
    G, s = decompose_sim3(T)
    R = Rotation.from_matrix(G[:3, :3])
    r = np.linalg.norm(R.as_rotvec()) / np.pi * 180
    t = np.linalg.norm(G[:3, 3])
    s = np.abs(np.log(s))
    return r, t, s


def extract_colmap_pose(colmap_im):
    rotation = qvec2rotmat(colmap_im.qvec)
    translation = colmap_im.tvec[:, None]
    w2c = complete_transform(np.concatenate((rotation, translation), axis=1))
    c2w = np.linalg.inv(w2c)
    # Convert from COLMAP's camera coordinate system (spec 1) to nerfstudio's (spec 2)
    c2w = c2w @ np.array([[1, 0, 0, 0],
                         [0, -1, 0, 0],
                         [0, 0, -1, 0],
                         [0, 0, 0, 1]])
    return c2w.astype(np.float32)


def write2json(cam_params, poses, output_dir, name='transforms'):
    out = {cam_param: cam_params[cam_param] for cam_param in ['fl_x', 'fl_y', 'cx', 'cy', 'w', 'h', 'k1', 'k2', 'p1', 'p2']}
    out['camera_model'] = 'OPENCV'
    out['frames'] = [{'file_path': im_name, 'transform_matrix': trans.tolist()} for im_name, trans in poses.items()]
    with open(output_dir / f'{name}.json', 'w') as f:
        json.dump(out, f, indent=4)


def rtg2xyz(rs, thetas, gammas):
    xs = rs * torch.cos(gammas) * torch.cos(thetas)
    ys = rs * torch.cos(gammas) * torch.sin(thetas)
    zs = rs * torch.sin(gammas)
    return xs, ys, zs


def xyz2rtg(xs, ys, zs):
    rs = torch.sqrt(xs**2 + ys**2 + zs**2)
    gammas = torch.asin(zs / rs)
    thetas = torch.atan2(ys, xs)
    return rs, gammas, thetas


def img_cat(imgs, axis, interval=0, color=255):
    assert axis in [0, 1], 'axis must be either 0 or 1'
    h, w, c = imgs[0].shape
    if axis:
        gap = np.broadcast_to(color, (h, interval, c))
    else:
        gap = np.broadcast_to(color, (interval, w, c))
    gap = gap.astype(imgs[0].dtype)
    n = len(imgs)
    t = [gap] * (n * 2 - 1)
    t[::2] = imgs
    return np.concatenate(t, axis=axis)


def dilute_image(img, t, b, l, r, p=0.5, color=(255, 255, 255)):
    img[t:b, l:r] = ((1 - p) * img[t:b, l:r] + p * np.asarray(color)).astype(img.dtype)


def idw(dists, g):
    t = dists.unsqueeze(1) / dists.unsqueeze(0)
    return 1 / (t**g).sum(dim=1)


def avg_trans(Ts, s=None, avg_func=np.mean):
    T = avg_func(Ts, axis=0)
    u, s_, vh = np.linalg.svd(T[:3, :3])
    T[:3, :3] = u * (s if s else s_) @ vh
    return T
