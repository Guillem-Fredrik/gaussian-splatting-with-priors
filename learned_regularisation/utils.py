import numpy as np
import torch

from copy import copy
from dataclasses import dataclass

from scipy.spatial.transform import Slerp, Rotation
from scene.cameras import Camera
from utils.graphics_utils import focal2fov

@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


def make_4x4_transform(rotation, translation):
    transform_4x4 = np.zeros((4, 4))
    transform_4x4[:3, :3] = rotation
    transform_4x4[:3, 3] = translation
    transform_4x4[3, 3] = 1.
    return transform_4x4

@torch.cuda.amp.autocast(enabled=False)
def get_rays(poses, intrinsics, H, W, N=-1, error_map=None):
    ''' get rays
    Args:
        poses: [B, 4, 4], cam2world
        intrinsics: [4]
        H, W, N: int
        error_map: [B, 128 * 128], sample probability based on training error
    Returns:
        rays_o, rays_d: [B, N, 3]
        inds: [B, N]
    '''

    device = poses.device
    B = poses.shape[0]
    fx, fy, cx, cy = intrinsics

    i, j = torch.meshgrid(torch.linspace(0, W - 1, W, device=device), torch.linspace(0, H - 1, H, device=device), indexing='ij')
    i = i.t().reshape([1, H*W]).expand([B, H*W]) + 0.5
    j = j.t().reshape([1, H*W]).expand([B, H*W]) + 0.5

    results = {}

    if N > 0:
        N = min(N, H*W)

        if error_map is None:
            inds = torch.randint(0, H*W, size=[N], device=device) # may duplicate
            inds = inds.expand([B, N])
        else:

            # weighted sample on a low-reso grid
            inds_coarse = torch.multinomial(error_map.to(device), N, replacement=False) # [B, N], but in [0, 128*128)

            # map to the original resolution with random perturb.
            inds_x, inds_y = inds_coarse // 128, inds_coarse % 128 # `//` will throw a warning in torch 1.10... anyway.
            sx, sy = H / 128, W / 128
            inds_x = (inds_x * sx + torch.rand(B, N, device=device) * sx).long().clamp(max=H - 1)
            inds_y = (inds_y * sy + torch.rand(B, N, device=device) * sy).long().clamp(max=W - 1)
            inds = inds_x * W + inds_y

            results['inds_coarse'] = inds_coarse # need this when updating error_map

        i = torch.gather(i, -1, inds)
        j = torch.gather(j, -1, inds)

    else:
        inds = torch.arange(H*W, device=device).expand([B, H*W])

    zs = torch.ones_like(i)
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    directions = torch.stack((xs, ys, zs), dim=-1)
    directions = directions / torch.norm(directions, dim=-1, keepdim=True)
    rays_d = directions @ poses[:, :3, :3].transpose(-1, -2) # (B, N, 3)

    rays_o = poses[..., :3, 3] # [B, 3]
    rays_o = rays_o[..., None, :].expand_as(rays_d) # [B, N, 3]

    results['rays_o'] = rays_o
    results['rays_d'] = rays_d
    results['rays_d_cam'] = directions
    results['rays_d_cam_z'] = directions[..., -1] # Useful because it's the conversion factor to go from spherical to planar depths
    results['inds'] = inds

    return results
    
    
def analyse_poses(poses):
    # Print out some information about where the camera centres are
    # Can be useful in helping to set the bounds
    # Poses should be camera-to-world and 4x4
    camera_centres = [np.array(p[:3, 3]) for p in poses]
    camera_centres = np.array(camera_centres)
    bounds_min = np.min(camera_centres, axis=0)
    bounds_max = np.max(camera_centres, axis=0)
    print('Pose bounds:', bounds_min, bounds_max)
    print('Avg of min and max bounds:', 0.5*bounds_min + 0.5*bounds_max)


def get_typical_deltas_between_poses(poses):
    # Get avg deltas between poses
    poses = np.array([np.array(p) for p in poses])
    delta_positions = []
    delta_orientations = []
    for pose, next_pose in zip(poses[:-1], poses[1:]):
        delta_position = next_pose[:3, 3] - pose[:3, 3]
        delta_orientation = Rotation.from_matrix(pose[:3, :3]) * Rotation.from_matrix(next_pose[:3, :3]).inv()
        delta_positions.append(delta_position)
        delta_orientations.append(delta_orientation)

    mean_delta_pos = np.mean(np.linalg.norm(delta_positions, axis=1))
    mean_delta_ori = np.mean([np.linalg.norm(rot.as_rotvec()) for rot in delta_orientations])
    return mean_delta_pos, mean_delta_ori



def apply_intrinsics_to_camera(intrinsics, camera):
    # Apply intrinsics to camera
    # The result will be a camera that with prescribed FOV (`fx`, `fy`) and whose center is (`cx`, `cy`)
    fov_x = focal2fov(intrinsics.fx, camera.image_width)
    fov_y = focal2fov(intrinsics.fy, camera.image_height)
    new_camera = Camera(camera.colmap_id, camera.R, camera.T, fov_x, fov_y, camera.original_image, None,
                 camera.image_name, camera.uid,
                #  trans=np.array([intrinsics.cx, intrinsics.cy, 0.0]),
                 trans=np.array([0.0, 0.0, 0.0]),
                 scale=1.0, data_device = "cuda"
                 )
    new_camera.image_width = intrinsics.width
    new_camera.image_height = intrinsics.height
    # print("fov_orig", camera.FoVx, camera.FoVy)
    # print("fov_modf", new_camera.FoVx, new_camera.FoVy)
    return new_camera