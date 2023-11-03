import numpy as np
import torch
import torchvision

from copy import copy
from dataclasses import dataclass

from scipy.spatial.transform import Slerp, Rotation
from scene.cameras import Camera
from utils.graphics_utils import focal2fov, getWorld2View2, getProjectionMatrix

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



def apply_intrinsics_to_camera(intrinsics, camera, old_intrinsics):
    # Apply intrinsics to camera
    # The result will be a camera that with prescribed FOV (`fx`, `fy`) and whose center is (`cx`, `cy`)
    fov_x = focal2fov(intrinsics.fx, intrinsics.width)
    fov_y = focal2fov(intrinsics.fy, intrinsics.height)

    # The conversion between the old screen space and the new one is given by
    # (old_x, old_y, 1) = pixel_conversion @ (x, y, 1)
    # for 0<=x,y<1, where:
    pixel_conversion = np.array([
        1.0/camera.image_width, 0.0,                     0.0, 0.0,
        0.0,                    1.0/camera.image_height, 0.0, 0.0,
        0.0,                    0.0,                     1.0, 0.0,
        0.0,                    0.0,                     0.0, 1.0,
    ]).reshape(4, 4) @ np.array([
        old_intrinsics.fx, 0.0,               0.0, old_intrinsics.cx,
        0.0,               old_intrinsics.fy, 0.0, old_intrinsics.cy,
        0.0,               0.0,               1.0,  0.0,
        0.0,               0.0,               0.0,  1.0,
    ]).reshape(4, 4) @ np.array([
        1.0/intrinsics.fx, 0.0,               0.0, -intrinsics.cx / intrinsics.fx,
        0.0,               1.0/intrinsics.fy, 0.0, -intrinsics.cy / intrinsics.fy,
        0.0,               0.0,               1.0,  0.0,
        0.0,               0.0,               0.0,  1.0,
    ]).reshape(4, 4) @ np.array([
        intrinsics.width, 0.0,               0.0, 0.0,
        0.0,              intrinsics.height, 0.0, 0.0,
        0.0,              0.0,               1.0, 0.0,
        0.0,              0.0,               0.0, 1.0,
    ]).reshape(4, 4)
    
    # Screen coordinates range from -1 to 1 instead of 0 to 1, so we must correct for that
    M = np.array([
        2.0, 0.0, 0.0, -1.0,
        0.0, 2.0, 0.0, -1.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ]).reshape(4, 4)
    pixel_conversion = M @ pixel_conversion @ np.linalg.inv(M)

    # For a point (x,y,z) in world space, we therefore want
    # camera.full_proj_transform @ (x,y,z) == pixel_conversion @ new_camera.full_proj_transform @ (x,y,z)
    # So
    # X := new_camera.full_proj_transform == pixel_conversion.inv() @ camera.full_proj_transform
    X = np.linalg.inv(pixel_conversion) @ camera.full_proj_transform.transpose(0,1).cpu().numpy()

    # As X = new_proj @ c2w, the camera2world matrix will be
    c2w = np.linalg.inv(getProjectionMatrix(znear=0.01, zfar=100.0, fovX=fov_x, fovY=fov_y)) @ X

    # So the new R and T will be
    R = c2w[:3, :3].transpose()
    T = c2w[:3, 3]

    new_camera = Camera(camera.colmap_id, R, T, fov_x, fov_y, camera.original_image, None,
                 camera.image_name, camera.uid,
                 trans=np.array([0.0, 0.0, 0.0]),
                 scale=1.0, data_device = "cuda"
                 )
    new_camera.image_width = intrinsics.width
    new_camera.image_height = intrinsics.height
    return new_camera

def averaged_depth_and_normal(depth_render, intrinsics, fov_radius = 0.01):
    # return depth_render[intrinsics.width//2, intrinsics.height//2].detach().cpu().numpy()

    fov_x = focal2fov(intrinsics.fx, intrinsics.width)
    fov_y = focal2fov(intrinsics.fy, intrinsics.height)
    std_x = intrinsics.width * fov_radius/fov_x
    std_y = intrinsics.height * fov_radius/fov_y
    std = int(0.5*(std_x + std_y))

    with torch.no_grad():
        depth_crop = depth_render[intrinsics.width//2-std-1:intrinsics.width//2+std+1, intrinsics.height//2-std-1:intrinsics.height//2+std+1]   
        blur_filter = torchvision.transforms.GaussianBlur(kernel_size=(2*std+1, 2*std+1), sigma=std)
        depth_blurred = blur_filter(depth_crop.unsqueeze(-1).permute(2,1,0)).permute(2,1,0).squeeze()
        return depth_blurred[depth_blurred.shape[0]//2, depth_blurred.shape[1]//2].detach().cpu().numpy()

    std = int(std)
    with torch.no_grad():
        depth = torch.mean(depth_render[intrinsics.width//2-std:intrinsics.width//2+std, intrinsics.height//2-std:intrinsics.height//2+std]).detach().cpu().numpy()

    return depth