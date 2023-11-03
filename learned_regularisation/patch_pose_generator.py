import random
from dataclasses import dataclass
from typing import Optional, List

import torch
import numpy as np
from scipy.spatial.transform import Rotation

from learned_regularisation.utils import Intrinsics, make_4x4_transform
from scene.cameras import Camera

@dataclass
class FrustumChecker:
    """
    Simple class to check whether a point lies within the camera frustum.
    Parameterised by the camera FOVs along the horizontal and vertical axes.
    """
    fov_x_rads: float
    fov_y_rads: float

    def is_in_frustum(self, pose_c2w, point_world) -> bool:
        rotation, translation = unpack_4x4_transform(np.linalg.inv(pose_c2w))
        point_cam = rotation @ point_world + translation
        x, y, z = point_cam
        alpha_x = np.arctan(x / (z + 1e-6))
        alpha_y = np.arctan(y / (z + 1e-6))
        # Point must be in front of camera, not behind it
        if z < 0:
            return False
        # Point must be within frustum
        if np.abs(alpha_x) < 0.5 * self.fov_x_rads and np.abs(alpha_y) < 0.5 * self.fov_y_rads:
            return True
        return False


class FrustumRegulariser:
    def __init__(self, cameras: List[Camera], intrinsics: Intrinsics, reg_strength: float,
                 min_near: float = 0.):
        """
        Frustum regulariser as described in the DiffusioNeRF paper. Exists to penalise placement of material
        that is visible only in one frustum.
        :param poses: List of poses for the training views, so that the i-th entry in poses if the pose of i-th training
        view.
        :param intrinsics: Intrinsics for the training views.
        :param reg_strength: Multiplier for the loss function.
        :param min_near: Points will be only considered to lie in a frustum if their depth is at least min_near. This should be the same
        value of min_near which is used in rendering (i.e. whatever opt.min_near is).
        """
        self.transforms_w2c = [camera.world_view_transform for camera in cameras]
        self.intrinsics = intrinsics
        self.reg_strength = reg_strength
        self.min_near = min_near
        assert self.min_near >= 0.
        print('Initialised FrustumRegulariser with min_near', self.min_near)

    def is_in_frustum(self, transform_w2c, points_world) -> float:
        rotation, translation = unpack_4x4_transform(transform_w2c)
        points_cam = points_world @ rotation.T + translation
        x = points_cam[..., 0]
        y = points_cam[..., 1]
        z = points_cam[..., 2]

        pixel_i = self.intrinsics.fx * (x/(z + 1e-12)) + self.intrinsics.cx
        pixel_j = self.intrinsics.fy * (y/(z + 1e-12)) + self.intrinsics.cy

        z_mask = z >= self.min_near
        pixel_i_mask = (0. <= pixel_i) & (pixel_i <= self.intrinsics.width)
        pixel_j_mask = (0. <= pixel_j) & (pixel_j <= self.intrinsics.height)
        return z_mask & pixel_i_mask & pixel_j_mask

    def count_frustums(self, xyzs):
        frustum_counts = torch.zeros(len(xyzs,)).to(xyzs.device)
        for tf in self.transforms_w2c:
            frustum_counts += self.is_in_frustum(tf, xyzs)
        return frustum_counts

    def __call__(self, xyzs: torch.Tensor, weights: torch.Tensor, frustum_count_thresh: int = 1,
                 debug_vis_name: Optional[str] = None) -> torch.Tensor:
        """
        Compute the frustum regularisation loss for some points.
        Will compute a loss proportional to the total amount of alpha-compositing weight which
        is visible from fewer than frustum_count_thresh frustums.

        :param xyzs: Points lying on rays which are being used in rendering.
        :param weights: The weights used for each of those points in alpha-compositing.
        :param frustum_count_thresh:
        :param debug_vis_name: Optional name for writing debug point clouds.
        :return: Frustum regularisation loss.
        """
        with torch.no_grad():
            frustum_counts = self.count_frustums(xyzs)
        print('Frustum count range', frustum_counts.min(), frustum_counts.max())

        penalty_size = frustum_count_thresh - frustum_counts
        penalty_size = torch.clip(penalty_size, min=0.)
        loss = self.reg_strength * weights * penalty_size

        # For debug: write points inside & outside frustum
        if debug_vis_name is not None:
            loss_mask = frustum_counts < frustum_count_thresh
            loss = self.reg_strength * weights[loss_mask].unsqueeze(-1)
            in_frustum = xyzs[~loss_mask].detach().cpu().numpy()
            outside_frustum = xyzs[loss_mask].detach().cpu().numpy()
            np.savetxt(f'in_frustum-{debug_vis_name}.txt', in_frustum)
            np.savetxt(f'outside_frustum-{debug_vis_name}.txt', outside_frustum)
            zero_frustum_xyzs = xyzs[frustum_counts < 0.5].detach().cpu().numpy()
            np.savetxt(f'zero_frustum_{debug_vis_name}.txt', zero_frustum_xyzs)

        return loss.sum()


class LenticularRegulariser:
    def __init__(self, cameras: List[Camera], reg_strength: float, epsilon: float = 1e-1):
        """
        Lenticular regulariser. Exists to penalise gaussians that are very flat (basically invisible) in the direction of one of the training cameras. As a side effect, gaussians tend to face the camera directly (i.e. be very flat in directions orthogonal to it), so we also penalise this.
        :param cameras: List of cameras for the training views, so that the i-th entry in cameras is the camera of i-th training
        view.
        :param reg_strength: Multiplier for the loss function.
        :param epsilon: Small value to avoid numerical errors and prevent overly large gradients. Should not be too small
        """
        self.cameras = cameras
        self.reg_strength = reg_strength
        self.epsilon = epsilon
        assert self.epsilon >= 0.

    def complete_orthonormal_basis(self, A: torch.Tensor) -> torch.Tensor:
            # A is of shape (N, 3)

            # Step 1: Create a set of arbitrary vectors U
            U = torch.ones_like(A)
            mask = torch.allclose(A, torch.tensor([1.0, 0.0, 0.0], device="cuda"), atol=1e-7)
            U[mask] = torch.tensor([0.0, 1.0, 0.0], device="cuda")

            # Step 2: Compute tensor B which is orthogonal to A
            B = torch.cross(U, A, dim=1)

            # Step 3: Normalize vectors in B
            B = torch.nn.functional.normalize(B, dim=1)

            # Step 4: Compute tensor C orthogonal to both A and B
            C = torch.cross(A, B, dim=1)

            # Step 5: Normalize vectors in C
            C = torch.nn.functional.normalize(C, dim=1)

            # Combine B and C into a single tensor of shape (N, 3, 2)
            BC = torch.stack((B, C), dim=2)

            return BC

    def product_of_two_largest(self, tensor):
        # Sort each row in descending order
        sorted_vals, _ = torch.sort(tensor, dim=1, descending=True)

        # Select the first two columns (largest values) and compute their product
        result = sorted_vals[:, 0] * sorted_vals[:, 1]

        return result, sorted_vals[:, 0]

    def __call__(self, xyzs: torch.Tensor, scales: torch.Tensor, weights: torch.Tensor, covariances: torch.Tensor) -> torch.Tensor:
        """
        Compute the lenticular regularisation loss for some gaussians.

        :param xyzs: Centers of the gaussians.
        :param scales: Scales of the gaussians in their principal directions.
        :param weights: The weights used for each of those points in alpha-compositing.
        :param covariances: Covariance matrices (or rather, array of its entries) for the gaussians in world space.
        :return: Lenticular regularisation loss.
        """

        loss = torch.zeros(len(self.cameras))

        covariance_matrices = torch.zeros((covariances.shape[0], 3, 3), dtype=torch.float, device="cuda")
        
        covariance_matrices[:, 0, 0] = covariances[:, 0]
        covariance_matrices[:, 0, 1] = covariances[:, 1]
        covariance_matrices[:, 1, 0] = covariances[:, 1]
        covariance_matrices[:, 0, 2] = covariances[:, 2]
        covariance_matrices[:, 2, 0] = covariances[:, 2]
        covariance_matrices[:, 1, 1] = covariances[:, 3]
        covariance_matrices[:, 1, 2] = covariances[:, 4]
        covariance_matrices[:, 2, 1] = covariances[:, 4]
        covariance_matrices[:, 2, 2] = covariances[:, 5]

        # guassian_volumes = scales[:, 0]*scales[:, 1]*scales[:, 2]

        # for i, camera in enumerate(self.cameras):
        #     camera_center = camera.camera_center
        #     camera_to_xyzs = xyzs - camera_center
        #     camera_to_xyzs = torch.nn.functional.normalize(camera_to_xyzs)
            
        #     gaussian_depth = torch.bmm(camera_to_xyzs.unsqueeze(-2), torch.bmm(covariance_matrices, camera_to_xyzs.unsqueeze(-1)))
        #     gaussian_areas_log = -torch.log(self.epsilon + guassian_volumes / (self.epsilon + gaussian_depth))

        #     loss[i] = torch.mean(weights * gaussian_areas_log)

        maximum_areas, maximum_lengths = self.product_of_two_largest(scales)

        for i, camera in enumerate(self.cameras):
            camera_center = camera.camera_center
            camera_to_xyzs = xyzs - camera_center
            camera_to_xyzs = torch.nn.functional.normalize(camera_to_xyzs)

            gaussian_depth_2 = torch.bmm(camera_to_xyzs.unsqueeze(-2), torch.bmm(covariance_matrices, camera_to_xyzs.unsqueeze(-1))).squeeze()
            gaussian_depths_log = torch.log(self.epsilon + maximum_lengths**2) - torch.log(self.epsilon + gaussian_depth_2)

            ON = self.complete_orthonormal_basis(camera_to_xyzs)
            orthogonal_covariance = torch.bmm(ON.transpose(1,2), torch.bmm(covariance_matrices, ON))
            gaussian_areas_2 = orthogonal_covariance[:,0,0] * orthogonal_covariance[:,1,1] - orthogonal_covariance[:,0,1] * orthogonal_covariance[:,1,0]
            gaussian_areas_log = torch.log(self.epsilon + maximum_areas**2) - torch.log(self.epsilon + gaussian_areas_2)

            camera_loss = 3e-1*gaussian_depths_log + gaussian_areas_log

            loss[i] = torch.mean(weights * camera_loss)

        return self.reg_strength * loss.mean()


class PatchPoseGenerator:
    """
    Generates poses at which to render patches, by taking the training poses and perturbing them.
    """
    def __init__(self, cameras, spatial_perturbation_magnitude: float, angular_perturbation_magnitude_rads: float,
                 no_perturb_prob: float = 0., frustum_checker: Optional[FrustumChecker] = None):
        """
        Initialise the pose generator with a set of cameras and an amount of jitter to apply to them.
        :param cameras: List of training cameras, s.t. the i-th entry is the pose of the i-th training view.
        :param spatial_perturbation_magnitude: Amount of jitter to apply to camera centres, in units of length.
        :param angular_perturbation_magnitude_rads: Amount of jitter to apply to camera orientations, in radians.
        :param no_perturb_prob: Fraction of the time to not apply any perturbation at all.
        :param frustum_checker: If given, will ensure that the perturbed camera centre lies within at least one
        training frustum.
        """
        self._cameras = cameras
        self._spatial_mag = spatial_perturbation_magnitude
        self._angular_mag = angular_perturbation_magnitude_rads
        self._no_perturb_prob = no_perturb_prob
        self._frustum_checker = frustum_checker
        self._screen_center_depths = [0.0 for camera in cameras]

    def __len__(self):
        return len(self._cameras)

    def _perturb_camera(self, camera_to_perturb):
        while True:
            new_camera = perturb_camera(camera=camera_to_perturb, spatial_mag=self._spatial_mag, angular_mag=self._angular_mag)
            return new_camera
            # TODO(guillem)
            # _, camera_centre = unpack_4x4_transform(new_camera)
            # for camera in self._cameras:
            #     if self._frustum_checker is None or self._frustum_checker.is_in_frustum(camera_c2w=camera,
            #                                                                             point_world=camera_centre):
            #         return new_camera
            #     else:
            #         pass

    def _perturb_camera_2(self, camera_to_perturb, depth):
        depth = 0.5*depth
        yaw_mag = camera_to_perturb.FoVx
        while True:
            new_camera = perturb_camera_2(camera=camera_to_perturb, depth=depth, yaw_mag=0.1*yaw_mag, spatial_mag=self._spatial_mag, angular_mag=self._angular_mag)
            return new_camera
            # TODO(guillem)
            # _, camera_centre = unpack_4x4_transform(new_camera)
            # for camera in self._cameras:
            #     if self._frustum_checker is None or self._frustum_checker.is_in_frustum(camera_c2w=camera,
            #                                                                             point_world=camera_centre):
            #         return new_camera
            #     else:
            #         pass


    def __getitem__(self, idx):
        # Generate a pose by perturbing the idx-th training view.
        camera = self._cameras[idx]
        if random.random() > self._no_perturb_prob:
            # new_camera = self._perturb_camera(camera)
            new_camera = self._perturb_camera_2(camera, self._screen_center_depths[idx])
        else:
            new_camera = camera
        return new_camera
    
    def generate_random(self):
        # Generate a pose by perturbing a random training view.
        idx = random.randint(0, len(self._cameras)-1)
        return self[idx]


def perturb_camera(camera, spatial_mag: float, angular_mag: float):
    # Sample perturbation to camera centre
    cam_centre_perturbation = spatial_mag * (2. * torch.rand(3,) - 1.)
    new_translation = camera.T + cam_centre_perturbation.numpy()

    # Sample perturbation to orientation
    rotation_perturbation = Rotation.random().as_rotvec() * (angular_mag / (2. * torch.pi))
    rotation_perturbation = Rotation.from_rotvec(rotation_perturbation).as_matrix()
    new_rotation = rotation_perturbation @ np.asarray(camera.R)

    new_camera = Camera(camera.colmap_id, new_rotation, new_translation, camera.FoVx, camera.FoVy, camera.original_image, None,
        camera.image_name, camera.uid,
        trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
        )

    return new_camera

def perturb_camera_2(camera, depth, yaw_mag: float, spatial_mag: float, angular_mag: float):
    # We will cast a ray of length `depth` in the direction of the camera, and rotate the camera around that point
    # Note that
    # camera.center = self.world_view_transform.inverse()[3, :3]
    #               = (0,0,0,1) @ self.world_view_transform.inverse()
    #               = v such that v @ R + T = 0
    #               = -T @ R.inv()
    # So T = -camera.center @ R

    old_camera_center = camera.camera_center.cpu().numpy()
    camera_direction = np.array([0.0, 0.0, 1.0]) @ np.linalg.inv(camera.R)
    mv = depth * camera_direction / np.linalg.norm(camera_direction)
    camera_focus_point = old_camera_center + mv

    yaw_perturbation = yaw_mag * (2.*np.random.random() - 1.)
    cos_yaw = np.cos(yaw_perturbation)
    sin_yaw = np.sin(yaw_perturbation)
    # new_rotation = np.array([
    #     [ cos_yaw, sin_yaw, 0.],
    #     [-sin_yaw, cos_yaw, 0.],
    #     [ 0.,      0.,      1.]
    # ])
    new_rotation = np.array([
        [ cos_yaw, 0., sin_yaw],
        [ 0.,      1., 0.],
        [-sin_yaw, 0., cos_yaw]
    ])
    # Without spatial or angular perturbations, the following would be the camera center
    new_camera_center = camera_focus_point + (old_camera_center - camera_focus_point) @ new_rotation

    # Sample perturbation to orientation
    rotation_perturbation = Rotation.random().as_rotvec() * (angular_mag / (2. * torch.pi))
    rotation_perturbation = Rotation.from_rotvec(rotation_perturbation).as_matrix()
    new_rotation = np.linalg.inv(new_rotation) @ rotation_perturbation @ np.asarray(camera.R)



    # Sample perturbation to camera centre
    cam_centre_perturbation = spatial_mag * (2. * torch.rand(3,) - 1.)
    new_translation = - new_camera_center @ new_rotation + cam_centre_perturbation.numpy()

    new_camera = Camera(camera.colmap_id, new_rotation, new_translation, camera.FoVx, camera.FoVy, camera.original_image, None,
        camera.image_name, camera.uid,
        trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
        )

    return new_camera

def unpack_4x4_transform(transform_mat):
    # Unpack transform into rotation & translation parts; obviously not valid if your matrix is more exotic
    # than an affine transform
    return transform_mat[:3, :3], transform_mat[:3, 3]