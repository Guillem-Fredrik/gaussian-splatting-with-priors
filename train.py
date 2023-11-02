#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
import torchvision
import random
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from utils.graphics_utils import fov2focal
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from pathlib import Path
import numpy as np
from learned_regularisation.patch_pose_generator import PatchPoseGenerator, FrustumChecker, FrustumRegulariser, LenticularRegulariser
from learned_regularisation.patch_regulariser import load_patch_diffusion_model, \
    PatchRegulariser, LLFF_DEFAULT_PSEUDO_INTRINSICS
from learned_regularisation.utils import Intrinsics
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    if opt.patch_regulariser_path:
        device = torch.device("cuda")

        camera = scene.getTrainCameras()[0]
        # pseudo_intrinsics = LLFF_DEFAULT_PSEUDO_INTRINSICS
        W = camera.image_width
        H = camera.image_height
        intrinsics = Intrinsics(
            fov2focal(camera.FoVx, W),
            fov2focal(camera.FoVy, H),
            W / 2,
            H / 2,
            W,
            H
        )

        frustum_checker = FrustumChecker(fov_x_rads=camera.FoVx, fov_y_rads=camera.FoVy)
        frustum_regulariser = FrustumRegulariser(
            intrinsics=intrinsics,
            reg_strength=1e-5,  # NB in the trainer this gets multiplied by the strength params passed in via the args
            min_near=opt.min_near,
            cameras=scene.getTrainCameras(),
        )
        lenticular_regulariser = LenticularRegulariser(
            reg_strength=1e-3,  # NB in the trainer this gets multiplied by the strength params passed in via the args TODO(guillem)
            cameras=scene.getTrainCameras(),
        )

        patch_diffusion_model = load_patch_diffusion_model(Path(opt.patch_regulariser_path))
        pose_generator = PatchPoseGenerator(cameras=scene.getTrainCameras(),
                                            spatial_perturbation_magnitude=0.2,
                                            angular_perturbation_magnitude_rads=0.2 * np.pi,
                                            no_perturb_prob=0.,
                                            frustum_checker=frustum_checker if opt.frustum_check_patches else None
                        )
        
        print('Using patch intrinsics', intrinsics)
        patch_regulariser = PatchRegulariser(pose_generator=pose_generator,
                                                patch_diffusion_model=patch_diffusion_model,
                                                full_image_intrinsics=intrinsics,
                                                device=device,
                                                planar_depths=True,
                                                frustum_regulariser=None,
                                                # frustum_regulariser=frustum_regulariser if opt.frustum_regularise_patches else None,
                                                lenticular_regulariser=None,
                                                # lenticular_regulariser=lenticular_regulariser,
                                                sample_downscale_factor=opt.patch_sample_downscale_factor,
                                                uniform_in_depth_space=opt.normalise_diffusion_losses,
                                                background=background,
                                                pipe=pipe)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        # if iteration % 10 == 0:
            # render_depth = render(viewpoint_cam, gaussians, pipe, background, render_depth=True)
            # xn = random.random()
            # torchvision.utils.save_image(render_depth["render"], 'debug/depth-{}.png'.format(xn))

            # print("xyz", gaussians.get_xyz.shape, gaussians.get_xyz)
            # print("_features_dc", gaussians._features_dc.shape, gaussians._features_dc)
            # print("_features_rest", gaussians._features_rest.shape, gaussians._features_rest)
            # print("features", gaussians.get_features.shape, gaussians.get_features)
            # print("scaling", gaussians.get_scaling.shape, gaussians.get_scaling)
            # print("rotation", gaussians.get_rotation.shape, gaussians.get_rotation)
            # print("opacity", gaussians.get_opacity.shape, gaussians.get_opacity)
            # print("covariance", gaussians.get_covariance().shape, gaussians.get_covariance())

        # Regularisation
        if opt.patch_regulariser_path:
            # t schedule
            initial_diffusion_time = opt.initial_diffusion_time
            patch_reg_start_step = opt.patch_reg_start_step
            patch_reg_finish_step = opt.patch_reg_finish_step
            weight_start = opt.patch_weight_start
            weight_finish = opt.patch_weight_finish

            lambda_t = (iteration - patch_reg_start_step) / (patch_reg_finish_step - patch_reg_start_step)
            lambda_t = np.clip(lambda_t, 0., 1.)
            weight = weight_start + (weight_finish - weight_start) * lambda_t

            if iteration > patch_reg_start_step:
                if iteration > patch_reg_finish_step:
                    time = 0.
                elif iteration > patch_reg_start_step:
                    time = initial_diffusion_time * (1. - lambda_t)
                else:
                    raise RuntimeError('Internal error')
                p_sample_patch = 0.25
                if random.random() >= p_sample_patch:
                    patch_outputs = patch_regulariser.get_diffusion_loss_with_rendered_patch(gaussians=gaussians,
                                                                                                  time=time)
                else:
                    patch_outputs = patch_regulariser.get_diffusion_loss_with_sampled_patch(
                        gaussians=gaussians, time=time, image=gt_image, image_intrinsics=intrinsics,
                        camera=viewpoint_cam
                    )
                loss += weight * patch_outputs.loss

                # # Geometric reg
                # if opt.apply_geom_reg_to_patches:
                #     loss += spread_loss_weight * patch_outputs.render_outputs['loss_dist']

                # Frustum reg        
                if patch_regulariser.frustum_regulariser is not None:
                    frustum_reg_weight = opt.frustum_reg_initial_weight if iteration < 100 else opt.frustum_reg_final_weight

                    xyzs_flat = patch_outputs.render_outputs['xyzs'].reshape(-1, 3)
                    weights_flat = patch_outputs.render_outputs['weights'].reshape(-1)

                    patch_frustum_reg_weight = frustum_reg_weight
                    patch_frustum_loss = patch_frustum_reg_weight * patch_regulariser.frustum_regulariser(
                        xyzs=xyzs_flat, weights=weights_flat, frustum_count_thresh=1,
                    )
                    print('Patch frustum loss', patch_frustum_loss)
                    loss += patch_frustum_loss

                # Lenticular reg        
                if patch_regulariser.lenticular_regulariser is not None:
                    lenticular_reg_weight = 1 # opt.frustum_reg_initial_weight if iteration < 100 else opt.frustum_reg_final_weight TODO(guillem)
                    # sampled_indices = np.random.choice(gaussians.get_xyz.shape[0], 200, replace=False)
                    sampled_indices = np.arange(gaussians.get_xyz.shape[0])
                    patch_lenticular_loss = lenticular_reg_weight * patch_regulariser.lenticular_regulariser(
                        xyzs=gaussians.get_xyz[sampled_indices].reshape(-1, 3), scales=gaussians.get_scaling[sampled_indices].reshape(-1, 3), weights=gaussians.get_opacity[sampled_indices].reshape(-1), covariances=gaussians.get_covariance()[sampled_indices].reshape(-1, 6),
                    )
                    loss += patch_lenticular_loss

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
