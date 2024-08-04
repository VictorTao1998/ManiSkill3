"""
Functions that map a observation to a particular format, e.g. mapping the raw images to rgbd or pointcloud formats
"""

import os
from typing import Dict

import numpy as np
import sapien.physx as physx
import torch
import cv2
from mani_skill.sensors.base_sensor import BaseSensor, BaseSensorConfig
from mani_skill.sensors.camera import Camera
from mani_skill.utils import common
import mani_skill.global_params as global_params
import imageio
import open3d as o3d

def sensor_data_to_rgbd(
    observation: Dict,
    sensors: Dict[str, BaseSensor],
    rgb=True,
    depth=True,
    segmentation=True,
):
    """
    Converts all camera data to a easily usable rgb+depth format

    Optionally can include segmentation
    """
    sensor_data = observation["sensor_data"]
    for (cam_uid, ori_images), (sensor_uid, sensor) in zip(
        sensor_data.items(), sensors.items()
    ):
        assert cam_uid == sensor_uid
        if isinstance(sensor, Camera):
            new_images = dict()
            ori_images: Dict[str, torch.Tensor]
            for key in ori_images:
                if key == "Color":
                    if rgb:
                        rgb_data = ori_images[key][..., :3].clone()  # [H, W, 4]
                        new_images["rgb"] = rgb_data  # [H, W, 4]
                elif key == "PositionSegmentation":
                    if depth:
                        depth_data = -ori_images[key][..., [2]]  # [H, W, 1]
                        # NOTE (stao): This is a bit of a hack since normally we have generic to_numpy call to convert
                        # internal torch tensors to numpy if we do not use GPU simulation
                        # but torch does not have a uint16 type so we convert that here earlier
                        # if not physx.is_gpu_enabled():
                        #     depth_data = depth_data.numpy().astype(np.uint16)
                        new_images["depth"] = depth_data
                    if segmentation:
                        segmentation_data = ori_images[key][..., [3]]
                        # if not physx.is_gpu_enabled():
                        #     segmentation_data = segmentation_data.numpy().astype(
                        #         np.uint16
                        #     )
                        new_images["segmentation"] = segmentation_data  # [H, W, 1]
                else:
                    new_images[key] = ori_images[key]
            sensor_data[cam_uid] = new_images
    return observation

def stereo_sensor_data_to_rgbd(
    observation: Dict,
    sensors: Dict[str, BaseSensor],
    rgb=True,
    depth=True,
    segmentation=True,
):
    """
    Converts all camera data to a easily usable rgb+depth format

    Optionally can include segmentation
    """
    sensor_data = observation["sensor_data"]
    for (cam_uid, ori_images), (sensor_uid, sensor) in zip(
        sensor_data.items(), sensors.items()
    ):
        assert cam_uid == sensor_uid
        if cam_uid == "hand_camera_right":
            rgb_data_right = ori_images["Color"][..., :3].clone().detach()
        else:
            if isinstance(sensor, Camera):
                new_images = dict()
                ori_images: Dict[str, torch.Tensor]
                for key in ori_images:
                    if key == "Color":
                        if rgb:
                            rgb_data = ori_images[key][..., :3].clone()  # [H, W, 4]
                            new_images["rgb"] = rgb_data  # [H, W, 4]
                        if cam_uid == "hand_camera":
                            rgb_data_left = ori_images[key][..., :3].clone().detach()
                            
                    elif key == "PositionSegmentation":
                        if depth:
                            depth_data = -ori_images[key][..., [2]]  # [H, W, 1] # negative to pos
                            # print(torch.max(ori_images[key][..., [2]][0,:,:,0]))
                            # assert 0
                            # NOTE (stao): This is a bit of a hack since normally we have generic to_numpy call to convert
                            # internal torch tensors to numpy if we do not use GPU simulation
                            # but torch does not have a uint16 type so we convert that here earlier
                            # if not physx.is_gpu_enabled():
                            #     depth_data = depth_data.numpy().astype(np.uint16)
                            new_images["depth"] = depth_data
                        if segmentation:
                            segmentation_data = ori_images[key][..., [3]]
                            # if not physx.is_gpu_enabled():
                            #     segmentation_data = segmentation_data.numpy().astype(
                            #         np.uint16
                            #     )
                            new_images["segmentation"] = segmentation_data  # [H, W, 1]
                    else:
                        new_images[key] = ori_images[key]
                sensor_data[cam_uid] = new_images
    
    # print(rgb_data_right.shape, rgb_data_left.shape)
    # print(torch.max(rgb_data_right[0,:,:,:]), torch.min(rgb_data_right[0,:,:,:]))

    if "hand_camera_right" in sensor_data.keys():
        del sensor_data["hand_camera_right"]

    # output = global_params.TRT_ENGINE.process(rgb_data_left, rgb_data_right)
    # out_path = "/home/jianyu/jianyu/pythonproject/ManiSkill3_Main/examples/baselines/ppo/runs/train_imgs"
    # for cuid in sensor_data:
    #     if "depth" in sensor_data[cuid].keys():
            #stereo_disparity = torch.Tensor(output).to(sensor_data[cuid]["depth"].device)[...,None]
            #stereo_depth_raw = (0.03 * 357.60738000000003) / stereo_disparity
            #stereo_depth = (stereo_depth_raw*1000).type(torch.int16)
            #stereo_depth_cv = stereo_depth.cpu().numpy()[0,:,:,0]
            #stereo_depth_cv = stereo_depth_cv.astype(float)/np.max(stereo_depth_cv)*255
            #stereo_depth_cv = stereo_depth_cv.astype(np.uint8)
            #stereo_disparity_cv = (stereo_disparity[0,:,:,0]/192.*255).type(torch.uint8)
            #stereo_disparity_cv = stereo_disparity_cv.cpu().numpy()
            # print(stereo_depth.shape)
            # tid = len(os.listdir(out_path))
            # os.makedirs(os.path.join(out_path, str(tid)))
            #imageio.imsave(os.path.join(out_path, str(tid), 'depth.png'), stereo_depth_cv)
            #imageio.imsave(os.path.join(out_path, str(tid), 'disparity.png'), stereo_disparity_cv)
            # imageio.imsave(os.path.join(out_path, str(tid), 'left.png'), rgb_data_left.cpu().numpy()[0,...])
            # imageio.imsave(os.path.join(out_path, str(tid), 'right.png'), rgb_data_right.cpu().numpy()[0,...])
            #pcd_pred = o3d.geometry.PointCloud()
            #intrinsic_l = [[357.60738000000003, 0., 384.],
            #               [0.,317.87322666666677,192.],
            #               [0.,0.,1.]]
            #pcd_pred.points = o3d.utility.Vector3dVector(depth2pts_np(stereo_depth_raw[0,:,:,0], intrinsic_l))
            #o3d.io.write_point_cloud(os.path.join(out_path, str(tid), 'depth.ply'), pcd_pred)

            #sensor_data[cuid]["depth"] = stereo_depth
            #print(stereo_depth[0,:50,:50,0])
            #print(sensor_data[cuid]["depth"].shape, sensor_data[cuid]["depth"].device, sensor_data[cuid]["depth"].dtype, sensor_data[cuid]["rgb"].dtype, sensor_data[cuid]["depth"][0,:50,:50,0])
            #assert 0
            
    return observation

def get_pixel_grids_np(height, width):
    x_linspace = np.linspace(0.5, width - 0.5, width)
    y_linspace = np.linspace(0.5, height - 0.5, height)
    x_coordinates, y_coordinates = np.meshgrid(x_linspace, y_linspace)
    x_coordinates = np.reshape(x_coordinates, (1, -1))
    y_coordinates = np.reshape(y_coordinates, (1, -1))
    ones = np.ones_like(x_coordinates).astype(float)
    grid = np.concatenate([x_coordinates, y_coordinates, ones], axis=0)

    return grid

def depth2pts_np(depth_map, cam_intrinsic, cam_extrinsic=np.eye(4)):
    feature_grid = get_pixel_grids_np(depth_map.shape[0], depth_map.shape[1])

    uv = np.matmul(np.linalg.inv(cam_intrinsic), feature_grid)
    depth_map = depth_map.cpu().detach().numpy()
    cam_points = uv * np.reshape(depth_map, (1, -1))

    R = cam_extrinsic[:3, :3]
    t = cam_extrinsic[:3, 3:4]
    R_inv = np.linalg.inv(R)

    world_points = np.matmul(R_inv, cam_points - t).transpose()
    return world_points



def sensor_data_to_pointcloud(observation: Dict, sensors: Dict[str, BaseSensor]):
    """convert all camera data in sensor to pointcloud data"""
    sensor_data = observation["sensor_data"]
    camera_params = observation["sensor_param"]
    pointcloud_obs = dict()

    for (cam_uid, images), (sensor_uid, sensor) in zip(
        sensor_data.items(), sensors.items()
    ):
        assert cam_uid == sensor_uid
        if isinstance(sensor, Camera):
            cam_pcd = {}

            # Each pixel is (x, y, z, actor_id) in OpenGL camera space
            # actor_id = 0 for the background
            images: Dict[str, torch.Tensor]
            position = images["PositionSegmentation"]
            segmentation = position[..., 3].clone()
            position = position.float()
            position[..., 3] = position[..., 3] != 0
            position[..., :3] = (
                position[..., :3] / 1000.0
            )  # convert the raw depth from millimeters to meters

            # Convert to world space
            cam2world = camera_params[cam_uid]["cam2world_gl"]
            xyzw = position.reshape(position.shape[0], -1, 4) @ cam2world.transpose(
                1, 2
            )
            cam_pcd["xyzw"] = xyzw

            # Extra keys
            if "Color" in images:
                rgb = images["Color"][..., :3].clone()
                cam_pcd["rgb"] = rgb.reshape(rgb.shape[0], -1, 3)
            if "PositionSegmentation" in images:
                cam_pcd["segmentation"] = segmentation.reshape(
                    segmentation.shape[0], -1, 1
                )

            pointcloud_obs[cam_uid] = cam_pcd
    for k in pointcloud_obs.keys():
        del observation["sensor_data"][k]
    pointcloud_obs = common.merge_dicts(pointcloud_obs.values())
    for key, value in pointcloud_obs.items():
        pointcloud_obs[key] = torch.concat(value, axis=1)
    observation["pointcloud"] = pointcloud_obs

    # if not physx.is_gpu_enabled():
    #     observation["pointcloud"]["segmentation"] = (
    #         observation["pointcloud"]["segmentation"].numpy().astype(np.uint16)
    #     )
    return observation
