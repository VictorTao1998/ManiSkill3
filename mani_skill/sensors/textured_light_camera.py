from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Sequence, Union

import numpy as np
import sapien
import sapien.render
import torchvision
from torch._tensor import Tensor
import sapien.physx as physx
from typing import cast
from mani_skill.utils.structs import Actor, Articulation, Link
from mani_skill.utils.structs.pose import Pose
import torch
from mani_skill.utils.structs.types import Array

if TYPE_CHECKING:
    from mani_skill.envs.scene import ManiSkillScene

from mani_skill.utils import sapien_utils, visualization

from .base_sensor import BaseSensor, BaseSensorConfig
from mani_skill.sensors.camera import Camera, CameraConfig

@dataclass
class TexturedLightCameraConfig(CameraConfig):
    ir_pose: Pose = sapien.Pose(p=[0, -0.02, 0], q=[1, 0, 0, 0]),
    light_pattern: str = os.path.join(os.path.dirname(__file__), "assets/patterns/d415.png")



def update_camera_cfgs_from_dict(
    camera_cfgs: Dict[str, CameraConfig], cfg_dict: Dict[str, dict]
):
    # Update CameraConfig to StereoDepthCameraConfig
    if cfg_dict.pop("use_stereo_depth", False):
        from .depth_camera import StereoDepthCameraConfig  # fmt: skip
        for name, cfg in camera_cfgs.items():
            camera_cfgs[name] = StereoDepthCameraConfig.fromCameraConfig(cfg)

    # First, apply global configuration
    for k, v in cfg_dict.items():
        if k in camera_cfgs:
            continue
        for cfg in camera_cfgs.values():
            if not hasattr(cfg, k):
                raise AttributeError(f"{k} is not a valid attribute of CameraConfig")
            else:
                setattr(cfg, k, v)
    # Then, apply camera-specific configuration
    for name, v in cfg_dict.items():
        if name not in camera_cfgs:
            continue

        # Update CameraConfig to StereoDepthCameraConfig
        if v.pop("use_stereo_depth", False):
            from .depth_camera import StereoDepthCameraConfig  # fmt: skip
            cfg = camera_cfgs[name]
            camera_cfgs[name] = StereoDepthCameraConfig.fromCameraConfig(cfg)

        cfg = camera_cfgs[name]
        for kk in v:
            assert hasattr(cfg, kk), f"{kk} is not a valid attribute of CameraConfig"
        cfg.__dict__.update(v)


def parse_camera_cfgs(camera_cfgs):
    if isinstance(camera_cfgs, (tuple, list)):
        return dict([(cfg.uid, cfg) for cfg in camera_cfgs])
    elif isinstance(camera_cfgs, dict):
        return dict(camera_cfgs)
    elif isinstance(camera_cfgs, CameraConfig):
        return dict([(camera_cfgs.uid, camera_cfgs)])
    else:
        raise TypeError(type(camera_cfgs))


class TexturedLightCamera(Camera):
    """Implementation of the Camera sensor which uses the sapien Camera."""

    cfg: TexturedLightCameraConfig

    def __init__(
        self,
        camera_cfg: TexturedLightCameraConfig,
        scene: ManiSkillScene,
        articulation: Articulation = None,
    ):
        super().__init__(camera_cfg=camera_cfg, scene=scene, articulation=articulation)

        print(len(self.camera._render_cameras))
        for i, cam in enumerate(self.camera._render_cameras):

            _alight = self._create_light(cam.entity, self.camera_cfg)
            #self.entity._objs[i].entity.add_component(_alight)


    def _create_light(self, mount: sapien.Entity, config: TexturedLightCameraConfig):
        # Active Light
        _alight = sapien.render.RenderTexturedLightComponent()
        _alight.color = np.array((255, 255, 255))
        _alight.inner_fov = 1.57
        _alight.outer_fov = 2.3
        _alight.texture = sapien.render.RenderTexture2D(config.light_pattern)
        _alight.local_pose = self.camera_cfg.ir_pose
        _alight.name = "active_light"
        mount.add_component(_alight)
        return _alight

    def get_obs(self):
        images = {}
        for name in self.texture_names:
            image = self.get_picture(name)
            if name == "Color":
                # image_grid = image.permute(0, 3, 1, 2)[:,:3,:,:]
                # image_grid = torchvision.utils.make_grid(image_grid)
                # image_grid_pil = torchvision.transforms.ToPILImage()(image_grid)
                # save_dir = "/home/jianyu/jianyu/pythonproject/ManiSkill3_Main/examples/baselines/ppo/test_results"
                # idx = len(os.listdir(save_dir))
                # image_grid_pil.save(os.path.join(save_dir, f"{idx}.png"))
                pass
            images[name] = image
        return images
    
    def get_picture(self, name: str):
        if physx.is_gpu_enabled():
            cast(physx.PhysxGpuSystem, self.scene.px).sync_poses_gpu_to_cpu()
            images = []
            for _, (subscene, cam) in enumerate(zip(self.scene.sub_scenes, self.camera._render_cameras)):
                subscene.update_render()
                cam.take_picture()
                c_img = cam.get_picture_cuda(name).torch()
                images.append(c_img[None,...])
            images = torch.cat(images, dim=0)
            return images
        else:
            return self.camera.get_picture(name)
