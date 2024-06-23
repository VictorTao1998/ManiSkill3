from dataclasses import dataclass, field
from typing import Any, Optional, cast

import cv2
import numpy as np
import sapien
import sapien.render
import torch
from sapien import CudaArray, physx
from sapien.sensor import StereoDepthSensorConfig
from sapien.sensor.simsense_component import SimSenseComponent

from mani_skill.envs.scene import ManiSkillScene
from mani_skill.sensors.base_sensor import BaseSensor
from mani_skill.sensors.camera import Camera, CameraConfig
from mani_skill.utils.sapien_utils import set_shader_dir
from mani_skill.utils.structs.articulation import Articulation


# from blazar.cfg import field
# from ussd.entity import EntityLoader, EntityNode, SceneLike, entity_builder
def take_picture(camera: Camera, depth: bool, take_cpu_picture: bool = False):
    position_seg = None
    if not physx.is_gpu_enabled() or take_cpu_picture:
        scene = camera.camera.scene
        assert isinstance(scene, ManiSkillScene)
        if physx.is_gpu_enabled():
            cast(physx.PhysxGpuSystem, scene.px).sync_poses_gpu_to_cpu()
        images = []
        position_seg = [] if depth else None
        for _, (subscene, cam) in enumerate(
            zip(scene.sub_scenes, camera.camera._render_cameras)
        ):
            subscene.update_render()
            cam.take_picture()
            images.append(cam.get_picture_cuda("Color").torch())
            if position_seg is not None:
                position_seg.append(
                    cam.get_picture_cuda("PositionSegmentation").torch()
                )

        rgb = torch.stack(images, 0)
        position_seg = (
            torch.stack(position_seg, 0) if position_seg is not None else None
        )
    else:
        camera.camera.camera_group.take_picture()
        rgb = (
            camera.camera.camera_group.get_picture_cuda("Color")
            .torch()[..., :3]
            .clone()
        )
        if depth:
            position_seg = camera.camera.camera_group.get_picture_cuda(
                "PositionSegmentation"
            ).torch()
    return rgb, position_seg


@dataclass
class D435StereoDepthCameraConfig(CameraConfig):
    min_depth: float = 0.05
    camera_type: str = "D435"
    width: int = 848
    height: int = 480

    @property
    def rgb_resolution(self):
        return (self.width, self.height)

    stereo_frame: int = 10
    fov: Optional[float] = 0.749

    @property
    def rgb_intrinsic(self):
        assert self.fov is not None
        fy = (self.height / 2) / np.tan(self.fov / 2)
        return np.array([[fy, 0, self.width / 2], [0, fy, self.height / 2], [0, 0, 1]])

    ir_camera_exposure: float = 0.01
    verbose: bool = False  # for debug only

    ir_resolution: tuple[int, int] = (1280, 720)
    trans_pose_l: list[list[float]] = field(
        default_factory=lambda: [
            [
                0.9999489188194275,
                0.009671091102063656,
                0.0029452370945364237,
                0.00015650546993128955,
            ],
            [
                -0.009709948673844337,
                0.999862015247345,
                0.013478035107254982,
                -0.014897654764354229,
            ],
            [
                -0.0028144833631813526,
                -0.013505944050848484,
                0.9999048113822937,
                -1.1531494237715378e-05,
            ],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    trans_pose_r: list[list[float]] = field(
        default_factory=lambda: [
            [
                0.9999489188194275,
                0.009671091102063656,
                0.0029452370945364237,
                -0.0003285693528596312,
            ],
            [
                -0.009709948673844337,
                0.999862015247345,
                0.013478035107254982,
                -0.06504792720079422,
            ],
            [
                -0.0028144833631813526,
                -0.013505944050848484,
                0.9999048113822937,
                0.0006658887723460793,
            ],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    ir_intrinsic: list[list[float]] = field(
        default_factory=lambda: [
            [920.0, 0.0, 640.0],
            [0.0, 920.0, 360.0],
            [0.0, 0.0, 1.0],
        ]
    )

    @classmethod
    def fromCameraConfig(cls, cfg: CameraConfig):
        return cls(**cfg.__dict__)


class D435StereoDepthCamera(BaseSensor):
    def __init__(
        self,
        cfg: D435StereoDepthCameraConfig,
        scene: ManiSkillScene,
        articulation: Articulation = None,
    ):
        print("creating stereo depth camera")
        # super().__init__(camera_cfg, scene, articulation)
        self.config = cfg
        # self.cameras = cameras
        sds_cfg = StereoDepthSensorConfig()
        trans_pose_l = sapien.Pose(np.array(cfg.trans_pose_l))
        trans_pose_r = sapien.Pose(np.array(cfg.trans_pose_r))
        if cfg.intrinsic is None:
            intrinsic = cfg.rgb_intrinsic
        else:
            intrinsic = cfg.intrinsic
        self.ss = SimSenseComponent(
            self.config.rgb_resolution,
            self.config.ir_resolution,
            np.array(intrinsic),
            np.array(intrinsic),
            trans_pose_l,
            trans_pose_r,
            sds_cfg.min_depth,
            sds_cfg.max_depth,
            sds_cfg.ir_noise_seed,
            sds_cfg.ir_speckle_noise,
            sds_cfg.ir_thermal_noise,
            sds_cfg.rectified,
            sds_cfg.census_width,
            sds_cfg.census_height,
            sds_cfg.max_disp,
            sds_cfg.block_width,
            sds_cfg.block_height,
            sds_cfg.p1_penalty,
            sds_cfg.p2_penalty,
            sds_cfg.uniqueness_ratio,
            sds_cfg.lr_max_diff,
            sds_cfg.median_filter_size,
            sds_cfg.depth_dilation,
        )
        # import ipdb;ipdb.set_trace()
        # create the 3 sub cameras
        cam_rgb_cfg = CameraConfig(
            # TODO (stao): why are near/far fixed @Zhiao
            uid=cfg.uid + "/cam_rgb",
            pose=cfg.pose,
            width=cfg.rgb_resolution[0],
            height=cfg.rgb_resolution[1],
            near=0.1,
            far=100.0,
            intrinsic=intrinsic,
            entity_uid=cfg.entity_uid,
            #hide_link=cfg.hide_link,
        )
        cam_rgb = Camera(cam_rgb_cfg, scene, articulation)

        set_shader_dir("default", textured=True)

        sapien.render.set_picture_format("Color", "r32g32b32a32sfloat")
        cam_ir_l_cfg = CameraConfig(
            uid=cfg.uid + "/cam_ir_l",
            pose=cfg.pose * trans_pose_l,
            width=cfg.ir_resolution[0],
            height=cfg.ir_resolution[1],
            near=0.1,
            far=100.0,
            intrinsic=np.array(intrinsic),
            entity_uid=cfg.entity_uid,
            #hide_link=cfg.hide_link,
            texture_names=("Color",),
        )
        cam_ir_l = Camera(cam_ir_l_cfg, scene, articulation)
        cam_ir_r_cfg = CameraConfig(
            uid=cfg.uid + "/cam_ir_r",
            pose=cfg.pose * trans_pose_r,
            width=cfg.ir_resolution[0],
            height=cfg.ir_resolution[1],
            near=0.1,
            far=100.0,
            intrinsic=np.array(intrinsic),
            entity_uid=cfg.entity_uid,
            #hide_link=cfg.hide_link,
            texture_names=("Color",),
        )
        cam_ir_r = Camera(cam_ir_r_cfg, scene, articulation)
        set_shader_dir("default", textured=False)

        sapien.render.set_picture_format("Color", "r8g8b8a8unorm")

        self.alights = [
            self._create_light(cam.entity, sds_cfg)
            for cam in cam_rgb.camera._render_cameras
        ]
        self.cameras = dict(cam_rgb=cam_rgb, cam_ir_l=cam_ir_l, cam_ir_r=cam_ir_r)

        self.scene = scene
        self.px = scene.sub_scenes[0].physx_system
        self.ss.on_add_to_scene(scene.sub_scenes[0])
        self.ratio = torch.tensor(
            np.array(self.config.ir_resolution) / np.array(self.config.rgb_resolution)
        ).cuda()

    def _create_light(self, mount: sapien.Entity, config: StereoDepthSensorConfig):
        # Active Light
        _alight = sapien.render.RenderTexturedLightComponent()
        _alight.color = np.array((100, 0, 0))
        _alight.inner_fov = 1.57
        _alight.outer_fov = 1.57
        _alight.texture = sapien.render.RenderTexture2D(config.light_pattern)
        _alight.local_pose = self.config.pose.sp
        _alight.name = "active_light"
        mount.add_component(_alight)
        return _alight

    def capture(
        self,
        compute_depth: bool = True,
        verbose: bool = False,
        object_ids: Optional[list[tuple[torch.Tensor, int]]] = None,
        take_cpu_picture: bool = False,
    ):
        print("enter")
        """assume already updated the renderer"""
        rgb, position_seg = take_picture(
            self.cameras["cam_rgb"], compute_depth, take_cpu_picture
        )

        if compute_depth:
            depths = []
            assert position_seg is not None

            if object_ids is not None and len(object_ids) > 0:
                seg = position_seg[..., 3]
                obj_seg = torch.zeros(seg.shape, device=seg.device, dtype=torch.bool)
                for _, (ind, _i) in enumerate(object_ids):
                    for k in ind:
                        obj_seg = obj_seg | (seg == k[:, None, None])
            else:
                obj_seg = None

            def compute_depth(left_cuda: CudaArray, right_cuda: CudaArray, idx: int):
                bbox_left, size = None, None
                if obj_seg is not None:
                    p = obj_seg[idx]

                    y, x = torch.stack(torch.where(p))
                    if len(y) == 0:
                        bbox_left = (0, 0)
                        size = (2, 2)
                    else:
                        xy = torch.stack([x, y])

                        min_xy = xy.min(1).values * self.ratio
                        max_xy = xy.max(1).values * self.ratio

                        frame = self.config.stereo_frame

                        bbox_left = (
                            max(int(min_xy[0]) - frame, 0),
                            max(int(min_xy[1]) - frame, 0),
                        )
                        res = self.config.ir_resolution
                        bbox_right = (
                            min(int(max_xy[0]) + frame, res[0] - 1),
                            min(int(max_xy[1]) + frame, res[1] - 1),
                        )
                        size = (
                            bbox_right[0] - bbox_left[0] + 1,
                            bbox_right[1] - bbox_left[1] + 1,
                        )

                self.ss.compute(left_cuda, right_cuda, bbox_left, size)  # type: ignore
                depth = self.ss.get_cuda().torch().clone()
                return depth
            print("here")
            if not physx.is_gpu_enabled() or take_cpu_picture:
                cam_ir_l = self.cameras["cam_ir_l"].camera._render_cameras
                cam_ir_r = self.cameras["cam_ir_r"].camera._render_cameras

                if physx.is_gpu_enabled():
                    cast(physx.PhysxGpuSystem, self.px).sync_poses_gpu_to_cpu()
                for idx, (scene, cam_l, cam_r) in enumerate(
                    zip(self.scene.sub_scenes, cam_ir_l, cam_ir_r)
                ):
                    scene.update_render()
                    cam_l.take_picture()
                    cam_r.take_picture()
                    left_cuda = cam_l.get_picture_cuda("Color")
                    right_cuda = cam_r.get_picture_cuda("Color")
                    print(left_cuda, type(left_cuda))
                    #assert 0
                    depths.append(compute_depth(left_cuda, right_cuda, idx))
            else:
                left_group = self.cameras["cam_ir_l"].camera.camera_group
                left_group.take_picture()

                right_group = self.cameras["cam_ir_r"].camera.camera_group
                right_group.take_picture()

                rgb_left = left_group.get_picture_cuda("Color").torch()
                rgb_right = right_group.get_picture_cuda("Color").torch()

                class Fake:
                    __cuda_array_interface__: dict[str, Any]

                def to_cuda_array(x: torch.Tensor):
                    y = Fake()
                    y.__cuda_array_interface__ = dict(x.__cuda_array_interface__)
                    y.__cuda_array_interface__["typestr"] = "f4"
                    return CudaArray(y)

                for i in range(len(self.scene.sub_scenes)):
                    left = to_cuda_array(rgb_left[i])
                    right = to_cuda_array(rgb_right[i])
                    depths.append(compute_depth(left, right, i))

            depths = (torch.stack(depths, 0) * 1000).to(dtype=torch.int16)

            if verbose or self.config.verbose:
                from blazar.utils.utils import tile_images

                rr = tile_images(list(rgb[..., :3].cpu().numpy()))
                rr = cv2.resize(rr, (0, 0), fx=0.25, fy=0.25)
                cv2.imshow("rgb", rr[..., [2, 1, 0]])

                d = tile_images(
                    [i / 1000.0 for i in depths.detach().cpu().numpy()]
                ).clip(0, 1)
                f = tile_images(
                    [i / 1000.0 for i in -position_seg[..., 2].detach().cpu().numpy()]
                ).clip(0, 1)

                all = cv2.resize(np.concatenate([d, f], 0), (0, 0), fx=0.25, fy=0.25)
                cv2.imshow("depth", all)
                cv2.waitKey(1)

            position_seg[..., 2] = -depths
        #

        # import ipdb;ipdb.set_trace()
        # plt.imshow(rgb.cpu().numpy()[0, ..., :3]); plt.show()
        # plt.imshow(depths.cpu().numpy()[0]); plt.show()
        self.last_rgb, self.last_position_seg = rgb, position_seg

    def get_obs(self):
        return dict(Color=self.last_rgb, PositionSegmentation=self.last_position_seg)

    def get_params(self):
        """Get camera parameters."""
        return self.cameras["cam_rgb"].get_params()
