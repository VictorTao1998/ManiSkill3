import numpy as np
import sapien

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.registration import register_agent
from mani_skill.sensors.camera import CameraConfig
from mani_skill.sensors.stereodepth_camera import D435StereoDepthCameraConfig
from mani_skill.sensors.textured_light_camera import TexturedLightCameraConfig
from mani_skill.utils import sapien_utils

from .panda import Panda


@register_agent()
class PandaWristCam(Panda):
    """Panda arm robot with the real sense camera attached to gripper"""

    uid = "panda_wristcam"
    urdf_path = f"{PACKAGE_ASSET_DIR}/robots/panda/panda_v3.urdf"

    @property
    def _sensor_configs(self):
        return [
            CameraConfig(
                uid="hand_camera",
                pose=sapien.Pose(p=[0, 0, 0], q=[1, 0, 0, 0]),
                width=128,
                height=128,
                fov=np.pi / 2,
                near=0.01,
                far=100,
                mount=self.robot.links_map["camera_link"],
            )
        ]

@register_agent()
class PandaWristStereoCam(Panda):
    """Panda arm robot with the real sense camera attached to gripper"""

    uid = "panda_wriststereocam"
    urdf_path = f"{PACKAGE_ASSET_DIR}/robots/panda/panda_v3.urdf"

    @property
    def _sensor_configs(self):
        return [
            CameraConfig(
                uid="hand_camera",
                pose=sapien.Pose(p=[0, 0, 0], q=[1, 0, 0, 0]),
                width=128,
                height=128,
                fov=np.pi / 2,
                near=0.01,
                far=100,
                mount=self.robot.links_map["camera_link"],
            ),
            CameraConfig(
                uid="hand_camera_right",
                pose=sapien.Pose(p=[0, -0.03, 0], q=[1, 0, 0, 0]),
                width=128,
                height=128,
                fov=np.pi / 2,
                near=0.01,
                far=100,
                mount=self.robot.links_map["camera_link"],
            )
        ]


@register_agent()
class PandaWristActiveStereoCam(Panda):
    """Panda arm robot with the real sense camera attached to gripper"""

    uid = "panda_wristactivestereocam"
    urdf_path = f"{PACKAGE_ASSET_DIR}/robots/panda/panda_v3.urdf"

    @property
    def _sensor_configs(self):
        return [
            TexturedLightCameraConfig(
                uid="hand_camera",
                pose=sapien.Pose(p=[0, 0, 0], q=[1, 0, 0, 0]),
                ir_pose=sapien.Pose(p=[0, -0.015, 0], q=[1, 0, 0, 0]),
                width=768,
                height=384,
                fov=np.pi / 2,
                # intrinsic=[[357.60738000000003, 0., 384.],
                #            [0.,317.87322666666677,192.],
                #            [0.,0.,1.]],
                near=0.01,
                far=100,
                mount=self.robot.links_map["camera_link"],
            ),
            CameraConfig(
                uid="hand_camera_right",
                pose=sapien.Pose(p=[0, -0.03, 0], q=[1, 0, 0, 0]),
                width=768,
                height=384,
                fov=np.pi / 2,
                # intrinsic=[[357.60738000000003, 0., 384.],
                #            [0.,317.87322666666677,192.],
                #            [0.,0.,1.]],
                near=0.01,
                far=100,
                mount=self.robot.links_map["camera_link"],
            )
        ]
