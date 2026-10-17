"""Race 继承作者 Go2；平地按用户要求增加 SLR 姿态奖励和速度课程。"""

from copy import deepcopy
import json
from math import isfinite
from pathlib import Path
import xml.etree.ElementTree as ET

from isaaclab.managers import (
    EventTermCfg as EventTerm, RewardTermCfg as RewTerm, SceneEntityCfg,
)
from isaaclab.utils import configclass

from robot_lab.assets import ISAACLAB_ASSETS_DATA_DIR
from robot_lab.assets.race import (
    RACE_CFG, JOINT_NAMES, RaceModelUrdfFileCfg,
)
from robot_lab.assets.unitree import GO2_CFG_UNITREE
from robot_lab.tasks.go2.env_cfg import (
    Go2EnvCfg, JOINT_NAMES as GO2_JOINT_NAMES, RewardsCfg,
)
from .mdp import SLRPoseReward
from .commands import RaceFlatOmniCommandCfg
from .events import RaceAdditiveJointReset

# 用户指定：初始释放高度与奖励目标分别设置。
BASE_RELEASE_HEIGHT = 0.36
BASE_RESET_Z_RANGE = (-0.01, 0.01)
BASE_HEIGHT_TARGET = 0.38
FLAT_BASE_HEIGHT_TARGET = 0.36
STAND_ANGLES = (0.0, -0.12, 0.30)
JOINT_KP = (30.0, 48.0, 51.0)
JOINT_KD = (1.15, 1.20, 1.00)
TORQUE_LIMITS = (22.0, 22.0, 33.0)

# 仅供当前复杂地形实验，平地仍使用独立的 SLR 姿态和原重置分布。
ROUGH_STAND_ANGLES = {
    f"{kind}_leg_{leg}_joint": value
    for leg, angles in (
        ("f_l", (0.10, 0.00, 0.10)),
        ("f_r", (-0.10, 0.00, 0.10)),
        ("b_l", (0.10, 0.20, 0.10)),
        ("b_r", (-0.10, 0.20, 0.10)),
    )
    for kind, value in zip(("roll", "pitch01", "pitch02"), angles, strict=True)
}
ROUGH_RESET_HALF_WIDTHS = {
    name: (0.0375 if name.startswith("roll_") else
           0.5625 if name.startswith("pitch02_") else
           0.3000 if "_leg_f_" in name else 0.3750)
    for name in JOINT_NAMES
}

# 快照来自 SLR-master/resources/race/poses/go2_link_direction.json。
# 训练只依赖本仓库，不要求服务器另有一份 SLR 工程。
SLR_STAND_ANGLES = json.loads(
    (Path(ISAACLAB_ASSETS_DATA_DIR)
     / "race/poses/slr_go2_link_direction.json").read_text(encoding="utf-8")
)
if (set(SLR_STAND_ANGLES) != set(JOINT_NAMES)
        or not all(isfinite(v) for v in SLR_STAND_ANGLES.values())):
    raise ValueError("SLR 站姿必须包含 Race 全部 12 个关节的有限角度")


def joint_parameter(values):
    """按关节名分配外展、髋俯仰、膝关节参数，不依赖导入后的排列。"""
    by_kind = dict(zip(("roll", "pitch01", "pitch02"), values, strict=True))
    return {name: by_kind[name.split("_leg_", 1)[0]] for name in JOINT_NAMES}


def urdf_velocity_limits(asset_path):
    """从当前 Race URDF 逐关节读取速度上限，缺失时立即报错。"""
    joints = {joint.attrib["name"]: joint
              for joint in ET.parse(asset_path).getroot().findall("joint")}
    limits = {}
    for name in JOINT_NAMES:
        limit = joints[name].find("limit")
        if limit is None or "velocity" not in limit.attrib:
            raise ValueError(f"Race URDF 关节 {name} 缺少速度上限")
        velocity = float(limit.attrib["velocity"])
        if not isfinite(velocity) or velocity <= 0:
            raise ValueError(f"Race URDF 关节 {name} 速度上限无效")
        limits[name] = velocity
    return limits


FOOT_NAMES = [f"pitch03_leg_{leg}_link" for leg in
              ("f_l", "f_r", "b_l", "b_r")]
NAME_MAP = dict(zip(GO2_JOINT_NAMES, JOINT_NAMES, strict=True))
NAME_MAP.update({
    "base": "base_link",
    ".*_foot": "pitch03_leg_[fb]_[lr]_link",
    ".*_thigh|.*_calf": "pitch0[12]_leg_[fb]_[lr]_link",
    ".*_hip_joint": "roll_leg_.*_joint",
    ".*_(thigh|calf)_joint": "pitch0[12]_leg_.*_joint",
    "^(?!.*base).*": "^(?!base_link$).*",
    "^(?!.*_hip_joint).*": "^(?!roll_leg_).*",
})

# 这些字段描述 Race 模型的导入方式；其余共有生成参数沿用原 Go2。
MODEL_SPAWN_FIELDS = {
    "asset_path", "usd_dir", "usd_file_name", "func", "root_link_name",
    "merge_fixed_joints", "replace_cylinders_with_capsules",
    "filter_base_hip_housings",
}


def adapt_entities(value):
    """只改实体选择器，不改奖励、随机化及课程的数值参数。"""
    if isinstance(value, SceneEntityCfg):
        for field in ("joint_names", "body_names"):
            names = getattr(value, field)
            if isinstance(names, str):
                setattr(value, field, NAME_MAP.get(names, names))
            elif isinstance(names, (tuple, list)):
                setattr(value, field, [NAME_MAP.get(n, n) for n in names])
    elif isinstance(value, dict):
        for child in value.values():
            adapt_entities(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            adapt_entities(child)
    elif hasattr(value, "__dict__") and not callable(value):
        for child in vars(value).values():
            adapt_entities(child)


@configclass
class RaceEnvCfg(Go2EnvCfg):
    """原 Go2 训练，适配 Race 模型、站姿、高度及用户指定电机参数。"""

    def __post_init__(self):
        super().__post_init__()
        robot = deepcopy(GO2_CFG_UNITREE)
        robot.prim_path = self.scene.robot.prim_path
        robot.spawn = RaceModelUrdfFileCfg()
        for name, value in vars(GO2_CFG_UNITREE.spawn).items():
            if name not in MODEL_SPAWN_FIELDS:
                setattr(robot.spawn, name, deepcopy(value))
        # 用户指定：主机身单盒按Go2比例收紧，保留原URDF供历史部署使用。
        robot.spawn.asset_path = str(Path(RACE_CFG.spawn.asset_path).with_name(
            "race_go2_single_box.urdf"
        ))
        robot.spawn.root_link_name = "base_link"
        robot.spawn.merge_fixed_joints = False
        robot.spawn.replace_cylinders_with_capsules = False
        # 只排除机身与四个髋壳，匹配 Go2 相邻髋部的过滤范围。
        robot.spawn.filter_base_hip_housings = True
        # 独立导入缓存，避免覆盖历史 Race 模型的缓存文件。
        robot.spawn.usd_dir = str(
            Path(RACE_CFG.spawn.usd_dir).with_name("usd_go2_training_single_box")
        )
        robot.spawn.usd_file_name = "race_go2.usd"
        # 原对照配置使用统一站姿；当前复杂地形在子类单独覆盖。
        robot.init_state.joint_pos = joint_parameter(STAND_ANGLES)
        robot.init_state.pos = (*robot.init_state.pos[:2], BASE_RELEASE_HEIGHT)
        actuator = robot.actuators["GO2HV"]
        actuator.stiffness = joint_parameter(JOINT_KP)
        actuator.damping = joint_parameter(JOINT_KD)
        # 保留原曲线算法；同向、反向都采用 Race 电机力矩上限。
        for field in ("Y1", "Y2", "effort_limit", "effort_limit_sim"):
            setattr(actuator, field, joint_parameter(TORQUE_LIMITS))
        # URDF 未提供满力矩拐点，保留原曲线 45% 的相对位置作为假设。
        knee_ratio = actuator.X1 / actuator.X2
        velocities = urdf_velocity_limits(robot.spawn.asset_path)
        actuator.X1 = {name: speed * knee_ratio
                       for name, speed in velocities.items()}
        for field in ("X2", "velocity_limit", "velocity_limit_sim"):
            setattr(actuator, field, deepcopy(velocities))
        robot.joint_sdk_names = [NAME_MAP[name] for name in
                                 GO2_CFG_UNITREE.joint_sdk_names]
        self.scene.robot = robot
        for scanner in (self.scene.height_scanner,
                        self.scene.height_scanner_small):
            scanner.prim_path = scanner.prim_path.replace("/base", "/base_link")
        self.actions.joint_pos.joint_names = list(JOINT_NAMES)
        self.actions.joint_pos.scale = {
            NAME_MAP.get(name, name): value
            for name, value in self.actions.joint_pos.scale.items()
        }
        for group in (self.observations, self.events, self.rewards,
                      self.terminations, self.curriculum):
            adapt_entities(group)
        self.rewards.base_height_l2.params["target_height"] = BASE_HEIGHT_TARGET
        self.rewards.feet_regulation.params["base_height_target"] = (
            BASE_HEIGHT_TARGET
        )
        # 相对地形原点在 0.35～0.37 m 释放，不叠加原版 0～0.2 m 扰动。
        self.events.reset_base.params["pose_range"]["z"] = BASE_RESET_Z_RANGE


@configclass
class RaceRoughEnvCfg(RaceEnvCfg):
    """复杂地形：用户新站姿、独立加法重置、0.39～0.55 m 释放。"""

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot.init_state.joint_pos = deepcopy(ROUGH_STAND_ANGLES)
        self.scene.robot.init_state.pos = (
            *self.scene.robot.init_state.pos[:2], 0.39,
        )
        self.events.reset_base.params["pose_range"]["z"] = (0.0, 0.16)
        self.events.reset_robot_joints = EventTerm(
            func=RaceAdditiveJointReset,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=list(JOINT_NAMES), preserve_order=True,
                ),
                "offset_half_widths": deepcopy(ROUGH_RESET_HALF_WIDTHS),
            },
        )


@configclass
class RaceFlatRewardsCfg(RewardsCfg):
    """保留原版十五项奖惩，新增 SLR 默认关节姿态正奖励。"""

    pose = RewTerm(
        func=SLRPoseReward,
        weight=0.5,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=list(JOINT_NAMES), preserve_order=True,
            ),
            "command_threshold": 0.05,
            "standing_std": {
                "roll_": 0.12, "pitch01_": 0.12, "pitch02_": 0.25,
            },
            "moving_std": {
                "roll_": 0.30, "pitch01_": 0.30, "pitch02_": 0.60,
            },
        },
    )


@configclass
class RaceFlatEnvCfg(RaceEnvCfg):
    """平地实验：SLR 站姿、pose=0.5，全方向跟踪达标后扩速。"""

    rewards: RaceFlatRewardsCfg = RaceFlatRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        generator = self.scene.terrain.terrain_generator
        flat = deepcopy(generator.sub_terrains["flat"])
        flat.proportion = 1.0
        generator.sub_terrains = {"flat": flat}
        self.scene.robot.init_state.joint_pos = deepcopy(SLR_STAND_ANGLES)
        self.rewards.base_height_l2.params["target_height"] = (
            FLAT_BASE_HEIGHT_TARGET
        )
        self.rewards.feet_regulation.params["base_height_target"] = (
            FLAT_BASE_HEIGHT_TARGET
        )
        self.rewards.feet_regulation.weight = -0.2
        self.commands.base_velocity = RaceFlatOmniCommandCfg()
        ranges = self.commands.base_velocity.ranges
        ranges.lin_vel_x = [-0.5, 0.5]
        ranges.lin_vel_y = [-0.5, 0.5]
        ranges.ang_vel_yaw = [-0.5, 0.5]
        # 原版平地横移上限为1；用户要求此实验三轴均可扩到1.5。
        self.commands.base_velocity.terrain_max_command_ranges["flat"] = {
            name: [-1.5, 1.5] for name in
            ("lin_vel_x", "lin_vel_y", "ang_vel_yaw")
        }
