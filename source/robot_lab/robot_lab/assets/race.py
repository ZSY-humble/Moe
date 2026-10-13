"""Race 训练资产：保留原始碰撞几何、质量惯量及优化后的默认姿态。"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import yaml

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.sim.converters import UrdfConverter
from isaaclab.sim.spawners.from_files.from_files import _spawn_from_usd_file
from isaaclab.sim.utils import clone
from isaaclab.utils import configclass

from robot_lab.assets import ISAACLAB_ASSETS_DATA_DIR
from robot_lab.assets.race_actuator import RaceActuatorCfg


_RESOURCE_DIR = Path(ISAACLAB_ASSETS_DATA_DIR) / "race"
_CONTROL_PATH = (
    Path(ISAACLAB_ASSETS_DATA_DIR).parent
    / "deploy/deploy_mujoco/configs/race.yaml"
)
with _CONTROL_PATH.open(encoding="utf-8") as _stream:
    RACE_CONTROL = yaml.safe_load(_stream)

JOINT_NAMES = list(RACE_CONTROL["joint_names"])
FOOT_LINK_NAMES = [
    "pitch03_leg_f_l_link",
    "pitch03_leg_f_r_link",
    "pitch03_leg_b_l_link",
    "pitch03_leg_b_r_link",
]
BASE_LINK_NAME = RACE_CONTROL["base_name"]

if len(JOINT_NAMES) != 12 or len(set(JOINT_NAMES)) != 12:
    raise ValueError("Race 控制配置必须包含 12 个互不重复的主动关节。")


def _joint_parameter(key: str) -> dict[str, float]:
    """以精确关节名映射参数，避免导入器改变关节排列后发生串位。"""
    values = RACE_CONTROL[key]
    if len(values) != len(JOINT_NAMES):
        raise ValueError(f"Race 参数 {key} 的长度与主动关节数不一致。")
    return dict(zip(JOINT_NAMES, values, strict=True))


class RaceUrdfConverter(UrdfConverter):
    """显式导入 URDF 完整惯量，避免由碰撞体自动估算惯量。"""

    def _get_urdf_import_config(self):
        import_config = super()._get_urdf_import_config()
        # IsaacLab 的通用配置未暴露这个开关，Race 单独明确启用。
        import_config.set_import_inertia_tensor(self.cfg.import_inertia_tensor)
        return import_config

    def _convert_asset(self, cfg):
        super()._convert_asset(cfg)
        from pxr import PhysxSchema, Usd, UsdPhysics

        # 碰撞子节点采用实例化；在生成层设置参数，避免写只读实例代理。
        # 接触偏移不改变几何尺寸与连杆变换。
        authored = 0
        for path in Path(self.usd_dir).rglob("*.usd"):
            stage = Usd.Stage.Open(str(path))
            changed = False
            for prim in stage.Traverse():
                if prim.IsInstanceProxy() or not prim.HasAPI(UsdPhysics.CollisionAPI):
                    continue
                api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
                api.CreateContactOffsetAttr(cfg.imported_contact_offset)
                api.CreateRestOffsetAttr(cfg.imported_rest_offset)
                authored += 1
                changed = True
            if changed:
                stage.GetRootLayer().Save()
        if authored == 0:
            raise RuntimeError("Race USD 转换后没有可配置的碰撞几何")


@clone
def spawn_race_urdf(
    prim_path: str,
    cfg: RaceUrdfFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
):
    """沿用 IsaacLab 的生成和克隆流程，仅替换 Race 的 URDF 转换器。"""
    converter = RaceUrdfConverter(cfg)
    return _spawn_from_usd_file(
        prim_path, converter.usd_path, cfg, translation, orientation
    )


@configclass
class RaceUrdfFileCfg(sim_utils.UrdfFileCfg):
    """保留固定足端刚体、原始圆柱及完整质量惯量的导入配置。"""

    func: Callable = spawn_race_urdf
    # 此字段也参与转换缓存哈希，避免误用未导入完整惯量的缓存。
    import_inertia_tensor: bool = True
    imported_contact_offset: float = 0.001
    imported_rest_offset: float = 0.0


@configclass
class RaceArticulationCfg(ArticulationCfg):
    """带有标准控制顺序的 Race 关节机构配置。"""

    joint_sdk_names: list[str] | None = None


RACE_CFG = RaceArticulationCfg(
    spawn=RaceUrdfFileCfg(
        asset_path=str(_RESOURCE_DIR / "urdf/race.urdf"),
        usd_dir=str(_RESOURCE_DIR / "usd"),
        usd_file_name="race.usd",
        fix_base=False,
        root_link_name=BASE_LINK_NAME,
        merge_fixed_joints=False,
        replace_cylinders_with_capsules=False,
        collision_from_visuals=False,
        self_collision=True,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=0.2,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            drive_type="force",
            target_type="none",
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=0.0, damping=0.0
            ),
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=tuple(RACE_CONTROL["base_init_pos"]),
        rot=tuple(RACE_CONTROL["base_init_quat"]),
        joint_pos=_joint_parameter("default_angles"),
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    joint_sdk_names=JOINT_NAMES,
    actuators={
        "legs": RaceActuatorCfg(
            joint_names_expr=JOINT_NAMES,
            stiffness=_joint_parameter("kps"),
            damping=_joint_parameter("kds"),
            effort_limit=_joint_parameter("torque_limits"),
            # 仿真器不再次限制合成力矩，以保留电机限幅后的被动项。
            effort_limit_sim=1.0e9,
            armature=_joint_parameter("armature"),
            passive_damping=_joint_parameter("passive_damping"),
            passive_friction=_joint_parameter("passive_friction"),
        ),
    },
)
