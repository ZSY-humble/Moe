"""Race 四足机器人的速度跟踪任务，默认适用于 8 GB 显存。"""

import math

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from robot_lab.assets.race import RACE_CFG
from robot_lab.tasks.race import mdp


# 显式关节顺序同时约束策略动作、观测与部署，避免导入器排序变化。
JOINT_NAMES = [
    f"{joint}_leg_{leg}_joint"
    for leg in ("f_l", "f_r", "b_l", "b_r")
    for joint in ("roll", "pitch01", "pitch02")
]
FOOT_NAMES = [f"pitch03_leg_{leg}_link" for leg in ("f_l", "f_r", "b_l", "b_r")]
BASE_LINK_NAME = "base_link"
FOOT_LINK_NAME = "pitch03_leg_[fb]_[lr]_link"
BASE_HEIGHT_TARGET = 0.33

# 此处仅定义地形；Race URDF 内的 26 个简化碰撞体保持原样。
RACE_TERRAIN_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=12.0,
    num_rows=6,
    num_cols=8,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    curriculum=True,
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.25),
        "rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.25,
            noise_range=(-0.025, 0.025),
            noise_step=0.005,
            border_width=0.25,
        ),
        "slope_up": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.15,
            slope_range=(0.0, 0.20),
            platform_width=3.0,
            border_width=0.25,
        ),
        "slope_down": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.15,
            slope_range=(0.0, 0.20),
            platform_width=3.0,
            border_width=0.25,
        ),
        "stairs_up": terrain_gen.HfPyramidStairsTerrainCfg(
            proportion=0.10,
            step_height_range=(0.025, 0.10),
            step_width=0.32,
            platform_width=3.0,
            border_width=0.25,
        ),
        "stairs_down": terrain_gen.HfInvertedPyramidStairsTerrainCfg(
            proportion=0.10,
            step_height_range=(0.025, 0.10),
            step_width=0.32,
            platform_width=3.0,
            border_width=0.25,
        ),
    },
)


@configclass
class RaceSceneCfg(InteractiveSceneCfg):
    """使用本地 Race 资源及不依赖远程材质下载的训练场景。"""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=RACE_TERRAIN_CFG,
        max_init_terrain_level=1,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.45, 0.45, 0.45), roughness=0.8
        ),
        debug_vis=False,
    )
    robot: ArticulationCfg = RACE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        mesh_prim_paths=["/World/ground"],
        debug_vis=False,
    )
    height_scanner_small = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[0.4, 0.3]),
        mesh_prim_paths=["/World/ground"],
        debug_vis=False,
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=750.0),
    )


@configclass
class CommandsCfg:
    """先学习低速步行与站立，再按需要提高命令范围。"""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 8.0),
        rel_standing_envs=0.15,
        heading_command=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 0.8),
            lin_vel_y=(-0.25, 0.25),
            ang_vel_z=(-0.6, 0.6),
        ),
        debug_vis=False,
    )


@configclass
class ActionsCfg:
    """策略输出为默认姿态周围的关节偏移，最终目标按 URDF 限位裁剪。"""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=JOINT_NAMES,
        scale=0.25,
        use_default_offset=True,
        preserve_order=True,
        clip={
            "roll_leg_.*_joint": (-0.733, 0.733),
            "pitch01_leg_.*_joint": (-2.2689, 2.2689),
            "pitch02_leg_.*_joint": (-1.0472, 0.8727),
        },
    )


def ordered_joints():
    """每次返回独立实体配置，避免管理器解析索引时共享可变状态。"""
    return SceneEntityCfg("robot", joint_names=JOINT_NAMES, preserve_order=True)


def ordered_feet(asset_name="robot"):
    return SceneEntityCfg(asset_name, body_names=FOOT_NAMES, preserve_order=True)


@configclass
class ObservationsCfg:
    """保持原框架的三组观测布局，全部实体替换为 Race 名称。"""

    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.2, n_max=0.2),
            clip=(-100.0, 100.0), scale=0.25,
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            clip=(-100.0, 100.0),
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            clip=(-100.0, 100.0),
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": ordered_joints()},
            noise=Unoise(n_min=-0.03, n_max=0.03),
            clip=(-100.0, 100.0),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": ordered_joints()},
            noise=Unoise(n_min=-2.0, n_max=2.0),
            clip=(-100.0, 100.0), scale=0.05,
        )
        actions = ObsTerm(func=mdp.last_action, clip=(-100.0, 100.0))

        def __post_init__(self):
            self.history_length = 10
            self.enable_corruption = True
            self.concatenate_terms = True
            self.flatten_history_dim = True

    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel, clip=(-100.0, 100.0), scale=2.0,
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel, clip=(-100.0, 100.0), scale=0.25,
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity, clip=(-100.0, 100.0),
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            clip=(-100.0, 100.0),
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": ordered_joints()}, clip=(-100.0, 100.0),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": ordered_joints()},
            clip=(-100.0, 100.0), scale=0.05,
        )
        actions = ObsTerm(func=mdp.last_action, clip=(-100.0, 100.0))
        joint_acc = ObsTerm(
            func=mdp.joint_acc,
            params={"asset_cfg": ordered_joints()},
            clip=(-100.0, 100.0), scale=1e-4,
        )
        joint_torque = ObsTerm(
            func=mdp.joint_effort,
            params={"asset_cfg": ordered_joints()},
            clip=(-100.0, 100.0), scale=0.01,
        )
        contact_force = ObsTerm(
            func=mdp.foot_contact_force_norm,
            params={"sensor_cfg": ordered_feet("contact_forces")},
            clip=(-100.0, 100.0), scale=1e-3,
        )
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0), scale=2.5,
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class SingleObsCfg(PolicyCfg):
        def __post_init__(self):
            super().__post_init__()
            self.history_length = 1

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    single_obs: SingleObsCfg = SingleObsCfg()


@configclass
class EventCfg:
    """保留训练随机化，但初期围绕已验证姿态进行小幅扰动。"""

    randomize_rigid_body_mass_base = EventTerm(
        func=mdp.randomize_rigid_body_mass, mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=BASE_LINK_NAME),
            "mass_distribution_params": (-0.5, 0.5),
            "operation": "add", "recompute_inertia": True,
        },
    )
    randomize_rigid_body_mass_others = EventTerm(
        func=mdp.randomize_rigid_body_mass, mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="^(?!base_link$).*"),
            "mass_distribution_params": (0.95, 1.05),
            "operation": "scale", "recompute_inertia": True,
        },
    )
    randomize_com_positions = EventTerm(
        func=mdp.randomize_rigid_body_com, mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=BASE_LINK_NAME),
            "com_range": {
                "x": (-0.01, 0.01), "y": (-0.01, 0.01),
                "z": (-0.01, 0.01),
            },
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset, mode="reset",
        params={
            "asset_cfg": ordered_joints(),
            "position_range": (-0.03, 0.03),
            "velocity_range": (-0.1, 0.1),
        },
    )
    randomize_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains, mode="reset",
        params={
            "asset_cfg": ordered_joints(),
            "stiffness_distribution_params": (0.95, 1.05),
            "damping_distribution_params": (0.95, 1.05),
            "operation": "scale", "distribution": "uniform",
        },
    )
    randomize_motor_zero_offset = EventTerm(
        func=mdp.randomize_action_joint_pos_offset, mode="reset",
        params={
            "action_term_name": "joint_pos", "offset_range": (-0.01, 0.01),
        },
    )
    randomize_push_robot = EventTerm(
        func=mdp.push_by_setting_velocity, mode="interval",
        interval_range_s=(8.0, 12.0),
        params={"velocity_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2)}},
    )
    randomize_rigid_body_material = EventTerm(
        func=mdp.randomize_rigid_body_material, mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.6, 1.2),
            "dynamic_friction_range": (0.6, 1.2),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64, "make_consistent": True,
        },
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform, mode="reset",
        params={
            "pose_range": {
                "x": (-0.2, 0.2), "y": (-0.2, 0.2), "z": (0.0, 0.02),
                "roll": (-0.03, 0.03), "pitch": (-0.03, 0.03),
                "yaw": (-math.pi, math.pi),
            },
            "velocity_range": {
                "x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (-0.05, 0.05),
                "roll": (-0.1, 0.1), "pitch": (-0.1, 0.1),
                "yaw": (-0.1, 0.1),
            },
        },
    )


@configclass
class RewardsCfg:
    """保留原 Go2 已开启的 15 个奖励项，使用 Race 的任务参数。"""

    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=2.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    joint_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2, weight=-1e-7,
        params={"asset_cfg": ordered_joints()},
    )
    joint_power = RewTerm(
        func=mdp.joint_power, weight=-2e-5,
        params={"asset_cfg": ordered_joints()},
    )
    joint_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2, weight=-1e-4,
        params={"asset_cfg": ordered_joints()},
    )
    base_height_l2 = RewTerm(
        func=mdp.base_height_l2, weight=-5.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=BASE_LINK_NAME),
            "target_height": BASE_HEIGHT_TARGET,
            "sensor_cfg": SceneEntityCfg("height_scanner_small"),
        },
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    action_smoothness_l2 = RewTerm(func=mdp.action_smoothness_l2, weight=-0.01)
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts, weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names="(roll|pitch01|pitch02)_leg_.*_link"
            ),
            "threshold": 5.0,
        },
    )
    joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits, weight=-2.0,
        params={"asset_cfg": ordered_joints()},
    )
    feet_regulation = RewTerm(
        func=mdp.feet_regulation, weight=-0.05,
        params={
            "base_height_target": BASE_HEIGHT_TARGET,
            "asset_cfg": ordered_feet(),
            "sensor_cfg": SceneEntityCfg("height_scanner_small"),
        },
    )
    roll_pos_penalty_l1 = RewTerm(
        func=mdp.hip_pos_penalty_l1, weight=-0.05,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names="roll_leg_.*_joint"),
            "stand_still_scale": 1.0, "command_threshold": 0.1,
        },
    )
    pitch_pos_penalty_l1 = RewTerm(
        func=mdp.joint_pos_penalty_l1, weight=-0.01,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names="pitch(01|02)_leg_.*_joint"
            ),
            "stand_still_scale": 2.0,
            "velocity_threshold": 0.1, "command_threshold": 0.1,
        },
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    illegal_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=BASE_LINK_NAME),
            "threshold": 1.0,
        },
    )
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation, params={"limit_angle": math.radians(60.0)},
    )


@configclass
class CurriculumCfg:
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


@configclass
class RaceEnvCfg(ManagerBasedRLEnvCfg):
    """粗糙地形训练：50 Hz 策略、200 Hz 仿真、默认 256 个并行环境。"""

    scene: RaceSceneCfg = RaceSceneCfg(num_envs=256, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 128 * 1024
        self.sim.physx.gpu_collision_stack_size = 64 * 1024 * 1024
        self.sim.physx.enable_external_forces_every_iteration = True
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        self.scene.height_scanner_small.update_period = self.decimation * self.sim.dt
        self.scene.contact_forces.update_period = self.sim.dt
        generator = self.scene.terrain.terrain_generator
        if generator is not None:
            generator.curriculum = self.curriculum.terrain_levels is not None


@configclass
class RaceFlatEnvCfg(RaceEnvCfg):
    """平地预训练与流程验证；观测和网络维度与粗糙地形保持一致。"""

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.terrain.max_init_terrain_level = None
        self.curriculum.terrain_levels = None
        self.events.randomize_push_robot = None
