"""Race 原 Go2 代码对照版：继承原任务，仅适配模型及实体名称。"""

from copy import deepcopy

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils

from robot_lab.assets.race import RACE_CFG, JOINT_NAMES
from robot_lab.assets.unitree import GO2_CFG_UNITREE
from robot_lab.tasks.go2.env_cfg import Go2EnvCfg, JOINT_NAMES as GO2_JOINT_NAMES
from robot_lab.tasks.go2.rsl_rl_cfg import MoECTSRunnerCfg

REFERENCE_HEIGHT = 0.321
NAME_MAP = dict(zip(GO2_JOINT_NAMES, JOINT_NAMES, strict=True))
NAME_MAP.update({
    'base': 'base_link',
    '.*_foot': 'pitch03_leg_[fb]_[lr]_link',
    '.*_thigh|.*_calf': 'pitch0[12]_leg_[fb]_[lr]_link',
    '.*_hip_joint': 'roll_leg_.*_joint',
    '.*_(thigh|calf)_joint': 'pitch0[12]_leg_.*_joint',
    '^(?!.*base).*': '^(?!base_link$).*',
})


def adapt_entities(value):
    """仅替换实体选择名称，保留原奖励函数、噪声、权重和课程参数。"""
    if isinstance(value, SceneEntityCfg):
        for field in ('joint_names', 'body_names'):
            names = getattr(value, field)
            if isinstance(names, str):
                setattr(value, field, NAME_MAP.get(names, names))
            elif isinstance(names, (tuple, list)):
                setattr(value, field, [NAME_MAP.get(n, n) for n in names])
        return
    if isinstance(value, dict):
        for child in value.values():
            adapt_entities(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            adapt_entities(child)
    elif hasattr(value, '__dict__') and not callable(value):
        for child in vars(value).values():
            adapt_entities(child)


@configclass
class RaceGo2ReferenceEnvCfg(Go2EnvCfg):
    """使用 Race 原碰撞模型，算法环境规则继承当前原 Go2 实现。"""

    def __post_init__(self):
        super().__post_init__()
        # 明确把宇树电机曲线作为对照假设，不混入 Race 被动执行器。
        robot = deepcopy(RACE_CFG)
        robot.prim_path = '{ENV_REGEX_NS}/Robot'
        robot.init_state.pos = (0., 0., REFERENCE_HEIGHT)
        robot.init_state.rot = (1., 0., 0., 0.)
        robot.init_state.joint_pos = dict.fromkeys(JOINT_NAMES, 0.)
        robot.init_state.joint_vel = {'.*': 0.}
        robot.actuators = deepcopy(GO2_CFG_UNITREE.actuators)
        robot.joint_sdk_names = list(JOINT_NAMES)
        robot.soft_joint_pos_limit_factor = GO2_CFG_UNITREE.soft_joint_pos_limit_factor
        self.scene.robot = robot
        # 原远程材质只影响显示，换成本地颜色以支持离线服务器。
        self.scene.terrain.visual_material = sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.35, 0.4, 0.45))
        self.scene.sky_light.spawn.texture_file = None
        for scanner in (self.scene.height_scanner, self.scene.height_scanner_small):
            scanner.prim_path = '{ENV_REGEX_NS}/Robot/base_link'
        self.actions.joint_pos.joint_names = list(JOINT_NAMES)
        self.actions.joint_pos.scale = {
            'roll_leg_.*_joint': 0.25, '^(?!roll_leg_).*': 0.25,
        }
        for group in (self.observations, self.events, self.rewards,
                      self.terminations, self.curriculum):
            adapt_entities(group)
        # 保留零位及与初始高度一致的目标，其余奖励参数继承原 Go2。
        self.rewards.base_height_l2.params['target_height'] = REFERENCE_HEIGHT
        self.rewards.feet_regulation.params['base_height_target'] = REFERENCE_HEIGHT


@configclass
class RaceGo2ReferenceRunnerCfg(MoECTSRunnerCfg):
    """继承原 Go2 算法、探索噪声和训练轮数，仅独立命名及保存间隔。"""

    experiment_name = 'race_go2_reference_moe_cts'
    save_interval = 250
    obs_groups = {'policy': ['policy'], 'critic': ['critic']}
