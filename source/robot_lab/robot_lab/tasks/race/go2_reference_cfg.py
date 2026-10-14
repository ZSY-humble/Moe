"""兼容旧的原版对照任务名称，统一使用当前 Race 默认配置。"""

from isaaclab.utils import configclass

from robot_lab.tasks.race.env_cfg import RaceEnvCfg
from robot_lab.tasks.race.rsl_rl_cfg import RaceMoECTSRunnerCfg

RaceGo2ReferenceEnvCfg = RaceEnvCfg


@configclass
class RaceGo2ReferenceRunnerCfg(RaceMoECTSRunnerCfg):
    experiment_name = "race_go2_reference_moe_cts"
