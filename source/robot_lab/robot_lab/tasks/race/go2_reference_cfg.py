"""保留本次复杂地形加法重置之前的 Race 原版对照配置。"""

from isaaclab.utils import configclass

from robot_lab.tasks.race.env_cfg import RaceEnvCfg
from robot_lab.tasks.race.rsl_rl_cfg import RaceMoECTSRunnerCfg

RaceGo2ReferenceEnvCfg = RaceEnvCfg


@configclass
class RaceGo2ReferenceRunnerCfg(RaceMoECTSRunnerCfg):
    experiment_name = "race_go2_reference_moe_cts"
