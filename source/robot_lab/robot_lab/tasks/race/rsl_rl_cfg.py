"""继承原版 MoE CTS；复杂地形每1000轮、平地每250轮保存。"""

from isaaclab.utils import configclass

from robot_lab.tasks.go2.rsl_rl_cfg import MoECTSRunnerCfg


@configclass
class RaceMoECTSRunnerCfg(MoECTSRunnerCfg):
    experiment_name = "race_go2_moe_cts"
    save_interval = 1000


@configclass
class RaceFlatMoECTSRunnerCfg(MoECTSRunnerCfg):
    experiment_name = "race_go2_flat_moe_cts"
    save_interval = 250
    max_iterations = 10000
