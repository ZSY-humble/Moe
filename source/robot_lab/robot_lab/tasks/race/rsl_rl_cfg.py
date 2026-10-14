"""继承原版 MoE CTS；用户指定每 1000 轮保存，保留独立日志名称。"""

from isaaclab.utils import configclass

from robot_lab.tasks.go2.rsl_rl_cfg import MoECTSRunnerCfg


@configclass
class RaceMoECTSRunnerCfg(MoECTSRunnerCfg):
    experiment_name = "race_go2_moe_cts"
    save_interval = 1000


@configclass
class RaceFlatMoECTSRunnerCfg(MoECTSRunnerCfg):
    experiment_name = "race_go2_flat_moe_cts"
    save_interval = 1000
