"""沿用现有 MoE CTS 算法，使用 Race 独立日志与较小探索噪声。"""

from isaaclab.utils import configclass

from robot_lab.tasks.go2.rsl_rl_cfg import MoECTSRunnerCfg


@configclass
class RaceMoECTSRunnerCfg(MoECTSRunnerCfg):
    experiment_name = "race_moe_cts"
    max_iterations = 10000
    save_interval = 250
    clip_actions = 100.0
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}

    def __post_init__(self):
        super().__post_init__()
        self.policy.init_noise_std = 0.5


@configclass
class RaceFlatMoECTSRunnerCfg(RaceMoECTSRunnerCfg):
    experiment_name = "race_flat_moe_cts"
