"""默认 Race 复用原 Go2 环境，历史定制任务只保留兼容入口。"""

import gymnasium as gym


for task_id, environment_config, runner_config in (
    ("RobotLab-Race-v0", "RaceRoughEnvCfg", "RaceMoECTSRunnerCfg"),
    ("RobotLab-Race-Flat-v0", "RaceFlatEnvCfg", "RaceFlatMoECTSRunnerCfg"),
):
    gym.register(
        id=task_id,
        entry_point="robot_lab.tasks.go2.env.go2_env:Go2Env",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.env_cfg:{environment_config}",
            "rsl_rl_cfg_entry_point": f"{__name__}.rsl_rl_cfg:{runner_config}",
        },
    )

# 原版对照入口保留旧站姿和缩放重置，便于核对本次复杂地形实验。
gym.register(
    id="RobotLab-Race-Go2Reference-v0",
    entry_point="robot_lab.tasks.go2.env.go2_env:Go2Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_reference_cfg:RaceGo2ReferenceEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.go2_reference_cfg:RaceGo2ReferenceRunnerCfg",
    },
)

# 历史权重按原训练接口回放，不作为新的默认训练任务。
for task_id, environment_config, runner_config in (
    ("RobotLab-Race-Legacy-v0", "RaceEnvCfg", "RaceMoECTSRunnerCfg"),
    ("RobotLab-Race-Legacy-Flat-v0", "RaceFlatEnvCfg",
     "RaceFlatMoECTSRunnerCfg"),
):
    gym.register(
        id=task_id,
        entry_point="robot_lab.tasks.race.env:RaceEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point":
                f"{__name__}.legacy_env_cfg:{environment_config}",
            "rsl_rl_cfg_entry_point":
                f"{__name__}.legacy_rsl_rl_cfg:{runner_config}",
        },
    )
