"""CTS 回放只构造策略，单机器人推理不需要训练采样器和优化器。"""

from types import SimpleNamespace

import torch
from rsl_rl.modules import ActorCriticMoECTS
from rsl_rl.utils import resolve_obs_groups


class CTSPlayback:
    def __init__(self, env, config, device):
        observations = env.get_observations()
        groups = resolve_obs_groups(
            observations, config["obs_groups"], ["critic"],
        )
        policy_config = dict(config["policy"])
        name = policy_config.pop("class_name")
        if name != "ActorCriticMoECTS":
            raise ValueError(f"不支持的 CTS 策略：{name}")
        policy = ActorCriticMoECTS(
            observations, groups, env.num_actions, **policy_config,
        ).to(device)
        self.alg = SimpleNamespace(policy=policy)
        self.device = device

    def load(self, path):
        checkpoint = torch.load(
            path, map_location=self.device, weights_only=True,
        )
        weights = checkpoint["model_state_dict"]
        if any(not torch.isfinite(value).all().item()
               for value in weights.values()
               if torch.is_floating_point(value)):
            raise ValueError("检查点权重含非有限数值")
        self.alg.policy.load_state_dict(weights, strict=True)
        self.alg.policy.eval()
        print(f"[INFO] 已加载第 {checkpoint.get('iter')} 轮权重，"
              "使用学生策略回放。")

    def get_inference_policy(self, device=None):
        if device is not None:
            self.alg.policy.to(device)
        return self.alg.policy.act_inference
