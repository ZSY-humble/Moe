"""Race 环境保留动作的二阶历史，供动作平滑奖励使用。"""

from collections.abc import Sequence

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ActionManager


class RaceActionManager(ActionManager):
    """在框架动作管理器上记录前两步原始动作。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._prev_prev_action = torch.zeros_like(self._action)

    @property
    def prev_prev_action(self):
        return self._prev_prev_action

    def reset(self, env_ids: Sequence[int] | None = None):
        result = super().reset(env_ids)
        if env_ids is None:
            self._prev_prev_action.zero_()
        else:
            self._prev_prev_action[env_ids] = 0.0
        return result

    def process_action(self, action: torch.Tensor):
        self._prev_prev_action[:] = self._prev_action
        super().process_action(action)


class RaceEnv(ManagerBasedRLEnv):
    """沿用管理器训练流程，安装支持动作二阶差分的管理器。"""

    def load_managers(self):
        super().load_managers()
        self.action_manager = RaceActionManager(self.cfg.actions, self)
