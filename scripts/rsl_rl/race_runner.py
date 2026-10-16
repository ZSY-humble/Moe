"""保存 Race 平地速度课程进度，学习算法仍使用原版 MoE CTS。"""

from rsl_rl.runners import OnPolicyRunnerCTS


class RaceFlatOnPolicyRunnerCTS(OnPolicyRunnerCTS):
    """只扩展检查点附加信息，不改优化器、网络或采样步数。"""

    def save(self, path, it, last_model, infos=None):
        command = self.env.unwrapped.command_manager.get_term("base_velocity")
        extra = dict(infos or {})
        extra["race_command_curriculum"] = command.curriculum_state()
        super().save(path, it=it, last_model=last_model, infos=extra)

    def load(self, path, load_optimizer=True, map_location=None):
        infos = super().load(path, load_optimizer, map_location)
        if infos and "race_command_curriculum" in infos:
            command = self.env.unwrapped.command_manager.get_term("base_velocity")
            command.load_curriculum_state(infos["race_command_curriculum"])
        return infos
