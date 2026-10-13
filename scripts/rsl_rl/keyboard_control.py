"""把键盘速度接入命令管理器，保持观测历史按物理步更新。"""

import torch


def attach_keyboard(env, keyboard, ranges):
    """替换回放实例的命令更新，保留训练中的速度上下限。"""
    term = env.command_manager.get_term("base_velocity")
    limits = torch.tensor(
        [ranges.lin_vel_x, ranges.lin_vel_y, ranges.ang_vel_z],
        device=env.device,
    )

    def update():
        command = keyboard.advance().to(env.device)
        command = torch.maximum(limits[:, 0], command)
        command = torch.minimum(limits[:, 1], command)
        term.command[:] = command
        term.is_standing_env[:] = False

    # 命令管理器在每个策略步、生成新观测前调用此方法。
    term._update_command = update
    return update
