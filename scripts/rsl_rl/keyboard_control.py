"""把键盘速度接入命令管理器，保持观测历史按物理步更新。"""

import torch


def attach_keyboard(env, keyboard, ranges):
    """替换回放实例的命令更新，保留训练中的速度上下限。"""
    term = env.command_manager.get_term("base_velocity")
    limits = torch.tensor(
        [ranges.lin_vel_x, ranges.lin_vel_y, ranges.ang_vel_z],
        device=env.device,
    )
    previous = None

    def update():
        nonlocal previous
        command = keyboard.advance().to(env.device)
        command = torch.maximum(limits[:, 0], command)
        command = torch.minimum(limits[:, 1], command)
        term.command[:] = command
        term.is_standing_env[:] = False
        values = command.detach().cpu().tolist()
        if values != previous:
            print(f"[键盘] 前后={values[0]:+.2f} m/s，"
                  f"左右={values[1]:+.2f} m/s，"
                  f"转向={values[2]:+.2f} rad/s", flush=True)
            previous = values

    # 命令管理器在每个策略步、生成新观测前调用此方法。
    term._update_command = update
    return update


def add_game_keys(keyboard):
    """在官方方向键映射上补充 WASD/QE，空格清零速度。"""
    mapping = keyboard._INPUT_KEY_MAPPING
    for key, original in {
        "W": "UP", "S": "DOWN", "A": "LEFT", "D": "RIGHT",
        "Q": "Z", "E": "X",
    }.items():
        mapping[key] = mapping[original].copy()
    keyboard.add_callback("SPACE", keyboard.reset)


def create_game_keyboard(config):
    """按当前按住的键计算速度，清零后松键也不会产生反向命令。"""
    import carb.input
    from isaaclab.devices import Se2Keyboard

    class GameKeyboard(Se2Keyboard):
        def __init__(self, cfg):
            self._held_keys = set()
            super().__init__(cfg)
            add_game_keys(self)

        def reset(self):
            self._held_keys.clear()
            super().reset()

        def _on_keyboard_event(self, event, *args, **kwargs):
            key = event.input.name
            if event.type == carb.input.KeyboardEventType.KEY_PRESS:
                if key in {"SPACE", "L"}:
                    self.reset()
                elif key in self._INPUT_KEY_MAPPING:
                    self._held_keys.add(key)
                callback = self._additional_callbacks.get(key)
                if callback:
                    callback()
            elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
                self._held_keys.discard(key)
            self._base_command.fill(0.0)
            for held in self._held_keys:
                self._base_command += self._INPUT_KEY_MAPPING[held]
            return True

    return GameKeyboard(config)
