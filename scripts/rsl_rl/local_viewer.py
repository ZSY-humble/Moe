"""配置本地键盘窗口，处理 NVIDIA 535 系列的 Vulkan 版本误报。"""

from pathlib import Path
import re
import sys


def configure_local_viewer(args):
    if not args.keyboard:
        return
    args.window_width = 1280
    args.window_height = 800
    args.width = 960
    args.height = 540
    if "--rendering_mode" not in sys.argv:
        args.rendering_mode = "performance"
    try:
        driver = Path("/proc/driver/nvidia/version").read_text()
    except OSError:
        return
    version = re.search(r"Kernel Module\s+(\d+)\.(\d+)\.(\d+)", driver)
    if version and int(version[1]) == 535 and int(version[2]) >= 256:
        setting = "/rtx/verifyDriverVersion/enabled"
        if setting not in args.kit_args:
            args.kit_args += f" --{setting}=false"
            print("[INFO] NVIDIA 535.256+ 的 Vulkan 版本误报："
                  "本次窗口使用官方驱动检查规避参数。")
