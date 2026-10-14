"""读取 Linux X11 窗口的按住状态，支持松键立即停止。"""

import ctypes as C
import ctypes.util


class WindowKeys:
    def __init__(self):
        self.x = C.CDLL(ctypes.util.find_library("X11"))
        self.x.XOpenDisplay.restype = C.c_void_p
        self.x.XOpenDisplay.argtypes = [C.c_char_p]
        self.display = self.x.XOpenDisplay(None)
        if not self.display:
            raise RuntimeError("键盘按住控制需要 X11 桌面，请在本地图形终端运行")
        self.x.XQueryKeymap.argtypes = [C.c_void_p, C.c_void_p]
        self.x.XGetInputFocus.argtypes = [
            C.c_void_p, C.POINTER(C.c_ulong), C.POINTER(C.c_int),
        ]
        self.x.XFetchName.argtypes = [
            C.c_void_p, C.c_ulong, C.POINTER(C.c_void_p),
        ]
        self.x.XFree.argtypes = [C.c_void_p]
        self.x.XKeysymToKeycode.argtypes = [C.c_void_p, C.c_ulong]
        self.x.XKeysymToKeycode.restype = C.c_ubyte
        self.codes = {
            key: self.x.XKeysymToKeycode(self.display, ord(key.lower()))
            for key in "WSADQE "
        }

    def read(self):
        """只接受获得焦点的 MuJoCo 窗口，切换窗口即停止。"""
        window, revert = C.c_ulong(), C.c_int()
        self.x.XGetInputFocus(self.display, C.byref(window), C.byref(revert))
        if window.value <= 1:
            return set()
        name = C.c_void_p()
        self.x.XFetchName(self.display, window.value, C.byref(name))
        title = C.string_at(name).decode(errors="replace") if name.value else ""
        if name.value:
            self.x.XFree(name)
        if "MuJoCo" not in title:
            return set()
        states = C.create_string_buffer(32)
        self.x.XQueryKeymap(self.display, states)
        values = states.raw
        return {
            key for key, code in self.codes.items()
            if values[code // 8] & (1 << (code % 8))
        }
