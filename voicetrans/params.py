"""参数定义、预设、配置文件读写。"""
from __future__ import annotations

import copy
import json
import os
import threading

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.json")


def _p(key, label, lo, hi, step, default, unit="", group="变声", kind="slider", hint=""):
    return dict(key=key, label=label, min=lo, max=hi, step=step, default=default,
                unit=unit, group=group, kind=kind, hint=hint)


PARAMS = [
    # ---------------- 输入 ----------------
    _p("input_gain", "输入增益", -24, 24, 0.5, 0.0, "dB", "输入", hint="麦克风太小/太大时先调这里"),
    _p("gate_on", "噪声门", 0, 1, 1, True, "", "输入", "switch", hint="不说话时自动闭麦，屏蔽键盘声与底噪"),
    _p("gate_db", "门限", -80, -10, 1, -52, "dB", "输入", hint="环境吵就往上调，漏气就往下降"),
    _p("hpf", "低切", 40, 400, 5, 80, "Hz", "输入", hint="切掉轰隆声/喷麦低频"),
    _p("lpf", "高切", 4000, 20000, 100, 16000, "Hz", "输入", hint="切掉嘶声，电话音靠它"),

    # ---------------- 变声 ----------------
    _p("pitch", "音高", -12, 12, 0.1, 0.0, "半音", "变声", hint="±12 半音 = 一个八度，正数变高"),
    _p("formant", "音色（共振峰）", -12, 12, 0.1, 0.0, "半音", "变声",
       hint="决定“像谁”。0 = 保持本人音色；配合音高可做男↔女"),

    # ---------------- 音色 ----------------
    _p("eq_low", "低频", -12, 12, 0.5, 0.0, "dB", "音色", hint="200Hz 附近，厚度"),
    _p("eq_mid", "中频", -12, 12, 0.5, 0.0, "dB", "音色", hint="2kHz 附近，清晰度/穿透"),
    _p("eq_high", "高频", -12, 12, 0.5, 0.0, "dB", "音色", hint="6kHz 以上，空气感"),

    # ---------------- 效果 ----------------
    _p("drive", "失真染色", 0, 100, 1, 0, "%", "效果", hint="加厚、加脏，怪兽/机械音必备"),
    _p("robot", "机械调制", 0, 100, 1, 0, "%", "效果", hint="环形调制，做出机器人/外星人"),
    _p("robot_freq", "调制频率", 20, 200, 1, 55, "Hz", "效果", hint="40~60 像机器人，80+ 像电子音"),
    _p("whisper", "气声", 0, 100, 1, 0, "%", "效果", hint="把声音变“嘶嘶”的耳语"),
    _p("reverb_mix", "混响量", 0, 100, 1, 0, "%", "效果", hint="湿度，直播聊天建议 < 20"),
    _p("reverb_size", "空间大小", 20, 100, 1, 50, "%", "效果", hint="房间越大尾巴越长"),
    _p("comp", "压缩", 0, 100, 1, 30, "%", "效果", hint="让大小声更均匀，直播必备"),

    # ---------------- 输出 ----------------
    _p("output_gain", "输出增益", -24, 24, 0.5, 0.0, "dB", "输出"),
    _p("limiter", "限幅保护", 0, 1, 1, True, "", "输出", "switch", hint="防止爆音，建议常开"),
    _p("mute", "静音", 0, 1, 1, False, "", "输出", "switch"),
    _p("bypass", "旁通", 0, 1, 1, False, "", "输出", "switch", hint="一键对比原声/变声"),
]

PARAM_BY_KEY = {p["key"]: p for p in PARAMS}
DEFAULTS = {p["key"]: p["default"] for p in PARAMS}
GROUPS = ["输入", "变声", "音色", "效果", "输出"]


def _preset(name, desc, **kw):
    vals = copy.deepcopy(DEFAULTS)
    vals.update(kw)
    return dict(name=name, desc=desc, values=vals)


PRESETS = [
    _preset("原声直通", "不做任何处理，用于对比", bypass=False),
    _preset("甜美女声", "男声 → 女声，自然不尖", pitch=6.0, formant=3.5, eq_high=2.5,
            comp=35, reverb_mix=12, hpf=90, output_gain=2.0),
    _preset("萝莉音", "高甜、偏薄，适合聊天/游戏", pitch=10.0, formant=6.5, eq_high=3.0,
            whisper=6, comp=30, reverb_mix=10),
    _preset("御姐音", "低沉女声，气场强", pitch=3.0, formant=2.0, eq_low=1.5, comp=40,
            reverb_mix=8),
    _preset("清亮少年", "轻微升高，少年感", pitch=2.5, formant=1.2, eq_high=2.0, comp=30),
    _preset("磁性大叔", "降调加厚，暖男嗓", pitch=-3.5, formant=-2.5, eq_low=3.0,
            eq_mid=1.5, drive=18, comp=45),
    _preset("播音主持", "电台质感，字字清楚", pitch=-1.5, formant=-0.8, comp=55,
            eq_low=2.0, eq_mid=2.0, eq_high=2.0, reverb_mix=6, hpf=100),
    _preset("机械音", "赛博机器人", pitch=-1.0, formant=-1.0, robot=70, robot_freq=45,
            drive=12, comp=35, reverb_mix=15),
    _preset("深渊怪兽", "低沉咆哮，恐怖游戏", pitch=-7.0, formant=-6.0, drive=45,
            reverb_mix=45, reverb_size=85, gate_on=False, output_gain=3.0),
    _preset("外星访客", "怪腔怪调，整活专用", pitch=7.0, formant=-5.0, robot=30,
            robot_freq=80, reverb_mix=30),
    _preset("电话听筒", "300~3.4k 窄带 + 失真", hpf=340, lpf=3400, drive=35, eq_mid=4.0,
            comp=55, reverb_mix=0),
    _preset("耳语密语", "只有气声，适合 ASMR", whisper=75, pitch=0.5, reverb_mix=25,
            gate_on=False, output_gain=2.0),
    _preset("空谷回声", "大混响，远景感", reverb_mix=55, reverb_size=90, comp=40,
            output_gain=5.0),
    _preset("电音花栗鼠", "极限升高，娱乐向", pitch=12.0, formant=8.0, comp=30),
]
PRESET_BY_NAME = {p["name"]: p for p in PRESETS}


DEFAULT_CONFIG = {
    "input_device": None,
    "output_device": None,
    "samplerate": 48000,
    "blocksize": 256,
    "params": copy.deepcopy(DEFAULTS),
    "last_preset": "原声直通",
    "ui_port": 8765,
}


class ParamStore:
    """线程安全的参数容器：音频线程每块取一次快照，UI 线程随时写。"""

    def __init__(self, initial=None):
        self._lock = threading.RLock()
        self._values = copy.deepcopy(DEFAULTS)
        if initial:
            for k, v in initial.items():
                if k in self._values:
                    self._values[k] = self._coerce(k, v)

    @staticmethod
    def _coerce(key, value):
        spec = PARAM_BY_KEY.get(key)
        if spec is None:
            return value
        if spec["kind"] == "switch":
            return bool(value)
        try:
            v = float(value)
        except (TypeError, ValueError):
            return spec["default"]
        return max(spec["min"], min(spec["max"], v))

    def snapshot(self):
        with self._lock:
            return dict(self._values)

    def get(self, key, default=None):
        with self._lock:
            return self._values.get(key, default)

    def set(self, key, value):
        if key not in PARAM_BY_KEY:
            return False
        with self._lock:
            self._values[key] = self._coerce(key, value)
        return True

    def update(self, values):
        with self._lock:
            for k, v in values.items():
                if k in self._values:
                    self._values[k] = self._coerce(k, v)
        return self.snapshot()


def load_config():
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            for k in ("input_device", "output_device", "samplerate", "blocksize",
                      "last_preset", "ui_port"):
                if k in data and data[k] is not None:
                    cfg[k] = data[k]
            if isinstance(data.get("params"), dict):
                cfg["params"].update({k: v for k, v in data["params"].items()
                                      if k in PARAM_BY_KEY})
        except Exception:
            pass
    return cfg


def save_config(cfg):
    tmp = CONFIG_PATH + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        os.replace(tmp, CONFIG_PATH)
        return True
    except Exception:
        return False


def schema():
    return {
        "groups": GROUPS,
        "params": PARAMS,
        "presets": PRESETS,
        "defaults": DEFAULTS,
    }
