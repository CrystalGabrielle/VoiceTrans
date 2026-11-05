"""音频流：WASAPI 输入 → 变声链 → 输出，带环形缓冲、电平表、录制。"""
from __future__ import annotations

import datetime as _dt
import os
import queue
import threading
import time
import wave

import numpy as np
import sounddevice as sd

from .dsp import VoiceChain, lin2db

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REC_DIR = os.path.join(PROJECT_ROOT, "recordings")

VIRTUAL_HINTS = ("cable input", "cable output", "voicemeeter", "virtual audio", "vb-audio")


def _hostapi_name(idx):
    try:
        return sd.query_hostapis(idx)["name"]
    except Exception:
        return ""


def list_devices():
    out = []
    apis = sd.query_hostapis()
    default_in, default_out = sd.default.device
    for i, d in enumerate(sd.query_devices()):
        try:
            api = apis[d["hostapi"]]["name"]
        except Exception:
            api = ""
        out.append({
            "index": i,
            "name": d["name"],
            "hostapi": api,
            "in": int(d["max_input_channels"]),
            "out": int(d["max_output_channels"]),
            "sr": int(d["default_samplerate"]),
            "is_default_in": i == default_in,
            "is_default_out": i == default_out,
            "virtual": any(h in d["name"].lower() for h in VIRTUAL_HINTS),
        })
    return out


def pick_defaults():
    """挑一个合理的输入（真麦克风，优先 WASAPI）和输出（优先虚拟声卡）。"""
    devs = list_devices()
    apis = sd.query_hostapis()
    in_cands = [d for d in devs if d["in"] > 0 and not d["virtual"]]
    out_cands = [d for d in devs if d["out"] > 0]

    def api_default(idx, key):
        try:
            info = sd.query_devices(idx)
            return apis[info["hostapi"]].get(key)
        except Exception:
            return None

    def score_in(d):
        s = 0
        if "WASAPI" in d["hostapi"]:
            s += 40
        if any(k in d["name"].lower() for k in ("麦克风", "microphone", "mic")):
            s += 30
        if api_default(d["index"], "default_input_device") == d["index"]:
            s += 35
        if d["is_default_in"]:
            s += 10
        if "声音映射器" in d["name"] or "主声音" in d["name"]:
            s -= 50
        return s

    def score_out(d):
        s = 0
        nm = d["name"].lower()
        if "cable input" in nm:
            s += 100
        elif "voicemeeter input" in nm or "voicemeeter aux input" in nm:
            s += 90
        if "WASAPI" in d["hostapi"]:
            s += 40
        if api_default(d["index"], "default_output_device") == d["index"]:
            s += 35
        if d["is_default_out"]:
            s += 10
        if "digital output" in nm or "spdif" in nm:
            s -= 20
        if "声音映射器" in d["name"] or "主声音" in d["name"]:
            s -= 50
        return s

    in_dev = max(in_cands, key=score_in)["index"] if in_cands else None
    out_dev = max(out_cands, key=score_out)["index"] if out_cands else None
    return in_dev, out_dev


def has_virtual_cable():
    return any(d["virtual"] and d["out"] > 0 for d in list_devices())


def dev_info(idx):
    """按序号取设备信息，越界或非法返回 None。"""
    try:
        return list_devices()[int(idx)]
    except Exception:
        return None


def resolve_device(saved_idx, saved_name, want_in, saved_api=""):
    """按名字把设备找回来。

    设备序号是会整体漂移的：装/卸虚拟声卡、插拔 USB 声卡都会让 index 变化，
    但设备名稳定。所以优先按名字匹配，同名多个宿主 API 时优先原来那个、其次 WASAPI。
    """
    devs = list_devices()
    pool = [d for d in devs if (d["in"] if want_in else d["out"]) > 0]
    if saved_name:
        same = [d for d in pool if d["name"] == saved_name]
        if same:
            pref = ([d for d in same if d["hostapi"] == saved_api]
                    or [d for d in same if "WASAPI" in d["hostapi"]]
                    or same)
            return pref[0]["index"]
    if saved_idx is not None:
        cur = dev_info(saved_idx)
        if cur and (cur["in"] if want_in else cur["out"]) > 0:
            if not saved_name or cur["name"] == saved_name:
                return int(saved_idx)
    return None


class RingBuffer:
    def __init__(self, capacity):
        self.cap = int(capacity)
        self.buf = np.zeros(self.cap, dtype=np.float32)
        self.r = 0
        self.w = 0
        self.count = 0
        self.overruns = 0
        self.underruns = 0

    def write(self, data):
        n = len(data)
        if n > self.cap:
            data = data[-self.cap:]
            n = len(data)
        if self.count + n > self.cap:
            drop = self.count + n - self.cap
            self._advance_read(drop)
            self.overruns += 1
        end = self.w + n
        if end <= self.cap:
            self.buf[self.w:end] = data
        else:
            first = self.cap - self.w
            self.buf[self.w:] = data[:first]
            self.buf[:end - self.cap] = data[first:]
        self.w = end % self.cap
        self.count += n

    def _advance_read(self, n):
        self.r = (self.r + n) % self.cap
        self.count -= n

    def read(self, n):
        out = np.zeros(n, dtype=np.float32)
        if self.count < n:
            self.underruns += 1
            if self.count <= 0:
                return out
            n = self.count
        end = self.r + n
        if end <= self.cap:
            out[:n] = self.buf[self.r:end]
        else:
            first = self.cap - self.r
            out[:first] = self.buf[self.r:]
            out[first:n] = self.buf[:n - first]
        self.r = end % self.cap
        self.count -= n
        return out

    def clear(self):
        self.r = self.w = 0
        self.count = 0


class Meter:
    def __init__(self):
        self.peak = 0.0
        self.rms = 0.0
        self.hold = 0.0
        self._hold_t = 0.0

    def update(self, x):
        if len(x) == 0:
            return
        pk = float(np.max(np.abs(x)))
        rms = float(np.sqrt(np.mean(np.square(x, dtype=np.float32))))
        if pk > self.peak:
            self.peak = pk
        else:
            self.peak = self.peak * 0.82 + pk * 0.18
        self.rms = self.rms * 0.82 + rms * 0.18
        now = time.time()
        if pk >= self.hold or now - self._hold_t > 1.2:
            self.hold = pk
            self._hold_t = now

    def values(self):
        return {
            "peak_db": round(lin2db(self.peak), 1),
            "rms_db": round(lin2db(self.rms), 1),
            "hold_db": round(lin2db(self.hold), 1),
            "peak": round(self.peak, 4),
        }


class Recorder:
    """把送往输出端的音频写成 16bit WAV（后台线程落盘，不阻塞音频回调）。"""

    def __init__(self, fs):
        self.fs = fs
        self.q = queue.Queue(maxsize=400)
        self.thread = None
        self.path = None
        self._stop = threading.Event()
        self.frames = 0

    @property
    def active(self):
        return self.thread is not None and self.thread.is_alive()

    def start(self):
        os.makedirs(REC_DIR, exist_ok=True)
        name = "rec_" + _dt.datetime.now().strftime("%Y%m%d_%H%M%S") + ".wav"
        self.path = os.path.join(REC_DIR, name)
        self._stop.clear()
        self.frames = 0
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return self.path

    def push(self, block):
        if not self.active:
            return
        try:
            self.q.put_nowait(np.clip(block, -1, 1))
        except queue.Full:
            pass

    def _run(self):
        with wave.open(self.path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.fs)
            while not self._stop.is_set():
                try:
                    blk = self.q.get(timeout=0.2)
                except queue.Empty:
                    continue
                pcm = (blk * 32767.0).astype("<i2").tobytes()
                wf.writeframes(pcm)
                self.frames += len(blk)
            while True:
                try:
                    blk = self.q.get_nowait()
                except queue.Empty:
                    break
                wf.writeframes((blk * 32767.0).astype("<i2").tobytes())
                self.frames += len(blk)

    def stop(self):
        self._stop.set()
        if self.thread:
            self.thread.join(timeout=2.0)
        self.thread = None
        return self.path


class AudioEngine:
    def __init__(self, store, fs=48000, block=256):
        self.store = store
        self.fs = fs
        self.block = block
        self.chain = VoiceChain(fs, block)
        self.ring = RingBuffer(max(4096, block * 16))
        self.in_meter = Meter()
        self.out_meter = Meter()
        self.recorder = Recorder(fs)
        self.istream = None
        self.ostream = None
        self.in_dev = None
        self.out_dev = None
        self.in_name = ""
        self.out_name = ""
        self.error = ""
        self._cpu = 0.0
        self._lock = threading.RLock()
        self.latency_ms = 0.0
        self.running = False

    # ---------------- 设备打开 ----------------
    def _settings_for(self, dev):
        info = sd.query_devices(dev)
        api = _hostapi_name(info["hostapi"])
        if "WASAPI" in api:
            try:
                return sd.WasapiSettings(auto_convert=True)
            except Exception:
                return None
        return None

    def start(self, in_dev=None, out_dev=None):
        with self._lock:
            self.stop()
            self.error = ""
            last = ""
            # 依次尝试：低延迟/当前采样率 → 默认延迟 → 设备默认采样率
            for force_sr, lat in ((False, "low"), (False, None), (True, None)):
                try:
                    self._open(in_dev, out_dev, force_default_sr=force_sr, latency=lat)
                    self.running = True
                    return True
                except Exception as exc:
                    last = str(exc)
                    self.stop()
            self.error = last
            self.running = False
            return False

    def _open(self, in_dev, out_dev, force_default_sr=False, latency="low"):
        if in_dev is None or out_dev is None:
            d_in, d_out = pick_defaults()
            in_dev = d_in if in_dev is None else in_dev
            out_dev = d_out if out_dev is None else out_dev
        din = sd.query_devices(in_dev)
        dout = sd.query_devices(out_dev)
        if din["max_input_channels"] < 1:
            raise RuntimeError(f"设备 [{din['name']}] 没有输入通道")
        if dout["max_output_channels"] < 1:
            raise RuntimeError(f"设备 [{dout['name']}] 没有输出通道")

        fs = self.fs
        if force_default_sr:
            fs = int(din["default_samplerate"])
        fs = int(fs)

        if fs != self.fs or self.chain.fs != fs:
            self.fs = fs
            self.chain = VoiceChain(fs, self.block)
            self.recorder = Recorder(fs)
        self.ring.clear()
        self.ring.cap = max(4096, self.block * 16)
        self.ring.buf = np.zeros(self.ring.cap, dtype=np.float32)
        if hasattr(self.chain, "reset"):
            self.chain.reset()

        in_ch = 2 if din["max_input_channels"] >= 2 else 1
        out_ch = 2 if dout["max_output_channels"] >= 2 else 1
        status = {"in": None, "out": None}

        def in_cb(indata, frames, time_info, st):
            status["in"] = st
            x = indata.mean(axis=1) if indata.ndim > 1 and indata.shape[1] > 1 else indata[:, 0]
            t0 = time.perf_counter()
            p = self.store.snapshot()
            y = self.chain.process(x, p)
            self.ring.write(y)
            self.in_meter.update(x)
            dt = time.perf_counter() - t0
            block_dur = frames / self.fs
            self._cpu = 0.9 * self._cpu + 0.1 * (dt / block_dur)

        def out_cb(outdata, frames, time_info, st):
            status["out"] = st
            y = self.ring.read(frames)
            if outdata.ndim > 1 and outdata.shape[1] > 1:
                outdata[:] = y[:, None]
            else:
                outdata[:, 0] = y
            self.out_meter.update(y)
            if self.recorder.active:
                self.recorder.push(y)

        self.istream = sd.InputStream(
            device=in_dev, channels=in_ch, samplerate=fs, blocksize=self.block,
            dtype="float32", latency=latency, callback=in_cb,
            extra_settings=self._settings_for(in_dev),
        )
        self.ostream = sd.OutputStream(
            device=out_dev, channels=out_ch, samplerate=fs, blocksize=self.block,
            dtype="float32", latency=latency, callback=out_cb,
            extra_settings=self._settings_for(out_dev),
        )
        self.istream.start()
        self.ostream.start()

        self.in_dev, self.out_dev = in_dev, out_dev
        self.in_name, self.out_name = din["name"], dout["name"]
        try:
            lat_in = self.istream.latency or 0.0
            lat_out = self.ostream.latency or 0.0
        except Exception:
            lat_in = lat_out = 0.0
        algo = (self.chain.shifter.n) / fs
        self.latency_ms = round((lat_in + lat_out + algo) * 1000, 1)

    def stop(self):
        with self._lock:
            for s in (self.istream, self.ostream):
                if s is not None:
                    try:
                        s.stop()
                        s.close()
                    except Exception:
                        pass
            self.istream = self.ostream = None
            self.running = False
            self.ring.clear()
            self._cpu = 0.0

    # ---------------- 录制 ----------------
    def record_start(self):
        if not self.running:
            return None
        return self.recorder.start()

    def record_stop(self):
        return self.recorder.stop()

    # ---------------- 状态 ----------------
    def status(self):
        rec = self.recorder
        return {
            "running": self.running,
            "fs": self.fs,
            "block": self.block,
            "cpu": round(self._cpu * 100.0, 1),
            "latency_ms": self.latency_ms,
            "input": self.in_meter.values(),
            "output": self.out_meter.values(),
            "in_dev": self.in_dev,
            "out_dev": self.out_dev,
            "in_name": self.in_name,
            "out_name": self.out_name,
            "error": self.error,
            "underruns": self.ring.underruns,
            "overruns": self.ring.overruns,
            "recording": rec.active,
            "record_path": rec.path if rec.active else None,
            "record_seconds": round(rec.frames / max(1, rec.fs), 1),
            "virtual_cable": has_virtual_cable(),
        }
