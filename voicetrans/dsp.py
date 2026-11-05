"""实时变声 DSP：变调 + 独立共振峰、噪声门、EQ、压缩、混响、特效、限幅。

设计要点
--------
* 纯 numpy / scipy，48k 下单块 256 点（5.3 ms），CPU 占用约 3~8%。
* 变调：相位声码器频域搬移（保持时长），叠加共振峰包络替换，
  这样"变调不变音色"和"男变女"都能做。
* 所有效果都是因果的、块处理，额外算法延迟只有 STFT 的一帧 ≈ 21 ms。
"""
from __future__ import annotations

import math

import numpy as np
from scipy.signal import butter, lfilter, sosfilt

EPS = 1e-9


def db2lin(db: float) -> float:
    return 10.0 ** (db / 20.0)


def lin2db(x: float, floor: float = 1e-7) -> float:
    return 20.0 * math.log10(max(abs(float(x)), floor))


def semitones_to_ratio(semi: float) -> float:
    return 2.0 ** (semi / 12.0)


# --------------------------------------------------------------------------- #
#  双二阶滤波器（RBJ）
# --------------------------------------------------------------------------- #
def _lowshelf(fs, f0, gain_db, S=1.0):
    A = 10 ** (gain_db / 40.0)
    w0 = 2 * math.pi * f0 / fs
    cw, sw = math.cos(w0), math.sin(w0)
    alpha = sw / 2 * math.sqrt(max((A + 1 / A) * (1 / S - 1) + 2, EPS))
    b0 = A * ((A + 1) - (A - 1) * cw + 2 * math.sqrt(A) * alpha)
    b1 = 2 * A * ((A - 1) - (A + 1) * cw)
    b2 = A * ((A + 1) - (A - 1) * cw - 2 * math.sqrt(A) * alpha)
    a0 = (A + 1) + (A - 1) * cw + 2 * math.sqrt(A) * alpha
    a1 = -2 * ((A - 1) + (A + 1) * cw)
    a2 = (A + 1) + (A - 1) * cw - 2 * math.sqrt(A) * alpha
    return [b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]


def _highshelf(fs, f0, gain_db, S=1.0):
    A = 10 ** (gain_db / 40.0)
    w0 = 2 * math.pi * f0 / fs
    cw, sw = math.cos(w0), math.sin(w0)
    alpha = sw / 2 * math.sqrt(max((A + 1 / A) * (1 / S - 1) + 2, EPS))
    b0 = A * ((A + 1) + (A - 1) * cw + 2 * math.sqrt(A) * alpha)
    b1 = -2 * A * ((A - 1) + (A + 1) * cw)
    b2 = A * ((A + 1) + (A - 1) * cw - 2 * math.sqrt(A) * alpha)
    a0 = (A + 1) - (A - 1) * cw + 2 * math.sqrt(A) * alpha
    a1 = 2 * ((A - 1) - (A + 1) * cw)
    a2 = (A + 1) - (A - 1) * cw - 2 * math.sqrt(A) * alpha
    return [b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]


def _peaking(fs, f0, gain_db, Q=1.0):
    A = 10 ** (gain_db / 40.0)
    w0 = 2 * math.pi * f0 / fs
    cw, sw = math.cos(w0), math.sin(w0)
    alpha = sw / (2 * Q)
    b0 = 1 + alpha * A
    b1 = -2 * cw
    b2 = 1 - alpha * A
    a0 = 1 + alpha / A
    a1 = -2 * cw
    a2 = 1 - alpha / A
    return [b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]


class EqChain:
    """低切 + 低频搁架 + 中频峰 + 高频搁架 + 高切，五段固定结构（状态可复用）。"""

    def __init__(self, fs):
        self.fs = fs
        self.sos = None
        self.zi = None
        self._sig = None

    def _build(self, hpf, lpf, low, mid, high):
        hpf = max(20.0, min(hpf, self.fs * 0.4))
        lpf = max(hpf * 1.5 + 100, min(lpf, self.fs * 0.45))
        sections = [
            butter(2, hpf, btype="highpass", fs=self.fs, output="sos")[0],
            _lowshelf(self.fs, 200.0, low),
            _peaking(self.fs, 2000.0, mid, Q=0.9),
            _highshelf(self.fs, 6000.0, high),
            butter(2, lpf, btype="lowpass", fs=self.fs, output="sos")[0],
        ]
        return np.asarray(sections, dtype=np.float64)

    def process(self, x, hpf, lpf, low, mid, high):
        sig = (round(hpf, 1), round(lpf, -1), round(low, 1), round(mid, 1), round(high, 1))
        if sig != self._sig:
            self.sos = self._build(hpf, lpf, low, mid, high)
            if self.zi is None or self.zi.shape[0] != self.sos.shape[0]:
                self.zi = np.zeros((self.sos.shape[0], 2))
            self._sig = sig
        if self.sos is None:
            return x
        y, self.zi = sosfilt(self.sos, x, zi=self.zi)
        return y


# --------------------------------------------------------------------------- #
#  控制率包络工具
# --------------------------------------------------------------------------- #
class _Control:
    """把一块音频按 ctrl 步长降采样到控制率，处理完再插值回采样率。"""

    def __init__(self, fs, block, ctrl=32):
        self.fs = fs
        self.block = block
        self.c = ctrl
        self.n = max(1, block // ctrl)
        self.c = max(1, block // self.n)

    def peaks(self, x):
        n, c = self.n, self.c
        body = x[: n * c].reshape(n, c)
        pk = np.max(np.abs(body), axis=1)
        if n * c < len(x):
            pk[-1] = max(pk[-1], float(np.max(np.abs(x[n * c:]))))
        return pk

    def upsample(self, series):
        return np.repeat(series, self.c, axis=0)[: self.block]


class Gate:
    def __init__(self, fs, block):
        self.fs, self.block = fs, block
        self.ctl = _Control(fs, block)
        self.env = 0.0
        self.g = 1.0
        self.zi = np.zeros(1)
        dt = self.ctl.c / fs
        self.rel = math.exp(-dt / 0.120)          # 120 ms 释放
        self.att_g = 1 - math.exp(-dt / 0.0015)   # 1.5 ms 打开
        self.rel_g = 1 - math.exp(-dt / 0.080)    # 80 ms 关闭
        self.a = 1 - math.exp(-1.0 / (fs * 0.0012))

    def process(self, x, thresh_db, enabled):
        if not enabled:
            self.env *= 0.85
            self.g += (1.0 - self.g) * 0.3
            return x
        pk = self.ctl.peaks(x)
        rel = self.rel
        gains = np.empty_like(pk)
        for i, p in enumerate(pk):
            if p > self.env:
                self.env = p
            else:
                self.env *= rel
            e = lin2db(self.env)
            t = (e - thresh_db) / 6.0
            target = 1.0 if t >= 1.0 else (0.0 if t <= 0.0 else t)
            coef = self.att_g if target > self.g else self.rel_g
            self.g += (target - self.g) * coef
            gains[i] = self.g
        g = self.ctl.upsample(gains)
        g, self.zi = lfilter([self.a], [1.0, self.a - 1.0], g, zi=self.zi)
        return x * g


class Compressor:
    """峰值检测 + 软拐点压缩，补偿增益按长期平均压缩量自动回补（响度基本持平）。"""

    def __init__(self, fs, block):
        self.fs, self.block = fs, block
        self.ctl = _Control(fs, block)
        self.env = 0.0
        self.gr = 0.0
        self.gr_avg = 0.0
        dt = self.ctl.c / fs
        self.rel = math.exp(-dt / 0.150)
        self.att = 1 - math.exp(-dt / 0.008)
        self.relg = 1 - math.exp(-dt / 0.150)
        self.a = 1 - math.exp(-1.0 / (fs * 0.002))
        self.zi = np.zeros(1)

    def process(self, x, amount):
        if amount <= 0.5:
            self.gr *= 0.9
            self.gr_avg *= 0.95
            return x
        thresh = -9.0 - amount * 0.14      # -9 .. -23 dB
        ratio = 1.6 + amount * 0.045       # 1.6 .. 6.1
        pk = self.ctl.peaks(x)
        out = np.empty_like(pk)
        for i, p in enumerate(pk):
            if p > self.env:
                self.env = p
            else:
                self.env *= self.rel
            e = lin2db(self.env)
            # 6 dB 软拐点：压缩不是"到点就折"，过渡圆滑，听感更接近人声的呼吸感
            knee = 6.0
            if e <= thresh - knee * 0.5:
                target = 0.0
            elif e >= thresh + knee * 0.5:
                target = -(e - thresh) * (1.0 - 1.0 / ratio)
            else:
                d = e - (thresh - knee * 0.5)
                target = -(d * d) / (2.0 * knee) * (1.0 - 1.0 / ratio)
            coef = self.att if target < self.gr else self.relg
            self.gr += (target - self.gr) * coef
            out[i] = self.gr
        # 固定补偿即可：整体响度由后级 AGC 统一对齐，避免两级积分互相打架
        makeup = amount * 0.05
        g = self.ctl.upsample(out) + makeup
        lin = 10.0 ** (g / 20.0)
        lin, self.zi = lfilter([self.a], [1.0, self.a - 1.0], lin, zi=self.zi)
        return x * lin


class DeEsser:
    """分频式齿音抑制：4 阶 Butterworth 在 5 kHz 分频，只压高频带。

    变调后 s/sh 的整体上移约 1.6 倍，齿音会挤到 7~9 kHz 变成刺耳的"沙"，
    这是听感"像机器"的主要来源之一。这里只衰减高频带的瞬态，不动 5 kHz 以下，
    所以不会像全带压缩那样把字音压扁。
    """

    def __init__(self, fs, block, freq=5000.0):
        self.fs, self.block = fs, block
        self.ctl = _Control(fs, block)
        nyq = fs / 2.0
        self.sos_lp = butter(6, freq / nyq, btype="lowpass", output="sos")
        self.sos_hp = butter(6, freq / nyq, btype="highpass", output="sos")
        self.zi_lp = np.zeros((self.sos_lp.shape[0], 2))
        self.zi_hp = np.zeros((self.sos_hp.shape[0], 2))
        self.env = 0.0
        self.gr = 0.0
        dt = self.ctl.c / fs
        self.rel = math.exp(-dt / 0.050)          # 包络释放 50 ms
        self.att_g = 1 - math.exp(-dt / 0.0015)   # 起控 1.5 ms
        self.relg = 1 - math.exp(-dt / 0.060)     # 恢复 60 ms
        self.a = 1 - math.exp(-1.0 / (fs * 0.0015))
        self.zi = np.zeros(1)

    def reset(self):
        self.zi_lp[:] = 0
        self.zi_hp[:] = 0
        self.env = 0.0
        self.gr = 0.0
        self.zi[:] = 0

    def process(self, x, amount):
        if amount <= 0.5:
            self.gr *= 0.9
            return x
        lo, self.zi_lp = sosfilt(self.sos_lp, x, zi=self.zi_lp)
        hi, self.zi_hp = sosfilt(self.sos_hp, x, zi=self.zi_hp)
        pk = self.ctl.peaks(hi)
        thr = 0.050 - amount * 0.00030            # 0.05 .. 0.02，越大越灵敏
        maxgr = 3.0 + amount * 0.13               # 最多压 3 .. 16 dB
        out = np.empty_like(pk)
        for i, p in enumerate(pk):
            self.env = p if p > self.env else self.env * self.rel
            over = 20.0 * math.log10(max(self.env, 1e-9) / max(thr, 1e-6))
            target = 0.0 if over <= 0.0 else -min(maxgr, over * 0.95)
            coef = self.att_g if target < self.gr else self.relg
            self.gr += (target - self.gr) * coef
            out[i] = self.gr
        g = 10.0 ** (self.ctl.upsample(out) / 20.0)
        g, self.zi = lfilter([self.a], [1.0, self.a - 1.0], g, zi=self.zi)
        return (lo + hi * g).astype(np.float32)


class Limiter:
    def __init__(self, fs, block, ceiling_db=-0.8):
        self.fs, self.block = fs, block
        self.ceil = db2lin(ceiling_db)
        self.g = 1.0
        dt = block / fs
        self.rel = 1 - math.exp(-dt / 0.120)
        self.a = 1 - math.exp(-1.0 / (fs * 0.0015))
        self.zi = np.zeros(1)

    def process(self, x, enabled=True):
        if not enabled:
            return np.clip(x, -1.0, 1.0)
        peak = float(np.max(np.abs(x))) if len(x) else 0.0
        target = 1.0 if peak <= self.ceil else self.ceil / max(peak, EPS)
        if target < self.g:
            self.g = target
        else:
            self.g += (target - self.g) * self.rel
        if self.g < 0.999:
            g = np.full(len(x), self.g, dtype=np.float32)
            g, self.zi = lfilter([self.a], [1.0, self.a - 1.0], g, zi=self.zi)
            x = x * g
        return np.clip(x, -1.0, 1.0)


# --------------------------------------------------------------------------- #
#  混响（块级梳状 + 全通，零额外延迟）
# --------------------------------------------------------------------------- #
class _Delay:
    def __init__(self, delay, dtype=np.float32):
        self.delay = max(1, int(delay))
        self.buf = np.zeros(self.delay, dtype=dtype)

    def read(self, n):
        return self.buf[:n].copy()

    def push(self, new):
        n = len(new)
        self.buf = np.concatenate((self.buf[n:], new.astype(np.float32)))


class _Comb:
    def __init__(self, delay, g, damp):
        self.d = _Delay(delay)
        self.g = g
        self.damp = damp
        self.lp = np.zeros(1)

    def process(self, x):
        d = self.d.read(len(x))
        y, self.lp = lfilter([1.0 - self.damp], [1.0, -self.damp], d, zi=self.lp)
        self.d.push(x + self.g * y)
        return d


class _Allpass:
    def __init__(self, delay, g=0.5):
        self.d = _Delay(delay)
        self.g = g

    def process(self, x):
        d = self.d.read(len(x))
        y = d - self.g * x
        self.d.push(x + self.g * y)
        return y


class Reverb:
    COMB_SEC = (0.0297, 0.0371, 0.0411, 0.0437, 0.0503, 0.0587, 0.0641, 0.0713)
    AP_SEC = (0.0050, 0.0017)

    def __init__(self, fs, block):
        self.fs, self.block = fs, block
        self._size = None
        self._build(50)

    def _build(self, size):
        s = max(0.2, size / 100.0)
        scale = 0.62 + s * 0.95
        g = 0.72 + s * 0.155
        self.combs = [_Comb(max(self.block + 1, self.COMB_SEC[i] * scale * self.fs),
                            g - i * 0.004, 0.22) for i in range(len(self.COMB_SEC))]
        self.aps = [_Allpass(max(self.block + 1, t * scale * self.fs), 0.5) for t in self.AP_SEC]
        self._size = size

    def process(self, x, size, mix):
        if mix <= 0.5:
            return x
        if abs(size - self._size) > 3:
            self._build(size)
        wet = np.zeros_like(x, dtype=np.float32)
        for c in self.combs:
            wet += c.process(x)
        wet *= 1.0 / len(self.combs)
        for a in self.aps:
            wet = a.process(wet)
        m = min(max(mix / 100.0, 0.0), 1.0)
        return (1.0 - m) * x + m * wet * 1.6


# --------------------------------------------------------------------------- #
#  变调 + 共振峰
# --------------------------------------------------------------------------- #
class VoiceShifter:
    """STFT 频域搬移做变调，叠加谱包络替换做独立共振峰控制。

    每帧输入 hop 个样本，输出 hop 个样本；固定延迟一帧（n_fft 样本）。
    """

    def __init__(self, fs, n_fft=1024, hop=256):
        self.fs = fs
        self.n = n_fft
        self.hop = hop
        self.w = np.hanning(n_fft + 1)[:-1].astype(np.float64)
        self.k = np.arange(n_fft // 2 + 1)
        self.bin_omega = 2 * np.pi * self.k / n_fft
        self.freqs = self.k * fs / n_fft
        self.prev_phase = np.zeros_like(self.freqs)
        self.synth_phase = np.zeros_like(self.freqs)
        self.inbuf = np.zeros(n_fft)
        self.ola = np.zeros(n_fft)
        self.wsum = self._window_sum()
        ksz = 11
        ker = np.hanning(ksz + 2)[1:-1]
        self.env_ker = ker / ker.sum()

    def _window_sum(self):
        """稳态 OLA 增益：在远离两端的区域统计 Σw²（Hann + 75% 重叠 = 1.5）。"""
        frames = int(3 * self.n / self.hop) + 2
        tmp = np.zeros(4 * self.n)
        for i in range(frames):
            s = i * self.hop
            if s + self.n <= len(tmp):
                tmp[s: s + self.n] += self.w ** 2
        seg = tmp[3 * self.n: 3 * self.n + self.hop]
        return float(np.mean(seg))

    def _envelope(self, mag):
        lm = np.log(mag + 1e-6)
        pad = len(self.env_ker) // 2
        lm = np.pad(lm, pad, mode="edge")
        return np.convolve(lm, self.env_ker, mode="valid")

    def reset(self):
        self.prev_phase[:] = 0
        self.synth_phase[:] = 0
        self.inbuf[:] = 0
        self.ola[:] = 0

    def process(self, block, pitch_ratio=1.0, formant_ratio=1.0):
        hop, n = self.hop, self.n
        self.inbuf = np.concatenate((self.inbuf[hop:], block.astype(np.float64)))
        frame = self.inbuf * self.w
        X = np.fft.rfft(frame)
        mag = np.abs(X)
        ph = np.angle(X)

        identity = abs(pitch_ratio - 1.0) < 1e-4 and abs(formant_ratio - 1.0) < 1e-4
        if identity:
            self.prev_phase = ph
            out_frame = frame
        else:
            dphi = ph - self.prev_phase - self.hop * self.bin_omega
            dphi = np.mod(dphi + np.pi, 2 * np.pi) - np.pi
            self.prev_phase = ph
            omega = self.bin_omega + dphi / self.hop

            K = len(mag)
            tgt = self.k * pitch_ratio
            lo = np.floor(tgt).astype(np.int64)
            frac = tgt - lo
            valid = (lo >= 0) & (lo <= K - 2)

            mag_out = np.zeros(K)
            num = np.zeros(K)
            den = np.zeros(K)
            lv, fv, mv, ov = lo[valid], frac[valid], mag[valid], omega[valid]
            np.add.at(mag_out, lv, mv * (1 - fv))
            np.add.at(mag_out, lv + 1, mv * fv)
            np.add.at(num, lv, ov * pitch_ratio * (1 - fv))
            np.add.at(num, lv + 1, ov * pitch_ratio * fv)
            np.add.at(den, lv, (1 - fv))
            np.add.at(den, lv + 1, fv)
            omega_out = np.where(den > 1e-9, num / np.maximum(den, 1e-9), self.bin_omega)

            if abs(formant_ratio - 1.0) > 1e-4:
                env_in = self._envelope(mag)
                env_out = self._envelope(np.maximum(mag_out, 1e-9))
                env_tgt = np.interp(self.freqs / formant_ratio, self.freqs, env_in)
                gain_db = np.clip((env_tgt - env_out) * 8.685889638, -24.0, 24.0)
                mag_out = mag_out * (10.0 ** (gain_db / 20.0))

            # 频点搬移后相位不再相关，重建帧的能量会塌陷；按帧能量归一，保证响度一致
            e_in = float(np.dot(mag, mag))
            e_out = float(np.dot(mag_out, mag_out))
            if e_in > 1e-12 and e_out > 1e-12:
                mag_out = mag_out * min(4.0, max(0.25, math.sqrt(e_in / e_out)))

            self.synth_phase = np.mod(self.synth_phase + omega_out * self.hop + np.pi, 2 * np.pi) - np.pi
            spec = mag_out * np.exp(1j * self.synth_phase)
            out_frame = np.fft.irfft(spec, n)

        self.ola += out_frame * self.w
        y = (self.ola[:hop] / max(self.wsum, EPS)).astype(np.float32)
        self.ola = np.concatenate((self.ola[hop:], np.zeros(hop)))
        return y


# --------------------------------------------------------------------------- #
#  整链
# --------------------------------------------------------------------------- #
class VoiceChain:
    def __init__(self, fs=48000, block=256):
        self.fs, self.block = fs, block
        n_fft = 1024 if block <= 256 else 2 * block
        self.shifter = VoiceShifter(fs, n_fft=n_fft, hop=block)
        self.eq = EqChain(fs)
        self.deess = DeEsser(fs, block)
        self.gate = Gate(fs, block)
        self.comp = Compressor(fs, block)
        self.rev = Reverb(fs, block)
        self.lim = Limiter(fs, block)
        self.rng = np.random.default_rng(12345)
        self.robot_phase = 0.0
        self._nz_prev = 0.0
        self._sm = {}
        self._fade = 0.0
        self._agc_db = 0.0
        self._agc_coef = self.block / (fs * 1.2)
        t = np.arange(block, dtype=np.float64) / fs
        self._t = t

    def reset(self):
        """开流时调用：清空 STFT 状态并重新淡入，避免首帧爆音。"""
        self.shifter.reset()
        self.deess.reset()
        self._fade = 0.0
        self._agc_db = 0.0
        self._nz_prev = 0.0

    def _smooth(self, key, target, alpha=0.25):
        cur = self._sm.get(key)
        if cur is None:
            self._sm[key] = target
            return target
        cur += (target - cur) * alpha
        self._sm[key] = cur
        return cur

    def process(self, x, p):
        x = np.asarray(x, dtype=np.float32)
        self._fade = min(1.0, self._fade + self.block / (self.fs * 0.04))
        fade = self._fade
        if p.get("mute"):
            return np.zeros_like(x)

        bypass = bool(p.get("bypass"))
        if bypass:
            out = self.shifter.process(x, 1.0, 1.0)
            return np.clip(out * fade, -1.0, 1.0)

        ig = db2lin(self._smooth("ig", float(p.get("input_gain", 0.0))))
        x = x * ig
        in_rms = float(np.sqrt(np.mean(np.square(x))))

        x = self.gate.process(x, float(p.get("gate_db", -52)), bool(p.get("gate_on", True)))
        x = self.eq.process(x, float(p.get("hpf", 80)), float(p.get("lpf", 16000)),
                            float(p.get("eq_low", 0.0)), float(p.get("eq_mid", 0.0)),
                            float(p.get("eq_high", 0.0)))

        pr = semitones_to_ratio(float(p.get("pitch", 0.0)))
        fr = semitones_to_ratio(float(p.get("formant", 0.0)))
        x = self.shifter.process(x, pr, fr)
        # 齿音抑制紧跟变声：只处理"变调后变刺"的那一段，且不被后面的气声噪声误触发
        x = self.deess.process(x, self._smooth("deess", float(p.get("deess", 0))))

        drive = self._smooth("drive", float(p.get("drive", 0.0)))
        if drive > 0.5:
            g = 1.0 + drive * 0.055
            x = (np.tanh(x * g) / math.tanh(g)).astype(np.float32)

        robot = self._smooth("robot", float(p.get("robot", 0.0)))
        if robot > 0.5:
            f = float(p.get("robot_freq", 55))
            ph = (self.robot_phase + 2 * np.pi * f * self._t) % (2 * np.pi)
            self.robot_phase = float((self.robot_phase + 2 * np.pi * f * self.block / self.fs) % (2 * np.pi))
            x = x * (1 - robot / 100.0) + x * np.cos(ph).astype(np.float32) * (robot / 100.0) * 1.8

        whisper = self._smooth("whisper", float(p.get("whisper", 0.0)))
        if whisper > 0.5:
            env = np.abs(x)
            # 简单包络平滑（块内）
            a = 1 - math.exp(-self.block / (self.fs * 0.02))
            env = lfilter([a], [1.0, a - 1.0], env)
            # 一阶高通的白噪：真实气声集中在 1.5 kHz 以上，全频白噪只会把声音弄"糊"，
            # 听着像底噪而不像呼吸。
            raw = self.rng.standard_normal(len(x) + 1).astype(np.float32)
            raw[0] = self._nz_prev
            self._nz_prev = float(raw[-1])
            nz = np.diff(raw) * 2.2 * env
            m = whisper / 100.0
            x = x * (1 - m * 0.55) + nz * m

        x = self.comp.process(x, self._smooth("comp", float(p.get("comp", 0))))

        mix = self._smooth("rev", float(p.get("reverb_mix", 0)))
        x = self.rev.process(x, float(p.get("reverb_size", 50)), mix)

        # 响度对齐：变调后帧间相位不再相关，OLA 增益略低于理论值，加上各效果
        # 自身的增减，换预设时音量会飘。这里用一条 1.5 s 的慢速 AGC 把
        # “输出响度”拉回“输入响度”（限幅器施压时暂停积分，避免卷绕）。
        out_rms = float(np.sqrt(np.mean(np.square(x))))
        out_pk = float(np.max(np.abs(x))) if len(x) else 0.0
        if in_rms > 3e-4 and out_rms > 3e-5 and out_pk < 0.92:
            err = lin2db(in_rms) - lin2db(out_rms) - self._agc_db
            self._agc_db = min(10.0, max(-6.0, self._agc_db + err * self._agc_coef))
        x = x * db2lin(self._agc_db)

        og = db2lin(self._smooth("og", float(p.get("output_gain", 0))))
        x = x * og
        return self.lim.process(x, bool(p.get("limiter", True))) * fade
