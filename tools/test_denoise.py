# -*- coding: utf-8 -*-
"""离线验证人声降噪：合成谐波人声 + 稳态底噪，对比开/关降噪的输出。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from voicetrans.dsp import VoiceChain
from voicetrans.params import DEFAULTS

FS, BLOCK = 48000, 256
rng = np.random.default_rng(7)

def make_signal(sec):
    """1.6s 纯噪声 + 2.4s 人声(谐波+颤音+音节包络)混 10dB 底噪。"""
    n = int(sec * FS)
    t = np.arange(n) / FS
    f0 = 120 * (1 + 0.02 * np.sin(2 * np.pi * 5 * t))       # 基频+颤音
    ph = 2 * np.pi * np.cumsum(f0) / FS
    voice = sum(np.sin(ph * h) / h for h in (1, 2, 3, 4, 5))
    syll = 0.55 + 0.45 * np.sign(np.sin(2 * np.pi * 2.5 * t))  # 2.5Hz 音节
    syll = np.clip(syll, 0, 1) * (0.7 + 0.3 * rng.random(n).repeat(1))
    voice *= syll
    noise = rng.standard_normal(n) * 0.08
    sig = np.zeros(n)
    sig[: int(1.6 * FS)] = noise[: int(1.6 * FS)]            # 前 1.6s 只有噪声
    sig[int(1.6 * FS):] = voice[int(1.6 * FS):] * 0.18 + noise[int(1.6 * FS):]
    return sig, n

def run(sig, n, denoise_on):
    chain = VoiceChain(FS, BLOCK)
    p = dict(DEFAULTS)
    p.update(pitch=8.5, formant=3.2, comp=16, deess=40, whisper=3,
             denoise_on=denoise_on, denoise=45)
    out = np.zeros(n)
    for i in range(0, n - BLOCK, BLOCK):
        out[i:i + BLOCK] = chain.process(sig[i:i + BLOCK], p)
    return out

def rms_db(x, a, b):
    seg = x[a:b]
    return 20 * np.log10(np.sqrt(np.mean(seg ** 2)) + 1e-12)

sig, n = make_signal(4.0)
off = run(sig, n, False)
on = run(sig, n, True)

# 各段：0.9-1.5s 纯噪声稳态（避开学习期），1.8-3.9s 人声
nz_off, nz_on = rms_db(off, int(0.9 * FS), int(1.5 * FS)), rms_db(on, int(0.9 * FS), int(1.5 * FS))
vc_off, vc_on = rms_db(off, int(1.8 * FS), int(3.9 * FS)), rms_db(on, int(1.8 * FS), int(3.9 * FS))
print(f"噪声段电平: 关 {nz_off:6.1f} dB -> 开 {nz_on:6.1f} dB   (压低 {nz_off - nz_on:5.1f} dB)")
print(f"人声段电平: 关 {vc_off:6.1f} dB -> 开 {vc_on:6.1f} dB   (变化 {vc_on - vc_off:+5.1f} dB)")
print(f"信噪比改善: {(nz_off - nz_on) - (vc_on - vc_off):.1f} dB")
# 峰值检查，确认无爆音
print(f"输出峰值: 开降噪 {np.max(np.abs(on)):.3f} / 关 {np.max(np.abs(off)):.3f}")
