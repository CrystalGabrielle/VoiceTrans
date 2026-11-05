"""离线调参台：拿一条录音当输入，用真实 DSP 链跑多组参数，导出可试听 WAV + 客观指标。

录音点是输出端（录的是变声后的声音），先用当前预设反变换回近似原声，
再当输入跑候选参数。反变换只为拿回嗓音特征，不追求波形保真。

用法: python tools/tune.py <wav> --pitch-semi 6 --formant-semi 3.5
"""
import argparse
import os
import sys
import wave

import numpy as np
from scipy.linalg import solve_toeplitz
from scipy.signal import periodogram, resample_poly

sys.path.insert(0, r"F:\create\voice-trans")
from voicetrans.dsp import VoiceChain                       # noqa: E402
from voicetrans.params import DEFAULTS, PRESET_BY_NAME      # noqa: E402

OUT_DIR = os.path.join(r"F:\create\voice-trans", "recordings", "tune")
BANDS = ((150, 400), (400, 1000), (1000, 2500), (2500, 5000), (5000, 12000))


def load(path):
    with wave.open(path, "rb") as w:
        fs = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
    return x, fs


def save(path, x, fs):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(fs)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())


def run(x, fs, params, block=256):
    ch = VoiceChain(fs, block)
    out = [ch.process(x[i:i + block], params) for i in range(0, len(x) - block + 1, block)]
    return np.concatenate(out) if out else np.zeros(0, np.float32)


def f0_median(x, fs, fmin=70, fmax=450, win=2048, hop=256):
    vals = []
    for i in range(0, len(x) - win, hop):
        s = x[i:i + win].astype(np.float64)
        if np.sqrt(np.mean(s ** 2)) < 0.004:
            continue
        s = (s - s.mean()) * np.hanning(len(s))
        ac = np.correlate(s, s, "full")[len(s) - 1:]
        ac /= (ac[0] + 1e-12)
        lo, hi = int(fs / fmax), int(fs / fmin)
        k = lo + int(np.argmax(ac[lo:hi]))
        if ac[k] > 0.35:
            vals.append(fs / k)
    return float(np.median(vals)) if vals else 0.0


def bands(x, fs):
    f, P = periodogram(x * np.hanning(len(x)), fs)
    tot = np.sum(P) + 1e-15
    return [10 * np.log10(np.sum(P[(f >= a) & (f < b)]) / tot + 1e-12) for a, b in BANDS]


def centroid(x, fs):
    f, P = periodogram(x * np.hanning(len(x)), fs)
    return float(np.sum(f * P) / (np.sum(P) + 1e-15))


def formants(x, fs):
    fs2 = 8000
    y = resample_poly(x.astype(np.float64), fs2, fs)
    F = []
    for i in range(0, len(y) - 400, 200):
        seg = y[i:i + 400]
        if np.sqrt(np.mean(seg ** 2)) < 0.01:
            continue
        s = seg * np.hanning(len(seg))
        s = np.append(s[0], s[1:] - 0.97 * s[:-1])
        order = 12
        r = np.correlate(s, s, "full")[len(s) - 1:len(s) + order]
        aa = np.concatenate(([1.0], solve_toeplitz(r[:order], -r[1:order + 1])))
        roots = np.roots(aa)
        roots = roots[np.imag(roots) > 0]
        fr = np.angle(roots) * fs2 / (2 * np.pi)
        bw = -0.5 * (fs2 / (2 * np.pi)) * np.log(np.abs(roots))
        ok = (fr > 200) & (fr < 3500) & (bw < 500)
        fr = np.sort(fr[ok])
        if len(fr) >= 2:
            F.append(fr[:2])
    if not F:
        return 0.0, 0.0
    F = np.array(F)
    return float(np.median(F[:, 0])), float(np.median(F[:, 1]))


def hnr(x, fs, win=2048, hop=512):
    """自相关峰值近似谐噪比，用来监控变调引入的粗糙感。"""
    vals = []
    for i in range(0, len(x) - win, hop):
        s = x[i:i + win].astype(np.float64)
        if np.sqrt(np.mean(s ** 2)) < 0.01:
            continue
        s = (s - s.mean()) * np.hanning(len(s))
        ac = np.correlate(s, s, "full")[len(s) - 1:]
        ac /= (ac[0] + 1e-12)
        lo, hi = int(fs / 450), int(fs / 70)
        k = lo + int(np.argmax(ac[lo:hi]))
        vals.append(ac[k])
    return float(np.median(vals)) if vals else 0.0


def make_dry(wet, fs, pitch_semi, formant_semi, block):
    """反变换：音高、音色都反向，去掉门限/压缩/混响，加 80 Hz 高通防止低频堆积。"""
    p = dict(DEFAULTS)
    p.update(pitch=-pitch_semi, formant=-formant_semi, gate_on=False, comp=0,
             reverb_mix=0, drive=0, robot=0, whisper=0, eq_low=0, eq_mid=0,
             eq_high=0, hpf=80.0, lpf=20000.0, limiter=False, output_gain=0.0)
    y = run(wet, fs, p, block)
    return y / (np.max(np.abs(y)) + 1e-9) * 0.45


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--pitch-semi", type=float, default=6.0)
    ap.add_argument("--formant-semi", type=float, default=3.5)
    ap.add_argument("--block", type=int, default=256)
    a = ap.parse_args()

    wet, fs = load(a.wav)
    print(f"输入 {os.path.basename(a.wav)}  {len(wet) / fs:.2f}s")
    dry = make_dry(wet, fs, a.pitch_semi, a.formant_semi, a.block)
    save(os.path.join(OUT_DIR, "00_原声_估计.wav"), dry, fs)
    print(f"近似原声 f0 {f0_median(dry, fs):.1f} Hz")

    base = dict(DEFAULTS)
    base.update(PRESET_BY_NAME["甜美女声"]["values"])

    cands = [
        ("01_现状", {}),
        ("02_A_推荐", {"pitch": 8.5, "formant": 3.2, "eq_low": -4.0, "eq_mid": 3.0,
                       "eq_high": 5.0, "hpf": 140, "comp": 25, "reverb_mix": 8,
                       "output_gain": 1.5}),
        ("03_B_保守", {"pitch": 7.5, "formant": 3.6, "eq_low": -2.0, "eq_mid": 2.0,
                       "eq_high": 3.5, "hpf": 120, "comp": 30, "reverb_mix": 10,
                       "output_gain": 2.0}),
        ("04_C_更幼", {"pitch": 9.5, "formant": 2.9, "eq_low": -5.0, "eq_mid": 4.0,
                       "eq_high": 5.5, "hpf": 150, "comp": 22, "reverb_mix": 5,
                       "output_gain": 1.0, "whisper": 10}),
    ]

    head = f"{'名称':<12}{'f0':>6}{'F1':>6}{'F2':>6}{'重心':>7}"
    head += "".join(f"{f'{a0//1000}k-{b0//1000}k' if b0 >= 1000 else f'{a0}-{b0}':>10}"
                    for a0, b0 in BANDS)
    head += f"{'HNR':>7}{'RMS':>7}"
    print("\n" + head)
    for name, over in cands:
        p = dict(base)
        p.update(over)
        y = run(dry, fs, p, a.block)
        save(os.path.join(OUT_DIR, name + ".wav"), y, fs)
        f1, f2 = formants(y, fs)
        be = bands(y, fs)
        rms = 20 * np.log10(np.sqrt(np.mean(y ** 2)) + 1e-9)
        print(f"{name:<12}{f0_median(y, fs):6.1f}{f1:6.0f}{f2:6.0f}{centroid(y, fs):7.0f}"
              + "".join(f"{v:10.1f}" for v in be) + f"{hnr(y, fs):7.2f}{rms:7.1f}")

    print(f"\n试听文件在 {OUT_DIR}")


if __name__ == "__main__":
    main()
