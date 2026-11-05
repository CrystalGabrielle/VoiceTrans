"""录音分析：音高分布、共振峰、频谱分布、周期性。用于调参前摸清嗓音特征。

用法: python tools/analyze_voice.py <wav> [--pitch-semi 6] [--formant-semi 3.5]
      给出 --pitch-semi 时，会反推出原始嗓音的估算值（因为录音点在输出端）。
"""
import sys
import wave

import numpy as np
from scipy.linalg import solve_toeplitz
from scipy.signal import periodogram, resample_poly

BANDS = [125, 250, 500, 1000, 2000, 3000, 4000, 6000, 8000, 10000]


def load(path):
    with wave.open(path, "rb") as w:
        fs = w.getframerate()
        n = w.getnframes()
        raw = w.readframes(n)
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0, fs


def f0_track(x, fs, fmin=70.0, fmax=450.0, win=2048, hop=256):
    out = []
    for i in range(0, len(x) - win, hop):
        s = x[i:i + win].astype(np.float64)
        if np.sqrt(np.mean(s ** 2)) < 0.004:
            out.append(np.nan)
            continue
        s = s - s.mean()
        s = s * np.hanning(len(s))
        ac = np.correlate(s, s, "full")[len(s) - 1:]
        ac = ac / (ac[0] + 1e-12)
        lo, hi = int(fs / fmax), int(fs / fmin)
        k = lo + int(np.argmax(ac[lo:hi]))
        out.append(fs / k if ac[k] > 0.35 else np.nan)
    return np.array(out)


def lpc_formants(seg, fs, order=18):
    s = seg * np.hanning(len(seg))
    s = np.append(s[0], s[1:] - 0.97 * s[:-1])
    r = np.correlate(s, s, "full")[len(s) - 1:len(s) + order]
    a = solve_toeplitz(r[:order], -r[1:order + 1])
    a = np.concatenate(([1.0], a))
    roots = np.roots(a)
    roots = roots[np.imag(roots) > 0]
    fr = np.angle(roots) * fs / (2 * np.pi)
    bw = -0.5 * (fs / (2 * np.pi)) * np.log(np.abs(roots))
    ok = (fr > 200) & (fr < 4500) & (bw < 500)
    return np.sort(fr[ok])


def formant_track(x, fs, win=1024, hop=256, floor=0.01):
    """降采样到 8 kHz 再用 12 阶 LPC：48 kHz 直算需要很高阶数，否则 F1/F2 会被抬高。"""
    fs2 = 8000
    y = resample_poly(x.astype(np.float64), fs2, fs)
    step = max(1, int(round(fs / fs2)))
    F = []
    for i in range(0, len(x) - win, hop):
        seg = x[i:i + win]
        if np.sqrt(np.mean(seg ** 2)) < floor:
            continue
        j0 = i // step
        jn = win // step
        if j0 + jn > len(y):
            break
        f = lpc_formants(y[j0:j0 + jn], fs2, order=12)
        if len(f) >= 3:
            F.append(f[:3])
    return np.array(F)


def report(path, pitch_semi=None, formant_semi=None):
    x, fs = load(path)
    print(f"文件 {path}")
    print(f"  采样 {fs} Hz  时长 {len(x) / fs:.2f}s  峰值 {20 * np.log10(np.max(np.abs(x)) + 1e-9):.1f} dBFS")
    seg = x[x != 0]
    print(f"  有效 RMS {20 * np.log10(np.sqrt(np.mean(seg ** 2)) + 1e-9):.1f} dBFS")

    f0 = f0_track(x, fs)
    v = f0[~np.isnan(f0)]
    print(f"  f0  浊音帧 {len(v)}/{len(f0)}  中位 {np.median(v):.1f} Hz"
          f"  p10 {np.percentile(v, 10):.0f}  p90 {np.percentile(v, 90):.0f}")

    F = formant_track(x, fs)
    med = np.median(F, axis=0)
    print(f"  共振峰 F1 {med[0]:.0f}  F2 {med[1]:.0f}  F3 {med[2]:.0f} Hz  (帧 {len(F)})")

    f, P = periodogram(x * np.hanning(len(x)), fs)
    tot = np.sum(P)
    parts = []
    for b in BANDS:
        e = np.sum(P[(f >= b / 1.12) & (f < b * 1.12)]) / tot
        parts.append(f"{b}Hz {10 * np.log10(e + 1e-12):+.1f}")
    print("  LTAS 1/3oct: " + "  ".join(parts))

    hn = []
    for i in range(0, len(x) - 2048, 1024):
        s = x[i:i + 2048].astype(np.float64)
        if np.sqrt(np.mean(s ** 2)) < 0.01:
            continue
        s = s - s.mean()
        ac = np.correlate(s, s, "full")[len(s) - 1:]
        ac = ac / (ac[0] + 1e-12)
        lo, hi = int(fs / 450), int(fs / 70)
        k = lo + int(np.argmax(ac[lo:hi]))
        hn.append(ac[k])
    print(f"  周期性峰值 {np.median(hn):.2f}   峰值因数 "
          f"{20 * np.log10(np.max(np.abs(x)) / np.sqrt(np.mean(x ** 2))):.1f} dB")

    if pitch_semi is not None:
        pr = 2 ** (pitch_semi / 12)
        fr = 2 ** ((formant_semi or 0) / 12)
        print("  —— 按当前参数反推原始嗓音 ——")
        print(f"  原始 f0 约 {np.median(v) / pr:.1f} Hz"
              f"（音高 {pitch_semi:+.1f} 半音 = ×{pr:.3f}）")
        print(f"  原始 F1/F2/F3 约 {med[0] / fr:.0f} / {med[1] / fr:.0f} / {med[2] / fr:.0f} Hz"
              f"（音色 {formant_semi:+.1f} 半音 = ×{fr:.3f}）")
    return {"f0": np.median(v), "formants": med, "fs": fs}


if __name__ == "__main__":
    args = sys.argv[1:]
    p = args[0] if args else "recordings/rec_20261105_230010.wav"
    ps = fs_ = None
    if "--pitch-semi" in args:
        ps = float(args[args.index("--pitch-semi") + 1])
    if "--formant-semi" in args:
        fs_ = float(args[args.index("--formant-semi") + 1])
    report(p, ps, fs_)
