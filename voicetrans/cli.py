"""命令行入口：run / list / test。"""
from __future__ import annotations

import argparse
import sys
import time
import webbrowser

import numpy as np

from . import params as P
from . import ui as UI
from .engine import AudioEngine, has_virtual_cable, list_devices, pick_defaults


def _say(msg=""):
    try:
        sys.stdout.write(str(msg) + "\n")
        sys.stdout.flush()
    except Exception:
        pass


def cmd_list():
    devs = list_devices()
    _say("可用音频设备：")
    for d in devs:
        flags = []
        if d["is_default_in"]:
            flags.append("默认输入")
        if d["is_default_out"]:
            flags.append("默认输出")
        if d["virtual"]:
            flags.append("虚拟声卡")
        tag = ("  [" + "/".join(flags) + "]") if flags else ""
        _say(f"  {d['index']:>2}  {d['hostapi']:<18} in={d['in']} out={d['out']} "
             f"{int(d['sr'])}Hz  {d['name']}{tag}")
    d_in, d_out = pick_defaults()
    _say("")
    _say(f"自动推荐：麦克风 = {d_in}，输出 = {d_out}")
    _say(f"虚拟声卡：{'已安装' if has_virtual_cable() else '未安装（直播进流需要，见 README）'}")


def _run_signal(fs, block, p, seconds=2.0):
    from .dsp import VoiceChain
    t = np.arange(int(fs * seconds)) / fs
    x = sum(np.sin(2 * np.pi * 180 * h * t) / h for h in range(1, 10))
    x = (0.22 * x / np.max(np.abs(x))).astype(np.float32)
    ch = VoiceChain(fs, block)
    out = []
    t0 = time.perf_counter()
    for i in range(0, len(x) - block + 1, block):
        out.append(ch.process(x[i:i + block], p))
    dt = time.perf_counter() - t0
    return x, np.concatenate(out), dt


def _peak_hz(y, fs):
    y = y[len(y) // 2:]
    sp = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    f = np.fft.rfftfreq(len(y), 1 / fs)
    lo = f > 60
    return float(f[lo][np.argmax(sp[lo])])


def _f0_hz(y, fs, fmin=60.0, fmax=500.0):
    """自相关基频检测：共振峰变化不会影响它，适合验证“只改音色不改音高”。"""
    y = np.asarray(y[len(y) // 3:], dtype=np.float64)
    y = y - y.mean()
    n = len(y)
    spec = np.fft.rfft(y, 2 * n)
    ac = np.fft.irfft(spec * np.conj(spec))[:n]
    if ac[0] <= 0:
        return 0.0
    ac = ac / ac[0]
    lo, hi = int(fs / fmax), min(int(fs / fmin), n - 1)
    lag = lo + int(np.argmax(ac[lo:hi]))
    return fs / max(lag, 1)


def _centroid_hz(y, fs):
    y = y[len(y) // 2:].astype(np.float64)
    sp = np.abs(np.fft.rfft(y * np.hanning(len(y)))) + 1e-12
    f = np.fft.rfftfreq(len(y), 1 / fs)
    return float(np.sum(f * sp) / np.sum(sp))


def cmd_test():
    fs, block = 48000, 256
    ok_all = True
    _say("== 1. 变调精度 ==")
    for semi in (-7.0, -3.0, 0.0, 3.0, 6.0, 12.0):
        p = dict(P.DEFAULTS)
        p["pitch"] = semi
        p["comp"] = 0
        _, y, _ = _run_signal(fs, block, p)
        got = _peak_hz(y, fs)
        want = 180.0 * 2 ** (semi / 12.0)
        err = abs(got - want) / want * 100
        ok = err < 2.0
        ok_all &= ok
        _say(f"  {'OK ' if ok else 'FAIL'} 音高 {semi:+.1f} 半音 → 期望 {want:7.1f} Hz，实测 {got:7.1f} Hz（偏差 {err:.2f}%）")

    _say("== 2. 共振峰独立控制（只改音色，不改音高）==")
    p0 = dict(P.DEFAULTS)
    p0["comp"] = 0
    _, y0, _ = _run_signal(fs, block, p0)
    p1 = dict(P.DEFAULTS)
    p1["comp"] = 0
    p1["formant"] = 6.0
    _, y1, _ = _run_signal(fs, block, p1)
    f0_0, f0_1 = _f0_hz(y0, fs), _f0_hz(y1, fs)
    c0, c1 = _centroid_hz(y0, fs), _centroid_hz(y1, fs)
    ok = abs(f0_0 - f0_1) < 3.0 and c1 > c0 * 1.12
    ok_all &= ok
    _say(f"  {'OK ' if ok else 'FAIL'} 基频 {f0_0:.1f} → {f0_1:.1f} Hz（应不变）；"
         f"谱重心 {c0:.0f} → {c1:.0f} Hz（应上升）")

    _say("== 3. 各预设性能与稳定性 ==")
    for pr in P.PRESETS:
        _, y, dt = _run_signal(fs, block, pr["values"])
        rtf = dt / 2.0
        bad = np.isnan(y).any() or np.max(np.abs(y)) > 1.001
        ok = (rtf < 0.6) and not bad
        ok_all &= ok
        _say(f"  {'OK ' if ok else 'FAIL'} {pr['name']:<6} CPU {rtf * 100:5.1f}%  峰值 {np.max(np.abs(y)):.2f}  "
             f"RMS {20 * np.log10(np.sqrt(np.mean(y ** 2)) + 1e-9):6.1f} dBFS")

    _say("== 4. 设备可打开性（静音测试 0.6 秒）==")
    d_in, d_out = pick_defaults()
    if d_in is None or d_out is None:
        _say("  FAIL 找不到可用设备")
        ok_all = False
    else:
        st = P.ParamStore({"mute": True})
        eng = AudioEngine(st, fs, block)
        ok = eng.start(d_in, d_out)
        if ok:
            time.sleep(0.6)
        s = eng.status()
        eng.stop()
        ok = ok and not s["error"]
        ok_all &= ok
        _say(f"  {'OK ' if ok else 'FAIL'} 输入 [{s['in_name']}] → 输出 [{s['out_name']}]，"
             f"{s['fs']} Hz / {s['block']} 采样，延迟约 {s['latency_ms']} ms，CPU {s['cpu']}%")

    _say("")
    _say("全部通过 ✔" if ok_all else "存在未通过项 ✘")
    return 0 if ok_all else 1


def cmd_run(args):
    cfg = P.load_config()
    if args.input is not None:
        cfg["input_device"] = args.input
    if args.output is not None:
        cfg["output_device"] = args.output
    if args.port:
        cfg["ui_port"] = args.port

    app = UI.App(cfg)
    app.boot()
    st = app.engine.status()
    if st["error"]:
        _say(f"⚠ 音频启动失败：{st['error']}")
        _say("  控制台仍会打开，可在“设备”里换一个输入/输出再应用。")

    port = int(cfg.get("ui_port", 8765))
    httpd = None
    for cand in range(port, port + 25):
        try:
            httpd = UI.serve(app, cand)
            port = cand
            break
        except OSError:
            continue
    if httpd is None:
        _say("✘ 端口全部被占用，无法启动控制台")
        return 1

    url = f"http://127.0.0.1:{port}/"
    _say("")
    _say("  VOICE TRANS · 本地实时变声台")
    _say(f"  控制台   {url}")
    _say(f"  麦克风   {st['in_name'] or '—'}")
    _say(f"  输出     {st['out_name'] or '—'}")
    _say(f"  延迟     {st['latency_ms']} ms   CPU {st['cpu']}%")
    _say(f"  虚拟声卡 {'已安装' if has_virtual_cable() else '未安装（直播进流请见 README）'}")
    _say("")
    _say("  关闭这个黑窗口即结束；Ctrl+C 也可以。")
    _say("")

    if not args.no_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    try:
        while True:
            time.sleep(0.6)
    except KeyboardInterrupt:
        pass
    finally:
        _say("正在退出…")
        app.engine.stop()
        app.save()
        try:
            httpd.shutdown()
        except Exception:
            pass
    return 0


def main(argv=None):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(prog="voicetrans", description="本地实时直播变声系统")
    sub = ap.add_subparsers(dest="cmd")
    pr = sub.add_parser("run", help="启动变声器与控制台")
    pr.add_argument("--port", type=int, default=None)
    pr.add_argument("--input", type=int, default=None, help="输入设备序号（python -m voicetrans list 查看）")
    pr.add_argument("--output", type=int, default=None, help="输出设备序号")
    pr.add_argument("--no-browser", action="store_true")
    sub.add_parser("list", help="列出音频设备")
    sub.add_parser("test", help="自检：变调精度 / 性能 / 设备")
    args = ap.parse_args(argv)

    if args.cmd == "list":
        cmd_list()
        return 0
    if args.cmd == "test":
        return cmd_test()
    return cmd_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
