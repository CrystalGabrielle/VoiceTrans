"""本地控制台：静态页面 + JSON API（纯标准库，不额外装 web 框架）。"""
from __future__ import annotations

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from . import params as P
from .engine import AudioEngine, has_virtual_cable, list_devices, pick_defaults

WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")


class App:
    def __init__(self, cfg=None):
        self.cfg = cfg or P.load_config()
        if not self.cfg.get("input_device") and not self.cfg.get("output_device"):
            d_in, d_out = pick_defaults()
            self.cfg["input_device"] = d_in
            self.cfg["output_device"] = d_out
        self.store = P.ParamStore(self.cfg.get("params"))
        self.engine = AudioEngine(self.store, int(self.cfg["samplerate"]), int(self.cfg["blocksize"]))
        self.last_preset = self.cfg.get("last_preset") or "原声直通"
        self._save_lock = threading.Lock()
        self._dirty = False
        self._saver = threading.Thread(target=self._save_loop, daemon=True)
        self._saver.start()

    # ---------------- 配置持久化 ----------------
    def touch(self):
        self._dirty = True

    def _save_loop(self):
        while True:
            time.sleep(1.2)
            if self._dirty:
                self._dirty = False
                self.save()

    def save(self):
        with self._save_lock:
            self.cfg["params"] = self.store.snapshot()
            self.cfg["input_device"] = self.engine.in_dev or self.cfg.get("input_device")
            self.cfg["output_device"] = self.engine.out_dev or self.cfg.get("output_device")
            self.cfg["samplerate"] = self.engine.fs
            self.cfg["blocksize"] = self.engine.block
            self.cfg["last_preset"] = self.last_preset
            P.save_config(self.cfg)

    # ---------------- API ----------------
    def boot(self):
        self.engine.start(self.cfg.get("input_device"), self.cfg.get("output_device"))

    def state(self):
        st = self.engine.status()
        st["values"] = self.store.snapshot()
        st["preset"] = self.last_preset
        st["port"] = int(self.cfg.get("ui_port", 8765))
        return st

    def set_param(self, key, value):
        ok = self.store.set(key, value)
        if ok:
            if key in ("pitch", "formant", "drive", "robot", "whisper", "reverb_mix", "comp"):
                self.last_preset = ""
            self.touch()
        return {"ok": ok, "values": self.store.snapshot(), "preset": self.last_preset}

    def apply_preset(self, name):
        pr = P.PRESET_BY_NAME.get(name)
        if not pr:
            return {"ok": False, "error": "unknown preset"}
        self.store.update(pr["values"])
        self.last_preset = name
        self.touch()
        return {"ok": True, "values": self.store.snapshot(), "preset": name}

    def set_devices(self, in_dev, out_dev, samplerate=None, blocksize=None):
        if samplerate:
            self.cfg["samplerate"] = int(samplerate)
        if blocksize:
            self.cfg["blocksize"] = int(blocksize)
            self.engine.block = int(blocksize)
            self.engine.chain = type(self.engine.chain)(self.engine.fs, self.engine.block)
        was_running = self.engine.running
        ok = self.engine.start(in_dev, out_dev)
        if not ok and was_running:
            self.engine.start(self.cfg.get("input_device"), self.cfg.get("output_device"))
        self.cfg["input_device"] = in_dev if ok else self.cfg.get("input_device")
        self.cfg["output_device"] = out_dev if ok else self.cfg.get("output_device")
        self.touch()
        return {"ok": ok, "error": self.engine.error, "state": self.state()}

    def control(self, action):
        if action == "start":
            ok = self.engine.start(self.engine.in_dev or self.cfg.get("input_device"),
                                   self.engine.out_dev or self.cfg.get("output_device"))
            return {"ok": ok, "error": self.engine.error, "state": self.state()}
        if action == "stop":
            self.engine.stop()
            return {"ok": True, "state": self.state()}
        if action == "record_start":
            path = self.engine.record_start()
            return {"ok": bool(path), "path": path, "state": self.state()}
        if action == "record_stop":
            path = self.engine.record_stop()
            return {"ok": True, "path": path, "state": self.state()}
        if action == "reset":
            self.store.update(P.DEFAULTS)
            self.last_preset = "原声直通"
            self.touch()
            return {"ok": True, "values": self.store.snapshot(), "state": self.state()}
        return {"ok": False, "error": "unknown action"}


class _Handler(BaseHTTPRequestHandler):
    server_version = "VoiceTrans/1.0"
    app: App = None

    def log_message(self, fmt, *args):
        pass

    # ---------- helpers ----------
    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path, ctype):
        try:
            with open(path, "rb") as f:
                data = f.read()
        except OSError:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            return {}
        try:
            return json.loads(self.rfile.read(n).decode("utf-8"))
        except Exception:
            return {}

    # ---------- routes ----------
    def do_GET(self):
        path = urlparse(self.path).path
        app = self.app
        if path in ("/", "/index.html"):
            self._file(os.path.join(WEB_DIR, "index.html"), "text/html; charset=utf-8")
        elif path == "/api/schema":
            data = P.schema()
            data["devices"] = list_devices()
            data["config"] = {
                "input_device": app.engine.in_dev or app.cfg.get("input_device"),
                "output_device": app.engine.out_dev or app.cfg.get("output_device"),
                "samplerate": app.engine.fs,
                "blocksize": app.engine.block,
                "last_preset": app.last_preset,
            }
            data["virtual_cable"] = has_virtual_cable()
            self._json(data)
        elif path == "/api/state":
            self._json(app.state())
        else:
            self.send_error(404)

    def do_POST(self):
        path = urlparse(self.path).path
        app = self.app
        body = self._body()
        if path == "/api/set":
            self._json(app.set_param(body.get("key"), body.get("value")))
        elif path == "/api/preset":
            self._json(app.apply_preset(body.get("name")))
        elif path == "/api/devices":
            self._json(app.set_devices(body.get("input"), body.get("output"),
                                       body.get("samplerate"), body.get("blocksize")))
        elif path == "/api/control":
            self._json(app.control(body.get("action")))
        else:
            self.send_error(404)


def serve(app: App, port=8765):
    handler = type("BoundHandler", (_Handler,), {"app": app})
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
    httpd.daemon_threads = True
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    return httpd
