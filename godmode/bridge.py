"""
Dashboard Bridge
================
Zero-dependency local server (Python stdlib only).

  python bridge.py          -> http://localhost:8765

Routes:
  GET  /                   -> god_dashboard.html
  GET  /api/status         -> god_data.json  (written by mt5_bot_v2.py)
  GET  /api/config         -> strategy_config.json
  POST /api/config         -> update strategy_config.json (hot-reloaded by bot)
  POST /api/cmd            -> queue cmd.json  {id, action: buy|sell|close_all|status, lots}
  GET  /api/cmd-result     -> cmd_result.json (bot's execution result)
  GET  /api/log            -> last lines of bot_live.log (the bot's "thoughts")
  POST /api/start          -> remove kill flag + reset profit cycle (Start button)
  POST /api/stop           -> create watchdog-disable.flag (watchdog stops bot+bridge,
                              both idle until the flag is deleted = START BOT)
"""

import json
import os
import time
from http.server import HTTPServer, BaseHTTPRequestHandler

BASE = os.path.dirname(os.path.abspath(__file__))
DASHBOARD = os.path.join(BASE, "god_dashboard.html")
DATA_PATH = os.path.join(BASE, "god_data.json")
CONFIG_PATH = os.path.join(BASE, "strategy_config.json")
CMD_PATH = os.path.join(BASE, "cmd.json")
CMD_RESULT_PATH = os.path.join(BASE, "cmd_result.json")
CMD_ACTIONS = {"buy", "sell", "close_all", "status"}
LOG_PATH = os.path.join(BASE, "bot_live.log")
WATCHDOG_FLAG = os.path.join(BASE, "watchdog-disable.flag")
PROFIT_GUARD = os.path.join(BASE, "profit_guard.json")


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj):
        self._send(code, json.dumps(obj).encode(), "application/json")

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            if os.path.exists(DASHBOARD):
                with open(DASHBOARD, "rb") as f:
                    self._send(200, f.read(), "text/html; charset=utf-8")
            else:
                self._send(404, b"god_dashboard.html missing", "text/plain")
        elif self.path == "/api/status":
            if os.path.exists(DATA_PATH):
                try:
                    with open(DATA_PATH, "rb") as f:
                        body = f.read()
                    json.loads(body)
                    self._send(200, body, "application/json")
                except Exception:
                    self._json(503, {"error": "god_data.json being written"})
            else:
                self._json(503, {"error": "no data yet - start mt5_bot_v2.py"})
        elif self.path == "/api/log":
            out = {"lines": []}
            if os.path.exists(LOG_PATH):
                try:
                    with open(LOG_PATH, "rb") as f:
                        f.seek(0, os.SEEK_END)
                        size = f.tell()
                        f.seek(max(0, size - 65536))
                        data = f.read().decode("utf-8", "replace")
                    out["lines"] = data.splitlines()[-100:]
                except OSError:
                    pass
            self._json(200, out)
        elif self.path == "/api/cmd-result":
            if os.path.exists(CMD_RESULT_PATH):
                try:
                    with open(CMD_RESULT_PATH, "rb") as f:
                        body = f.read()
                    json.loads(body)
                    self._send(200, body, "application/json")
                except Exception:
                    self._json(503, {"error": "cmd_result.json being written"})
            else:
                self._json(200, {"pending": True})
        elif self.path == "/api/stop":
            self._json(405, {"error": "use POST /api/stop to STOP the bot (watchdog halt)"})
        elif self.path == "/api/config":
            if os.path.exists(CONFIG_PATH):
                with open(CONFIG_PATH, "rb") as f:
                    self._send(200, f.read(), "application/json")
            else:
                self._json(404, {"error": "no config"})
        else:
            self._json(404, {"error": "not found"})

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_POST(self):
        if self.path == "/api/stop":
            try:
                with open(WATCHDOG_FLAG, "w") as f:
                    f.write("stop requested via dashboard "
                            + time.strftime("%Y-%m-%d %H:%M:%S"))
                self._json(200, {"ok": True, "stopped": True,
                                 "note": "watchdog halts bot + bridge within ~40s; "
                                         "everything stays off until START BOT is pressed"})
            except OSError as e:
                self._json(500, {"ok": False, "error": str(e)})
            return
        if self.path == "/api/start":
            removed = []
            for path in (WATCHDOG_FLAG, PROFIT_GUARD, CMD_PATH,
                         CMD_RESULT_PATH):
                try:
                    if os.path.exists(path):
                        os.remove(path)
                        removed.append(os.path.basename(path))
                except OSError:
                    pass
            self._json(200, {"ok": True, "removed": removed,
                             "note": "watchdog starts the bot within ~35s "
                                     "(needs MT5 open + Algo Trading ON)"})
            return
        if self.path == "/api/cmd":
            length = int(self.headers.get("Content-Length", 0))
            if length <= 0 or length > 10_000:
                self._json(400, {"error": "bad length"})
                return
            raw = self.rfile.read(length)
            try:
                obj = json.loads(raw)
                if not isinstance(obj, dict):
                    raise ValueError("cmd must be an object")
                action = str(obj.get("action", "")).lower()
                if action not in CMD_ACTIONS:
                    raise ValueError(f"action must be one of {sorted(CMD_ACTIONS)}")
                lots = obj.get("lots", 0.01)
                float(lots)  # validate
            except Exception as e:
                self._json(400, {"error": f"bad cmd: {e}"})
                return
            cmd = {"id": obj.get("id") or int(time.time() * 1000),
                   "action": action, "lots": float(lots)}
            tmp = CMD_PATH + ".tmp"
            with open(tmp, "w") as f:
                json.dump(cmd, f)
            os.replace(tmp, CMD_PATH)
            self._json(200, {"ok": True, "queued": cmd})
            return
        if self.path != "/api/config":
            self._json(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", 0))
        if length <= 0 or length > 100_000:
            self._json(400, {"error": "bad length"})
            return
        raw = self.rfile.read(length)
        try:
            obj = json.loads(raw)
            if not isinstance(obj, dict):
                raise ValueError("config must be an object")
        except Exception as e:
            self._json(400, {"error": f"bad json: {e}"})
            return
        # merge onto current config so nothing is lost
        current = {}
        if os.path.exists(CONFIG_PATH):
            try:
                with open(CONFIG_PATH, "r") as f:
                    current = json.load(f)
            except Exception:
                current = {}
        current.update(obj)
        tmp = CONFIG_PATH + ".tmp"
        with open(tmp, "w") as f:
            json.dump(current, f, indent=2)
        os.replace(tmp, CONFIG_PATH)
        self._json(200, {"ok": True, "config": current})

    def log_message(self, *args):
        pass


def main():
    addr = ("127.0.0.1", 8765)
    print("=" * 60)
    print("  GOD-MODE DASHBOARD BRIDGE")
    print(f"  open:  http://localhost:{addr[1]}")
    print("  POST /api/config -> bot hot-reloads on next cycle")
    print("  Ctrl+C to stop")
    print("=" * 60)
    try:
        HTTPServer(addr, Handler).serve_forever()
    except KeyboardInterrupt:
        print("\nbridge stopped")


if __name__ == "__main__":
    main()
