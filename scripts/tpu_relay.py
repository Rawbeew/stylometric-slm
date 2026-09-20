"""
tpu_relay.py — single HTTP server exposing both secrets and the dataset
over a single cloudflared tunnel.

Routes:
  GET /secret/<KEY>     -> returns value of KEY from .env (404 if absent)
  GET /data/<filename>  -> serves files from splits/ directory (no directory listing)
  GET /health           -> returns status JSON (no secrets leaked)

This collapses what was previously secret_relay.py + a separate http.server
into one process with one tunnel endpoint.
"""

import argparse
import http.server
import json
import os
import sys
import time
from pathlib import Path


def load_env(env_path: Path) -> dict:
    env = {}
    if not env_path.exists():
        return env
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        env[key.strip()] = value.strip().strip('"').strip("'")
    return env


SECRETS = {}
DATA_DIR = None
ACCESS_LOG = []


class RelayHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # suppress default stderr access log

    def _record(self, ok: bool, **extra):
        entry = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "ip": self.client_address[0],
            "path": self.path,
            "ok": ok,
        }
        entry.update(extra)
        ACCESS_LOG.append(entry)

    def do_GET(self):
        # /secret/<KEY>
        if self.path.startswith("/secret/"):
            key = self.path[len("/secret/"):].split("?")[0]
            if key in SECRETS:
                self._record(True, kind="secret", key=key)
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(SECRETS[key].encode("utf-8"))
            else:
                self._record(False, kind="secret", key=key)
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b"key not found")
            return

        # /data/<filename>
        if self.path.startswith("/data/"):
            name = self.path[len("/data/"):].split("?")[0]
            # Reject any path traversal
            if ".." in name or name.startswith("/") or name.startswith("\\"):
                self._record(False, kind="data", reason="path-traversal", name=name)
                self.send_response(400)
                self.end_headers()
                return
            target = DATA_DIR / name
            target = target.resolve()
            # Make sure resolved path is still under DATA_DIR
            if not str(target).startswith(str(DATA_DIR.resolve())):
                self._record(False, kind="data", reason="escape", name=name)
                self.send_response(400)
                self.end_headers()
                return
            if not target.exists() or not target.is_file():
                self._record(False, kind="data", reason="not-found", name=name)
                self.send_response(404)
                self.end_headers()
                return
            self._record(True, kind="data", name=name, size=target.stat().st_size)
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(target.stat().st_size))
            self.end_headers()
            with target.open("rb") as f:
                while chunk := f.read(64 * 1024):
                    self.wfile.write(chunk)
            return

        # /health
        if self.path == "/health":
            self._record(True, kind="health")
            body = json.dumps({
                "status": "ok",
                "secrets": sorted(SECRETS.keys()),
                "data_dir": str(DATA_DIR),
                "access_count": len(ACCESS_LOG),
                "recent": ACCESS_LOG[-10:],
            }, indent=2)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body.encode("utf-8"))
            return

        # Unknown path
        self._record(False, kind="unknown")
        self.send_response(404)
        self.end_headers()
        self.wfile.write(b"not found")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--env-file", default=None)
    ap.add_argument("--data-dir", default=None,
                    help="Directory served at /data/ (default: <repo>/splits)")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    env_path = Path(args.env_file) if args.env_file else (repo_root / ".env")
    data_dir = Path(args.data_dir) if args.data_dir else (repo_root / "splits")

    global SECRETS, DATA_DIR
    SECRETS = load_env(env_path)
    DATA_DIR = data_dir.resolve()

    print(f"[relay] loaded {len(SECRETS)} secret(s) from {env_path}", file=sys.stderr)
    print(f"[relay] data dir: {DATA_DIR}", file=sys.stderr)
    print(f"[relay] listening on 127.0.0.1:{args.port}", file=sys.stderr)

    server = http.server.HTTPServer(("127.0.0.1", args.port), RelayHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("[relay] shutting down", file=sys.stderr)
        server.shutdown()


if __name__ == "__main__":
    main()
