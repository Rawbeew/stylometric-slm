"""
secret_relay.py — serves secrets from local .env over the cloudflared tunnel.

Single-purpose HTTP server. Only handles /secret/<KEY> paths. Returns 404 otherwise.
Logs every access (which key, which remote IP, which timestamp) but never logs values.

Usage:
  python scripts/secret_relay.py --port 8001

  # from cloudflared tunnel:
  curl https://<random>.trycloudflare.com/secret/HF_TOKEN

The server reads secrets from .env in the repo root (one KEY=VALUE per line, # for comments).
Secrets are NEVER logged, echoed, or returned unless the path exactly matches /secret/<KEY>.

This is intentional: the URL is unguessable in practice (random subdomain) and HTTPS-encrypted,
but anyone with the tunnel URL has read access to whatever .env contains. Treat .env contents
as scoped to "things I'm OK with the cloudflare edge seeing during this session."
"""

import argparse
import http.server
import json
import os
import sys
import time
from pathlib import Path


def load_env(env_path: Path) -> dict:
    """Read KEY=VALUE pairs from a .env file. No shell expansion, no value echo."""
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
        key = key.strip()
        value = value.strip()
        # Strip optional quotes
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            value = value[1:-1]
        env[key] = value
    return env


SECRETS = {}
ACCESS_LOG = []


class RelayHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Suppress default stderr logging — we do our own
        pass

    def do_GET(self):
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        remote = self.client_address[0]

        if self.path.startswith("/secret/"):
            key = self.path[len("/secret/"):]
            if key in SECRETS:
                ACCESS_LOG.append({"ts": ts, "ip": remote, "key": key, "ok": True})
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(SECRETS[key].encode("utf-8"))
                return
            else:
                ACCESS_LOG.append({"ts": ts, "ip": remote, "key": key, "ok": False})
                self.send_response(404)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(f"key not found: {key}".encode("utf-8"))
                return

        if self.path == "/health":
            ACCESS_LOG.append({"ts": ts, "ip": remote, "path": "/health", "ok": True})
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "status": "ok",
                "keys_available": sorted(SECRETS.keys()),
                "access_count": len(ACCESS_LOG),
            }).encode("utf-8"))
            return

        if self.path == "/access-log":
            # For debugging — only available if a debug token is in env
            debug = os.environ.get("RELAY_DEBUG", "").lower() in ("1", "true", "yes")
            if not debug:
                self.send_response(403)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(ACCESS_LOG, indent=2).encode("utf-8"))
            return

        ACCESS_LOG.append({"ts": ts, "ip": remote, "path": self.path, "ok": False})
        self.send_response(404)
        self.end_headers()
        self.wfile.write(b"not found")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--env-file", default=None,
                    help="Path to .env file (default: <repo>/.env)")
    args = ap.parse_args()

    env_path = Path(args.env_file) if args.env_file else (
        Path(__file__).resolve().parents[1] / ".env"
    )

    global SECRETS
    SECRETS = load_env(env_path)
    if not SECRETS:
        print(f"[relay] WARNING: no secrets loaded from {env_path}", file=sys.stderr)
        print(f"[relay] create the file with: KEY=value (one per line)", file=sys.stderr)
    else:
        # Log only KEYS, never values
        print(f"[relay] loaded {len(SECRETS)} secrets: {sorted(SECRETS.keys())}")
        print(f"[relay] serving on port {args.port}")

    server = http.server.HTTPServer(("127.0.0.1", args.port), RelayHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("[relay] shutting down")
        server.shutdown()


if __name__ == "__main__":
    main()
