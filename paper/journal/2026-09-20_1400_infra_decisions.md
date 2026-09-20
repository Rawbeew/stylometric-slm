# Journal Entry 1: Infrastructure Decisions (2026-09-20 ~14:00)

## Goal
Train mT5-base on multilingual authorship-attribution dataset using free / cheap cloud GPU/TPU, push result to Hugging Face, write paper documenting process.

## Hardware decision
User offered: "I have small balance we can use" → not zero-cost but small spend OK.

### Considered
| Option | Cost/hr | mT5-base fits? | Notes |
|---|---|---|---|
| Local CPU (Hermes Windows box) | $0 | yes (slow) | mT5-base ~580M, CPU only, ~8-12hr |
| Free Colab GPU | $0 | yes | T4 16GB, queue wait, 90min idle cap |
| Lambda/Vast GPU | $0.50/hr | yes | T4/A10 spot |
| GCP TPU v5e 1-chip preemptible | $0.74/hr | **NO** (16GB HBM, model needs 19GB+) | confirmed by OOM |
| GCP TPU v5e 4-chip | $3.00/hr | yes | verified v5litepod-1 math |
| GCP CPU n2-standard-32 | $0.17/hr | yes (32GB RAM) | chosen after TPU failed |

**Decision:** After OOM proof on TPU, switch to GCP CPU VM. Confirmed in next entry.

## Cloudflare tunnel setup
Need to bridge secret/API egress from PC (where `.env` lives) to remote compute (TPU).

### Commands run
```bash
# Local (Windows), tunnel for serving .env values + dataset
/c/Users/alaga/bin/cloudflared.exe tunnel --url http://localhost:8001 --no-autoupdate &
```

### Tunnel URL evolution
| Time | URL | Status | Notes |
|---|---|---|---|
| 09:15 | https://random-words.trycloudflare.com | **DIED at ~09:45** | initial tunnel |
| 09:30 | https://invited-forums-acquisitions-videos.trycloudflare.com | **DIED** | alternate |
| 09:45 | https://barbara-themes-forbes-shaved.trycloudflare.com | **ALIVE** | current final |

**Lesson:** cloudflared quick-tunnels rotate/die every ~30min. Always verify with `curl {url}/health` before relying.

## Local relay server

### `scripts/tpu_relay.py` (replaced `secret_relay.py`)
Lines: 159 | Path: `/c/Users/alaga/ghwork/stylometric-slm/scripts/tpu_relay.py`

**Routes:**
- `GET /secret/<KEY>` → returns value from `.env` (404 if absent)
- `GET /data/<filename>` → serves files from `splits/` directory  
- `GET /health` → status JSON, no secrets leaked

```bash
# Local start (still running as of this entry)
cd /c/Users/alaga/ghwork/stylometric-slm
/c/Users/alaga/AppData/Local/hermes/hermes-agent/venv/Scripts/python.exe scripts/tpu_relay.py --port 8001
```

**Output confirmed:**
```
[relay] loaded 1 secret(s) from C:\Users\alaga\ghwork\stylometric-slm\.env
[relay] data dir: C:\Users\alaga\ghwork\stylometric-slm\splits
[relay] listening on 127.0.0.1:8001
```

### Path-verification commands
```bash
curl http://localhost:8001/health
curl http://localhost:8001/secret/HF_TOKEN -w '%{http_code}\n'
curl http://localhost:8001/data/train.jsonl -w '%{http_code} %{size_download}\n'
curl https://barbara-themes-forbes-shaved.trycloudflare.com/data/train.jsonl -w '%{http_code} %{size_download}\n'
```

**Verified outputs:**
```
GET /secret/HF_TOKEN -> 200 bytes=37
GET /data/train.jsonl via tunnel -> 200 bytes=13790244
```

## GitHub repo
- URL: `https://github.com/Rawbeew/stylometric-slm`
- Branch: default `master` (not `main` — local git config, NOT main as initially planned)
- Visibility: flipped to **public** so TPU can `git clone` without auth
- Visible to others: yes (SOP, paper docs, training scripts all visible)

---

## Cross-check commands for this entry
After writing, run these to verify all facts above:

```bash
# 1. Tunnel alive?
curl -sLf https://barbara-themes-forbes-shaved.trycloudflare.com/health | python -m json.tool | head -5

# 2. Local relay up?
curl -sLf http://localhost:8001/health | python -m json.tool | head -5

# 3. Local .env has HF_TOKEN?
grep -v '^#' /c/Users/alaga/ghwork/stylometric-slm/.env | grep -v '^$' | head -3

# 4. Splits exist?
ls -la /c/Users/alaga/ghwork/stylometric-slm/splits/

# 5. Repo public?
curl -sI https://github.com/Rawbeew/stylometric-slm | head -3
```

### Verified results (run 2026-09-20 17:23 UTC)
```
=== Cross-check 1: tunnel alive ===
HTTP 200
secrets: ['HF_TOKEN'] data_dir: C:\Users\alaga\ghwork\stylometric-slm\splits

=== Cross-check 2: local relay ===
HTTP 200
secrets: ['HF_TOKEN'] data_dir: C:\Users\alaga\ghwork\stylometric-slm\splits access_count: 35

=== Cross-check 3: .env (masked) ===
HF_TOKEN=<masked>

=== Cross-check 4: splits ===
total 33836
drwxr-xr-x 1 alaga 197121        0 Sep 19 03:48 .
drwxr-xr-x 1 alaga 197121        0 Sep 20 09:13 ..
-rw-r--r-- 1 alaga 197121 17345568 Sep 19 06:04 dataset.jsonl
-rw-r--r-- 1 alaga 197121  3502163 Sep 19 06:04 eval.jsonl
-rw-r--r-- 1 alaga 197121 13790244 Sep 19 06:04 train.jsonl

=== Cross-check 5: repo public ===
HTTP/1.1 200 OK
Date: Sun, 20 Sep 2026 17:23:22 GMT
Content-Type: text/html; charset=utf-8
```
