#!/usr/bin/env python3
"""
Elevator 4-channel dashboard + anomaly alert hub
- 4-way preview grid (ports 5000~5003)
- DeepStream log tailing (fall/fight events)
- In-browser live alert feed + optional Telegram/webhook alerts
- Severity-based alert routing (critical=immediate, warning=digest, info=log-only)
- App-level health monitoring with auto-recovery
- Role-based access control (admin/viewer)

API examples:
- GET /api/events?since=120
- GET /api/events?channel=rtsp&type=fall&severity=critical&min_score=0.85&limit=30
- GET /api/events?from_ts=1710000000&to_ts=1710003600&q=신뢰도&sort=asc
- GET /api/events/stats?from_ts=1710000000&to_ts=1710086400
- POST /api/channels/<channel_id>/restart  (admin only)
"""

import argparse
import json
import logging
import logging.handlers
import os
import re
import sqlite3
import subprocess
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from flask import Flask, jsonify, render_template_string, request

# ---------------------------------------------------------------------------
# Module-level logger (configured in main())
# ---------------------------------------------------------------------------
logger = logging.getLogger("elevator-dashboard")

CHANNELS = [
    {
        "id": "webcam",
        "name": "Webcam",
        "port": 5000,
        "ds_log": "/home/ppak/projects/elevator/deepstream_pose/logs/elevator-ds-webcam.out.log",
    },
    {
        "id": "rtsp",
        "name": "RTSP",
        "port": 5001,
        "ds_log": "/home/ppak/projects/elevator/deepstream_pose/logs/elevator-ds-rtsp.out.log",
    },
    {
        "id": "video1",
        "name": "Video 1",
        "port": 5002,
        "ds_log": "/home/ppak/projects/elevator/deepstream_pose/logs/elevator-ds-video1.out.log",
    },
    {
        "id": "video2",
        "name": "Video 2",
        "port": 5003,
        "ds_log": "/home/ppak/projects/elevator/deepstream_pose/logs/elevator-ds-video2.out.log",
    },
]

FALL_RE = re.compile(r"\[쓰러짐 감지\].*신뢰도:\s*([0-9.]+)")
FIGHT_RE = re.compile(r"\[싸움 감지\].*신뢰도:\s*([0-9.]+)")

# ---------------------------------------------------------------------------
# Alert digest interval (seconds). WARNING events are batched.
# ---------------------------------------------------------------------------
ALERT_DIGEST_INTERVAL_SEC = int(os.getenv("ALERT_DIGEST_INTERVAL_SEC", "300"))

# ---------------------------------------------------------------------------
# Health-check settings
# ---------------------------------------------------------------------------
HEALTH_CHECK_INTERVAL_SEC = int(os.getenv("HEALTH_CHECK_INTERVAL_SEC", "30"))
HEALTH_MAX_FAILURES = int(os.getenv("HEALTH_MAX_FAILURES", "3"))


def _service_active(service_name: str) -> bool:
    try:
        result = subprocess.run(
            ["systemctl", "--user", "is-active", service_name],
            capture_output=True, text=True, timeout=5,
        )
        return result.returncode == 0 and result.stdout.strip() == "active"
    except Exception:
        return False


def _augment_channel_health(channel_health: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    enriched: Dict[str, Dict[str, Any]] = {}
    for ch in CHANNELS:
        state = dict(channel_health.get(ch["id"], {}))
        ds_service = f"elevator-ds-{ch['id']}.service"
        state["inference"] = {
            "service": ds_service,
            "active": _service_active(ds_service),
        }
        enriched[ch["id"]] = state
    return enriched


CHANNEL_CONFIG_PATH = Path("/home/ppak/projects/elevator/channel_sources.json")
SYSTEMD_USER_DIR = Path.home() / ".config/systemd/user"
OVERLAY_DIR = Path("/home/ppak/projects/elevator/runtime/overlays")
PREVIEW_CAPTURE_OPTIONS = "rtsp_transport;udp|fflags;nobuffer|flags;low_delay|max_delay;500000|reorder_queue_size;0"
SOURCE_TYPE_CHOICES = ("disabled", "rtsp", "file", "webcam")
CHANNEL_BY_ID = {ch["id"]: ch for ch in CHANNELS}


def _infer_source_type(source: str) -> str:
    src = (source or "").strip()
    if not src:
        return "disabled"
    lower = src.lower()
    if src.isdigit():
        return "webcam"
    if lower.startswith(("rtsp://", "rtmp://", "udp://", "http://", "https://")):
        return "rtsp"
    return "file"


def _default_channel_sources() -> Dict[str, Dict[str, Any]]:
    return {
        ch["id"]: {
            "id": ch["id"],
            "name": ch["name"],
            "sourceType": "disabled",
            "source": "",
            "enabled": False,
        }
        for ch in CHANNELS
    }


def _normalize_channel_source(channel_id: str, raw: Any) -> Dict[str, Any]:
    entry = raw if isinstance(raw, dict) else {}
    source = str(entry.get("source") or "").strip()
    source_type = str(entry.get("sourceType") or "").strip().lower()
    if source_type not in SOURCE_TYPE_CHOICES:
        source_type = _infer_source_type(source)
    enabled = source_type != "disabled" and bool(source)
    return {
        "id": channel_id,
        "name": CHANNEL_BY_ID[channel_id]["name"],
        "sourceType": source_type if enabled else "disabled",
        "source": source if enabled else "",
        "enabled": enabled,
    }


def _extract_execstart_source(service_path: Path) -> str:
    if not service_path.exists():
        return ""
    try:
        for line in service_path.read_text().splitlines():
            if not line.startswith("ExecStart="):
                continue
            m = re.search(r'--source\s+(?P<value>"(?:\\.|[^"])*"|\S+)', line)
            if not m:
                continue
            value = m.group("value").strip()
            if value.startswith('"') and value.endswith('"'):
                value = value[1:-1].replace('\\"', '"').replace('\\\\', '\\')
            return value
    except Exception:
        return ""
    return ""


def _save_channel_sources(config: Dict[str, Dict[str, Any]]) -> None:
    payload = {
        ch["id"]: {
            "sourceType": config[ch["id"]]["sourceType"],
            "source": config[ch["id"]]["source"],
        }
        for ch in CHANNELS
    }
    CHANNEL_CONFIG_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _load_channel_sources() -> Dict[str, Dict[str, Any]]:
    config = _default_channel_sources()
    if CHANNEL_CONFIG_PATH.exists():
        try:
            raw = json.loads(CHANNEL_CONFIG_PATH.read_text())
        except Exception:
            raw = {}
    else:
        raw = {}
        for ch in CHANNELS:
            service_path = SYSTEMD_USER_DIR / f"elevator-preview-{ch['id']}.service"
            raw[ch["id"]] = {"source": _extract_execstart_source(service_path)}
        inferred = {cid: _normalize_channel_source(cid, raw.get(cid, {})) for cid in config}
        _save_channel_sources(inferred)
    for ch in CHANNELS:
        config[ch["id"]] = _normalize_channel_source(ch["id"], raw.get(ch["id"], {}))
    return config


def _quote_unit_arg(value: str) -> str:
    escaped = str(value).replace('\\', '\\\\').replace('"', '\\"')
    return f'"{escaped}"'


def _preview_service_text(channel: Dict[str, Any], source: str) -> str:
    overlay_json = OVERLAY_DIR / f"{channel['id']}.json"
    return f'''[Unit]
Description=Elevator preview channel: elevator-preview-{channel['id']}.service
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=/home/ppak/projects/elevator
Environment=PYTHONUNBUFFERED=1
Environment=OPENCV_FFMPEG_CAPTURE_OPTIONS={PREVIEW_CAPTURE_OPTIONS}
ExecStart=/home/ppak/projects/elevator/venv/bin/python /home/ppak/projects/elevator/preview_server.py --source {_quote_unit_arg(source)} --port {channel['port']} --width 640 --jpeg-quality 65 --overlay-json {_quote_unit_arg(str(overlay_json))}
Restart=always
RestartSec=3
StandardOutput=append:/home/ppak/projects/elevator/logs/elevator-preview-{channel['id']}.out.log
StandardError=append:/home/ppak/projects/elevator/logs/elevator-preview-{channel['id']}.err.log

[Install]
WantedBy=default.target
'''


def _ds_service_text(channel: Dict[str, Any], source_type: str, source: str) -> str:
    overlay_json = OVERLAY_DIR / f"{channel['id']}.json"
    ds_source = f"http://127.0.0.1:{channel['port']}/video_feed" if source_type == "webcam" else source
    extra_unit = ""
    extra_service = ""
    if source_type == "webcam":
        extra_unit = f"After=network-online.target elevator-preview-{channel['id']}.service\nWants=network-online.target\nRequires=elevator-preview-{channel['id']}.service\nPartOf=elevator-preview-{channel['id']}.service\n"
        extra_service = "ExecStartPre=/bin/sleep 2\n"
    else:
        extra_unit = "After=network-online.target\nWants=network-online.target\n"
    return f'''[Unit]
Description=Elevator DeepStream channel: elevator-ds-{channel['id']}.service
{extra_unit}
[Service]
Type=simple
WorkingDirectory=/home/ppak/projects/elevator/deepstream_pose
Environment=PYTHONUNBUFFERED=1
{extra_service}ExecStart=/usr/bin/python3 /home/ppak/projects/elevator/deepstream_pose/deepstream_pose_simple.py --source {_quote_unit_arg(ds_source)} --no-display --overlay-json {_quote_unit_arg(str(overlay_json))}
Restart=always
RestartSec=3
StandardOutput=append:/home/ppak/projects/elevator/deepstream_pose/logs/elevator-ds-{channel['id']}.out.log
StandardError=append:/home/ppak/projects/elevator/deepstream_pose/logs/elevator-ds-{channel['id']}.err.log

[Install]
WantedBy=default.target
'''


def _restart_channel_services(channel_id: str, config: Optional[Dict[str, Dict[str, Any]]] = None) -> Tuple[bool, str]:
    config = config or _load_channel_sources()
    if channel_id not in config:
        return False, f"unknown channel: {channel_id}"

    channel = CHANNEL_BY_ID[channel_id]
    entry = config[channel_id]
    preview_service = f"elevator-preview-{channel_id}.service"
    ds_service = f"elevator-ds-{channel_id}.service"
    preview_path = SYSTEMD_USER_DIR / preview_service
    ds_path = SYSTEMD_USER_DIR / ds_service
    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
    if entry["enabled"]:
        preview_path.write_text(_preview_service_text(channel, entry["source"]))
        ds_path.write_text(_ds_service_text(channel, entry["sourceType"], entry["source"]))

    subprocess.run(["systemctl", "--user", "daemon-reload"], capture_output=True, text=True, timeout=20)

    if not entry["enabled"]:
        subprocess.run(["systemctl", "--user", "stop", ds_service], capture_output=True, text=True, timeout=20)
        subprocess.run(["systemctl", "--user", "stop", preview_service], capture_output=True, text=True, timeout=20)
        return True, f"{channel_id} disabled"

    subprocess.run(["systemctl", "--user", "enable", preview_service, ds_service], capture_output=True, text=True, timeout=20)
    preview_result = subprocess.run(["systemctl", "--user", "restart", preview_service], capture_output=True, text=True, timeout=20)
    if preview_result.returncode != 0:
        return False, f"preview restart failed: {preview_result.stderr.strip()}"
    ds_result = subprocess.run(["systemctl", "--user", "restart", ds_service], capture_output=True, text=True, timeout=20)
    if ds_result.returncode != 0:
        return False, f"inference restart failed: {ds_result.stderr.strip()}"
    return True, f"{channel_id} applied ({entry['sourceType']})"


def _validate_channel_sources(config: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    errors: Dict[str, str] = {}
    for channel_id, entry in config.items():
        source_type = entry["sourceType"]
        source = entry["source"]
        if source_type == "disabled":
            continue
        if not source:
            errors[channel_id] = "source is required"
        elif source_type == "webcam" and not source.isdigit():
            errors[channel_id] = "webcam source must be a device index like 0"
        elif source_type == "rtsp" and not source.lower().startswith(("rtsp://", "rtmp://", "udp://", "http://", "https://")):
            errors[channel_id] = "rtsp source must start with rtsp://, rtmp://, udp://, http://, or https://"
    return errors


def _apply_channel_sources(config: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    _save_channel_sources(config)
    results: Dict[str, Dict[str, Any]] = {}
    for ch in CHANNELS:
        ok, message = _restart_channel_services(ch["id"], config)
        results[ch["id"]] = {"ok": ok, "message": message}
    return results

HTML = """
<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Elevator Safety Dashboard</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
  <style>
    *, *::before, *::after { box-sizing:border-box; margin:0; padding:0; }
    :root {
      --bg-primary: #0a0e1a;
      --bg-secondary: #111827;
      --bg-card: rgba(17,24,39,0.7);
      --bg-card-hover: rgba(17,24,39,0.9);
      --bg-input: rgba(15,23,42,0.8);
      --border: rgba(99,102,241,0.15);
      --border-hover: rgba(99,102,241,0.3);
      --accent: #6366f1;
      --accent-glow: rgba(99,102,241,0.15);
      --text-primary: #f1f5f9;
      --text-secondary: #94a3b8;
      --text-muted: #64748b;
      --green: #10b981;
      --red: #ef4444;
      --amber: #f59e0b;
      --blue: #3b82f6;
      --radius: 12px;
      --radius-sm: 8px;
      --shadow: 0 4px 24px rgba(0,0,0,0.3);
      --shadow-lg: 0 8px 40px rgba(0,0,0,0.4);
      --glass: saturate(180%) blur(16px);
    }

    body {
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
      background: var(--bg-primary);
      color: var(--text-primary);
      min-height: 100vh;
      overflow-x: hidden;
    }

    /* Animated background */
    body::before {
      content: '';
      position: fixed;
      top: -50%;
      left: -50%;
      width: 200%;
      height: 200%;
      background: radial-gradient(circle at 20% 50%, rgba(99,102,241,0.04) 0%, transparent 50%),
                  radial-gradient(circle at 80% 20%, rgba(16,185,129,0.03) 0%, transparent 50%),
                  radial-gradient(circle at 50% 80%, rgba(239,68,68,0.02) 0%, transparent 50%);
      z-index: -1;
      animation: bgShift 30s ease-in-out infinite alternate;
    }
    @keyframes bgShift { to { transform: translate(5%, 3%) rotate(2deg); } }

    /* ─── Header ─── */
    .top {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 16px 24px;
      background: linear-gradient(135deg, rgba(17,24,39,0.95) 0%, rgba(10,14,26,0.95) 100%);
      backdrop-filter: var(--glass);
      border-bottom: 1px solid var(--border);
      position: sticky;
      top: 0;
      z-index: 100;
    }
    .top-left { display: flex; align-items: center; gap: 16px; }
    .logo {
      width: 40px; height: 40px;
      background: linear-gradient(135deg, var(--accent), #818cf8);
      border-radius: 10px;
      display: flex; align-items: center; justify-content: center;
      font-size: 20px;
      box-shadow: 0 0 20px var(--accent-glow);
    }
    .title { font-weight: 700; font-size: 18px; letter-spacing: -0.3px; }
    .sub { font-size: 12px; color: var(--text-secondary); margin-top: 2px; }
    .role-badge {
      padding: 3px 10px;
      border-radius: 99px;
      font-size: 10px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      margin-left: 10px;
      vertical-align: middle;
    }
    .role-badge.admin { background: rgba(239,68,68,0.15); color: #fca5a5; border: 1px solid rgba(239,68,68,0.3); }
    .role-badge.viewer { background: rgba(59,130,246,0.15); color: #93c5fd; border: 1px solid rgba(59,130,246,0.3); }
    #clock { font-size: 13px; color: var(--text-secondary); font-variant-numeric: tabular-nums; font-weight: 500; }

    /* ─── Layout ─── */
    .layout {
      display: grid;
      grid-template-columns: 1fr 380px;
      gap: 16px;
      padding: 16px 24px 24px;
      max-width: 1920px;
      margin: 0 auto;
    }
    .main { display: flex; flex-direction: column; gap: 16px; }
    .side { display: flex; flex-direction: column; gap: 16px; }

    /* ─── Cards ─── */
    .card {
      background: var(--bg-card);
      backdrop-filter: var(--glass);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      overflow: hidden;
      box-shadow: var(--shadow);
      transition: border-color 0.2s, box-shadow 0.2s;
    }
    .card:hover { border-color: var(--border-hover); }
    .card-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 12px 16px;
      border-bottom: 1px solid var(--border);
      background: rgba(0,0,0,0.2);
    }
    .card-header h3 {
      font-size: 13px;
      font-weight: 600;
      color: var(--text-primary);
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .card-header .badge {
      font-size: 11px;
      color: var(--text-muted);
      font-weight: 400;
    }

    /* ─── Controls ─── */
    .controls-toggle {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 10px 16px;
      cursor: pointer;
      user-select: none;
    }
    .controls-toggle:hover { background: rgba(255,255,255,0.02); }
    .controls-body { padding: 0 16px 16px; }
    .filter-grid {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 10px;
    }
    .field label {
      display: block;
      font-size: 10px;
      font-weight: 600;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.5px;
      margin-bottom: 4px;
    }
    .field input, .field select {
      width: 100%;
      background: var(--bg-input);
      color: var(--text-primary);
      border: 1px solid var(--border);
      border-radius: var(--radius-sm);
      padding: 7px 10px;
      font-size: 12px;
      font-family: inherit;
      transition: border-color 0.2s, box-shadow 0.2s;
      outline: none;
    }
    .field input:focus, .field select:focus {
      border-color: var(--accent);
      box-shadow: 0 0 0 3px var(--accent-glow);
    }
    .actions {
      display: flex;
      gap: 8px;
      margin-top: 12px;
      flex-wrap: wrap;
    }
    .btn {
      font-family: inherit;
      font-size: 12px;
      font-weight: 500;
      padding: 7px 14px;
      border: none;
      border-radius: var(--radius-sm);
      cursor: pointer;
      transition: all 0.15s;
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }
    .btn-primary { background: var(--accent); color: white; }
    .btn-primary:hover { background: #5558e6; box-shadow: 0 0 16px var(--accent-glow); }
    .btn-secondary { background: rgba(255,255,255,0.06); color: var(--text-secondary); }
    .btn-secondary:hover { background: rgba(255,255,255,0.1); color: var(--text-primary); }
    .btn-outline { background: transparent; color: var(--text-secondary); border: 1px solid var(--border); }
    .btn-outline:hover { border-color: var(--accent); color: var(--accent); }
    .btn-outline.active { background: rgba(16,185,129,0.1); border-color: var(--green); color: var(--green); }
    .btn-danger { background: rgba(239,68,68,0.15); color: #fca5a5; border: 1px solid rgba(239,68,68,0.2); }
    .btn-danger:hover { background: rgba(239,68,68,0.25); }
    .btn-sm {
      font-family: inherit;
      background: rgba(255,255,255,0.06);
      color: var(--text-secondary);
      border: 1px solid transparent;
      border-radius: 6px;
      padding: 3px 10px;
      font-size: 11px;
      cursor: pointer;
      transition: all 0.15s;
    }
    .btn-sm:hover { background: rgba(255,255,255,0.1); color: var(--text-primary); border-color: var(--border); }
    .btn-sm.pin-on { background: rgba(124,58,237,0.15); color: #c4b5fd; border-color: rgba(124,58,237,0.3); }
    .btn-sm.restart { background: rgba(239,68,68,0.1); color: #fca5a5; }
    .btn-sm.restart:hover { background: rgba(239,68,68,0.2); }

    .status-bar {
      display: flex;
      justify-content: space-between;
      padding: 8px 16px;
      font-size: 11px;
      color: var(--text-muted);
      border-top: 1px solid var(--border);
    }

    /* ─── Video Grid ─── */
    .grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }
    .grid.focus-mode .vid-card { display: none; }
    .grid.focus-mode .vid-card.focused { display: block; grid-column: 1 / -1; }
    .grid.focus-mode .vid-card.focused img { height: 560px; }

    .vid-card {
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      overflow: hidden;
      transition: all 0.3s;
      position: relative;
    }
    .vid-card:hover { border-color: var(--border-hover); transform: translateY(-1px); box-shadow: var(--shadow-lg); }
    .vid-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 10px 14px;
      background: rgba(0,0,0,0.3);
    }
    .vid-title { display: flex; align-items: center; gap: 8px; font-size: 13px; font-weight: 600; }
    .vid-tools { display: flex; gap: 4px; }
    .status-dot {
      width: 8px; height: 8px;
      border-radius: 50%;
      display: inline-block;
    }
    .status-dot.online { background: var(--green); box-shadow: 0 0 8px rgba(16,185,129,0.5); }
    .status-dot.offline { background: var(--red); box-shadow: 0 0 8px rgba(239,68,68,0.5); animation: pulse 2s infinite; }
    .status-dot.checking { background: var(--amber); }
    @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }

    .status-label { font-size: 11px; font-weight: 600; }
    .status-label.ok { color: var(--green); }
    .status-label.bad { color: var(--red); }
    .status-label.warn { color: var(--amber); }

    .vid-card img {
      width: 100%;
      height: 260px;
      object-fit: contain;
      background: #000;
      display: block;
      border-top: 1px solid rgba(255,255,255,0.03);
      border-bottom: 1px solid rgba(255,255,255,0.03);
    }
    .vid-meta {
      font-size: 11px;
      color: var(--text-muted);
      padding: 8px 14px;
      font-variant-numeric: tabular-nums;
    }

    /* ─── Events ─── */
    .events { max-height: 520px; overflow-y: auto; padding: 12px; }
    .events::-webkit-scrollbar { width: 4px; }
    .events::-webkit-scrollbar-track { background: transparent; }
    .events::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 4px; }

    .ev {
      border-radius: var(--radius-sm);
      padding: 10px 12px;
      margin-bottom: 8px;
      background: rgba(0,0,0,0.25);
      border-left: 3px solid transparent;
      transition: all 0.2s;
      animation: slideIn 0.3s ease-out;
    }
    @keyframes slideIn { from { opacity: 0; transform: translateX(-8px); } to { opacity: 1; transform: translateX(0); } }
    .ev:hover { background: rgba(0,0,0,0.35); }
    .ev.fall { border-left-color: var(--red); }
    .ev.fight { border-left-color: var(--amber); }
    .ev.channel_recovery { border-left-color: var(--blue); }
    .ev .line {
      font-size: 13px;
      font-weight: 500;
      display: flex;
      align-items: center;
      gap: 6px;
      flex-wrap: wrap;
      margin-bottom: 4px;
    }
    .ev .meta {
      font-size: 11px;
      color: var(--text-muted);
      line-height: 1.5;
    }

    .pill {
      padding: 2px 8px;
      border-radius: 99px;
      font-size: 10px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.3px;
    }
    .pill.fall { background: rgba(239,68,68,0.15); color: #fca5a5; }
    .pill.fight { background: rgba(245,158,11,0.15); color: #fcd34d; }
    .pill.channel_recovery { background: rgba(59,130,246,0.15); color: #93c5fd; }
    .pill.critical { background: rgba(239,68,68,0.15); color: #fca5a5; }
    .pill.warning { background: rgba(245,158,11,0.15); color: #fcd34d; }
    .pill.normal { background: rgba(59,130,246,0.15); color: #93c5fd; }

    .hint {
      font-size: 11px;
      color: var(--text-muted);
      padding: 10px 16px;
      border-top: 1px solid var(--border);
      line-height: 1.6;
    }

    /* ─── Stats ─── */
    .stats-grid { padding: 12px 16px; }
    .stat-section { margin-bottom: 12px; }
    .stat-section:last-child { margin-bottom: 0; }
    .stat-section strong {
      font-size: 11px;
      color: var(--text-secondary);
      text-transform: uppercase;
      letter-spacing: 0.5px;
      display: block;
      margin-bottom: 6px;
    }
    .stat-bar-container { margin-bottom: 4px; }
    .stat-bar-label {
      display: flex;
      justify-content: space-between;
      font-size: 11px;
      color: var(--text-secondary);
      margin-bottom: 2px;
    }
    .stat-bar {
      height: 6px;
      background: rgba(255,255,255,0.05);
      border-radius: 3px;
      overflow: hidden;
    }
    .stat-bar-fill {
      height: 100%;
      border-radius: 3px;
      transition: width 0.6s ease;
      background: linear-gradient(90deg, var(--accent), #818cf8);
    }
    .stat-bar-fill.type-fall { background: linear-gradient(90deg, #ef4444, #f87171); }
    .stat-bar-fill.type-fight { background: linear-gradient(90deg, #f59e0b, #fbbf24); }
    .stat-bar-fill.sev-critical { background: linear-gradient(90deg, #ef4444, #f87171); }
    .stat-bar-fill.sev-warning { background: linear-gradient(90deg, #f59e0b, #fbbf24); }
    .stat-bar-fill.sev-normal { background: linear-gradient(90deg, #3b82f6, #60a5fa); }

    @media (max-width: 1400px) {
      .layout { grid-template-columns: 1fr; }
      .side { flex-direction: row; flex-wrap: wrap; }
      .side > .card { flex: 1; min-width: 300px; }
    }
    @media (max-width: 900px) {
      .filter-grid { grid-template-columns: repeat(2, 1fr); }
      .grid { grid-template-columns: 1fr; }
      .layout { padding: 12px; }
    }
  </style>
</head>
<body>
  <div class="top">
    <div class="top-left">
      <div class="logo">🛗</div>
      <div>
        <div class="title">
          승강기 이상감지 관제
          <span id="roleBadge" class="role-badge" style="display:none;"></span>
        </div>
        <div class="sub">4채널 실시간 모니터링 · 이벤트 관제 · 자동 복구</div>
      </div>
    </div>
    <div id="clock"></div>
  </div>

  <div class="layout">
    <div class="main">
      <!-- Controls -->
      <div class="card">
        <div class="controls-toggle" onclick="document.getElementById('controlsBody').style.display = document.getElementById('controlsBody').style.display === 'none' ? 'block' : 'none'">
          <h3 style="font-size:13px;font-weight:600;display:flex;align-items:center;gap:8px;">
            ⚙️ 운영 제어
          </h3>
          <span id="modeText" class="badge" style="font-size:11px;color:var(--text-muted);"></span>
        </div>
        <div id="controlsBody" class="controls-body">
          <div class="filter-grid">
            <div class="field">
              <label>채널</label>
              <select id="fChannel"><option value="">전체</option></select>
            </div>
            <div class="field">
              <label>이벤트 유형</label>
              <select id="fType">
                <option value="">전체</option>
                <option value="fall">fall (쓰러짐)</option>
                <option value="fight">fight (싸움)</option>
              </select>
            </div>
            <div class="field">
              <label>심각도</label>
              <select id="fSeverity">
                <option value="">전체</option>
                <option value="critical">🔴 critical</option>
                <option value="warning">🟡 warning</option>
                <option value="normal">🔵 normal</option>
              </select>
            </div>
            <div class="field">
              <label>최소 신뢰도</label>
              <input id="fMinScore" type="number" min="0" max="1" step="0.01" placeholder="0.85" />
            </div>
            <div class="field">
              <label>키워드 검색</label>
              <input id="fQ" type="text" placeholder="raw line 검색..." />
            </div>
            <div class="field">
              <label>정렬</label>
              <select id="fSort">
                <option value="desc">최신순 ↓</option>
                <option value="asc">과거순 ↑</option>
              </select>
            </div>
            <div class="field">
              <label>조회 구간</label>
              <select id="fWindowMin">
                <option value="0">전체</option>
                <option value="10">최근 10분</option>
                <option value="30">최근 30분</option>
                <option value="60">최근 1시간</option>
                <option value="180">최근 3시간</option>
                <option value="1440">최근 24시간</option>
              </select>
            </div>
            <div class="field">
              <label>순환 주기</label>
              <select id="cycleSec">
                <option value="5">5초</option>
                <option value="10" selected>10초</option>
                <option value="15">15초</option>
                <option value="30">30초</option>
              </select>
            </div>
          </div>
          <div class="actions">
            <button class="btn btn-primary" id="btnApply">필터 적용</button>
            <button class="btn btn-secondary" id="btnReset">초기화</button>
            <button class="btn btn-outline" id="btnCycle">🔄 순환 시작</button>
            <button class="btn btn-outline" id="btnExitFocus">↩ 확대 해제</button>
          </div>
        </div>
        <div class="status-bar">
          <span id="focusInfo">확대: 없음</span>
          <span id="pinInfo">고정: 없음</span>
        </div>
      </div>

      <!-- Video Grid -->
      <div class="grid" id="grid"></div>
    </div>

    <div class="side">
      <!-- Events -->
      <div class="card">
        <div class="card-header">
          <h3>🔔 실시간 알람</h3>
          <span id="evCount" class="badge"></span>
        </div>
        <div class="events" id="events"></div>
        <div class="hint">필터 비활성 시 since 기반 증분 폴링</div>
      </div>

      <!-- Stats -->
      <div class="card">
        <div class="card-header">
          <h3>📊 이벤트 통계</h3>
          <span class="badge" id="statsWindowText"></span>
        </div>
        <div class="stats-grid" id="statsBody">로딩중...</div>
      </div>

      <!-- Source Config -->
      <div class="card">
        <div class="card-header">
          <h3>🎛️ 채널 입력 소스 설정</h3>
          <span class="badge">preview + inference 공통 적용</span>
        </div>
        <div class="controls-body">
          <div id="sourceConfigBody">로딩중...</div>
          <div style="display:flex; gap:8px; margin-top:12px; flex-wrap:wrap;">
            <button class="btn" id="btnSaveSources">💾 저장 후 적용</button>
            <button class="btn btn-outline" id="btnReloadSources">↻ 다시 불러오기</button>
          </div>
        </div>
      </div>

      <!-- Memo -->
      <div class="card">
        <div class="card-header"><h3>📋 운영 정보</h3></div>
        <div class="hint">
          알림 정책: 🔴 critical → 즉시 전송 · 🟡 warning → 5분 다이제스트 · 🔵 info → 로그만<br/>
          헬스체크: 30초 주기 · 3회 연속 실패 시 자동 복구<br/>
          Webcam은 장치 점유 특성상 추론/미리보기 동시 사용 제한
        </div>
      </div>
    </div>
  </div>

<script>
const channels = {{ channels|tojson }};
const pageToken = new URLSearchParams(location.search).get('token') || '';
const userRole = '{{ user_role }}';
const state = {
  lastEventId: 0,
  lastRenderedTopId: 0,
  filters: {
    channel: '',
    type: '',
    severity: '',
    minScore: '',
    q: '',
    sort: 'desc',
    windowMin: '0',
  },
  focusedChannelId: null,
  pinned: new Set(),
  cycleEnabled: false,
  cycleIndex: 0,
  cycleTimer: null,
  channelHealth: {},
  channelSources: {},
};

function streamUrl(port){ return `http://${location.hostname}:${port}/video_feed`; }
function statsUrl(port){ return `http://${location.hostname}:${port}/stats`; }

function formatTime(ts){
  if(!ts) return '-';
  return new Date(ts).toLocaleTimeString();
}

function escapeHtml(value){
  return String(value || '').replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');
}

function withToken(params){
  if(pageToken) params.set('token', pageToken);
  return params;
}

function showRoleBadge(){
  const badge = document.getElementById('roleBadge');
  if(userRole === 'admin'){
    badge.textContent = 'Admin';
    badge.className = 'role-badge admin';
    badge.style.display = 'inline';
  } else if(userRole === 'viewer'){
    badge.textContent = 'Viewer';
    badge.className = 'role-badge viewer';
    badge.style.display = 'inline';
  }
}

function initChannelFilterOptions(){
  const sel = document.getElementById('fChannel');
  channels.forEach(ch => {
    const opt = document.createElement('option');
    opt.value = ch.id;
    opt.textContent = `${ch.name} (${ch.id})`;
    sel.appendChild(opt);
  });
}

function makeGrid(){
  const grid = document.getElementById('grid');
  grid.innerHTML = '';
  channels.forEach(ch => {
    const restartBtn = userRole === 'admin'
      ? `<button class="btn-sm restart" data-action="restart" data-channel="${ch.id}">재시작</button>`
      : '';
    const card = document.createElement('div');
    card.className = 'vid-card';
    card.id = `card_${ch.id}`;
    card.dataset.channelId = ch.id;
    card.innerHTML = `
      <div class="vid-header">
        <div class="vid-title">
          <span class="status-dot checking" id="dot_${ch.id}"></span>
          <strong>${ch.name}</strong>
          <span id="st_${ch.id}" class="status-label warn">확인중</span>
        </div>
        <div class="vid-tools">
          <button class="btn-sm" data-action="focus" data-channel="${ch.id}">확대</button>
          <button class="btn-sm" id="pin_${ch.id}" data-action="pin" data-channel="${ch.id}">고정</button>
          ${restartBtn}
        </div>
      </div>
      <img id="img_${ch.id}" src="${streamUrl(ch.port)}" alt="${ch.name}"/>
      <div class="vid-meta" id="meta_${ch.id}">port: ${ch.port}</div>
    `;
    grid.appendChild(card);
  });

  grid.addEventListener('click', (ev) => {
    const btn = ev.target.closest('button[data-action]');
    if(!btn) return;
    const action = btn.dataset.action;
    const channelId = btn.dataset.channel;
    if(action === 'focus'){
      if(state.focusedChannelId === channelId) setFocusedChannel(null);
      else setFocusedChannel(channelId);
      if(state.cycleEnabled) stopCycle();
    }
    if(action === 'pin') togglePin(channelId);
    if(action === 'restart') restartChannel(channelId);
  });
}

async function restartChannel(channelId){
  if(!confirm(`${channelId} 채널 미리보기를 재시작하시겠습니까?`)) return;
  try {
    const params = new URLSearchParams();
    if(pageToken) params.set('token', pageToken);
    const r = await fetch(`/api/channels/${channelId}/restart?${params.toString()}`, {method:'POST'});
    const j = await r.json();
    if(r.ok) alert(`${channelId} 재시작 완료`);
    else alert(`재시작 실패: ${j.error || j.message || r.status}`);
  } catch(e) { alert(`재시작 요청 실패: ${e.message}`); }
}

function setFocusedChannel(channelId){
  state.focusedChannelId = channelId || null;
  const grid = document.getElementById('grid');
  if(state.focusedChannelId) grid.classList.add('focus-mode');
  else grid.classList.remove('focus-mode');

  channels.forEach(ch => {
    const card = document.getElementById(`card_${ch.id}`);
    if(!card) return;
    card.classList.toggle('focused', state.focusedChannelId === ch.id);
  });

  const focusName = channels.find(c => c.id === state.focusedChannelId)?.name || '없음';
  document.getElementById('focusInfo').textContent = `확대: ${focusName}`;
}

function togglePin(channelId){
  if(state.pinned.has(channelId)) state.pinned.delete(channelId);
  else state.pinned.add(channelId);

  channels.forEach(ch => {
    const btn = document.getElementById(`pin_${ch.id}`);
    if(!btn) return;
    btn.classList.toggle('pin-on', state.pinned.has(ch.id));
    btn.textContent = state.pinned.has(ch.id) ? '📌 고정됨' : '고정';
  });

  const pinNames = channels.filter(ch => state.pinned.has(ch.id)).map(ch => ch.name);
  document.getElementById('pinInfo').textContent = `고정: ${pinNames.length ? pinNames.join(', ') : '없음'}`;
}

function cycleCandidates(){
  const pinned = channels.filter(ch => state.pinned.has(ch.id)).map(ch => ch.id);
  if(pinned.length > 0) return pinned;
  return channels.map(ch => ch.id);
}

function advanceCycle(){
  const candidates = cycleCandidates();
  if(!candidates.length) return;
  const channelId = candidates[state.cycleIndex % candidates.length];
  state.cycleIndex += 1;
  setFocusedChannel(channelId);
}

function startCycle(){
  const sec = parseInt(document.getElementById('cycleSec').value || '10', 10);
  const intervalMs = Math.max(3, sec) * 1000;
  state.cycleEnabled = true;
  state.cycleIndex = 0;
  advanceCycle();
  if(state.cycleTimer) clearInterval(state.cycleTimer);
  state.cycleTimer = setInterval(advanceCycle, intervalMs);
  document.getElementById('btnCycle').textContent = '⏹ 순환 중지';
  document.getElementById('btnCycle').classList.add('active');
  updateModeText();
}

function stopCycle(){
  state.cycleEnabled = false;
  if(state.cycleTimer){ clearInterval(state.cycleTimer); state.cycleTimer = null; }
  document.getElementById('btnCycle').textContent = '🔄 순환 시작';
  document.getElementById('btnCycle').classList.remove('active');
  updateModeText();
}

function updateModeText(){
  const filterOn = hasActiveFilters();
  const parts = [];
  parts.push(filterOn ? '필터 모드' : '실시간 모드');
  if(state.cycleEnabled) parts.push('순환 ON');
  if(state.focusedChannelId) parts.push(`확대: ${state.focusedChannelId}`);
  document.getElementById('modeText').textContent = parts.join(' · ');
}

async function refreshStatus(){
  let runtimeHealth = {};
  try {
    const params = withToken(new URLSearchParams());
    const hr = await fetch(`/api/health?${params.toString()}`, {cache:'no-store'});
    if(hr.ok){
      const hj = await hr.json();
      runtimeHealth = hj.channels || {};
    }
  } catch(e) {}

  for (const ch of channels){
    const stEl = document.getElementById(`st_${ch.id}`);
    const dotEl = document.getElementById(`dot_${ch.id}`);
    const meta = document.getElementById(`meta_${ch.id}`);
    const started = performance.now();
    const inferenceActive = !!(runtimeHealth[ch.id] && runtimeHealth[ch.id].inference && runtimeHealth[ch.id].inference.active);
    try {
      const r = await fetch(statsUrl(ch.port), {cache:'no-store'});
      if(!r.ok) throw new Error(`HTTP ${r.status}`);
      const j = await r.json();
      const latencyMs = Math.round(performance.now() - started);
      state.channelHealth[ch.id] = { online:true, latencyMs, fps:j.fps||0, frames:j.frames||0, lastOkAt:Date.now(), inferenceActive };
      stEl.textContent = inferenceActive ? 'LIVE + AI' : 'LIVE';
      stEl.className = inferenceActive ? 'status-label ok' : 'status-label warn';
      dotEl.className = 'status-dot online';
      const fpsText = (j.fps || 0).toFixed ? (j.fps || 0).toFixed(1) : j.fps;
      meta.textContent = `${fpsText} fps · ${j.frames||0} frames · ${latencyMs}ms 지연 · ${inferenceActive ? 'AI ON' : 'AI OFF'}`;
    } catch(e){
      const prev = state.channelHealth[ch.id] || {};
      state.channelHealth[ch.id] = { ...prev, online:false, inferenceActive };
      const lastOk = prev.lastOkAt ? formatTime(prev.lastOkAt) : '-';
      if(inferenceActive){
        stEl.textContent = 'AI ONLY';
        stEl.className = 'status-label warn';
        dotEl.className = 'status-dot checking';
        meta.textContent = `미리보기 연결 실패 · 마지막 정상: ${lastOk} · AI ON`;
      } else {
        stEl.textContent = 'OFFLINE';
        stEl.className = 'status-label bad';
        dotEl.className = 'status-dot offline';
        meta.textContent = `연결 실패 · 마지막 정상: ${lastOk} · AI OFF`;
      }
    }
  }
}

function beep(){
  try {
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const o = ctx.createOscillator();
    const g = ctx.createGain();
    o.type = 'sine'; o.frequency.value = 880; g.gain.value = 0.04;
    o.connect(g); g.connect(ctx.destination);
    o.start(); setTimeout(() => { o.stop(); ctx.close(); }, 150);
  } catch(e){}
}

function eventCardHtml(ev){
  const type = ev.type || ev.event_type || 'unknown';
  const severity = ev.severity || 'warning';
  const score = Number(ev.score || 0).toFixed(2);
  const timeStr = ev.time_str || '-';
  return `
    <div class="ev ${type}">
      <div class="line">
        <span class="pill ${type}">${escapeHtml(String(type).toUpperCase())}</span>
        <span class="pill ${severity}">${escapeHtml(severity)}</span>
        <span style="color:var(--text-secondary)">${escapeHtml(ev.channel_name || ev.channel_id || ev.channel || '-')}</span>
      </div>
      <div class="meta">신뢰도 ${score} · ${escapeHtml(timeStr)} · ${escapeHtml(ev.source || '-')}</div>
    </div>
  `;
}

function setEvents(events){
  const box = document.getElementById('events');
  const count = document.getElementById('evCount');
  box.innerHTML = events.map(eventCardHtml).join('');
  count.textContent = `${events.length}건`;
  if(events.length > 0){
    const topId = Math.max(...events.map(e => e.id || 0));
    if(topId > state.lastRenderedTopId) beep();
    state.lastRenderedTopId = Math.max(state.lastRenderedTopId, topId);
  }
}

function prependEvents(events){
  if(!events || !events.length) return;
  const box = document.getElementById('events');
  const count = document.getElementById('evCount');
  events.slice().reverse().forEach(ev => {
    box.insertAdjacentHTML('afterbegin', eventCardHtml(ev));
  });
  while(box.children.length > 160) box.removeChild(box.lastChild);
  count.textContent = `${box.children.length}건`;
  beep();
}

function hasActiveFilters(){
  const f = state.filters;
  return Boolean(f.channel || f.type || f.severity || f.minScore || f.q || f.windowMin !== '0' || f.sort === 'asc');
}

function readFilterInputs(){
  state.filters.channel = document.getElementById('fChannel').value || '';
  state.filters.type = document.getElementById('fType').value || '';
  state.filters.severity = document.getElementById('fSeverity').value || '';
  state.filters.minScore = document.getElementById('fMinScore').value || '';
  state.filters.q = document.getElementById('fQ').value.trim() || '';
  state.filters.sort = document.getElementById('fSort').value || 'desc';
  state.filters.windowMin = document.getElementById('fWindowMin').value || '0';
  updateModeText();
}

function resetFilterInputs(){
  document.getElementById('fChannel').value = '';
  document.getElementById('fType').value = '';
  document.getElementById('fSeverity').value = '';
  document.getElementById('fMinScore').value = '';
  document.getElementById('fQ').value = '';
  document.getElementById('fSort').value = 'desc';
  document.getElementById('fWindowMin').value = '0';
  readFilterInputs();
}

function buildEventQuery(incremental){
  const params = new URLSearchParams();
  params.set('limit', hasActiveFilters() ? '160' : '80');
  if(incremental){ params.set('since', String(state.lastEventId)); return params; }
  const f = state.filters;
  if(f.channel) params.set('channel', f.channel);
  if(f.type) params.set('type', f.type);
  if(f.severity) params.set('severity', f.severity);
  if(f.minScore) params.set('min_score', f.minScore);
  if(f.q) params.set('q', f.q);
  if(f.sort) params.set('sort', f.sort);
  const windowMin = parseInt(f.windowMin || '0', 10);
  if(windowMin > 0){
    const nowSec = Math.floor(Date.now() / 1000);
    params.set('from_ts', String(nowSec - windowMin * 60));
    params.set('to_ts', String(nowSec));
  }
  return params;
}

async function pollEvents(){
  try {
    const incremental = !hasActiveFilters();
    const params = buildEventQuery(incremental);
    const r = await fetch(`/api/events?${withToken(params).toString()}`, {cache:'no-store'});
    if(!r.ok) throw new Error(`HTTP ${r.status}`);
    const j = await r.json();
    const events = Array.isArray(j.events) ? j.events : [];
    if(events.length > 0){
      const maxId = Math.max(...events.map(e => e.id || 0));
      state.lastEventId = Math.max(state.lastEventId, maxId);
    }
    if(incremental) prependEvents(events);
    else setEvents(events);
  } catch(e) {}
}

function buildStatsQuery(){
  const params = new URLSearchParams();
  const f = state.filters;
  const windowMin = parseInt(f.windowMin || '0', 10);
  if(windowMin > 0){
    const nowSec = Math.floor(Date.now() / 1000);
    params.set('from_ts', String(nowSec - windowMin * 60));
    params.set('to_ts', String(nowSec));
    document.getElementById('statsWindowText').textContent = `${windowMin}분`;
  } else {
    document.getElementById('statsWindowText').textContent = '전체';
  }
  return params;
}

function renderStatBars(title, data, cssPrefix){
  const entries = Object.entries(data || {});
  if(!entries.length) return `<div class="stat-section"><strong>${title}</strong><div style="font-size:11px;color:var(--text-muted);">데이터 없음</div></div>`;
  const maxVal = Math.max(...entries.map(([_,v])=>v), 1);
  const bars = entries.map(([k,v]) => {
    const pct = Math.round(v / maxVal * 100);
    const cls = cssPrefix ? `${cssPrefix}-${k}` : '';
    return `<div class="stat-bar-container">
      <div class="stat-bar-label"><span>${escapeHtml(k)}</span><span>${v}</span></div>
      <div class="stat-bar"><div class="stat-bar-fill ${cls}" style="width:${pct}%"></div></div>
    </div>`;
  }).join('');
  return `<div class="stat-section"><strong>${title}</strong>${bars}</div>`;
}

async function pollStats(){
  try {
    const params = buildStatsQuery();
    const r = await fetch(`/api/events/stats?${withToken(params).toString()}`, {cache:'no-store'});
    if(!r.ok) throw new Error(`HTTP ${r.status}`);
    const j = await r.json();
    const s = j.stats || {};
    document.getElementById('statsBody').innerHTML = [
      renderStatBars('유형별', s.by_type, 'type'),
      renderStatBars('채널별', s.by_channel, ''),
      renderStatBars('심각도별', s.by_severity, 'sev'),
    ].join('');
  } catch(e){
    document.getElementById('statsBody').textContent = '통계 조회 실패';
  }
}

function bindControlEvents(){
  document.getElementById('btnApply').addEventListener('click', () => { readFilterInputs(); pollEvents(); pollStats(); });
  document.getElementById('btnReset').addEventListener('click', () => { resetFilterInputs(); pollEvents(); pollStats(); });
  document.getElementById('btnCycle').addEventListener('click', () => { if(state.cycleEnabled) stopCycle(); else startCycle(); });
  document.getElementById('btnExitFocus').addEventListener('click', () => { setFocusedChannel(null); updateModeText(); });
}

function tickClock(){
  const now = new Date();
  const opts = { year:'numeric', month:'2-digit', day:'2-digit', hour:'2-digit', minute:'2-digit', second:'2-digit', hour12:false };
  document.getElementById('clock').textContent = now.toLocaleString('ko-KR', opts);
}

showRoleBadge();
initChannelFilterOptions();
makeGrid();
bindControlEvents();
readFilterInputs();
updateModeText();
refreshStatus();
pollEvents();
pollStats();
setInterval(refreshStatus, 2000);
setInterval(pollEvents, 1200);
setInterval(pollStats, 5000);
setInterval(tickClock, 1000);
tickClock();
</script>
</body>
</html>
"""



# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _clamp_int(value: Optional[str], default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value) if value is not None else default
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def _parse_float(value: Optional[str]) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_request_token() -> str:
    """Extract token from Authorization header or `token` query parameter."""
    auth = request.headers.get("Authorization", "").strip()
    if auth.lower().startswith("bearer "):
        return auth.split(" ", 1)[1].strip()
    return (request.args.get("token") or "").strip()


def infer_severity(event_type: str, score: float) -> str:
    """Infer severity from event type and confidence score."""
    event_type = (event_type or "").lower().strip()
    if event_type == "fall":
        return "critical" if score >= 0.9 else "warning"
    if event_type == "fight":
        return "critical" if score >= 0.85 else "warning"
    return "normal"


# ---------------------------------------------------------------------------
# Channel Health Monitor (Feature 2)
# ---------------------------------------------------------------------------

class ChannelHealthMonitor:
    """Monitor preview channels and auto-recover on repeated failures."""

    def __init__(self, hub: "AlertHub"):
        self.hub = hub
        self._channel_state: Dict[str, Dict[str, Any]] = {}
        for ch in CHANNELS:
            self._channel_state[ch["id"]] = {
                "online": False,
                "consecutive_failures": 0,
                "last_check_ts": None,
                "last_recovery_ts": None,
                "last_ok_ts": None,
            }
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self._thread.start()

    def _loop(self):
        while True:
            for ch in CHANNELS:
                self._check_channel(ch)
            time.sleep(HEALTH_CHECK_INTERVAL_SEC)

    def _check_channel(self, ch: Dict[str, Any]):
        channel_id = ch["id"]
        port = ch["port"]
        now = time.time()

        ok = False
        try:
            resp = requests.get(f"http://127.0.0.1:{port}/stats", timeout=5)
            ok = resp.status_code == 200
        except Exception:
            ok = False

        with self._lock:
            state = self._channel_state[channel_id]
            state["last_check_ts"] = now

            if ok:
                state["online"] = True
                state["consecutive_failures"] = 0
                state["last_ok_ts"] = now
            else:
                state["online"] = False
                state["consecutive_failures"] += 1
                failures = state["consecutive_failures"]

                logger.warning(
                    "Channel %s health check failed (%d/%d)",
                    channel_id, failures, HEALTH_MAX_FAILURES,
                )

                if failures >= HEALTH_MAX_FAILURES:
                    self._attempt_recovery(channel_id, state)

    def _attempt_recovery(self, channel_id: str, state: Dict[str, Any]):
        service_name = f"elevator-preview-{channel_id}.service"
        logger.error(
            "Channel %s failed %d consecutive checks — attempting auto-recovery: systemctl --user restart %s",
            channel_id, state["consecutive_failures"], service_name,
        )

        try:
            result = subprocess.run(
                ["systemctl", "--user", "restart", service_name],
                capture_output=True, text=True, timeout=15,
            )
            state["last_recovery_ts"] = time.time()
            state["consecutive_failures"] = 0  # reset after attempt

            if result.returncode == 0:
                logger.info("Auto-recovery of %s succeeded", service_name)
                recovery_msg = f"자동 복구 시도 성공: {service_name}"
            else:
                logger.error(
                    "Auto-recovery of %s failed (rc=%d): %s",
                    service_name, result.returncode, result.stderr.strip(),
                )
                recovery_msg = f"자동 복구 실패: {service_name} (rc={result.returncode})"
        except Exception as exc:
            state["last_recovery_ts"] = time.time()
            state["consecutive_failures"] = 0
            logger.error("Auto-recovery of %s raised exception: %s", service_name, exc)
            recovery_msg = f"자동 복구 예외: {service_name}: {exc}"

        # Push a CRITICAL alert about the recovery
        self.hub.push_system_alert(
            channel_id=channel_id,
            event_type="channel_recovery",
            severity="critical",
            message=recovery_msg,
        )

    def get_status(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return {k: dict(v) for k, v in self._channel_state.items()}


# ---------------------------------------------------------------------------
# AlertHub
# ---------------------------------------------------------------------------

class AlertHub:
    """Collect, persist, query, and relay anomaly events."""

    def __init__(self, cooldown_sec: int = 30, max_events: int = 500):
        self.cooldown_sec = cooldown_sec
        self.events = deque(maxlen=max_events)
        self.last_alert_at: Dict[str, float] = {}
        self._seq = 0
        self._lock = threading.Lock()
        self.start_ts = time.time()
        self.last_event_ts: Optional[float] = None

        # optional external alert targets
        self.telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        self.telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
        self.alert_webhook_url = os.getenv("ALERT_WEBHOOK_URL", "").strip()

        self.channel_map = {c["id"]: c for c in CHANNELS}

        # SQLite persistence
        self.db_path = os.getenv("EVENT_DB_PATH", "events.db").strip() or "events.db"
        self._db_lock = threading.Lock()
        self.db_conn: Optional[sqlite3.Connection] = None
        self.db_enabled = self._init_db()
        self._load_recent_from_db(max_events)

        # failed external alert retry queue
        self._retry_items: List[Dict[str, Any]] = []
        self._retry_lock = threading.Lock()
        self._retry_thread = threading.Thread(target=self._retry_loop, daemon=True)
        self._retry_thread.start()

        # --- Feature 1: warning digest queue ---
        self._warning_queue: List[Dict[str, Any]] = []
        self._warning_lock = threading.Lock()
        self._digest_thread = threading.Thread(target=self._digest_loop, daemon=True)
        self._digest_thread.start()

        # --- Feature 2: channel health monitor ---
        self.health_monitor = ChannelHealthMonitor(hub=self)
        self.health_monitor.start()

    # ------------------------------------------------------------------
    # SQLite
    # ------------------------------------------------------------------

    def _init_db(self) -> bool:
        try:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts REAL NOT NULL,
                    time_str TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    channel_id TEXT NOT NULL,
                    channel_name TEXT NOT NULL,
                    score REAL NOT NULL,
                    source TEXT NOT NULL,
                    raw_line TEXT,
                    metadata_json TEXT
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_channel ON events(channel_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_severity ON events(severity)")
            conn.commit()
            self.db_conn = conn
            logger.info("SQLite database initialized: %s", self.db_path)
            return True
        except Exception as exc:
            logger.warning("SQLite init failed: %s", exc)
            self.db_conn = None
            return False

    def _load_recent_from_db(self, limit: int):
        if not self.db_enabled or not self.db_conn:
            return
        try:
            with self._db_lock:
                rows = self.db_conn.execute(
                    """
                    SELECT id, ts, time_str, event_type, severity, channel_id, channel_name, score, source, raw_line, metadata_json
                    FROM events
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
                row_max_id = self.db_conn.execute("SELECT COALESCE(MAX(id), 0) AS max_id FROM events").fetchone()
                self._seq = int(row_max_id["max_id"] if row_max_id else 0)

            for row in rows:
                self.events.append(self._row_to_event(row))

            if self.events:
                self.last_event_ts = max(event["timestamp"] for event in self.events)

            logger.info("Loaded %d recent events from DB (seq=%d)", len(rows), self._seq)
        except Exception as exc:
            logger.warning("Failed to bootstrap in-memory queue from DB: %s", exc)

    def _row_to_event(self, row: sqlite3.Row) -> Dict[str, Any]:
        metadata = {}
        metadata_json = row["metadata_json"]
        if metadata_json:
            try:
                metadata = json.loads(metadata_json)
            except Exception:
                metadata = {}

        return {
            "id": int(row["id"]),
            "event_type": row["event_type"],
            "type": row["event_type"],
            "severity": row["severity"],
            "channel_id": row["channel_id"],
            "channel": row["channel_id"],
            "channel_name": row["channel_name"],
            "score": float(row["score"]),
            "timestamp": float(row["ts"]),
            "time": float(row["ts"]),
            "ts": float(row["ts"]),
            "time_str": row["time_str"],
            "source": row["source"],
            "metadata": metadata,
            "raw_line": row["raw_line"] or "",
        }

    def _next_id(self) -> int:
        with self._lock:
            self._seq += 1
            return self._seq

    def _should_emit(self, channel_id: str, event_type: str, now: float) -> bool:
        key = f"{channel_id}:{event_type}"
        last = self.last_alert_at.get(key, 0)
        if now - last < self.cooldown_sec:
            return False
        self.last_alert_at[key] = now
        return True

    def _persist_event(self, event: Dict[str, Any]) -> Optional[int]:
        if not self.db_enabled or not self.db_conn:
            return None
        try:
            metadata_json = json.dumps(event.get("metadata") or {}, ensure_ascii=False)
            with self._db_lock:
                cursor = self.db_conn.execute(
                    """
                    INSERT INTO events (ts, time_str, event_type, severity, channel_id, channel_name, score, source, raw_line, metadata_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event["timestamp"],
                        event["time_str"],
                        event["event_type"],
                        event["severity"],
                        event["channel_id"],
                        event["channel_name"],
                        event["score"],
                        event["source"],
                        event["raw_line"],
                        metadata_json,
                    ),
                )
                self.db_conn.commit()
                return int(cursor.lastrowid)
        except Exception as exc:
            logger.warning("Event persistence failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Event ingestion
    # ------------------------------------------------------------------

    def push(self, channel_id: str, event_type: str, score: float, raw_line: str, source: str = "deepstream-log"):
        now = time.time()
        if not self._should_emit(channel_id, event_type, now):
            return

        channel = self.channel_map.get(channel_id, {"name": channel_id})
        severity = infer_severity(event_type, score)
        time_iso = datetime.fromtimestamp(now, tz=timezone.utc).astimezone().isoformat(timespec="seconds")

        event = {
            "id": 0,
            "event_type": event_type,
            "type": event_type,
            "severity": severity,
            "channel_id": channel_id,
            "channel": channel_id,
            "channel_name": channel.get("name", channel_id),
            "score": float(score),
            "timestamp": now,
            "time": now,
            "ts": now,
            "time_str": time_iso,
            "source": source,
            "metadata": {"line_length": len(raw_line or "")},
            "raw_line": (raw_line or "").strip(),
        }

        persisted_id = self._persist_event(event)
        event["id"] = persisted_id if persisted_id is not None else self._next_id()

        self.events.appendleft(event)
        self.last_event_ts = now

        logger.info(
            "Event: type=%s severity=%s channel=%s score=%.2f",
            event_type, severity, channel_id, score,
        )

        self._route_alert(event)

    def push_system_alert(self, channel_id: str, event_type: str, severity: str, message: str):
        """Push a system-generated alert (e.g. auto-recovery)."""
        now = time.time()
        time_iso = datetime.fromtimestamp(now, tz=timezone.utc).astimezone().isoformat(timespec="seconds")
        channel = self.channel_map.get(channel_id, {"name": channel_id})

        event = {
            "id": 0,
            "event_type": event_type,
            "type": event_type,
            "severity": severity,
            "channel_id": channel_id,
            "channel": channel_id,
            "channel_name": channel.get("name", channel_id),
            "score": 0.0,
            "timestamp": now,
            "time": now,
            "ts": now,
            "time_str": time_iso,
            "source": "health-monitor",
            "metadata": {"message": message},
            "raw_line": message,
        }

        persisted_id = self._persist_event(event)
        event["id"] = persisted_id if persisted_id is not None else self._next_id()

        self.events.appendleft(event)
        self.last_event_ts = now

        logger.warning("System alert: %s — %s", event_type, message)
        self._route_alert(event)

    # ------------------------------------------------------------------
    # Feature 1: severity-based alert routing
    # ------------------------------------------------------------------

    def _route_alert(self, event: Dict[str, Any]):
        """Route alert based on severity: critical→immediate, warning→digest, info/normal→log only."""
        severity = event.get("severity", "normal").lower()

        if severity == "critical":
            self._send_external_immediate(event)
        elif severity == "warning":
            with self._warning_lock:
                self._warning_queue.append(event)
            logger.debug("Warning event queued for digest (queue size: %d)", len(self._warning_queue))
        else:
            # info/normal — log only, no external alert
            logger.debug("Info/normal event — no external alert")

    def _send_external_immediate(self, event: Dict[str, Any]):
        """Send CRITICAL alert immediately via Telegram AND webhook."""
        if self.telegram_bot_token and self.telegram_chat_id:
            text = (
                "🚨 [CRITICAL] 승강기 이상상황\n"
                f"- 채널: {event['channel_name']}\n"
                f"- 유형: {event['event_type']}\n"
                f"- 심각도: {event['severity']}\n"
                f"- 신뢰도: {event['score']:.2f}\n"
                f"- 시각: {event['time_str']}"
            )
            endpoint = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
            payload = {"chat_id": self.telegram_chat_id, "text": text}
            ok = self._send_target("telegram", endpoint, payload)
            if not ok:
                self._queue_retry("telegram", payload, endpoint, attempt=1)

        if self.alert_webhook_url:
            ok = self._send_target("webhook", self.alert_webhook_url, event)
            if not ok:
                self._queue_retry("webhook", event, self.alert_webhook_url, attempt=1)

    def _digest_loop(self):
        """Background thread: flush warning digest every ALERT_DIGEST_INTERVAL_SEC."""
        while True:
            time.sleep(ALERT_DIGEST_INTERVAL_SEC)
            self._flush_warning_digest()

    def _flush_warning_digest(self):
        """Collect queued warnings and send as a single digest message."""
        with self._warning_lock:
            if not self._warning_queue:
                return
            batch = list(self._warning_queue)
            self._warning_queue.clear()

        count = len(batch)
        logger.info("Flushing warning digest: %d events", count)

        lines = [f"⚠️ 승강기 경고 다이제스트 ({count}건)\n"]
        for ev in batch[:20]:  # cap at 20 to avoid huge messages
            lines.append(
                f"  • [{ev.get('event_type', '?')}] {ev.get('channel_name', '?')} "
                f"score={ev.get('score', 0):.2f} @ {ev.get('time_str', '-')}"
            )
        if count > 20:
            lines.append(f"  ... 외 {count - 20}건")

        digest_text = "\n".join(lines)

        if self.telegram_bot_token and self.telegram_chat_id:
            endpoint = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
            payload = {"chat_id": self.telegram_chat_id, "text": digest_text}
            ok = self._send_target("telegram", endpoint, payload)
            if not ok:
                self._queue_retry("telegram", payload, endpoint, attempt=1)

        if self.alert_webhook_url:
            digest_payload = {
                "type": "warning_digest",
                "count": count,
                "events": batch[:50],
                "text": digest_text,
            }
            ok = self._send_target("webhook", self.alert_webhook_url, digest_payload)
            if not ok:
                self._queue_retry("webhook", digest_payload, self.alert_webhook_url, attempt=1)

    # ------------------------------------------------------------------
    # Retry queue (unchanged logic, logging updated)
    # ------------------------------------------------------------------

    def _queue_retry(self, target: str, payload: Dict[str, Any], endpoint: str, attempt: int = 1):
        if attempt > 5:
            logger.error("Alert retry exhausted (5 attempts) for target=%s endpoint=%s", target, endpoint)
            return
        delay = min(2 ** attempt, 60)
        item = {
            "target": target,
            "payload": payload,
            "endpoint": endpoint,
            "attempt": attempt,
            "next_retry_at": time.time() + delay,
        }
        with self._retry_lock:
            self._retry_items.append(item)
        logger.debug("Queued retry #%d for %s (delay=%.0fs)", attempt, target, delay)

    def _retry_loop(self):
        while True:
            due_items: List[Dict[str, Any]] = []
            now = time.time()
            with self._retry_lock:
                remaining = []
                for item in self._retry_items:
                    if item["next_retry_at"] <= now:
                        due_items.append(item)
                    else:
                        remaining.append(item)
                self._retry_items = remaining

            for item in due_items:
                ok = self._send_target(item["target"], item["endpoint"], item["payload"])
                if not ok:
                    self._queue_retry(
                        target=item["target"],
                        payload=item["payload"],
                        endpoint=item["endpoint"],
                        attempt=item["attempt"] + 1,
                    )

            time.sleep(1.0)

    def _send_target(self, target: str, endpoint: str, payload: Dict[str, Any]) -> bool:
        try:
            if target == "telegram":
                requests.post(endpoint, json=payload, timeout=3)
                return True
            if target == "webhook":
                requests.post(endpoint, json=payload, timeout=3)
                return True
            return False
        except Exception as exc:
            logger.debug("External alert send failed (%s): %s", target, exc)
            return False

    # ------------------------------------------------------------------
    # Query / stats (unchanged logic, logging updated)
    # ------------------------------------------------------------------

    def _filter_memory_events(
        self,
        since_id: int,
        limit: int,
        channel: Optional[str],
        event_type: Optional[str],
        min_score: Optional[float],
        severity: Optional[str],
        from_ts: Optional[float],
        to_ts: Optional[float],
        query_text: Optional[str],
        sort: str,
    ) -> List[Dict[str, Any]]:
        rows = list(self.events)

        def matched(e: Dict[str, Any]) -> bool:
            if since_id > 0 and e["id"] <= since_id:
                return False
            if channel and e["channel_id"] != channel:
                return False
            if event_type and e["event_type"] != event_type:
                return False
            if min_score is not None and e["score"] < min_score:
                return False
            if severity and e["severity"] != severity:
                return False
            ts = e["timestamp"]
            if from_ts is not None and ts < from_ts:
                return False
            if to_ts is not None and ts > to_ts:
                return False
            if query_text and query_text not in (e.get("raw_line") or ""):
                return False
            return True

        filtered = [e for e in rows if matched(e)]
        reverse = sort != "asc"
        filtered.sort(key=lambda e: e["id"], reverse=reverse)
        return filtered[:limit]

    def query_events(
        self,
        since_id: int = 0,
        limit: int = 50,
        channel: Optional[str] = None,
        event_type: Optional[str] = None,
        min_score: Optional[float] = None,
        severity: Optional[str] = None,
        from_ts: Optional[float] = None,
        to_ts: Optional[float] = None,
        query_text: Optional[str] = None,
        sort: str = "desc",
    ) -> List[Dict[str, Any]]:
        if not self.db_enabled or not self.db_conn:
            return self._filter_memory_events(
                since_id=since_id, limit=limit, channel=channel, event_type=event_type,
                min_score=min_score, severity=severity, from_ts=from_ts, to_ts=to_ts,
                query_text=query_text, sort=sort,
            )

        sql = (
            "SELECT id, ts, time_str, event_type, severity, channel_id, channel_name, score, source, raw_line, metadata_json "
            "FROM events WHERE 1=1"
        )
        params: List[Any] = []

        if since_id > 0:
            sql += " AND id > ?"
            params.append(since_id)
        if channel:
            sql += " AND channel_id = ?"
            params.append(channel)
        if event_type:
            sql += " AND event_type = ?"
            params.append(event_type)
        if min_score is not None:
            sql += " AND score >= ?"
            params.append(min_score)
        if severity:
            sql += " AND severity = ?"
            params.append(severity)
        if from_ts is not None:
            sql += " AND ts >= ?"
            params.append(from_ts)
        if to_ts is not None:
            sql += " AND ts <= ?"
            params.append(to_ts)
        if query_text:
            sql += " AND raw_line LIKE ?"
            params.append(f"%{query_text}%")

        direction = "ASC" if sort == "asc" else "DESC"
        sql += f" ORDER BY id {direction} LIMIT ?"
        params.append(limit)

        try:
            with self._db_lock:
                rows = self.db_conn.execute(sql, tuple(params)).fetchall()
            return [self._row_to_event(row) for row in rows]
        except Exception as exc:
            logger.warning("DB query failed, fallback to memory queue: %s", exc)
            return self._filter_memory_events(
                since_id=since_id, limit=limit, channel=channel, event_type=event_type,
                min_score=min_score, severity=severity, from_ts=from_ts, to_ts=to_ts,
                query_text=query_text, sort=sort,
            )

    def event_stats(self, from_ts: Optional[float], to_ts: Optional[float]) -> Dict[str, Dict[str, int]]:
        by_type = defaultdict(int)
        by_channel = defaultdict(int)
        by_severity = defaultdict(int)

        if self.db_enabled and self.db_conn:
            where = "WHERE 1=1"
            params: List[Any] = []
            if from_ts is not None:
                where += " AND ts >= ?"
                params.append(from_ts)
            if to_ts is not None:
                where += " AND ts <= ?"
                params.append(to_ts)
            try:
                with self._db_lock:
                    for row in self.db_conn.execute(
                        f"SELECT event_type, COUNT(*) AS cnt FROM events {where} GROUP BY event_type",
                        tuple(params),
                    ).fetchall():
                        by_type[row["event_type"]] = int(row["cnt"])
                    for row in self.db_conn.execute(
                        f"SELECT channel_id, COUNT(*) AS cnt FROM events {where} GROUP BY channel_id",
                        tuple(params),
                    ).fetchall():
                        by_channel[row["channel_id"]] = int(row["cnt"])
                    for row in self.db_conn.execute(
                        f"SELECT severity, COUNT(*) AS cnt FROM events {where} GROUP BY severity",
                        tuple(params),
                    ).fetchall():
                        by_severity[row["severity"]] = int(row["cnt"])
            except Exception as exc:
                logger.warning("Stats aggregation failed: %s", exc)
        else:
            for event in self.events:
                ts = event["timestamp"]
                if from_ts is not None and ts < from_ts:
                    continue
                if to_ts is not None and ts > to_ts:
                    continue
                by_type[event["event_type"]] += 1
                by_channel[event["channel_id"]] += 1
                by_severity[event["severity"]] += 1

        return {
            "by_type": dict(by_type),
            "by_channel": dict(by_channel),
            "by_severity": dict(by_severity),
        }

    def retry_queue_size(self) -> int:
        with self._retry_lock:
            return len(self._retry_items)

    def warning_queue_size(self) -> int:
        with self._warning_lock:
            return len(self._warning_queue)


# ---------------------------------------------------------------------------
# Log tail
# ---------------------------------------------------------------------------

def tail_file(path: str, on_line):
    p = Path(path)
    while True:
        if not p.exists():
            time.sleep(1.0)
            continue

        try:
            with p.open("r", encoding="utf-8", errors="ignore") as f:
                f.seek(0, os.SEEK_END)
                inode = os.fstat(f.fileno()).st_ino

                while True:
                    line = f.readline()
                    if line:
                        on_line(line)
                        continue

                    try:
                        if p.exists() and p.stat().st_ino != inode:
                            break
                    except Exception:
                        break

                    time.sleep(0.2)
        except Exception as exc:
            logger.debug("tail_file error on %s: %s", path, exc)
            time.sleep(1.0)


def parse_line(channel_id: str, line: str, hub: AlertHub):
    m = FALL_RE.search(line)
    if m:
        hub.push(channel_id, "fall", float(m.group(1)), line)
        return

    m = FIGHT_RE.search(line)
    if m:
        hub.push(channel_id, "fight", float(m.group(1)), line)
        return


# ---------------------------------------------------------------------------
# Feature 4: Role-based access control
# ---------------------------------------------------------------------------

def _resolve_role(token: str, admin_token: str, viewer_token: str, legacy_token: str) -> Tuple[bool, str]:
    """
    Determine role from provided token.
    Returns (authorized: bool, role: str).
    If no tokens are configured, everyone is admin (open access).
    """
    token_enabled = bool(admin_token or viewer_token or legacy_token)

    if not token_enabled:
        return True, "admin"

    if not token:
        return False, ""

    if admin_token and token == admin_token:
        return True, "admin"
    if legacy_token and token == legacy_token:
        return True, "admin"  # backward compat
    if viewer_token and token == viewer_token:
        return True, "viewer"

    return False, ""


def create_app(hub: AlertHub):
    app = Flask(__name__)

    # --- Feature 4: role-based tokens ---
    admin_token = os.getenv("DASHBOARD_ADMIN_TOKEN", "").strip()
    viewer_token = os.getenv("DASHBOARD_VIEWER_TOKEN", "").strip()
    legacy_token = os.getenv("DASHBOARD_ACCESS_TOKEN", "").strip()  # backward compat → admin
    token_enabled = bool(admin_token or viewer_token or legacy_token)

    def _get_role() -> Tuple[bool, str]:
        provided = _extract_request_token()
        return _resolve_role(provided, admin_token, viewer_token, legacy_token)

    def any_role_required(view_func):
        """Allow any authenticated user (admin or viewer)."""
        @wraps(view_func)
        def _wrapped(*args, **kwargs):
            authorized, role = _get_role()
            if not authorized:
                return jsonify({"error": "unauthorized", "message": "valid token required"}), 401
            return view_func(*args, role=role, **kwargs)
        return _wrapped

    def admin_required(view_func):
        """Allow only admin users."""
        @wraps(view_func)
        def _wrapped(*args, **kwargs):
            authorized, role = _get_role()
            if not authorized:
                return jsonify({"error": "unauthorized", "message": "valid token required"}), 401
            if role != "admin":
                return jsonify({"error": "forbidden", "message": "admin access required"}), 403
            return view_func(*args, role=role, **kwargs)
        return _wrapped

    @app.route("/")
    def index():
        authorized, role = _get_role()
        if not authorized:
            return jsonify({"error": "unauthorized", "message": "valid token required"}), 401
        return render_template_string(HTML, channels=CHANNELS, user_role=role)

    @app.route("/api/events")
    @any_role_required
    def api_events(role="viewer"):
        since_id = _clamp_int(request.args.get("since_id") or request.args.get("since"), default=0, minimum=0, maximum=10**9)
        limit = _clamp_int(request.args.get("limit"), default=50, minimum=1, maximum=500)

        channel = (request.args.get("channel") or "").strip() or None
        event_type = (request.args.get("type") or "").strip() or None
        severity = (request.args.get("severity") or "").strip() or None
        min_score = _parse_float(request.args.get("min_score"))
        from_ts = _parse_float(request.args.get("from_ts"))
        to_ts = _parse_float(request.args.get("to_ts"))
        query_text = (request.args.get("q") or "").strip() or None
        sort = (request.args.get("sort") or "desc").lower().strip()
        if sort not in ("asc", "desc"):
            sort = "desc"

        events = hub.query_events(
            since_id=since_id, limit=limit, channel=channel, event_type=event_type,
            min_score=min_score, severity=severity, from_ts=from_ts, to_ts=to_ts,
            query_text=query_text, sort=sort,
        )

        return jsonify({"events": events, "count": len(events)})

    @app.route("/api/events/stats")
    @any_role_required
    def api_event_stats(role="viewer"):
        from_ts = _parse_float(request.args.get("from_ts"))
        to_ts = _parse_float(request.args.get("to_ts"))
        stats = hub.event_stats(from_ts=from_ts, to_ts=to_ts)
        return jsonify({"from_ts": from_ts, "to_ts": to_ts, "stats": stats})

    @app.route("/api/channel-sources")
    @any_role_required
    def api_channel_sources(role="viewer"):
        return jsonify({"channels": _load_channel_sources()})

    @app.route("/api/channel-sources", methods=["POST"])
    @admin_required
    def api_channel_sources_update(role="admin"):
        payload = request.get_json(silent=True) or {}
        raw_channels = payload.get("channels") or {}
        config = _load_channel_sources()
        for ch in CHANNELS:
            if ch["id"] in raw_channels:
                config[ch["id"]] = _normalize_channel_source(ch["id"], raw_channels.get(ch["id"]))

        errors = _validate_channel_sources(config)
        if errors:
            return jsonify({"error": "validation_failed", "message": "invalid channel source settings", "errors": errors}), 400

        results = _apply_channel_sources(config)
        return jsonify({"status": "ok", "channels": _load_channel_sources(), "results": results})

    @app.route("/api/health")
    @any_role_required
    def health(role="viewer"):
        channel_health = _augment_channel_health(hub.health_monitor.get_status())
        return jsonify({
            "status": "ok",
            "time": time.time(),
            "uptime_sec": int(time.time() - hub.start_ts),
            "event_queue_size": len(hub.events),
            "db_enabled": hub.db_enabled,
            "last_event_ts": hub.last_event_ts,
            "retry_queue_size": hub.retry_queue_size(),
            "warning_queue_size": hub.warning_queue_size(),
            "token_enabled": token_enabled,
            "channels": channel_health,
        })

    @app.route("/api/channels/<channel_id>/restart", methods=["POST"])
    @admin_required
    def restart_channel(channel_id, role="admin"):
        """Admin-only: restart preview + inference services for a channel."""
        valid_ids = {ch["id"] for ch in CHANNELS}
        if channel_id not in valid_ids:
            return jsonify({"error": "not_found", "message": f"Unknown channel: {channel_id}"}), 404

        logger.info("Admin requested restart of channel %s", channel_id)
        try:
            ok, message = _restart_channel_services(channel_id, _load_channel_sources())
            if ok:
                logger.info("Manual restart of %s succeeded: %s", channel_id, message)
                return jsonify({"status": "ok", "message": message})
            logger.error("Manual restart of %s failed: %s", channel_id, message)
            return jsonify({"error": "restart_failed", "message": message}), 500
        except Exception as exc:
            logger.error("Manual restart of %s raised exception: %s", channel_id, exc)
            return jsonify({"error": "exception", "message": str(exc)}), 500

    return app


# ---------------------------------------------------------------------------
# Feature 3: Logging setup
# ---------------------------------------------------------------------------

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """Configure structured logging with rotation."""
    level = getattr(logging, log_level.upper(), logging.INFO)

    fmt = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
    formatter = logging.Formatter(fmt)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Console handler (always)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler with rotation (if configured)
    if log_file is None:
        log_file = os.getenv("LOG_FILE", "").strip()

    if not log_file:
        log_file = "logs/dashboard.log"

    try:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        logger.info("Log file: %s (rotation: 10MB x 5)", log_file)
    except Exception as exc:
        logger.warning("Failed to set up file logging at %s: %s", log_file, exc)

    # Quieten noisy libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("werkzeug").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Elevator 4-channel dashboard + alert hub")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7000)
    parser.add_argument("--cooldown", type=int, default=30, help="alert cooldown per channel/event")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="logging level (default: INFO)")
    args = parser.parse_args()

    # Feature 3: structured logging
    setup_logging(log_level=args.log_level)

    logger.info("Starting Elevator Dashboard on %s:%d (cooldown=%ds)", args.host, args.port, args.cooldown)
    logger.info("Alert digest interval: %ds, Health check interval: %ds, Max failures: %d",
                ALERT_DIGEST_INTERVAL_SEC, HEALTH_CHECK_INTERVAL_SEC, HEALTH_MAX_FAILURES)

    hub = AlertHub(cooldown_sec=args.cooldown)

    # log tail workers
    for ch in CHANNELS:
        t = threading.Thread(
            target=tail_file,
            args=(ch["ds_log"], lambda line, cid=ch["id"]: parse_line(cid, line, hub)),
            daemon=True,
        )
        t.start()
        logger.info("Log tailer started for channel %s: %s", ch["id"], ch["ds_log"])

    app = create_app(hub)
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
