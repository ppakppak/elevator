#!/usr/bin/env python3
"""
Elevator 4-channel dashboard + anomaly alert hub
- 4-way preview grid (ports 5000~5003)
- DeepStream log tailing (fall/fight events)
- In-browser live alert feed + optional Telegram/webhook alerts
"""

import argparse
import json
import os
import re
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import requests
from flask import Flask, jsonify, render_template_string, request


CHANNELS = [
    {"id": "webcam", "name": "Webcam", "port": 5000, "ds_log": "/home/ppak/projects/elevator/deepstream_pose/logs/elevator-ds-webcam.out.log"},
    {"id": "rtsp", "name": "RTSP", "port": 5001, "ds_log": "/home/ppak/projects/elevator/deepstream_pose/logs/elevator-ds-rtsp.out.log"},
    {"id": "video1", "name": "Video 1", "port": 5002, "ds_log": "/home/ppak/projects/elevator/deepstream_pose/logs/elevator-ds-video1.out.log"},
    {"id": "video2", "name": "Video 2", "port": 5003, "ds_log": "/home/ppak/projects/elevator/deepstream_pose/logs/elevator-ds-video2.out.log"},
]

FALL_RE = re.compile(r"\[쓰러짐 감지\].*신뢰도:\s*([0-9.]+)")
FIGHT_RE = re.compile(r"\[싸움 감지\].*신뢰도:\s*([0-9.]+)")

HTML = """
<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Elevator Safety Dashboard</title>
  <style>
    body { margin:0; font-family:Arial,sans-serif; background:#0f172a; color:#e2e8f0; }
    .top { display:flex; justify-content:space-between; align-items:center; padding:10px 14px; background:#111827; border-bottom:1px solid #1f2937; }
    .title { font-weight:700; font-size:18px; }
    .sub { font-size:12px; color:#93c5fd; }
    .layout { display:grid; grid-template-columns:2fr 1fr; gap:10px; padding:10px; }
    .grid { display:grid; grid-template-columns:1fr 1fr; gap:10px; }
    .card { background:#111827; border:1px solid #1f2937; border-radius:10px; overflow:hidden; }
    .head { display:flex; justify-content:space-between; padding:8px 10px; font-size:13px; background:#0b1220; }
    .ok { color:#34d399; }
    .bad { color:#f87171; }
    .warn { color:#fbbf24; }
    img { width:100%; height:280px; object-fit:contain; background:#000; display:block; }
    .stats { font-size:12px; color:#cbd5e1; padding:6px 10px; border-top:1px solid #1f2937; }
    .side { display:flex; flex-direction:column; gap:10px; }
    .events { max-height:620px; overflow:auto; padding:8px; }
    .ev { border:1px solid #334155; border-radius:8px; padding:8px; margin-bottom:8px; background:#0b1220; }
    .ev.fall { border-color:#f43f5e; }
    .ev.fight { border-color:#f59e0b; }
    .ev .meta { font-size:12px; color:#cbd5e1; }
    .ev .line { font-size:14px; font-weight:600; }
    .pill { padding:2px 8px; border-radius:99px; font-size:11px; font-weight:700; }
    .pill.fall { background:#7f1d1d; color:#fecaca; }
    .pill.fight { background:#78350f; color:#fde68a; }
    .hint { font-size:12px; color:#94a3b8; padding:8px 10px; border-top:1px solid #1f2937; }
  </style>
</head>
<body>
  <div class="top">
    <div>
      <div class="title">승강기 이상상황 통합 대시보드 (4채널)</div>
      <div class="sub">실시간 모니터링 + 이벤트 알람</div>
    </div>
    <div id="clock" class="sub"></div>
  </div>

  <div class="layout">
    <div class="grid" id="grid"></div>

    <div class="side">
      <div class="card">
        <div class="head"><strong>실시간 알람</strong><span id="evCount" class="sub"></span></div>
        <div class="events" id="events"></div>
        <div class="hint">알람은 채널/이벤트별 쿨다운 적용(중복 폭주 방지)</div>
      </div>
      <div class="card">
        <div class="head"><strong>운영 메모</strong></div>
        <div class="hint">
          - Webcam은 장치 점유 특성상 추론/미리보기 동시 사용이 제한될 수 있음<br/>
          - Telegram/Webhook 알람은 서버 환경변수 설정 시 자동 전송
        </div>
      </div>
    </div>
  </div>

<script>
const channels = {{ channels|tojson }};
let lastEventId = 0;

function streamUrl(port){ return `http://${location.hostname}:${port}/video_feed`; }
function statsUrl(port){ return `http://${location.hostname}:${port}/stats`; }

function makeGrid(){
  const grid = document.getElementById('grid');
  grid.innerHTML = '';
  channels.forEach(ch => {
    const card = document.createElement('div');
    card.className = 'card';
    card.innerHTML = `
      <div class="head">
        <div><strong>${ch.name}</strong></div>
        <div id="st_${ch.id}" class="warn">확인중...</div>
      </div>
      <img id="img_${ch.id}" src="${streamUrl(ch.port)}" alt="${ch.name}"/>
      <div class="stats" id="meta_${ch.id}">port:${ch.port}</div>
    `;
    grid.appendChild(card);
  });
}

async function refreshStatus(){
  for (const ch of channels){
    const stEl = document.getElementById(`st_${ch.id}`);
    const meta = document.getElementById(`meta_${ch.id}`);
    try {
      const r = await fetch(statsUrl(ch.port), {cache:'no-store'});
      if(!r.ok) throw new Error(`HTTP ${r.status}`);
      const j = await r.json();
      stEl.textContent = 'ONLINE';
      stEl.className = 'ok';
      meta.textContent = `fps:${(j.fps||0).toFixed ? (j.fps||0).toFixed(2) : j.fps} | frames:${j.frames||0}`;
    } catch(e){
      stEl.textContent = 'OFFLINE';
      stEl.className = 'bad';
      meta.textContent = `port:${ch.port} 연결 실패`;
    }
  }
}

function beep(){
  const ctx = new (window.AudioContext || window.webkitAudioContext)();
  const o = ctx.createOscillator();
  const g = ctx.createGain();
  o.type = 'sine';
  o.frequency.value = 880;
  g.gain.value = 0.05;
  o.connect(g); g.connect(ctx.destination);
  o.start();
  setTimeout(() => { o.stop(); ctx.close(); }, 220);
}

function addEvents(events){
  if(!events || !events.length) return;
  const box = document.getElementById('events');
  const count = document.getElementById('evCount');

  // 새 이벤트 먼저(내림차순) 전달됨
  events.slice().reverse().forEach(ev => {
    const d = document.createElement('div');
    d.className = `ev ${ev.type}`;
    d.innerHTML = `
      <div class="line"><span class="pill ${ev.type}">${ev.type.toUpperCase()}</span> ${ev.channel_name}</div>
      <div class="meta">score=${ev.score.toFixed(2)} | ${ev.time_str}</div>
      <div class="meta">${ev.raw_line || ''}</div>
    `;
    box.prepend(d);
  });

  while (box.children.length > 120) box.removeChild(box.lastChild);
  count.textContent = `최근 ${box.children.length}건`;
  beep();
}

async function pollEvents(){
  try {
    const r = await fetch(`/api/events?since=${lastEventId}`, {cache:'no-store'});
    const j = await r.json();
    if(j.events && j.events.length){
      addEvents(j.events);
      lastEventId = Math.max(lastEventId, ...j.events.map(e => e.id));
    }
  } catch(e) {}
}

function tickClock(){
  document.getElementById('clock').textContent = new Date().toLocaleString();
}

makeGrid();
refreshStatus();
setInterval(refreshStatus, 2000);
setInterval(pollEvents, 1000);
setInterval(tickClock, 1000);
tickClock();
</script>
</body>
</html>
"""


class AlertHub:
    def __init__(self, cooldown_sec: int = 30, max_events: int = 500):
        self.cooldown_sec = cooldown_sec
        self.events = deque(maxlen=max_events)
        self.last_alert_at: Dict[str, float] = {}
        self._seq = 0
        self._lock = threading.Lock()

        # optional external alert targets
        self.telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        self.telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
        self.alert_webhook_url = os.getenv("ALERT_WEBHOOK_URL", "").strip()

        self.channel_map = {c["id"]: c for c in CHANNELS}

    def _next_id(self) -> int:
        with self._lock:
            self._seq += 1
            return self._seq

    def _should_emit(self, channel_id: str, event_type: str, now: float) -> bool:
        k = f"{channel_id}:{event_type}"
        last = self.last_alert_at.get(k, 0)
        if now - last < self.cooldown_sec:
            return False
        self.last_alert_at[k] = now
        return True

    def push(self, channel_id: str, event_type: str, score: float, raw_line: str):
        now = time.time()
        if not self._should_emit(channel_id, event_type, now):
            return

        ch = self.channel_map.get(channel_id, {"name": channel_id})
        event = {
            "id": self._next_id(),
            "time": now,
            "time_str": datetime.fromtimestamp(now).strftime("%Y-%m-%d %H:%M:%S"),
            "channel": channel_id,
            "channel_name": ch.get("name", channel_id),
            "type": event_type,
            "score": float(score),
            "raw_line": raw_line.strip(),
        }
        self.events.appendleft(event)

        self._send_external(event)

    def _send_external(self, event: Dict):
        if self.telegram_bot_token and self.telegram_chat_id:
            try:
                text = (
                    f"🚨 승강기 이상상황\n"
                    f"- 채널: {event['channel_name']}\n"
                    f"- 유형: {event['type']}\n"
                    f"- 신뢰도: {event['score']:.2f}\n"
                    f"- 시각: {event['time_str']}"
                )
                requests.post(
                    f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage",
                    json={"chat_id": self.telegram_chat_id, "text": text},
                    timeout=3,
                )
            except Exception:
                pass

        if self.alert_webhook_url:
            try:
                requests.post(self.alert_webhook_url, json=event, timeout=3)
            except Exception:
                pass

    def recent(self, limit: int = 50, since_id: int = 0) -> List[Dict]:
        if since_id > 0:
            return [e for e in list(self.events) if e["id"] > since_id][:limit]
        return list(self.events)[:limit]


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

                    # rotated/truncated check
                    try:
                        if p.exists() and p.stat().st_ino != inode:
                            break
                    except Exception:
                        break

                    time.sleep(0.2)
        except Exception:
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


def create_app(hub: AlertHub):
    app = Flask(__name__)

    @app.route("/")
    def index():
        return render_template_string(HTML, channels=CHANNELS)

    @app.route("/api/events")
    def api_events():
        since = int(request.args.get("since", "0") or 0)
        limit = int(request.args.get("limit", "50") or 50)
        return jsonify({"events": hub.recent(limit=limit, since_id=since)})

    @app.route("/api/health")
    def health():
        return jsonify({"status": "ok", "events": len(hub.events), "time": time.time()})

    return app


def main():
    parser = argparse.ArgumentParser(description="Elevator 4-channel dashboard + alert hub")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7000)
    parser.add_argument("--cooldown", type=int, default=30, help="alert cooldown per channel/event")
    args = parser.parse_args()

    hub = AlertHub(cooldown_sec=args.cooldown)

    # log tail workers
    for ch in CHANNELS:
        t = threading.Thread(
            target=tail_file,
            args=(ch["ds_log"], lambda line, cid=ch["id"]: parse_line(cid, line, hub)),
            daemon=True,
        )
        t.start()

    app = create_app(hub)
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
