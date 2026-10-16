#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Chay radar.py va stream giao dien len web (MJPEG).
Chay:  python web.py [--cam 0] [--port 8000] [--hfov 70 --height 1.5 --pitch 25]
Mo:    http://localhost:8000  (hoac http://<IP-laptop>:8000 tu dien thoai cung wifi)
"""
import argparse
import math
import socket
import subprocess
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2

import radar as R

PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Webcam Radar</title>
<style>body{margin:0;background:#111;color:#ddd;font:14px sans-serif;text-align:center}
img{max-width:100%;cursor:crosshair}button{margin:4px;padding:10px 14px;font-size:15px}
#calib.on{background:#fc0}</style></head><body>
<img id="v" src="/stream"><div>
<button id="calib" onclick="cmd('calib');this.classList.toggle('on')">Hieu chinh (click diem cham san)</button>
<button onclick="cmd('pitch_up')">Cui +1°</button><button onclick="cmd('pitch_down')">Cui -1°</button>
<button onclick="cmd('h_up')">Cao +5cm</button><button onclick="cmd('h_down')">Cao -5cm</button>
<button onclick="cmd('flip')">Lat</button><button onclick="cmd('mute')">Am thanh</button></div>
<script>
const v=document.getElementById('v'), cb=document.getElementById('calib');
function cmd(c,q=''){fetch('/cmd?c='+c+q)}
v.onclick=e=>{if(!cb.classList.contains('on'))return;
 const r=v.getBoundingClientRect(), s=v.naturalWidth/r.width;
 cmd('click','&x='+Math.round((e.clientX-r.left)*s)+'&y='+Math.round((e.clientY-r.top)*s));
 cb.classList.remove('on')}
</script></body></html>"""


class App:
    def __init__(self, args):
        from ultralytics import YOLO
        self.model = YOLO(args.model)
        self.pose = None if args.no_pose else YOLO(args.pose_model)
        self.imgsz = args.imgsz
        self.cap = cv2.VideoCapture(args.video if args.video else args.cam)
        if not args.video:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        ok, frame = self.cap.read()
        if not ok:
            raise SystemExit("Khong mo duoc camera/video.")
        self.video = args.video
        self.h, self.w = frame.shape[:2]
        self.cam = R.Camera(self.w, self.h, args.hfov, args.height, args.pitch)
        self.tracker = R.SectorTracker(args.hfov)
        self.beeper = R.Beeper()
        self.beeper.start()
        self.calib, self.click, self.flip = False, None, False
        self.jpg = None
        self.cond = threading.Condition()
        self.tilt, self.roll, self.offset = None, 0.0, 0.0   # goc cui tu cam bien (do)
        if not args.no_sensor:
            threading.Thread(target=self.read_sensor, daemon=True).start()

    def read_sensor(self):
        """Doc gia toc ke man hinh qua PowerShell (WinRT) -> goc cui cua webcam.
        Man hinh dung thang: y = -1g. Ngua ra sau -> z am -> cam huong len."""
        ps = ("[Windows.Devices.Sensors.Accelerometer,Windows.Devices.Sensors,ContentType=WindowsRuntime]|Out-Null;"
              "$s=[Windows.Devices.Sensors.Accelerometer]::GetDefault(); if(!$s){exit};"
              "$c=[cultureinfo]::InvariantCulture; while($true){$r=$s.GetCurrentReading();"
              "[Console]::WriteLine([string]::Format($c,'{0} {1} {2}',$r.AccelerationX,$r.AccelerationY,$r.AccelerationZ));"
              "[Console]::Out.Flush(); Start-Sleep -Milliseconds 100}")
        try:
            p = subprocess.Popen(["powershell", "-NoProfile", "-Command", ps],
                                 stdout=subprocess.PIPE, text=True, creationflags=subprocess.CREATE_NO_WINDOW)
        except (OSError, AttributeError):
            return
        for line in p.stdout:
            try:
                x, y, z = map(float, line.split())
            except ValueError:
                continue
            pitch = math.degrees(math.atan2(z, -y))
            roll = math.degrees(math.atan2(x, -y))
            # ponytail: EMA chong rung khi cam lap di bo; tang 0.2 neu muon phan ung nhanh hon
            self.tilt = pitch if self.tilt is None else self.tilt + 0.2 * (pitch - self.tilt)
            self.roll += 0.2 * (roll - self.roll)

    def loop(self):
        while True:                    # loi bat ngo -> in ra va chay tiep, khong de UI dung hinh
            try:
                self._loop()
            except Exception:
                traceback.print_exc()
                self.beeper.level = 2  # trong luc loi coi nhu nguy hiem

    def _loop(self):
        fps, t_prev = 0.0, time.time()
        while True:
            ok, frame = self.cap.read()
            if not ok:
                if self.video:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                self.beeper.level = 2               # mat tin hieu -> bao DO
                time.sleep(0.2)
                continue
            if self.flip:
                frame = cv2.flip(frame, 1)
            if self.tilt is not None:
                self.cam.pitch = math.radians(self.tilt + self.offset)
            # Co pose: model detect chi tim do vat, nguoi lay tu model pose (co keypoint)
            classes = [c for c in R.CLASSES if c] if self.pose else list(R.CLASSES)
            res = self.model.predict(frame, imgsz=self.imgsz, conf=R.CONF,
                                     classes=classes, verbose=False)[0]
            dets = [(b, c, None) for b, c in zip(res.boxes.xyxy.cpu().numpy(),
                                                 res.boxes.cls.cpu().numpy().astype(int))]
            if self.pose:
                pr = self.pose.predict(frame, imgsz=self.imgsz, conf=R.CONF, verbose=False)[0]
                kps = pr.keypoints.data.cpu().numpy() if pr.keypoints is not None else []
                dets += [(b, 0, k) for b, k in zip(pr.boxes.xyxy.cpu().numpy(), kps)]
            objs = []
            for box, c, kp in dets:
                e = R.estimate(self.cam, box, c)
                if kp is not None and (e is None or e[2] in ("sat", "hc", "cao")):
                    p = R.person_distance(self.cam, box, kp)    # chan khong tin duoc -> dung kich thuoc co the
                    if p:
                        e = (*p, "pose")
                    for x, y, s in kp:
                        if s > R.KP_CONF:
                            cv2.circle(frame, (int(x), int(y)), 3, (255, 255, 0), -1)
                if e:
                    X, Z, mode = e
                    objs.append(dict(box=box, cls=c, name=R.CLASSES[c], X=X, Z=Z,
                                     r=math.hypot(X, Z), mode=mode))
            R.resolve_occlusion(objs)
            overall = self.tracker.update(objs)
            self.beeper.level = overall
            if self.click:
                self.cam.calibrate_pitch(*self.click, R.CALIB_DIST)
                if self.tilt is not None:                # luu do lech cam bien <-> truc camera
                    self.offset = math.degrees(self.cam.pitch) - self.tilt
                self.click, self.calib = None, False
            now = time.time()
            fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-6)
            t_prev = now
            out = R.compose(R.draw_camera(frame, self.cam, objs, self.calib),
                            R.draw_radar(self.h, self.cam, objs, self.tracker, overall),
                            self.cam, fps, self.beeper, self.flip)
            sensor = (f"cam bien {self.tilt:+.1f} lech {self.offset:+.1f} "
                      f"ngang {self.roll:+.1f}deg | mu {self.cam.min_visible():.1f}m"
                      if self.tilt is not None else "CAM BIEN: khong co (dung --pitch)")
            R.put(out, sensor, (self.w + 10, self.h - 8), (200, 200, 200), 0.45)
            with self.cond:
                self.jpg = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
                self.cond.notify_all()

    def command(self, c, q):
        cam = self.cam
        if c == "calib":
            self.calib = not self.calib
        elif c == "click" and self.calib:
            x, y = int(q["x"][0]), int(q["y"][0])
            if x < self.w and y < self.h:
                self.click = (x, y)
        elif c in ("pitch_up", "pitch_down"):
            d = 1 if c == "pitch_up" else -1
            if self.tilt is not None:
                self.offset += d                         # co cam bien -> chinh do lech
            else:
                cam.pitch += math.radians(d)
        elif c == "h_up":
            cam.height += 0.05
        elif c == "h_down":
            cam.height = max(0.2, cam.height - 0.05)
        elif c == "flip":
            self.flip = not self.flip
        elif c == "mute":
            self.beeper.muted = not self.beeper.muted


def make_handler(app):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            url = urlparse(self.path)
            if url.path == "/":
                body = PAGE.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif url.path == "/cmd":
                q = parse_qs(url.query)
                app.command(q.get("c", [""])[0], q)
                self.send_response(204)
                self.end_headers()
            elif url.path == "/stream":
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                try:
                    while True:
                        with app.cond:
                            app.cond.wait(timeout=2)
                            jpg = app.jpg
                        if jpg:
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass
            else:
                self.send_error(404)
    return H


def main():
    ap = argparse.ArgumentParser(description="Radar 2D tu webcam - giao dien web")
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--video", type=str, default=None)
    ap.add_argument("--hfov", type=float, default=R.HFOV_DEG)
    ap.add_argument("--height", type=float, default=R.CAM_HEIGHT)
    ap.add_argument("--pitch", type=float, default=R.PITCH_DEG)
    ap.add_argument("--model", type=str, default=R.MODEL)
    ap.add_argument("--imgsz", type=int, default=R.IMGSZ)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--pose-model", type=str, default="yolo11n-pose.pt")
    ap.add_argument("--no-pose", action="store_true", help="tat model pose (nhanh hon)")
    ap.add_argument("--no-sensor",action="store_true", help="bo cam bien nghieng, dung --pitch co dinh")
    args = ap.parse_args()

    app = App(args)
    threading.Thread(target=app.loop, daemon=True).start()
    try:
        ip = socket.gethostbyname(socket.gethostname())
    except OSError:
        ip = "<IP-laptop>"
    print(f"Mo: http://localhost:{args.port}   (dien thoai cung wifi: http://{ip}:{args.port})", flush=True)
    ThreadingHTTPServer(("0.0.0.0", args.port), make_handler(app)).serve_forever()


if __name__ == "__main__":
    main()
