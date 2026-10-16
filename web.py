#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Chay radar.py va stream giao dien len web (MJPEG).
Chay:  python web.py [--cam 0] [--port 8000] [--hfov 70 --height 1.5 --pitch 25]
Mo:    http://localhost:8000  (hoac http://<IP-laptop>:8000 tu dien thoai cung wifi)
"""
import argparse
import json
import math
import os
import socket
import subprocess
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

import depth as D
import radar as R

PAGE = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "web.html"), encoding="utf-8").read()


class App:
    def __init__(self, args):
        import torch
        from ultralytics import YOLO
        # Do tren may nay: 4 luong nhanh nhat (detect 26 ms, pose 42 ms); mac dinh 14 luong cham gap 3-5 lan
        torch.set_num_threads(args.threads)
        self.model = YOLO(args.model)
        self.pose = None if args.no_pose else YOLO(args.pose_model)
        self.imgsz = args.imgsz
        self.det_every, self.k = max(1, args.det_every), 0
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
        self.kf = R.KalmanTracker()
        self.depth = None
        if not args.no_depth:
            try:
                self.depth = D.Depth(self.cam)
            except ImportError as ex:
                print("Khong co torch/transformers -> tat depth:", ex)
        self.beeper = R.Beeper()
        self.beeper.start()
        self.calib, self.click, self.flip = False, None, False
        self.jpg, self.state = {}, {}
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
              "try{[Console]::WriteLine([string]::Format($c,'{0} {1} {2}',$r.AccelerationX,$r.AccelerationY,$r.AccelerationZ));"
              "[Console]::Out.Flush()}catch{exit}; Start-Sleep -Milliseconds 100}")
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
            # Co pose: model detect chi tim do vat (do dung yen -> chay moi det_every khung,
            # khung bo qua Kalman tu du doan), nguoi lay tu model pose (co keypoint) moi khung
            self.k += 1
            dets = []
            if not self.pose or self.k % self.det_every == 0:
                classes = [c for c in R.CLASSES if c] if self.pose else list(R.CLASSES)
                res = self.model.predict(frame, imgsz=self.imgsz, conf=R.CONF,
                                         classes=classes, verbose=False)[0]
                dets = [(b, c, None) for b, c in zip(res.boxes.xyxy.cpu().numpy(),
                                                     res.boxes.cls.cpu().numpy().astype(int))]
            if self.pose:
                pr = self.pose.predict(frame, imgsz=self.imgsz, conf=R.CONF, verbose=False)[0]
                kps = pr.keypoints.data.cpu().numpy() if pr.keypoints is not None else []
                dets += [(b, 0, k) for b, k in zip(pr.boxes.xyxy.cpu().numpy(), kps)]
            vis = frame.copy()                           # ve keypoint len ban sao, depth dung anh goc
            objs = []
            for box, c, kp in dets:
                e = R.estimate(self.cam, box, c)
                if kp is not None and (e is None or e[2] in ("sat", "hc", "cao")):
                    p = R.person_distance(self.cam, box, kp)    # chan khong tin duoc -> dung kich thuoc co the
                    if p:
                        e = (*p, "pose")
                    for x, y, s in kp:
                        if s > R.KP_CONF:
                            cv2.circle(vis, (int(x), int(y)), 3, (255, 255, 0), -1)
                if e is None and self.depth and self.depth.ready():
                    p = self.depth.measure(box)
                    e = (*p, "depth") if p else None
                if e:
                    X, Z, mode = e
                    objs.append(dict(box=box, cls=c, name=R.CLASSES[c], X=X, Z=Z,
                                     r=math.hypot(X, Z), mode=mode))
            R.resolve_occlusion(objs)

            # Moi vat: phep do hinh hoc/pose + (neu do khong tot) phep do depth -> Kalman gop
            scan = []
            for o in objs:
                o["meas"] = [(np.array([o["X"], o["Z"]]), R.meas_cov(o["Z"], o["mode"]))]
                if self.depth and o["mode"] not in D.ANCHOR_MODES + ("depth",):
                    p = self.depth.measure(o["box"])
                    if p:
                        o["meas"].append((np.array(p), R.meas_cov(p[1], "depth")))
                        o["dep"] = True
            if self.depth:
                self.depth.step(frame, objs)
                scan = self.depth.scan(R.RADAR_RANGE)
            tracks = self.kf.step(objs, time.time())
            overall = self.tracker.update(tracks + [dict(X=x, Z=z, r=math.hypot(x, z)) for x, z in scan])
            self.beeper.level = overall
            if self.click:
                self.cam.calibrate_pitch(*self.click, R.CALIB_DIST)
                if self.tilt is not None:                # luu do lech cam bien <-> truc camera
                    self.offset = math.degrees(self.cam.pitch) - self.tilt
                self.click, self.calib = None, False
            now = time.time()
            fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-6)
            t_prev = now
            cam_img = R.draw_camera(vis, self.cam, tracks, self.calib)
            radar_img = R.draw_radar(self.h, self.cam, tracks, self.tracker, overall, scan, labels=False)
            self.state = self.snapshot(overall, fps, len(tracks))
            enc = [cv2.IMWRITE_JPEG_QUALITY, 80]
            with self.cond:
                self.jpg = {"cam": cv2.imencode(".jpg", cam_img, enc)[1].tobytes(),
                            "radar": cv2.imencode(".jpg", radar_img, enc)[1].tobytes()}
                self.cond.notify_all()

    def snapshot(self, overall, fps, n):
        """Trang thai cho giao dien web (JSON)."""
        tr, d = self.tracker, self.depth
        return dict(
            t=time.time(), level=overall, fps=round(fps, 1), n=n,
            sectors=[dict(d=None if v == math.inf else round(v, 2), level=lv)
                     for v, lv in zip(tr.d, tr.level)],
            pitch=round(math.degrees(self.cam.pitch), 1), height=round(self.cam.height, 2),
            blind=round(self.cam.min_visible(), 2),
            tilt=None if self.tilt is None else round(self.tilt, 1),
            offset=round(self.offset, 1), roll=round(self.roll, 1),
            depth=None if not d else dict(ready=d.ready(), ms=round(d.ms), anchors=d.n_anchor),
            muted=self.beeper.muted, flip=self.flip, calib=self.calib)

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
            if url.path in ("/", "/state"):
                html = url.path == "/"
                st = dict(app.state)
                if not html and time.time() - st.get("t", 0) > 1.5:   # xu ly treo / mat camera -> bao DO
                    st.update(level=2, stale=True)
                body = PAGE.encode() if html else json.dumps(st).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8" if html else "application/json")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif url.path == "/cmd":
                q = parse_qs(url.query)
                app.command(q.get("c", [""])[0], q)
                self.send_response(204)
                self.end_headers()
            elif url.path == "/stream":
                view = parse_qs(url.query).get("v", ["cam"])[0]   # cam | radar
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                try:
                    while True:
                        with app.cond:
                            app.cond.wait(timeout=2)
                            jpg = app.jpg.get(view)
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
    ap.add_argument("--imgsz", type=int, default=320, help="nho -> nhanh (416 neu can thay nguoi xa)")
    ap.add_argument("--det-every", type=int, default=3, help="chay model do vat moi N khung")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--pose-model", type=str, default="yolo11n-pose.pt")
    ap.add_argument("--no-pose", action="store_true", help="tat model pose (nhanh hon)")
    ap.add_argument("--no-depth", action="store_true", help="tat depth model (nhanh hon)")
    ap.add_argument("--threads", type=int, default=4, help="so luong CPU cho torch (YOLO + depth)")
    ap.add_argument("--no-sensor", action="store_true",help="bo cam bien nghieng, dung --pitch co dinh")
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
