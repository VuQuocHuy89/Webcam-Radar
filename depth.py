# -*- coding: utf-8 -*-
"""Depth Anything V2 Small (CPU, ~70 ms o 196x252) -> do sau TUONG DOI (nghich dao, khong co don vi).
Doi ra met bang cac "neo" da biet khoang cach (vat do tot bang hinh hoc/pose, san o mep duoi anh):
    1/zc = a * disp + b        (zc: do sau theo truc camera)
Roi: (1) do khoang cach cho tung box -> them 1 phep do cho Kalman
     (2) quet vat can theo tung cot anh -> thay ca tuong, cua, cot (khong co trong COCO)."""
import math
import threading
import time

import cv2
import numpy as np

MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
IN_W, IN_H = 252, 196            # boi so cua 14; to hon -> chinh xac hon nhung cham hon
GW, GH = 84, 64                  # luoi tinh toan (du cho radar)
MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
ANCHOR_MODES = ("ok", "pose", "san")
OBST_H = (0.15, 1.9)             # m tren san: bo mat san va tran nha
FIT_EMA = 0.3                    # lam muot he so a, b giua cac khung


class Depth(threading.Thread):
    def __init__(self, cam, threads=4):
        super().__init__(daemon=True)
        import torch
        from transformers import AutoModelForDepthEstimation
        torch.set_num_threads(threads)
        self.torch = torch
        self.model = AutoModelForDepthEstimation.from_pretrained(MODEL_ID).eval()
        self.cam = cam
        self.job, self.ev = None, threading.Event()
        self.disp, self.fit, self.ms, self.n_anchor = None, None, 0.0, 0
        # Toa do tia cua moi o luoi (x, y chuan hoa) - tinh 1 lan
        u = (np.arange(GW) + 0.5) * cam.w / GW
        v = (np.arange(GH) + 0.5) * cam.h / GH
        self.gx = np.tile((u - cam.cx) / cam.f, (GH, 1))
        self.gy = np.tile(((v - cam.cy) / cam.f)[:, None], (1, GW))

    def submit(self, frame, objs):
        """Gui khung moi (bo qua neu dang ban) kem cac vat do tot lam neo."""
        self.job = (frame, [(o["box"], o["Z"], o["mode"]) for o in objs])
        self.ev.set()

    def run(self):
        while True:
            self.ev.wait()
            self.ev.clear()
            frame, anchors = self.job
            t = time.time()
            x = (cv2.resize(frame, (IN_W, IN_H))[:, :, ::-1] / 255.0 - MEAN) / STD
            with self.torch.inference_mode():
                d = self.model(pixel_values=self.torch.from_numpy(
                    x.transpose(2, 0, 1)[None].astype(np.float32))).predicted_depth[0].numpy()
            disp = cv2.medianBlur(cv2.resize(d, (GW, GH)).astype(np.float32), 5)
            self._fit(disp, anchors)
            self.disp = disp
            self.ms = (time.time() - t) * 1000

    # ---------- doi do sau tuong doi -> met ----------
    def _cell(self, box):
        """Vung giua box (tranh nen phia sau) tren luoi."""
        x1, y1, x2, y2 = box
        sx, sy = GW / self.cam.w, GH / self.cam.h
        w, h = x2 - x1, y2 - y1
        return (slice(int((y1 + 0.3 * h) * sy), max(int((y1 + 0.7 * h) * sy), int((y1 + 0.3 * h) * sy) + 1)),
                slice(int((x1 + 0.3 * w) * sx), max(int((x1 + 0.7 * w) * sx), int((x1 + 0.3 * w) * sx) + 1)))

    def _ray(self, gy):
        cp, sp = math.cos(self.cam.pitch), math.sin(self.cam.pitch)
        return cp - gy * sp                        # Z_san = zc * ray

    def _fit(self, disp, anchors):
        pts = []                                   # (disp, 1/zc, trong so)
        for box, Z, mode in anchors:
            if mode in ANCHOR_MODES:
                r, c = self._cell(box)
                ray = self._ray(float(self.gy[r, c].mean()))
                if ray > 0.05:
                    pts.append((float(np.median(disp[r, c])), ray / Z, 1.0))
        # San o mep duoi giua anh (khi camera nhin thay san gan)
        cam = self.cam
        if cam.min_visible() < 4.0:
            row = GH - 2
            p = cam.floor_point(cam.cx, (row + 0.5) * cam.h / GH)
            if p:
                ray = self._ray(float(self.gy[row, 0]))
                pts.append((float(np.median(disp[row, GW // 3: 2 * GW // 3])), ray / p[1], 0.5))
        self.n_anchor = len(pts)
        if not pts:
            return
        d, iz, w = map(np.array, zip(*pts))
        if len(pts) >= 2 and np.ptp(d) > 1e-3 * max(abs(d).max(), 1e-6):
            A = np.stack([d, np.ones_like(d)], 1) * w[:, None]
            a, b = np.linalg.lstsq(A, iz * w, rcond=None)[0]
        elif self.fit:                             # 1 neo: giu b cu, chinh ti le a
            b = self.fit[1]
            a = float(np.sum(w * (iz - b) * d) / max(np.sum(w * d * d), 1e-9))
        else:
            a, b = float(np.sum(w * iz * d) / max(np.sum(w * d * d), 1e-9)), 0.0
        if a <= 0:                                 # vo ly (gan hon phai disp lon hon) -> bo
            return
        self.fit = (a, b) if self.fit is None else tuple(
            f + FIT_EMA * (n - f) for f, n in zip(self.fit, (a, b)))

    def _zc(self, disp):
        a, b = self.fit
        inv = a * disp + b
        return np.where(inv > 1e-3, 1 / np.maximum(inv, 1e-3), np.inf)

    # ---------- dau ra ----------
    def ready(self):
        return self.disp is not None and self.fit is not None

    def measure(self, box):
        """Khoang cach (X, Z) tren san cua vat trong box, theo depth."""
        if not self.ready():
            return None
        r, c = self._cell(box)
        zc = float(np.median(self._zc(self.disp[r, c])))
        if not math.isfinite(zc):
            return None
        gx, gy = float(self.gx[r, c].mean()), float(self.gy[r, c].mean())
        return zc * gx, zc * self._ray(gy)

    def scan(self, max_r):
        """Diem vat can gan nhat moi cot anh: [(X, Z)] (giong tia lidar)."""
        if not self.ready():
            return []
        cam = self.cam
        zc = self._zc(self.disp)
        cp, sp = math.cos(cam.pitch), math.sin(cam.pitch)
        hgt = cam.height - zc * (self.gy * cp + sp)        # do cao diem so voi san
        X, Z = zc * self.gx, zc * (cp - self.gy * sp)
        r = np.hypot(X, Z)
        r[~((hgt > OBST_H[0]) & (hgt < OBST_H[1]) & (Z > 0.2) & (r < max_r))] = np.inf
        rows = r.argmin(0)
        cols = np.arange(GW)
        keep = np.isfinite(r[rows, cols])
        return list(zip(X[rows, cols][keep], Z[rows, cols][keep]))
