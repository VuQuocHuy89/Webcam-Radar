#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RADAR 2D TU WEBCAM - Proof of Concept
--------------------------------------
Webcam -> YOLO nano (phat hien nguoi, ban, ghe...) -> uoc luong vi tri tren san
bang hinh hoc "diem cham san" -> ban do nhin tu tren xuong (BEV, kieu radar)
-> canh bao 3 vung XANH / VANG / DO theo 3 cung TRAI / GIUA / PHAI.

Cai dat:   pip install ultralytics opencv-python
Chay:      python radar.py                   (webcam mac dinh)
           python radar.py --cam 1           (webcam khac)
           python radar.py --video lop.mp4   (thu bang video quay san)

Phim tat:
  c       : che do hieu chinh -> dat 1 vat cach dung CALIB_DIST met,
            click chuot vao diem vat cham san tren anh camera
  + / -   : tang / giam goc cui camera 1 do
  ] / [   : tang / giam chieu cao camera 5 cm
  f       : lat anh trai-phai
  m       : bat / tat am thanh
  q / ESC : thoat
"""
import argparse
import math
import threading
import time

import cv2
import numpy as np

try:
    import winsound  # chi co tren Windows
except ImportError:
    winsound = None

# ======================= CAU HINH =======================
HFOV_DEG = 70.0       # goc nhin ngang cua webcam (webcam laptop thuong 60-80)
CAM_HEIGHT = 1.5      # chieu cao webcam so voi san (m) khi cam laptop
PITCH_DEG = 25.0      # goc cui xuong (do). Cao 1.5 m can cui ~25 do de vung mu chi ~1.1 m
D_RED = 1.5           # < 1.5 m  -> DO  (vung mu duoi chan thuong ~1.0-1.3 m)
D_YELLOW = 2.5        # < 2.5 m  -> VANG
RADAR_RANGE = 5.0     # ban kinh hien thi radar (m)
CALIB_DIST = 2.0      # khoang cach vat mau khi hieu chinh (m)
HYST = 0.15           # do tre chuyen vung (m), chong nhap nhay
MODEL = "yolo11n.pt"  # model nhe nhat; doi "yolov8n.pt" neu muon
IMGSZ = 416           # nho hon -> nhanh hon tren CPU (320 neu may yeu)
CONF = 0.35

# Lop COCO coi la vat can (bo laptop/chai... vi nam tren ban, khong cham san)
CLASSES = {0: "nguoi", 13: "ghe dai", 24: "ba lo", 28: "vali",
           56: "ghe", 57: "sofa", 58: "chau cay", 60: "ban"}

# Chieu cao trung binh (m) +- dung sai cho tung lop. Nguoi co 2 tu the: dung / ngoi;
# tu the nao khop voi diem cham san hon thi duoc chon.
HEIGHTS = {
    0:  [(1.65, 0.15), (1.25, 0.15)],   # nguoi: dung, ngoi
    13: [(0.45, 0.10)],                 # ghe dai
    24: [(0.45, 0.10)],                 # ba lo
    28: [(0.65, 0.15)],                 # vali
    56: [(0.85, 0.15)],                 # ghe (co tua lung)
    57: [(0.85, 0.15)],                 # sofa
    58: [(0.60, 0.20)],                 # chau cay
    60: [(0.75, 0.20)],                 # ban (dung sai lon: nhin tu tren thay ca mat ban)
}

LEVEL_NAME = ["AN TOAN", "CHU Y", "NGUY HIEM"]
LEVEL_COLOR = [(80, 200, 80), (0, 200, 255), (0, 0, 255)]  # BGR
SECTOR_NAME = ["TRAI", "GIUA", "PHAI"]
WIN = "Radar 2D - webcam PoC"


# ======================= HINH HOC CAMERA =======================
class Camera:
    """Mo hinh camera pinhole cam o do cao h, cui xuong goc pitch."""

    def __init__(self, w, h, hfov_deg, height, pitch_deg):
        self.w, self.h = w, h
        self.hfov_deg = hfov_deg
        self.f = (w / 2) / math.tan(math.radians(hfov_deg) / 2)
        self.cx, self.cy = w / 2, h / 2
        self.height = height
        self.pitch = math.radians(pitch_deg)

    def floor_point(self, u, v):
        """Pixel (u, v) -> diem tren san (X: ngang, Z: phia truoc), don vi met.
        Tra ve None neu tia nhin nam tren duong chan troi (khong cat san)."""
        x = (u - self.cx) / self.f
        y = (v - self.cy) / self.f
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        down = y * cp + sp
        if down <= 1e-3:
            return None
        t = self.height / down
        return t * x, t * (cp - y * sp)

    def min_visible(self):
        """Khoang cach san gan nhat camera con nhin thay (mep duoi anh)."""
        p = self.floor_point(self.cx, self.h - 1)
        return p[1] if p else 0.0

    def horizon_v(self):
        return self.cy - self.f * math.tan(self.pitch)

    def calibrate_pitch(self, u, v, dist):
        """Tim goc cui sao cho diem (u, v) tren san cach camera dung `dist` m."""
        lo, hi = math.radians(-20), math.radians(75)
        for _ in range(60):
            self.pitch = (lo + hi) / 2
            p = self.floor_point(u, v)
            if p is None or math.hypot(*p) > dist:
                lo = self.pitch      # dang qua xa -> phai cui nhieu hon
            else:
                hi = self.pitch
        self.pitch = (lo + hi) / 2


def size_distance(cam, v_top, v_bot, H):
    """Khoang cach den vat cao H dung tren san, tu goc nhin dinh va chan vat:
       tan(goc chan) = h/Z,  tan(goc dinh) = (h-H)/Z  ->  Z = H / (tan_chan - tan_dinh).
    Dung duoc ca khi dinh vat cao hon camera (tan_dinh am)."""
    tb = math.tan(cam.pitch + math.atan((v_bot - cam.cy) / cam.f))
    tt = math.tan(cam.pitch + math.atan((v_top - cam.cy) / cam.f))
    diff = tb - tt
    return H / diff if diff > 1e-4 else None


def estimate(cam, box, cls):
    """Uoc luong vi tri (X, Z) cua vat, tra ve them che do:
      'sat' : box bi cat mep duoi -> vat gan hon vung mu -> ep vao vung DO
      'ok'  : diem cham san va chieu cao trung binh khop nhau -> lay trung binh
      'hc'  : lech nhau (outlier, vd chan bi ban che) -> keo ve khoang hop le
      'san' : chi dung diem cham san (dau vat bi cat mep tren)
      'cao' : chi dung chieu cao (diem cham san khong tinh duoc)"""
    x1, y1, x2, y2 = box
    u = (x1 + x2) / 2
    if y2 >= cam.h - 4:
        Z = min(cam.min_visible(), D_RED - 0.05)
        return (u - cam.cx) / cam.f * Z, Z, "sat"

    p = cam.floor_point(u, y2)
    Zg = p[1] if p else None

    best = None                                   # (zs, lo, hi, gap)
    if y1 > 2:                                    # dinh vat nam trong anh
        for H, tol in HEIGHTS.get(cls, []):
            zs = size_distance(cam, y1, y2, H)
            lo = size_distance(cam, y1, y2, H - tol)
            hi = size_distance(cam, y1, y2, H + tol)
            if zs is None:
                continue
            gap = 0.0 if Zg is None or lo <= Zg <= hi else min(abs(Zg - lo), abs(Zg - hi))
            if best is None or gap < best[3]:
                best = (zs, lo, hi, gap)

    if best is None:
        if Zg is None:
            return None
        Z, mode = Zg, "san"
    elif Zg is None:
        Z, mode = best[0], "cao"
    elif best[3] == 0.0:
        Z, mode = (Zg + best[0]) / 2, "ok"
    else:
        Z, mode = min(max(Zg, best[1]), best[2]), "hc"

    X = p[0] / p[1] * Z if p else (u - cam.cx) / cam.f * Z
    return X, Z, mode


# Kich thuoc co the trung binh (m) + sai so tuong doi, cho model pose (keypoint COCO).
# Chinh L neu do thuc te lech deu (vd nguoi Viet vai hep hon -> 0.36).
TORSO = (0.50, 0.12)     # vai -> hong, doan DOC (ngoi gu lung thi ngan lai)
SHOULDER = (0.38, 0.10)  # vai trai - vai phai, doan NGANG (quay nghieng thi ngan lai)
HEAD = (0.15, 0.15)      # tai trai - tai phai
BOX_W = (0.45, 0.35)     # be rong box nguoi (tay, ba lo... lam sai nhieu -> trong so thap)
KP_CONF = 0.5


def person_distance(cam, box, kp):
    """Vi tri nguoi (X, Z) tu kich thuoc co the, KHONG can thay chan.
      - doan doc (than)  : Z = L / (tan goc duoi - tan goc tren)   (dung moi goc cui)
      - doan ngang       : do sau zc = f*L/pixel -> doi ra Z tren san
    Moi tin hieu cho 1 Z voi sai so s*Z; bo tin hieu lech trung vi > 35%,
    roi gop trung binh co trong so 1/(s*Z)^2."""
    ok = kp[:, 2] > KP_CONF
    cp, sp = math.cos(cam.pitch), math.sin(cam.pitch)
    cues = []

    def horiz(a, b, L, s, v):                   # doan ngang dai L, tam o dong anh v
        px = math.hypot(*(a - b))
        if px > 3:
            cues.append((cam.f * L / px * (cp - (v - cam.cy) / cam.f * sp), s))

    for sh, hp in ((5, 11), (6, 12)):
        if ok[sh] and ok[hp] and kp[hp, 1] - kp[sh, 1] > 3:
            z = size_distance(cam, kp[sh, 1], kp[hp, 1], TORSO[0])
            if z:
                cues.append((z, TORSO[1]))
    for (a, b), (L, s) in (((5, 6), SHOULDER), ((3, 4), HEAD)):
        if ok[a] and ok[b]:
            horiz(kp[a, :2], kp[b, :2], L, s, (kp[a, 1] + kp[b, 1]) / 2)
    x1, y1, x2, y2 = box
    if x1 > 4 and x2 < cam.w - 4:              # box khong bi cat mep trai/phai
        horiz(np.array([x1, 0]), np.array([x2, 0]), *BOX_W, (y1 + y2) / 2)

    cues = [(z, s) for z, s in cues if z > 0.1]  # keypoint tren chan troi -> Z am, bo
    if not cues:
        return None
    # Nguoi quay nghieng lam TAT CA doan ngang cung ngan lai (cung sai 1 phia) -> khong cho
    # chung bo phieu thang than; than (doc, khong doi khi xoay) co thi lam moc.
    n_torso = sum(1 for _, s in cues if s == TORSO[1])
    med = float(np.median([z for z, _ in (cues[:n_torso] or cues)]))
    good = ([(z, s) for z, s in cues if abs(z - med) <= 0.35 * med]
            or [min(cues, key=lambda c: abs(c[0] - med))])   # 2 than lech nhau -> lay cai gan moc
    w = [1 / (s * z) ** 2 for z, s in good]
    Z = sum(wi * z for wi, (z, _) in zip(w, good)) / sum(w)
    vis = kp[ok]                                # tam nguoi theo chieu ngang
    u = float(vis[:, 0].mean()) if len(vis) else (x1 + x2) / 2
    v = float(vis[:, 1].mean()) if len(vis) else (y1 + y2) / 2
    return (u - cam.cx) / cam.f / (cp - (v - cam.cy) / cam.f * sp) * Z, Z


OCCLUDED_GAP = 0.5  # vat bi che (vd nguoi ngoi sau ban) coi nhu nam sau vat che 0.5 m


def resolve_occlusion(objs):
    """Neu chan vat A nam trong box cua vat B o gan hon (chan B thap hon trong anh),
    A bi B che chan -> diem cham san cua A sai. Dat A = B + OCCLUDED_GAP."""
    for a in objs:
        if a["mode"] in ("sat", "pose"):           # pose khong dung chan -> khong bi che sai
            continue
        ax1, _, ax2, ay2 = a["box"]
        au = (ax1 + ax2) / 2
        for b in objs:
            if b is a:
                continue
            bx1, by1, bx2, by2 = b["box"]
            if bx1 <= au <= bx2 and by1 <= ay2 < by2 - 8 and b["Z"] < a["Z"]:
                z_new = b["Z"] + OCCLUDED_GAP
                if z_new < a["Z"]:
                    a["X"] *= z_new / a["Z"]
                    a["Z"], a["mode"] = z_new, "che"
                    a["r"] = math.hypot(a["X"], a["Z"])
    return objs


def level_of(d, prev=0):
    """Phan vung co do tre: vao vung nguy hiem ngay, ra vung an toan phai vuot them HYST."""
    if d < D_RED or (prev == 2 and d < D_RED + HYST):
        return 2
    if d < D_YELLOW or (prev >= 1 and d < D_YELLOW + HYST):
        return 1
    return 0


class SectorTracker:
    """Giu khoang cach gan nhat moi cung TRAI/GIUA/PHAI, lam muot bat doi xung."""

    def __init__(self, hfov_deg):
        self.edge = hfov_deg / 6
        self.d = [math.inf] * 3
        self.miss = [0] * 3
        self.level = [0] * 3

    def sector_of(self, X, Z):
        a = math.degrees(math.atan2(X, Z))
        return 0 if a < -self.edge else (2 if a > self.edge else 1)

    def update(self, objs):
        nearest = [math.inf] * 3
        for o in objs:
            s = self.sector_of(o["X"], o["Z"])
            nearest[s] = min(nearest[s], o["r"])
        for i in range(3):
            n = nearest[i]
            if n < math.inf:
                self.miss[i] = 0
                if self.d[i] == math.inf or n < self.d[i]:
                    self.d[i] = n                       # nguy hiem hon -> cap nhat ngay
                else:
                    self.d[i] += 0.3 * (n - self.d[i])  # an toan hon -> tu tu
            else:
                self.miss[i] += 1
                if self.miss[i] > 8:                    # mat dau ~8 khung moi xoa
                    self.d[i] = math.inf
            self.level[i] = level_of(self.d[i], self.level[i])
        return max(self.level)


# ======================= AM THANH =======================
class Beeper(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.level = 0
        self.muted = False

    def run(self):
        while True:
            lv = 0 if self.muted else self.level
            if lv == 0:
                time.sleep(0.1)
                continue
            freq, dur, gap = (1800, 120, 0.08) if lv == 2 else (1000, 100, 0.45)
            if winsound:
                winsound.Beep(freq, dur)
            else:
                print("\a", end="", flush=True)
                time.sleep(dur / 1000)
            time.sleep(gap)


# ======================= VE GIAO DIEN =======================
def put(img, text, org, color=(230, 230, 230), scale=0.5, thick=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def draw_camera(frame, cam, objs, calib):
    img = frame.copy()
    hv = int(cam.horizon_v())
    if 0 <= hv < cam.h:
        cv2.line(img, (0, hv), (cam.w, hv), (150, 150, 150), 1)
        put(img, "chan troi", (5, hv - 5), (180, 180, 180), 0.4)
    for o in objs:
        x1, y1, x2, y2 = map(int, o["box"])
        col = LEVEL_COLOR[level_of(o["r"])]
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
        cv2.circle(img, ((x1 + x2) // 2, y2), 4, col, -1)  # diem cham san
        mark = {"sat": "<", "hc": "~", "che": "~", "pose": "*"}.get(o["mode"], "")
        tag = f'{o["name"]} {mark}{o["r"]:.1f}m'
        put(img, tag, (x1 + 2, max(y1 - 6, 14)), col, 0.5)
    if calib:
        cv2.rectangle(img, (0, 0), (cam.w, 50), (0, 0, 0), -1)
        put(img, f"HIEU CHINH: dat vat cach dung {CALIB_DIST:.1f} m,", (8, 20), (0, 255, 255))
        put(img, "click vao diem vat cham san (c de huy)", (8, 42), (0, 255, 255))
    return img


def draw_radar(size, cam, objs, tracker, overall):
    img = np.full((size, size, 3), 22, np.uint8)
    ox, oy = size // 2, size - 25
    R = size - 60
    scale = R / RADAR_RANGE
    half = cam.hfov_deg / 2
    a0, a1 = -90 - half, -90 + half              # goc OpenCV: 0 = sang phai, theo chieu kim dong ho
    edges = [a0, -90 - half / 3, -90 + half / 3, a1]

    # To mau cung theo muc canh bao
    overlay = img.copy()
    for i in range(3):
        lv = tracker.level[i]
        if lv > 0:
            cv2.ellipse(overlay, (ox, oy), (R, R), 0, edges[i], edges[i + 1], LEVEL_COLOR[lv], -1)
    img = cv2.addWeighted(overlay, 0.22, img, 0.78, 0)

    # Vung mu (camera khong thay san)
    rb = int(cam.min_visible() * scale)
    if rb > 0:
        cv2.ellipse(img, (ox, oy), (rb, rb), 0, a0, a1, (60, 60, 60), -1)
        put(img, "vung mu", (ox - 28, oy - rb // 2), (140, 140, 140), 0.4)

    # Vong khoang cach
    for d, col in ((D_RED, (0, 0, 220)), (D_YELLOW, (0, 180, 230))):   # nguong canh bao
        cv2.ellipse(img, (ox, oy), (int(d * scale),) * 2, 0, a0, a1, col, 2, cv2.LINE_AA)
    for r in range(1, int(RADAR_RANGE) + 1):
        col = (70, 70, 70)
        cv2.ellipse(img, (ox, oy), (int(r * scale),) * 2, 0, a0, a1, col, 1, cv2.LINE_AA)
        ang = math.radians(a1)
        put(img, f"{r}m", (int(ox + r * scale * math.cos(ang)) + 4, int(oy + r * scale * math.sin(ang))),
            (150, 150, 150), 0.4)
    for e in edges:
        ang = math.radians(e)
        cv2.line(img, (ox, oy), (int(ox + R * math.cos(ang)), int(oy + R * math.sin(ang))),
                 (90, 90, 90) if e in (a0, a1) else (50, 50, 50), 1, cv2.LINE_AA)

    # Vat can
    for o in objs:
        if o["Z"] > RADAR_RANGE + 0.3:
            continue
        px, py = int(ox + o["X"] * scale), int(oy - o["Z"] * scale)
        col = LEVEL_COLOR[level_of(o["r"])]
        if o["cls"] == 0:
            cv2.circle(img, (px, py), 8, col, -1)
        else:
            cv2.rectangle(img, (px - 7, py - 7), (px + 7, py + 7), col, -1)
        put(img, o["name"], (px + 10, py + 4), (220, 220, 220), 0.4)

    # Camera (minh)
    cv2.fillPoly(img, [np.array([[ox, oy - 12], [ox - 9, oy + 6], [ox + 9, oy + 6]])], (255, 255, 255))

    # Tieu de
    put(img, LEVEL_NAME[overall], (12, 32), LEVEL_COLOR[overall], 0.9, 2)
    for i, name in enumerate(SECTOR_NAME):
        d = tracker.d[i]
        txt = f"{name}: {d:.1f}m" if d < math.inf else f"{name}: --"
        put(img, txt, (size - 130, 22 + i * 20), LEVEL_COLOR[tracker.level[i]], 0.5)
    return img


def compose(cam_img, radar_img, cam, fps, beeper, flip):
    h = cam_img.shape[0]
    bar = np.full((44, cam_img.shape[1] + radar_img.shape[1], 3), 35, np.uint8)
    info = (f"FPS {fps:4.1f} | cao {cam.height:.2f}m  cui {math.degrees(cam.pitch):.1f}deg  "
            f"FOV {cam.hfov_deg:.0f}deg | am thanh {'TAT' if beeper.muted else 'BAT'}"
            f"{' | LAT' if flip else ''}")
    put(bar, info, (10, 18), (220, 220, 220), 0.5)
    put(bar, "[c] hieu chinh  [+/-] goc cui  [ [/] ] chieu cao  [f] lat  [m] am thanh  [q] thoat",
        (10, 37), (150, 150, 150), 0.45)
    return np.vstack([np.hstack([cam_img, radar_img[:h]]), bar])


# ======================= CHUONG TRINH CHINH =======================
def main():
    ap = argparse.ArgumentParser(description="Radar 2D tu webcam (PoC)")
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--video", type=str, default=None)
    ap.add_argument("--hfov", type=float, default=HFOV_DEG)
    ap.add_argument("--height", type=float, default=CAM_HEIGHT)
    ap.add_argument("--pitch", type=float, default=PITCH_DEG)
    ap.add_argument("--model", type=str, default=MODEL)
    ap.add_argument("--imgsz", type=int, default=IMGSZ)
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.model)

    cap = cv2.VideoCapture(args.video if args.video else args.cam)
    if not args.video:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("Khong mo duoc camera/video.")
    h, w = frame.shape[:2]

    cam = Camera(w, h, args.hfov, args.height, args.pitch)
    tracker = SectorTracker(args.hfov)
    beeper = Beeper()
    beeper.start()
    state = {"calib": False, "click": None}
    flip = False
    fps, t_prev = 0.0, time.time()

    def on_mouse(event, x, y, flags, st):
        if event == cv2.EVENT_LBUTTONDOWN and st["calib"] and x < w and y < h:
            st["click"] = (x, y)

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WIN, on_mouse, state)

    while True:
        ok, frame = cap.read()
        if not ok:
            if args.video:                       # video: phat lai tu dau
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            # Mat tin hieu camera -> mac dinh an toan: bao DO
            beeper.level = 2
            err = np.zeros((h, w + h, 3), np.uint8)
            put(err, "MAT TIN HIEU CAMERA - DUNG!", (20, h // 2), (0, 0, 255), 1.0, 2)
            cv2.imshow(WIN, err)
            if cv2.waitKey(200) & 0xFF in (ord("q"), 27):
                break
            continue
        if flip:
            frame = cv2.flip(frame, 1)

        res = model.predict(frame, imgsz=args.imgsz, conf=CONF,
                            classes=list(CLASSES), verbose=False)[0]
        objs = []
        for box, c in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy().astype(int)):
            e = estimate(cam, box, c)
            if e is None:
                continue
            X, Z, mode = e
            objs.append(dict(box=box, cls=c, name=CLASSES[c], X=X, Z=Z,
                             r=math.hypot(X, Z), mode=mode))

        resolve_occlusion(objs)
        overall = tracker.update(objs)
        beeper.level = overall

        if state["click"]:
            cam.calibrate_pitch(*state["click"], CALIB_DIST)
            state["click"], state["calib"] = None, False

        now = time.time()
        fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-6)
        t_prev = now

        out = compose(draw_camera(frame, cam, objs, state["calib"]),
                      draw_radar(h, cam, objs, tracker, overall), cam, fps, beeper, flip)
        cv2.imshow(WIN, out)

        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):
            break
        elif k == ord("c"):
            state["calib"] = not state["calib"]
        elif k in (ord("+"), ord("=")):
            cam.pitch += math.radians(1)
        elif k == ord("-"):
            cam.pitch -= math.radians(1)
        elif k == ord("]"):
            cam.height += 0.05
        elif k == ord("["):
            cam.height = max(0.2, cam.height - 0.05)
        elif k == ord("f"):
            flip = not flip
        elif k == ord("m"):
            beeper.muted = not beeper.muted

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
