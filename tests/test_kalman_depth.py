import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import depth as D
import radar as R

rng = np.random.default_rng(0)

# 1) Kalman: nguoi di tu 4 m lai gan voi vZ = -1 m/s, do co nhieu 15% -> loc phai tot hon do tho
kf = R.KalmanTracker()
raw_err, kf_err = [], []
for k in range(40):
    t = k * 0.1
    X, Z = 0.5, 4.0 - 1.0 * t
    zx, zz = X + rng.normal(0, 0.1), Z + rng.normal(0, 0.15 * Z)
    o = dict(cls=0, box=None, mode="pose", meas=[(np.array([zx, zz]), R.meas_cov(zz, "pose"))])
    tr = kf.step([o], t)
    assert len(tr) == 1, "phai giu 1 track duy nhat"
    if k > 10:
        raw_err.append(abs(zz - Z))
        kf_err.append(abs(tr[0]["Z"] - Z))
print(f"Kalman: sai so tho {np.mean(raw_err):.3f} m -> loc {np.mean(kf_err):.3f} m, vZ = {tr[0]['vz']:+.2f} m/s")
assert np.mean(kf_err) < np.mean(raw_err) and abs(tr[0]["vz"] + 1.0) < 0.35

# 2) Hai vat gan nhau khong bi tron, nguoi khong ghep voi ghe; mat dau qua TTL thi xoa
kf = R.KalmanTracker()
for k in range(10):
    objs = [dict(cls=0, mode="ok", meas=[(np.array([-0.5, 2.0]), R.meas_cov(2.0, "ok"))]),
            dict(cls=56, mode="ok", meas=[(np.array([-0.4, 2.1]), R.meas_cov(2.1, "ok"))])]
    tr = kf.step(objs, k * 0.1)
assert len(tr) == 2 and {o["cls"] for o in tr} == {0, 56}
assert len(kf.step([], 1.0 + R.TRACK_TTL + 0.1)) == 0
# do sat (vung mu, sai so lon) + do depth (2.3 m) -> ket qua gan depth hon
z, _ = R.fuse([(np.array([0, 1.45]), R.meas_cov(1.45, "sat")), (np.array([0, 2.3]), R.meas_cov(2.3, "depth"))])
assert 2.0 < z[1] < 2.3, z
print("Kalman: ghep/xoa track, gop do OK; sat+depth ->", round(float(z[1]), 2))

# 3) Depth: san + tuong cach 3 m, model gia cho disp = 2/zc + 0.5 (ti le/do lech la an so)
cam = R.Camera(640, 480, 70, 1.5, 20)
d = object.__new__(D.Depth)                           # khong load model, chi gan thuoc tinh can thiet
d.cam, d.fit, d.disp, d.n_anchor = cam, None, None, 0
u = (np.arange(D.GW) + 0.5) * cam.w / D.GW
v = (np.arange(D.GH) + 0.5) * cam.h / D.GH
d.gx = np.tile((u - cam.cx) / cam.f, (D.GH, 1))
d.gy = np.tile(((v - cam.cy) / cam.f)[:, None], (1, D.GW))
cp, sp = math.cos(cam.pitch), math.sin(cam.pitch)
ray = cp - d.gy * sp
down = d.gy * cp + sp
zc_floor = np.where(down > 1e-3, cam.height / np.maximum(down, 1e-3), np.inf)
zc_wall = 3.0 / ray                                   # mat phang dung Z = 3 m
zc = np.minimum(zc_floor, zc_wall)
disp = (2.0 / zc + 0.5).astype(np.float32)
# Neo: 1 nguoi o Z = 2 m (box giua anh) do tot
box = (300, 150, 340, 330)
r, c = d._cell(box)
Zp = float(np.median(zc[r, c] * ray[r, c]))
for _ in range(15):
    d._fit(disp, [(box, Zp, "pose")])
d.disp = disp
print(f"Depth: fit a={d.fit[0]:.3f} b={d.fit[1]:.3f} (dung: a=0.5 b=-0.25), neo={d.n_anchor}")
X, Z = d.measure(box)
assert abs(Z - Zp) < 0.1, (Z, Zp)
pts = d.scan(5.0)
wall = [z for x, z in pts if abs(x) < 1.0]
assert wall and abs(np.median(wall) - 3.0) < 0.15, np.median(wall)
assert all(z > 0.2 for _, z in pts)
print(f"Depth: tuong o Z = {np.median(wall):.2f} m (that: 3.00), {len(pts)} diem quet, san bi loai OK")
