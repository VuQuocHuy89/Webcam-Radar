import sys, math
import os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import radar as R

cam = R.Camera(640, 480, 70, 1.5, 25)

def proj(X, down, Z):
    cp, sp = math.cos(cam.pitch), math.sin(cam.pitch)
    yc = down*cp - Z*sp; zc = down*sp + Z*cp
    return cam.cx + cam.f*X/zc, cam.cy + cam.f*yc/zc

# 1) floor_point round-trip
for X, Z in [(0,2),(1,3),(-1.5,4),(0.3,1.2)]:
    u, v = proj(X, 1.5, Z)
    gx, gz = cam.floor_point(u, v)
    assert abs(gx-X) < 1e-6 and abs(gz-Z) < 1e-6, (X, Z, gx, gz)
print("floor_point round-trip OK, min_visible = %.2f m" % cam.min_visible())

# 2) estimate on synthetic boxes (standing person 1.65 m, chair 0.85 m)
print(f"{'cls':>5} {'X':>5} {'Z':>5} | {'estX':>6} {'estZ':>6} mode  errZ")
for cls, H in [(0,1.65),(0,1.25),(56,0.85),(60,0.75)]:
    for X, Z in [(0,1.3),(0,2),(0.8,3),(-1,4),(0,6)]:
        u, vb = proj(X, 1.5, Z); _, vt = proj(X, 1.5-H, Z)
        w = 0.25*cam.f/Z
        box = (u-w, vt, u+w, vb)
        if vb >= cam.h: box = (u-w, vt, u+w, cam.h-1)   # cut at bottom like real camera
        if vt < 0: box = (box[0], 0, box[2], box[3])
        e = R.estimate(cam, box, cls)
        ex, ez, m = e
        print(f"{cls:>5} {X:>5} {Z:>5} | {ex:6.2f} {ez:6.2f} {m:4} {ez-Z:+.2f}")

# 3) calibrate_pitch recovers true pitch
u, v = proj(0, 1.5, 2.0)
c2 = R.Camera(640, 480, 70, 1.5, 10); c2.calibrate_pitch(u, v, 2.0)
print("calib pitch: true 25.0, got %.2f" % math.degrees(c2.pitch))

# 4) sector / level
t = R.SectorTracker(70)
objs=[dict(X=-1,Z=1.2,r=math.hypot(-1,1.2)),dict(X=0,Z=2,r=2),dict(X=2,Z=4,r=math.hypot(2,4))]
print("overall", t.update(objs), "levels", t.level, "d", [round(d,2) for d in t.d])
