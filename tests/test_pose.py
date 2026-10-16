import sys, math, numpy as np
import os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import radar as R

def person_kp(cam, X, Z, yaw=0.0):
    """Khung nguoi 3D (cao vai 1.40, hong 0.90, tai 1.60) dung tai (X, Z), xoay yaw do."""
    cp, sp = math.cos(cam.pitch), math.sin(cam.pitch)
    def proj(dx, H):
        c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
        x, z = X + dx * c, Z + dx * s
        down = cam.height - H
        yc, zc = down * cp - z * sp, down * sp + z * cp
        return cam.cx + cam.f * x / zc, cam.cy + cam.f * yc / zc
    kp = np.zeros((17, 3))
    for i, dx, H in [(3, -.075, 1.6), (4, .075, 1.6), (5, -.19, 1.4), (6, .19, 1.4),
                     (11, -.19, .9), (12, .19, .9)]:
        kp[i] = (*proj(dx, H), 0.9)
    bl, br = proj(-.225, 1.4)[0], proj(.225, 1.4)[0]
    return kp, (min(bl, br), proj(0, 1.7)[1], max(bl, br), cam.h - 1)

for pitch in (0, -12, 25):
    cam = R.Camera(640, 480, 70, 1.5, pitch)
    for X, Z, yaw in [(0, 1.5, 0), (0.5, 2.0, 0), (-1, 3.0, 0), (0, 2.0, 60), (0.3, 5.0, 30)]:
        ex, ez = R.person_distance(cam, *person_kp(cam, X, Z, yaw)[::-1])
        print(f"pitch {pitch:3} X={X:+.1f} Z={Z:.1f} yaw={yaw:2} -> X={ex:+.2f} Z={ez:.2f} errZ={ez-Z:+.2f}")
        assert abs(ez - Z) < 0.10 * Z and abs(ex - X) < 0.15, "sai > 15%"
print("OK")

# Ca loi chia 0 cu: 2 than lech nhau xa + keypoint tren chan troi (Z am)
cam = R.Camera(640, 480, 70, 1.5, -12)
kp, box = person_kp(cam, 0, 2.0)
kp[11, 1] = kp[5, 1] + 5          # than trai gan nhu 0 px -> Z rat xa
kp[12, 1] = kp[6, 1] + 400        # than phai rat dai -> Z rat gan
assert R.person_distance(cam, box, kp) is not None
kp2 = np.zeros((17, 3)); kp2[5] = (300, 10, .9); kp2[6] = (340, 10, .9)   # vai sat mep tren
cam2 = R.Camera(640, 480, 70, 1.5, 60)
R.person_distance(cam2, (290, 0, 350, 479), kp2)   # khong duoc crash
print("edge cases OK")
