"""Module 3 -- Fully Worked Example, Step 1: the running point (2,3,4) through
every coordinate space.

Model order matches Example E1: translate (1,0,-1) FIRST, then yaw 90 deg,
i.e.  M = Ry(90) @ T(1,0,-1).

Run:  python worked_example.py
"""
from __future__ import annotations

import numpy as np

from transforms import (translate, rot_y, look_at, perspective, project, scale,
                        viewport, perspective_divide_2d)

np.set_printoptions(precision=3, suppress=True)

p = np.array([2.0, 3.0, 4.0])
print(f"object p        = {np.r_[p, 1.0]}")

M = rot_y(90) @ translate((1, 0, -1))
world = (M @ np.r_[p, 1.0])
print(f"model  M = Ry(90) . T(1,0,-1)        ; world  = M p       = {world}")

V = look_at(eye=(0, 0, 8), center=(0, 0, 0))
eye = V @ world
print(f"view   V = lookAt(eye=[0,0,8], ctr=0); eye    = V world   = {eye}   "
      f"(z={eye[2]:.0f}: {'in front' if eye[2] < 0 else 'BEHIND'}, dist {abs(eye[2]):.0f})")

P = perspective(60, 1.0, 1.0, 50.0)
clip = P @ eye
print(f"proj   P = perspective(60deg, 1, 1, 50)")
print(f"clip            = P eye              = {clip}")
ndc = clip[:3] / clip[3]
print(f"ndc  = clip/clip.w                   = {ndc}")

W, H = 800, 600
scr = viewport(ndc[None, :], W, H)[0]
print(f"screen ({W}x{H}) = [{scr[0]:.0f} {scr[1]:.0f}]  depth {(ndc[2] + 1) * 0.5:.3f}")

print()
print("--- Example E1 cross-check (bare similar-triangles projection) ---")
print(f"world point (pre-view) = {world[:3]}   -> after Ry(90) of (3,3,3): (3,3,-3)")
proj2d = perspective_divide_2d(world[:3], d=2)
print(f"perspective divide d=2 of (3,3,-3)   = {proj2d}   (expected [2. 2.])")

print("\n=== LAB 1: TELUSURAN 1 VERTEX PENDOPO JOGLO (data/mymodel.obj, idx 2083) ===")
p_my = np.array([1.0000, 0.7036, 0.7376])
print(f"object p        = {np.r_[p_my, 1.0]}")

M_my = translate((0.2, 0.0, -0.4)) @ rot_y(90) @ scale((1.4, 1.4, 1.4))
world_my = M_my @ np.r_[p_my, 1.0]
print(f"world  = M p    = {world_my}")

V_my = look_at(eye=(3.0, 1.0, 4.0), center=(0.0, 1.0, 0.0))
eye_my = V_my @ world_my
print(f"eye    = V w    = {eye_my}   (z={eye_my[2]:.3f}: di depan kamera)")

P_my = perspective(45.0, 1.0, 1.0, 14.0)
clip_my = P_my @ eye_my
print(f"clip   = P eye  = {clip_my}")

ndc_my = clip_my[:3] / clip_my[3]
print(f"ndc  = clip/w   = {ndc_my}")

scr_my = viewport(ndc_my[None, :], 512, 512)[0]
print(f"screen (512x512)= [{scr_my[0]:.1f}, {scr_my[1]:.1f}]   depth(0..1) = {(ndc_my[2] + 1) * 0.5:.4f}")

print("\n--- Telusuran Kolom Matriks M ---")
for idx, label in enumerate(["+x model", "+y model", "+z model", "origin  "]):
    print(f"  kolom {idx+1} ({label}) -> {M_my[:, idx]}")

print("\n--- Telusuran Kolom Matriks V ---")
for idx, label in enumerate(["+x world", "+y world", "+z world", "origin W"]):
    print(f"  kolom {idx+1} ({label}) -> {V_my[:, idx]}")