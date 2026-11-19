"""Lab 2 -- Eksperimen Butir 3 (E4 perspective-correct), Butir 4 (Mipmap 4-level),
dan Butir 5 (Back-face culling fragment reduction).

Jalankan:  python lab2_experiments.py
"""
from __future__ import annotations

import os
import numpy as np

import raster3d
from objio import Mesh, load_obj
from ppm_io import read_ppm, write_ppm
from transforms import translate, rot_y, scale
from metrics import psnr, ssim


def checkerboard_ppm(n=256, tiles=16):
    y, x = np.mgrid[0:n, 0:n]
    step = n // tiles
    c = (((x // step) + (y // step)) % 2 == 0)
    img = np.where(c[..., None], 235, 35).astype(np.uint8)
    return np.repeat(img, 3, axis=2)


def exp_e4_perspective_correct():
    print("\n=== LAB 2 BUTIR 3: PERSPECTIVE-CORRECT VS NAIF (CONTOH E4) ===")
    # Hitungan eksak E4: tepi lantai wL=1 (u=0) -> wR=9 (u=1) di titik tengah layar
    u_naive = 0.5 * 0.0 + 0.5 * 1.0
    u_over_w = 0.5 * (0.0 / 1.0) + 0.5 * (1.0 / 9.0)
    inv_w = 0.5 * (1.0 / 1.0) + 0.5 * (1.0 / 9.0)
    u_pc = u_over_w / inv_w
    print(f"Tengah layar (wL=1, wR=9): u naif = {u_naive:.3f} | u perspective-correct = {u_pc:.3f}")

    # Render quad lantai yang menjauh dari z=-1 (w=1) ke z=-9 (w=9)
    pos = np.array([
        [-1.0, -0.45, -1.0], [ 1.0, -0.45, -1.0],
        [ 1.0, -0.45, -9.0], [-1.0, -0.45, -9.0],
    ])
    uv = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    floor_mesh = Mesh(pos, uv, [[0, 1, 2], [0, 2, 3]], [[0, 1, 2], [0, 2, 3]])
    checker = checkerboard_ppm(256, tiles=8)

    cam = dict(w=256, h=256, tex_u8=checker, near=0.5, far=20.0,
               eye=(0, 0, 0), center=(0, 0, -1), cull_back=False, light_dir=(0, 1, 0.3))
    img_pc, _, _ = raster3d.render(floor_mesh, perspective_correct=True, **cam)
    img_na, _, _ = raster3d.render(floor_mesh, perspective_correct=False, **cam)

    write_ppm("out/e4_persp.ppm", img_pc)
    write_ppm("out/e4_naive.ppm", img_na)
    diff = np.clip(np.abs(img_pc.astype(int) - img_na.astype(int)) * 3, 0, 255).astype(np.uint8)
    write_ppm("out/e4_diff.ppm", diff)
    print(f"SSIM(persp, naive) = {ssim(img_pc, img_na):.3f} | PSNR = {psnr(img_pc, img_na):.1f} dB")
    print("wrote out/e4_persp.ppm, out/e4_naive.ppm, out/e4_diff.ppm")


def high_freq_energy(img_u8):
    """Rata-rata energi gradien spasial (indikator aliasing/kelap-kelip)."""
    g = img_u8.astype(np.float64).mean(axis=2) / 255.0
    gx = np.diff(g, axis=1)
    gy = np.diff(g, axis=0)
    return float(np.mean(gx ** 2) + np.mean(gy ** 2))


def exp_mipmap_far():
    print("\n=== LAB 2 BUTIR 4: MIPMAP 4-LEVEL (RENDER JARAK JAUH) ===")
    mesh = load_obj("data/mymodel.obj")
    tex = checkerboard_ppm(256, tiles=32)  # pola frekuensi tinggi untuk memicu aliasing minification
    M = translate((0, 0, 0)) @ rot_y(35) @ scale((1.4, 1.4, 1.4))

    # Kamera jauh agar 1 pixel menampung banyak texel (minification)
    cam = dict(w=256, h=256, tex_u8=tex, model=M,
               eye=(5.5, 4.2, 7.2), center=(0, 0.7, 0), near=1.0, far=25.0)
    img_off, _, _ = raster3d.render(mesh, use_mipmap=False, **cam)
    img_on, _, _ = raster3d.render(mesh, use_mipmap=True, **cam)

    write_ppm("out/mip_off.ppm", img_off)
    write_ppm("out/mip_on.ppm", img_on)
    diff = np.clip(np.abs(img_off.astype(int) - img_on.astype(int)) * 4, 0, 255).astype(np.uint8)
    write_ppm("out/mip_diff.ppm", diff)

    e_off, e_on = high_freq_energy(img_off), high_freq_energy(img_on)
    drop = (1.0 - e_on / max(e_off, 1e-12)) * 100.0
    print(f"Energi frekuensi tinggi: tanpa mipmap = {e_off:.6f} | dengan mipmap 4-level = {e_on:.6f} (turun {drop:.1f}%)")
    print(f"SSIM(off, on) = {ssim(img_off, img_on):.3f} | PSNR = {psnr(img_off, img_on):.1f} dB")
    print("wrote out/mip_off.ppm, out/mip_on.ppm, out/mip_diff.ppm")


def exp_backface_culling():
    print("\n=== LAB 2 BUTIR 5: PITFALL FIX -- BACK-FACE CULLING ===")
    mesh = load_obj("data/mymodel.obj")
    tex = read_ppm("data/relief.ppm") if os.path.exists("data/relief.ppm") else None
    M = translate((0, 0, 0)) @ rot_y(35) @ scale((1.4, 1.4, 1.4))

    img_no, _, f_no = raster3d.render(mesh, w=256, h=256, tex_u8=tex, model=M, cull_back=False)
    img_yes, _, f_yes = raster3d.render(mesh, w=256, h=256, tex_u8=tex, model=M, cull_back=True)
    red = (1.0 - f_yes / max(f_no, 1)) * 100.0
    print(f"Fragment terisi: cull_back=False -> {f_no} | cull_back=True -> {f_yes} (turun {red:.1f}%)")
    print(f"SSIM(no_cull, cull) = {ssim(img_no, img_yes):.4f} | PSNR = {psnr(img_no, img_yes):.1f} dB")


if __name__ == "__main__":
    os.makedirs("out", exist_ok=True)
    exp_e4_perspective_correct()
    exp_mipmap_far()
    exp_backface_culling()