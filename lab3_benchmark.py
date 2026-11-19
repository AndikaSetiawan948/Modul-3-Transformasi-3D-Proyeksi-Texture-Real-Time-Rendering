"""Lab 3 Butir 3 -- Benchmark frame time raster3d.py vs raytrace.py pada data/mymodel.obj."""
import os
import time
import numpy as np

import raster3d
import raytrace
from objio import load_obj
from ppm_io import read_ppm, write_ppm
from transforms import translate, rot_y, scale
from metrics import psnr, ssim


def main():
    mesh = load_obj("data/mymodel.obj")
    tex = read_ppm("data/relief.ppm") if os.path.exists("data/relief.ppm") else None
    M = translate((0, 0, 0)) @ rot_y(35) @ scale((1.4, 1.4, 1.4))
    w, h = 256, 256

    print(f"Model: data/mymodel.obj ({mesh.n_tris} triangles), resolusi {w}x{h}")

    t0 = time.perf_counter()
    img_r, _, filled = raster3d.render(mesh, w=w, h=h, tex_u8=tex, model=M, cull_back=True)
    t_raster = time.perf_counter() - t0

    t0 = time.perf_counter()
    img_rt, t_hit = raytrace.render(mesh, w=w, h=h, tex_u8=tex, model=M)
    t_rt = time.perf_counter() - t0

    os.makedirs("out", exist_ok=True)
    write_ppm("out/mymodel_raster_256.ppm", img_r)
    write_ppm("out/mymodel_rt_256.ppm", img_rt)
    diff = np.clip(np.abs(img_r.astype(int) - img_rt.astype(int)) * 6, 0, 255).astype(np.uint8)
    write_ppm("out/mymodel_rt_diff.ppm", diff)

    ratio = t_rt / max(t_raster, 1e-9)
    print(f"raster3d.py : {t_raster:.3f} s  ({1.0/t_raster:.2f} FPS, {filled} fragments)")
    print(f"raytrace.py : {t_rt:.3f} s  ({1.0/t_rt:.2f} FPS, {int(np.isfinite(t_hit).sum())} pixels hit)")
    print(f"Rasio waktu : raytrace {ratio:.1f}x lebih lambat dari raster3d")
    print(f"Kecocokan   : PSNR = {psnr(img_r, img_rt):.1f} dB | SSIM = {ssim(img_r, img_rt):.4f}")


if __name__ == "__main__":
    main()