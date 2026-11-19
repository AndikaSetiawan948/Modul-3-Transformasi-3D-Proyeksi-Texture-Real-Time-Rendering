"""Bersihkan 'o Cube' bawaan Blender + normalisasi skala -> data/mymodel.obj."""
import sys
import numpy as np
from objio import _parse_index, save_obj


def load_without_cube(path):
    positions, uvs, faces, face_uvs = [], [], [], []
    cur_obj = None
    skipped_faces = 0

    with open(path, "r") as fh:
        for line in fh:
            if line.startswith("o "):
                cur_obj = line.split()[1].strip()
            elif line.startswith("v "):
                positions.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("vt "):
                uvs.append([float(x) for x in line.split()[1:3]])
            elif line.startswith("f "):
                if cur_obj == "Cube":
                    skipped_faces += 1
                    continue
                toks = line.split()[1:]
                idx = [_parse_index(t, len(positions), len(uvs)) for t in toks]
                for k in range(1, len(idx) - 1):
                    faces.append([idx[0][0], idx[k][0], idx[k + 1][0]])
                    if idx[0][1] is not None:
                        face_uvs.append([idx[0][1], idx[k][1], idx[k + 1][1]])

    pos = np.asarray(positions, float)
    uv = np.asarray(uvs, float)
    F = np.asarray(faces, np.int64)
    FUV = np.asarray(face_uvs, np.int64)

    # Buang vertex & UV yatim (termasuk milik o Cube) dan rapikan indeks
    used_v, inv_v = np.unique(F, return_inverse=True)
    pos = pos[used_v]
    F = inv_v.reshape(-1, 3)

    used_uv, inv_uv = np.unique(FUV, return_inverse=True)
    uv = uv[used_uv]
    FUV = inv_uv.reshape(-1, 3)

    return pos, uv, F, FUV, skipped_faces


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "data/mymodel_raw.obj"
    dst = "data/mymodel.obj"

    pos, uv, F, FUV, skipped = load_without_cube(src)
    print(f"Input           : {src}")
    print(f"Dibuang (o Cube): {skipped} segitiga")
    print(f"Vertex bersih   : {len(pos)} | Segitiga: {len(F)} (syarat >= 200: {'OK' if len(F) >= 200 else 'KURANG'})")
    print(f"UV (vt) bersih  : {len(uv)} (syarat UV: {'OK' if len(uv) > 0 else 'TIDAK ADA'})")

    # Pusatkan X & Z ke 0, dudukkan alas tepat di Y = 0
    pmin, pmax = pos.min(axis=0), pos.max(axis=0)
    center_xz = np.array([(pmin[0] + pmax[0]) / 2.0, pmin[1], (pmin[2] + pmax[2]) / 2.0])
    pos = pos - center_xz

    # Skalakan agar dimensi terbesar = 2.0 satuan (pas dengan kamera raster3d.py & Three.js)
    extent = float(np.max(pmax - pmin))
    pos = pos * (2.0 / extent)

    save_obj(dst, pos, uv, F, FUV)
    print(f"\nTersimpan       : {dst}")
    print(f"Bounding box    : min={pos.min(axis=0).round(3)}, max={pos.max(axis=0).round(3)}")

    # Rekomendasi 1 vertex puncak atap & 1 vertex sudut teras untuk telusuran Lab 1
    top_idx = int(np.argmax(pos[:, 1]))
    corner_idx = int(np.argmax(pos[:, 0] + pos[:, 2]))
    print(f"Vertex puncak   (idx {top_idx}): {pos[top_idx].round(4)}")
    print(f"Vertex sudut +XZ(idx {corner_idx}): {pos[corner_idx].round(4)}")


if __name__ == "__main__":
    main()