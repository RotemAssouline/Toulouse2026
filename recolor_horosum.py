"""
Recolor horosum_text_blue.png to match fig_complex_euclidean_disc palette:
  A_0                    -> blue   (#5B9BD5)
  A_1                    -> orange (#ED7D31)
  A_lambda (geodesic)    -> gold   (#FFC000)
  A_lambda (horocyclic)  -> green  (#70AD47)

The three gray blobs (A_0 left, A_1 right, geodesic arc top) are
disambiguated by connected-component labeling. The light-blue horocyclic
blob is recolored by color match. Text and thin outlines stay untouched.
"""

import numpy as np
from PIL import Image
from scipy.ndimage import label


SRC = "horosum_text_blue.png"
DST = "horosum_text_recolored.png"

TARGET_A0  = (0x5B, 0x9B, 0xD5)
TARGET_A1  = (0xED, 0x7D, 0x31)
TARGET_GEO = (0xFF, 0xC0, 0x00)
TARGET_HOR = (0x70, 0xAD, 0x47)


def recolor_mask(arr_rgb, alpha, mask, target):
    """Blend `target` into rgba pixels under `mask`, preserving original luminance variation."""
    if not mask.any():
        return
    orig = arr_rgb[mask].astype(np.float32)
    lum = orig.mean(axis=1) / 255.0  # 0..1 — keeps anti-aliased edges soft

    # Map: pure target where lum ~ mid-gray (~0.65 for the body), fading
    # toward white at lum = 1 to preserve soft edges. Anything darker than
    # mid-gray keeps the target color.
    t = np.array(target, dtype=np.float32)
    white = np.array([255.0, 255.0, 255.0], dtype=np.float32)

    # Edge factor: 0 deep in shape, 1 at the white background.
    # Calibrate against the typical body luminance to avoid washing things out.
    body_lum = np.median(lum)
    edge_t = np.clip((lum - body_lum) / max(1e-3, 1.0 - body_lum), 0.0, 1.0)
    new = (1.0 - edge_t[:, None]) * t + edge_t[:, None] * white
    arr_rgb[mask] = np.clip(new, 0, 255).astype(np.uint8)


def main():
    im = Image.open(SRC).convert("RGBA")
    arr = np.array(im)
    rgb = arr[..., :3].copy()
    a = arr[..., 3]

    R, G, B = rgb[..., 0].astype(np.int16), rgb[..., 1].astype(np.int16), rgb[..., 2].astype(np.int16)

    # --- Light-blue horocyclic blob: B > R and B > G by a margin, and it's not white.
    blue_mask = (B > R + 25) & (B > G + 10)  # B>R+25 already excludes white

    # --- Gray blobs: R ~ G ~ B, not too dark (avoid text/outlines), not white.
    max_ch = np.maximum.reduce([R, G, B])
    min_ch = np.minimum.reduce([R, G, B])
    is_grayish = (max_ch - min_ch <= 15)
    is_midtone = (max_ch > 120) & (max_ch < 235)
    gray_mask = is_grayish & is_midtone

    # Label gray connected components and pick the three biggest.
    lbl, n = label(gray_mask)
    sizes = np.bincount(lbl.ravel())
    # component 0 is background; pick top 3 by size.
    comp_ids = np.argsort(sizes[1:])[::-1][:3] + 1

    # For each component find centroid x and area.
    comps = []
    for cid in comp_ids:
        ys, xs = np.where(lbl == cid)
        comps.append({
            "id": int(cid),
            "x": float(xs.mean()),
            "y": float(ys.mean()),
            "area": int(len(xs)),
            "mask": (lbl == cid),
        })

    # Geodesic arc is the largest by area; among the remaining two, leftmost = A_0, rightmost = A_1.
    comps_by_area = sorted(comps, key=lambda c: -c["area"])
    geo = comps_by_area[0]
    rest = sorted(comps_by_area[1:], key=lambda c: c["x"])
    a0, a1 = rest[0], rest[1]

    print(f"geo: x={geo['x']:.0f} y={geo['y']:.0f} area={geo['area']}")
    print(f"A_0: x={a0['x']:.0f} y={a0['y']:.0f} area={a0['area']}")
    print(f"A_1: x={a1['x']:.0f} y={a1['y']:.0f} area={a1['area']}")

    recolor_mask(rgb, a, geo["mask"],  TARGET_GEO)
    recolor_mask(rgb, a, a0["mask"],   TARGET_A0)
    recolor_mask(rgb, a, a1["mask"],   TARGET_A1)
    recolor_mask(rgb, a, blue_mask,    TARGET_HOR)

    arr[..., :3] = rgb
    Image.fromarray(arr).save(DST)
    print(f"Wrote {DST}")


if __name__ == "__main__":
    main()
