"""
Split horosum_text_recolored.png into three transparent layers for
fragment-based reveal in the slide:

  horosum_layer_base.png  — A_0, A_1, their labels, (lambda=1/2), bottom line
  horosum_layer_geo.png   — geodesic Minkowski average arc and label
  horosum_layer_horo.png  — horocyclic Minkowski average blob and label

Within each layer, alpha is computed per pixel from darkness/saturation so
that white background is fully transparent and anti-aliased edges remain
smooth.
"""

import numpy as np
from PIL import Image


SRC = "horosum_text_recolored.png"


def main():
    im = np.array(Image.open(SRC).convert("RGB"))
    H, W = im.shape[:2]
    ys, xs = np.indices((H, W))

    R, G, B = im[..., 0].astype(int), im[..., 1].astype(int), im[..., 2].astype(int)
    maxc = np.maximum.reduce([R, G, B])
    minc = np.minimum.reduce([R, G, B])
    sat = maxc - minc
    darkness = 255 - maxc
    # Per-pixel alpha: opaque where dark (text/lines) or saturated (colored shapes),
    # transparent on pure white. Multiplied saturation a bit so faded edges of
    # colored shapes stay reasonably opaque.
    alpha_full = np.clip(np.maximum(darkness, 2 * sat), 0, 255).astype(np.uint8)

    # Region masks (rectangular, non-overlapping where it matters).
    base = np.zeros((H, W), dtype=bool)
    base |= (ys < 500) & (xs < 750)                                   # (lambda=1/2) label
    base |= (ys >= 1450) & (ys < 1880) & (xs < 700)                   # A_0 disc + label
    base |= (ys >= 1450) & (ys < 1880) & (xs >= 3500)                 # A_1 disc + label
    base |= (ys >= 1870)                                              # bottom horizontal line

    geo = (ys < 1450) & ~base                                         # arc + "A_lambda (geodesic)"

    horo = (ys >= 1450) & (ys < 1880) & (xs >= 700) & (xs < 3500)     # green blob + label + small circle

    def save_layer(region_mask, name):
        rgba = np.zeros((H, W, 4), dtype=np.uint8)
        rgba[..., :3] = im
        rgba[..., 3] = alpha_full * region_mask.astype(np.uint8) // 1  # zero outside region
        # Actually: alpha = alpha_full where region_mask, else 0.
        rgba[..., 3] = np.where(region_mask, alpha_full, 0).astype(np.uint8)
        Image.fromarray(rgba).save(name)
        print(f"Wrote {name}  (opaque pixels: {(rgba[..., 3] > 0).sum()})")

    save_layer(base, "horosum_layer_base.png")
    save_layer(geo,  "horosum_layer_geo.png")
    save_layer(horo, "horosum_layer_horo.png")


if __name__ == "__main__":
    main()
