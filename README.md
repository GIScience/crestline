# Crestline

Polygon centerlines by flooding: no skeletonization, linear time, for shapely polygons and raster masks alike.

## Install

```bash
uv sync
```

## How it works

1. **Flood in.** The shape is flooded from its boundary one ring per step. A pixel's depth is the number of steps it
   stays dry.
1. **Crest.** Pixels with nothing deeper within 2 px are the crest: where the flood runs out.
1. **Flood out.** A second flood runs back from the crest, staying inside the shape. `climb / (climb + to_go)` gives a
   0 → 1 center raster that peaks along the middle of every branch, whatever its width. Its peaks are exactly the crest
   pixels at least 2 steps deep, so the vector pipeline reads them straight from step 2.
1. **Link.** Crest points are thinned and linked along the ridges, both scaled by each point's own depth, so narrow and
   wide branches are treated alike. The longest link of every triangle or quad is dropped, and dead ends whose disc fits
   inside the disc of a much deeper point (edge noise) are pruned.

## Benchmark

![](docs/benchmark.png)

Synthetic shapes with known centerlines at 1 m pixels, six seeds each: road networks (15 m half-width), road networks
with mixed half-widths (6 to 30 m) and tapering rivers (5 to 30 m), plus a 4 km road network (17 Mpx). Every shape is
traced as the clean polygon and as a noisy raster mask (about 2.5 m of boundary noise), and scored against the true
centerlines: F1 at 5 m and 2 m tolerance, mean offset of the matched line, and dead ends and loops against the truth.
`pygeoops.centerline` and `centerline` (Voronoi) run with their default settings, on the vectorised mask for noisy
inputs (vectorising is included in their runtime). Skeletonize + prune is scikit-image's `skeletonize` with branches
shorter than twice the median half-width pruned from their tips. Means over the 1 km shapes, runtimes on 2 CPU cores,
best in bold.

|                                               | crestline               | pygeoops.centerline | skeletonize + prune | centerline (Voronoi) |
|-----------------------------------------------|-------------------------|---------------------|---------------------|----------------------|
| F1 @ 5 m, polygon / noisy mask                | **0.929** / **0.909**   | 0.884 / 0.758       | 0.918 / 0.815       | 0.903 / 0.536        |
| F1 @ 2 m, polygon / noisy mask                | **0.846** / **0.699**   | 0.581 / 0.500       | 0.808 / 0.596       | 0.816 / 0.315        |
| Mean offset, polygon / noisy mask             | **0.55 m** / **1.32 m** | 1.59 m / 1.71 m     | 0.77 m / 1.53 m     | 0.58 m / 2.11 m      |
| Dead ends, polygon / noisy mask (true: 11.4)  | **11.2** / **19.2**     | 11.8 / 134.6        | 11.1 / 44.1         | 257.2 / 3488.4       |
| Loops, polygon / noisy mask (true holes: 3.8) | 4.3 / **4.7**           | **4.0** / 19.6      | 5.9 / 40.8          | 4.3 / 25.4           |
| Runtime, 1.2 Mpx, polygon / noisy mask        | **17 ms** / **18 ms**   | 64 ms / 806 ms      | 161 ms / 163 ms     | 472 ms / 1.9 s       |
| Runtime, 17 Mpx, polygon / noisy mask         | **275 ms** / **344 ms** | 459 ms / 18.2 s     | 2.7 s / 2.7 s       | -                    |

Synthetic data only so far...

## Limitations

TBD

## Background

Crestline combines established ideas:

- Blum's grassfire transform (https://en.wikipedia.org/wiki/Grassfire_transform),
- Digital distance transforms (Rosenfeld & Pfaltz, 1968),
- Boundary-to-skeleton ratio field like the skeleton-aware distance transform of Lin et al. (MICCAI 2023).

What it adds is a skeletonization-free, linear-time pipeline that normalises inside the shape and works on both vectors
and rasters.
