# vfold

Fold a large unordered image set down to the *k* views that best cover it, so a
sparse feed-forward reconstruction model can ingest the whole scene in one pass
instead of chunk-and-stitch.

Detached by design: it reads a directory and writes a newline-delimited list of
paths to stdout. Nothing links against it. Delete the binary and the rest of the
pipeline is unchanged. Dependencies are `image`, `jpeg-decoder`, `rayon`, and
`ort` (ONNX Runtime) — no Python, no torch.

```
vfold <IMAGE_DIR> -k 64 --out skeleton.txt --json diag.json
```

- **[USAGE.md](USAGE.md)** — step-by-step Windows guide. Setup, every flag in
  plain English, copy-paste PowerShell recipes, troubleshooting table. Start
  here if you just want to run it.
- **[AGENTS.md](AGENTS.md)** — for coding agents wiring this into a
  view-count-limited feed-forward 3D model. How to measure the model's real
  ceiling (rather than guess it), how to size `k`, which JSON fields gate a
  "did it actually work" decision, and what may not be claimed without an A/B.

The rest of this file is *why* the internals are the way they are.

## What it does, and why each piece is the way it is

```
JPEG dir ──▶ scaled decode ──▶ DINOv3 ViT-S/16 ──▶ grid pool ──▶ whiten
              (1/8 via DCT)      (448px, patch16)    (4×4×384)      (drop PC1)
                                                                       │
   skeleton.txt ◀── saturated-coverage lazy greedy ◀── rectify ───────┘
                    (uniform cap, support-gated)       (frame-based floor)
```

- **Scaled decode.** JPEG stores 8×8 DCT blocks; reconstructing from the
  low-frequency corner yields the image directly at 1/8 size for ~1/64th of the
  decode work. On 6000×4000 frames this is the single biggest lever.
- **DINOv3, not the reconstruction model's own features.** Mean-pooled tokens
  from the sparse model encode *which scene*, not *which viewpoint* — measured on
  this project's own data at 0.982 adjacent vs 0.789 distant cosine, with dedup
  thresholds landing on a cliff. DINOv3's Gram-anchored dense features and its 4
  register tokens (patch tokens start at index 5, not 1) avoid that.
- **4×4 grid pooling.** A global mean makes two overlapping nadir frames look
  identical. Preserving coarse spatial layout is what tells them apart.
- **Whiten.** Self-supervised ViT features have a dominant leading component that
  says "same scene" and compresses every similarity into a narrow band. Dropping
  it restores ~4× the dynamic range (measured).
- **Frame-based rectification.** "Covering" must mean genuine overlap, not weak
  correlation, so low similarities are floored to zero. The floor is set from
  each frame's *`--overlap`-th* strongest neighbour, guaranteeing 95% of frames
  keep that many neighbours **regardless of region** — an all-pairs quantile
  instead starves a minority region when one region is oversampled.
- **Saturated coverage with a uniform cap**, not the textbook per-item cap of
  Lin & Bilmes. The per-item cap scales with how photographed a region already
  is, so the densest region takes the whole budget — reproducible with
  `--objective proportional` (12/12 onto one cluster in the test). The uniform
  cap sends budget to whatever is still under-covered.
- **Support gate as a count, not a threshold.** A frame needs ≥2 genuine
  neighbours to be registrable at all; blur/sky/obstruction frames have zero and
  are excluded from *selection* while still counting as *covered*. A count needs
  no tuning and rejects nothing on clean data — a relative threshold slices the
  tail of any distribution and always finds victims.
- **Lazy greedy.** Submodularity makes Minoux's lazy evaluation return the
  bit-identical answer to naive greedy; on the 444-frame set, 1029 gain
  evaluations instead of 28,416.
- **GPS as a false-edge filter** (`--gps auto`, when EXIF positions exist).
  Visual similarity is only a *proxy* for co-visibility, and on repetitive ground
  the proxy breaks — two frames 300 m apart over identical crop rows are
  neighbours in feature space, so one wheat-field frame "covers" every wheat
  field on the site. A spatial plausibility weight multiplied into the similarity
  matrix removes those false edges; every downstream stage runs unchanged. No
  stratifier, no clusterer, no grid.

## The cap calibrates itself

`--sat-mult` expresses the saturation cap as a multiple of median
nearest-neighbour similarity. That sounds scale-free and is not: the right
multiple moves with coverage-graph density. Measured on a 482-frame UAV set,
worst-case ground gap by multiplier —

| sat-mult | 0.5 | 0.75 | 1.0 | 1.5 | 2.0 |
|---|---|---|---|---|---|
| density 5.1% (`--gps off`) | 39.0 | **36.5** | 38.2 | 57.4 | 50.8 |
| density 2.5% (`--gps auto`) | **27.6** | 39.1 | 50.1 | 64.9 | 87.2 |

A constant tuned at one density misbehaves badly at another — which is how
*enabling* GPS initially made results worse. The quantity that actually predicts
quality is the **fraction of frames whose coverage reaches the cap**: every good
run above sits at 56–85% saturation, every poor one at 3–45%. The mechanism is
mechanical — when little saturates there are no diminishing returns, the
objective degenerates towards a linear sum, and greedy loses its incentive to
spread.

So the default targets the fraction and bisects for `T` (saturation is monotone
decreasing in `T`; selection is instant, so probes are free). The result is a
plateau instead of a cliff:

| `--target-sat` | 0.55 | 0.70 | 0.85 |
|---|---|---|---|
| `--gps auto` worst | 27.6 | 27.4 | **25.3** |
| `--gps off` worst | 38.2 | **36.5** | 39.0 |

The cap it selects varies 4× between scenes (1.00× median nn1 on ETH3D facade,
0.26× on the UAV set). That variation is the point.

## Spatial mode (`--gps`)

Nothing in the code knows what a lawnmower pattern is, and that is deliberate:
orbits, expanding spirals, sector searches and scattered swarm patterns are all
just point sets. Two properties make it topology-agnostic.

**The scale is per-frame.** `rᵢ` is the distance to frame *i*'s own
`--overlap`-th nearest neighbour, giving `w_ij = exp(−u²)` with
`u = d²ᵢⱼ / (σ²·rᵢ·rⱼ)` — the self-tuning kernel of Zelnik-Manor & Perona
(2004), symmetric because `rᵢ·rⱼ` is. A single global radius assumes uniform
capture density; real flights mix dense hovering with sparse transit. Even on a
near-uniform crosshatch, the 8th-neighbour distance ranged 14.0–22.8 m between
p05 and p95, with nearest-neighbour spacing from 0.2 m (hover) to 10 m.

**The exponent is load-bearing.** A plain Gaussian returns `exp(−1) = 0.37` at
`d = r`, attenuating *genuine* overlapping pairs along with false ones. Measured:
graph density fell 5.1% → 2.7%, only 8% of frames reached the saturation cap,
and with nothing saturating the objective degenerates to a linear sum, loses its
diminishing returns, stops spreading, and left a 77.5 m hole. `exp(−u²)` holds
~0.94 at `d = r` and collapses beyond `1.5 r`.

Frames without GPS get neutral weight, so partial tagging degrades smoothly to
appearance-only behaviour. Use `--gps off` for tight orbits around a single
object, where cameras on opposite sides are far apart yet do see the same thing.

**Always read the spatial audit against the random baseline**, which is printed
on every run with GPS present. Absolute metres are uninterpretable alone: a
median distance-to-nearest-selected of 13.6 m sounds excellent until random
scores 14.2 m — which is exactly what appearance-only selection achieved here,
and the only reason the aliasing was caught.

## Key flags

| flag | default | meaning |
|---|---|---|
| `-k` | 64 | views to keep |
| `--size` | 448 | model input edge (multiple of 16) |
| `--grid` | 4 | pooling grid edge |
| `--objective` | saturated | `saturated` \| `facility` \| `proportional` |
| `--sat-mult` | 0 (auto) | uniform cap = this × median nn similarity; 0 auto-calibrates |
| `--target-sat` | 0.70 | fraction of frames to saturate (the knob that matters) |
| `--overlap` | 8 | expected views seeing the same ground; sets graph density |
| `--min-neighbours` | 2 | structural minimum to be selectable |
| `--gps` | auto | `auto` \| `off` |
| `--gps-scale` | 1.0 | co-visibility radius relative to local point spacing |

## Model

Expects the DINOv3 ViT-S/16 ONNX export at
`%USERPROFILE%\.cache\vfold\model.onnx` (+ `model.onnx_data`), from
`onnx-community/dinov3-vits16-pretrain-lvd1689m-ONNX`. Override with `--model`.

## Where it fits

Drop-in replacement for the FPS skeleton selector in the dvlt_long hub topology:
point `--skeleton-size`'s consumer at `skeleton.txt`. It does **not** claim a
single 64-view pass beats chunk-and-stitch — DVLT is trained on 2–18 views and 64
is out of distribution. It claims the skeleton is chosen by scene coverage rather
than by a descriptor that cannot see viewpoint.

Note on aerial nadir strips specifically: when terrain is near-uniform along
track, uniform stride is already a strong baseline and saturated coverage mostly
matches it. The gain shows up where content is *uneven* — mixed structures, not
flat farmland.

That baseline result has a cause, and it is worth stating plainly: on repetitive
terrain, appearance-only selection is spatially **no better than chance**
(measured on a 482-frame UAV crosshatch: 13.6 m median / 50.8 m worst, versus
14.2 m / 48.9 m for random). Feature-space coverage looked healthy the whole
time. This is the failure `--gps` exists to fix, and it is why the random
baseline is now printed rather than left to be discovered.
