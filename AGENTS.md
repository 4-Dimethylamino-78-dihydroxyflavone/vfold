# Instructions for coding agents

You are integrating `vfold` into a pipeline built around a **feed-forward 3D
foundation model** — VGGT, DUSt3R/MASt3R, Fast3R, CUT3R, DVLT, MapAnything, π³,
AnySplat, or similar. These models take N images and emit geometry in one
forward pass, and every one of them has a hard ceiling on N. `vfold` decides
*which* N.

Read `USAGE.md` for the human-facing CLI reference. This file is about judgement:
when to reach for it, how to size the budget, how to tell whether it worked, and
what not to claim.

---

## 1. The problem this solves

Feed-forward reconstruction models use **global attention across views**, so
compute and memory scale as O(N²). That produces two separate ceilings, and
conflating them is the most common integration mistake:

| ceiling | what it is | what happens past it |
|---|---|---|
| **VRAM ceiling** | Where activations stop fitting | Hard OOM crash. Obvious. |
| **Training-distribution ceiling** | Largest N the model was trained on | Silent quality degradation. No error. |

The second is lower and far more dangerous. DVLT, for example, was trained on
**2–18 views** but will physically accept ~80 at 504 px on a 16 GB card. Feeding
it 64 runs it 3.5× outside its training distribution; it produces output, the
output is worse, and nothing warns you.

**`vfold` addresses neither ceiling.** It does not raise N. It makes a given N
buy more scene. Keep that distinction sharp when reporting results.

---

## 2. Decision rule

```
Does the capture exceed the model's view ceiling?
├── No  → do not use vfold. Feed everything.
└── Yes → is the model's training-distribution ceiling also exceeded by your target k?
    ├── No  → PATTERN A: direct fold. Single pass over k views.
    └── Yes → PATTERN B: anchor/skeleton fold. Keep all N in play.
```

### Pattern A — direct fold

The capture is too big, but your budget sits inside the model's trained range.
Fold once, run once.

```
N images ──vfold──▶ k images ──▶ model ──▶ geometry
```

### Pattern B — anchor / skeleton fold (usually the right one)

Your budget would push the model out of distribution. Don't. Use `vfold` to
choose the **reference set** for a chunked, hub, or anchor-based topology, and
keep every original image in the reconstruction.

```
N images ──vfold──▶ k anchors ──┐
                                ├──▶ chunked / hub reconstruction ──▶ geometry
N images ───────────────────────┘
```

This is the higher-value use and the less obvious one. Chunked pipelines'
accuracy ceiling is usually **seam alignment**, and seams are determined by which
frames anchor the chunks. Replacing an arbitrary or stride-based anchor choice
with a coverage-maximal one improves the reconstruction without ever running the
model out of distribution. It is also the design ZPressor validates
(arXiv 2505.23734, NeurIPS 2025): partition into anchor and support sets, and
~50 anchors carry a 1000-view scene.

---

## 3. Find the model's real ceiling before choosing k

Do not guess, and do not trust a number from a README — it depends on your GPU,
resolution, and precision. Measure it. Bisect on N until you find the OOM edge,
then back off.

```powershell
# Replace the inner command with your model's actual invocation.
$lo = 8; $hi = 256
while ($hi - $lo -gt 4) {
  $mid = [math]::Floor(($lo + $hi) / 2)
  vfold "D:\Photos\Site" -k $mid --out "$env:TEMP\probe.txt" | Out-Null
  # materialise and attempt one forward pass
  $ok = $true
  try { python run_model.py --list "$env:TEMP\probe.txt" 2>&1 | Out-Null; if ($LASTEXITCODE -ne 0) { $ok = $false } }
  catch { $ok = $false }
  if ($ok) { $lo = $mid } else { $hi = $mid }
  Write-Host "N=$mid -> $(if($ok){'ok'}else{'FAIL'})  (bracket $lo..$hi)"
}
Write-Host "VRAM ceiling ~= $lo views"
```

Then set your budget:

```
k = min( VRAM_ceiling × 0.8 ,  training_distribution_ceiling )
```

The 0.8 leaves headroom — reconstruction memory varies with scene content, and a
run that OOMs at frame 300 of a batch job is worse than one that used fewer
views. If the training ceiling is the binding constraint and it is far below the
VRAM ceiling, that is your signal to switch to Pattern B.

---

## 4. Running it

```powershell
$vfold = "C:\path\to\vfold\target\release\vfold.exe"
& $vfold "D:\Photos\Site" -k 64 --out "$env:TEMP\pick.txt" --json "$env:TEMP\pick.json"
if ($LASTEXITCODE -ne 0) { throw "vfold failed" }
```

**stdout is the path list; stderr is diagnostics.** Never parse stderr for the
answer — use `--out` or redirect stdout.

Most models want a directory, not a list. Materialise with hard links so you
don't duplicate gigabytes (same volume required):

```powershell
$dst = "D:\Photos\Site_k64"
New-Item -ItemType Directory -Force -Path $dst | Out-Null
Get-Content "$env:TEMP\pick.txt" | ForEach-Object {
  New-Item -ItemType HardLink -Path (Join-Path $dst (Split-Path $_ -Leaf)) -Target $_ | Out-Null
}
```

**Ordering.** Output is in selection order (most scene-carrying first), not
filename order. Models that assume sequential input — anything doing temporal
windowing or treating frame 0 as a reference — need it sorted:

```powershell
Get-Content "$env:TEMP\pick.txt" | Sort-Object | Set-Content "$env:TEMP\pick_sorted.txt"
```

Selection order is itself meaningful: if you need a *reference* or *anchor* frame,
take rank 0.

---

## 5. Verify before you believe it

Parse `--json`. Schema:

```jsonc
{
  "dir": "...", "n": 482, "k": 64,
  "params": { "size":448, "grid":4, "objective":"saturated", "sat_mult":0,
              "overlap":8, "min_neighbours":2, "gps":"auto", "gps_scale":1.0,
              "power":1.0, "drop_pcs":1, "sigma_floor":0.3 },
  "similarity": { "p05":…, "p50":…, "p95":…, "mean_nn1":… },
  "gate": { "min_neighbours":2, "median_neighbours":12, "kept":482, "rejected":0 },
  "saturated_frac": 0.70,
  "spatial_m": { "median":11.8, "p95":19.9, "max":27.4,       // only when GPS present
                 "random_median":14.2, "random_max":48.3 },
  "curve": [ … ],                       // objective value after each pick
  "rejected": [ "path", … ],
  "selected": [ { "rank":0, "index":17, "path":"…", "focus":…, "nn1":… }, … ]
}
```

### Gate checks

| check | threshold | meaning if it fails |
|---|---|---|
| `gate.rejected / n` | > 0.15 | Many frames judged unusable. Inspect `rejected[]` before proceeding. |
| `saturated_frac` | < 0.5 | Calibration did not converge; selection may not have spread. |
| `spatial_m.max` | ≥ `spatial_m.random_max` | **Selection is no better than random.** Do not ship. |
| `similarity.p95` | ≈ 0 | Descriptors failed to discriminate. Investigate before using output. |

The spatial check is the one that matters and the only one that can catch a
fully plausible-looking failure. It exists because appearance-only selection on
repetitive terrain (farmland, water, uniform roofing) once scored 13.6 m median
distance-to-nearest-selected against random's 14.2 m — statistically
indistinguishable — while every feature-space diagnostic looked healthy. If the
images carry GPS, this check is free. **Run it.**

### If there is no GPS

You have no ground-truth spatial check. Fall back to comparing against a uniform
stride selection on your actual downstream metric, and be explicit in any report
that the spatial claim is unverified.

---

## 6. Failure modes and responses

| observation | do this |
|---|---|
| `NO BETTER THAN RANDOM` in stderr | Scene is visually self-similar. If GPS exists it is already applied — the honest conclusion is that a uniform stride is equally good here; say so rather than shipping a null result as a win. |
| `gate left only N eligible frames for k=…` | Genuinely poor capture. Inspect `rejected[]`. Only lower `--min-neighbours 1` after confirming the rejects are salvageable. |
| `coverage-graph density` < 1% | Too little overlap for the default. Try `--overlap 4`. |
| Results look wrong on a tight orbit around one object | Try `--gps off`. Spatial gating assumes distance implies non-co-visibility, which is false for cameras encircling a small subject. |
| Tempted to hand-set `--sat-mult` | Don't. The correct value moves with graph density (measured: 1.98× median-nn on one scene, 0.26× on another). Auto-calibration exists because a fixed constant silently misbehaves. Adjust `--target-sat` instead. |
| Someone suggests `--objective proportional` | It is a deliberately preserved failure case. It puts the entire budget on the most-photographed region. Never for production. |

---

## 7. Honesty guardrails

These constrain what you may claim in reports and commit messages.

- **Selection quality ≠ reconstruction quality.** `vfold` improving coverage
  metrics does not establish that the reconstruction is better. Only an A/B on
  the downstream output does. If you have not run one, say the downstream effect
  is unmeasured.
- **Always report the random baseline** alongside any spatial number. A bare
  "median 13.6 m" is not a result.
- **On near-uniform captures, uniform stride is a strong baseline** and may match
  or beat `vfold`. This is expected — the advantage appears where content is
  *uneven*. Report it as a tie when it is one.
- **`vfold` does not fix out-of-distribution view counts.** If the model was
  trained on 2–18 views, a beautifully chosen 64 is still 64. Prefer Pattern B.
- **Do not fabricate model ceilings.** Measure them (§3) or state that they are
  unmeasured. The only figures verified in this project are DVLT's 2–18 training
  range and ~80 frames @504 px on 16 GB.

---

## 8. Cost model

Runtime is dominated by the DINOv3 forward pass, not by selection.

| stage | ~cost (500 images) | notes |
|---|---|---|
| Decode | ~10 s | DCT-scaled; already ~64× cheaper than naive |
| Embed | **~27 s CUDA / ~180–240 s CPU** | The bottleneck either way. Measured 482 frames @448 px on an RTX 4060 Ti: 169 s CPU → 27 s CUDA (batch 32 gains ~10% more; the floor is host-device copies). |
| Whiten + similarity | < 1 s | |
| Selection | ~0.01 s | Lazy greedy; ~1.2k gain evaluations vs 30k naive |

Device handling — read the `[vfold] device` line, never assume:

- `--device auto` (default) tries **CUDA → CPU**, each rung hard-failed via
  `error_on_failure` so the achieved device is known, not hoped. Fallback prints
  the exact init error in parentheses.
- **DirectML is opt-in only** (`--device dml`), excluded from auto: on the
  current DINOv3 export it passes session init then dies on a dynamic-shape
  Reshape at run time — nondeterministically as an error *or an access
  violation*, and an AV is uncatchable in-process. Do not put it back in the
  auto ladder without verifying a new model export end-to-end.
- `--batch 0` (default) sizes per device and **halves on OOM**, so low-VRAM
  machines self-throttle. A failure persisting at batch 1 under auto triggers
  device demotion mid-run; already-computed descriptors are kept (cross-device
  float drift ~1e-6, far below the similarity floor).
- If you relocate `vfold.exe`, take the provider DLLs beside it
  (`onnxruntime_providers_cuda.dll`, `onnxruntime_providers_shared.dll`,
  `DirectML.dll`) or CUDA silently becomes unavailable and auto lands on CPU.

Consequences for agents:

- **There is no descriptor cache.** Every parameter sweep re-embeds from
  scratch. On CUDA that is ~40 s per run per 500 images (tolerable to sweep);
  on CPU ~4 minutes (plan a single batch of runs instead).
- `--size 224 --grid 2` is ~4× faster for a smoke test (`--grid 2` is required:
  the pooling grid must divide 224/16 = 14). Use 448 + grid 4 for anything
  reported.
- Selection being effectively free is why `--target-sat` auto-calibration can
  afford ~14 full selection probes per run.
