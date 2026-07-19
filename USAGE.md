# vfold on Windows — the no-assumptions guide

You have 4,000 photos. Your 3D reconstruction model accepts maybe 64. `vfold`
picks *which* 64, by choosing the ones that between them best represent the whole
set. It prints their file paths. That's all it does.

Everything below is copy-pasteable into PowerShell.

---

## 1. One-time setup

### 1.1 Build the program

You need Rust. If `cargo --version` prints nothing, install from
<https://rustup.rs> first, then reopen PowerShell.

```powershell
cd C:\path\to\vfold
cargo build --release
```

Takes a couple of minutes the first time. The program lands at
`C:\path\to\vfold\target\release\vfold.exe`.

### 1.2 Download the AI model it uses

`vfold` looks at images using a model called DINOv3. That model is not included
(it's 82 MB). Download it once:

```powershell
$dir = "$env:USERPROFILE\.cache\vfold"
New-Item -ItemType Directory -Force -Path $dir | Out-Null
$base = "https://huggingface.co/onnx-community/dinov3-vits16-pretrain-lvd1689m-ONNX/resolve/main"
curl.exe -sSL -o "$dir\model.onnx"      "$base/onnx/model.onnx"
curl.exe -sSL -o "$dir\model.onnx_data" "$base/onnx/model.onnx_data"
Get-ChildItem $dir | Select-Object Name, @{n='MB';e={[math]::Round($_.Length/1MB,1)}}
```

You should see **two** files: `model.onnx` (~0.1 MB) and `model.onnx_data`
(~82 MB). **You need both.** The first is just the blueprint; the second holds
the actual weights. With only the first, the program fails at startup.

> Use `curl.exe`, not PowerShell's `Invoke-WebRequest`. The `.exe` matters —
> plain `curl` in PowerShell is an alias for `Invoke-WebRequest` and takes
> different arguments.

### 1.3 Make it easy to type

Optional, but saves pain:

```powershell
Set-Alias vfold "C:\path\to\vfold\target\release\vfold.exe"
```

That lasts until you close the window. To make it permanent, add the same line
to your profile (`notepad $PROFILE`). The rest of this guide assumes the alias
exists; if not, just type the full path instead.

---

## 2. Your first run

```powershell
vfold "D:\Photos\MySite" -k 64
```

Two kinds of output appear, and the difference matters:

- **The chosen file paths** go to *stdout* — this is the actual answer.
- **Everything starting with `[vfold]`** goes to *stderr* — progress and
  diagnostics.

They are separated deliberately so you can pipe the answer somewhere without the
chatter coming along:

```powershell
vfold "D:\Photos\MySite" -k 64 > chosen.txt      # only the paths land in the file
```

Or ask for a file directly and still watch the progress on screen:

```powershell
vfold "D:\Photos\MySite" -k 64 --out chosen.txt --json report.json
```

**Quote any path containing spaces.** `D:\My Photos\Site` without quotes will be
read as two separate arguments and fail.

---

## 3. Reading the output

A healthy run looks roughly like this:

```
[vfold] 482 images in D:\Photos\MySite
[vfold] decoded 482 frames at 448px in 10.3s (10 threads)
[vfold] embedded 482 frames -> 6144-d (4x4 grid x 384) in 240.1s
[vfold] gps         482/482 tagged, site 307x388 m, co-visibility radius
                    p05/median/p95 = 14.0/18.7/22.8 m (1.6x spread) in 0.1s
[vfold] cosine      raw p05/p50/p95 = +0.650/+0.746/+0.830   whitened = -0.173/-0.014/+0.203
[vfold] rectified   p50/p95 = 0.000/0.000  mean nn1 = 0.652  coverage-graph density = 2.5%
[vfold] support gate need >=2 neighbours (corpus median 12): kept 482, rejected 0
[vfold] calibrate   cap 0.172 (0.26x median nn1) -> 70% saturated (target 70%)
[vfold] selected 64 of 482 in 0.01s (1222 gain evaluations vs 30848 naive, saturated 70%)
[vfold] spatial     nearest selected frame: median 11.8 m, p95 19.9 m, worst 27.4 m
                    [random: 14.2 / 48.3]  clearly beats chance
```

### The four lines actually worth checking

**`support gate ... rejected N`** — how many photos were ruled out as unusable
(blurred, pointed at the sky, lens covered). `rejected 0` on clean photos is
correct and expected. If it rejects a large fraction, your photo set has a
problem — look at the filenames it lists.

**`coverage-graph density`** — how interconnected your photos are. Anywhere from
about 2% to 70% is normal; it depends on how much your photos overlap. Below
~1% means the program can barely tell which photos go together, so the selection
is close to arbitrary.

**`saturated 70%`** — should land near 70%. This is auto-tuned; if it lands far
off (say under 30%), see Troubleshooting.

**`spatial ... clearly beats chance`** — only appears when your photos have GPS.
This is the honest scorecard. It compares your selection against picking 64
photos at random. Three possible verdicts:

| verdict | meaning |
|---|---|
| `clearly beats chance` | Good. Selection is genuinely spread over the site. |
| `marginally beats chance` | Weak. Works, but barely. |
| `NO BETTER THAN RANDOM - appearance aliasing likely` | **Bad.** See Troubleshooting. |

Without GPS this line is absent — that isn't an error, there's just nothing to
measure against.

---

## 4. Copy-paste recipes

### Pick 64 photos and put them in a new folder

Most reconstruction programs want a *folder* of images, not a list. This copies
the winners into one:

```powershell
$src = "D:\Photos\MySite"
$dst = "D:\Photos\MySite_64"
New-Item -ItemType Directory -Force -Path $dst | Out-Null
vfold $src -k 64 --out "$env:TEMP\pick.txt" | Out-Null
Get-Content "$env:TEMP\pick.txt" | ForEach-Object { Copy-Item $_ $dst }
"copied $((Get-ChildItem $dst).Count) files to $dst"
```

### Same, without duplicating gigabytes on disk

Hard links point at the same data, so the second "copy" costs nothing. Both
folders must be on the same drive.

```powershell
Get-Content "$env:TEMP\pick.txt" | ForEach-Object {
  New-Item -ItemType HardLink -Path (Join-Path $dst (Split-Path $_ -Leaf)) -Target $_ | Out-Null
}
```

### Keep the original shooting order

`vfold` outputs in the order it chose them (most important first), not in
filename order. If your next tool expects sequential frames:

```powershell
Get-Content "$env:TEMP\pick.txt" | Sort-Object | Set-Content "$env:TEMP\pick_sorted.txt"
```

### Try several budgets and compare

```powershell
foreach ($k in 32,48,64,96) {
  Write-Host "--- k=$k ---"
  vfold "D:\Photos\MySite" -k $k 2>&1 |
    Select-String "saturated |spatial " | Out-Null
}
```

Watch the `saturated` percentage in the `selected` line. If it's near 100% at
k=32, your budget is already enough and more photos add little. If it's still
low at k=96, the scene is bigger than your budget can cover.

### Faster, lower-quality pass for a quick look

```powershell
vfold "D:\Photos\MySite" -k 64 --size 224 --grid 2 --batch 32
```

Roughly 4× faster. `--grid 2` is required at 224 — the pooling grid must divide
the 14×14 patch grid, and the default `--grid 4` does not (the program stops and
says so). Fine for a sanity check; use the default 448 for real work.

### Checking the GPU is actually being used

The `device` line in the output says where the model ran:

```
[vfold] device      CUDA (batch 16, auto)
```

`CUDA` is the fast lane. If it says `CPU (higher rungs unavailable: ...)`, read
the parenthesis — it names exactly what failed (usually a missing CUDA runtime
or no NVIDIA GPU). The run still works on CPU, just ~6× slower.

---

## 5. Every option, and whether you should touch it

### The ones you'll actually use

| flag | default | what it does |
|---|---|---|
| `-k`, `--keep` | 64 | **How many photos to pick.** The main setting. Must be fewer than you have. |
| `-o`, `--out` | — | Also write the list to this file. |
| `--json` | — | Write a detailed report file. Useful for records. |
| `--gps` | auto | `off` disables GPS. Try this if results look wrong on close-up orbits (see below). |
| `--threads` | cores−2 | Fewer threads if you want to keep using your PC. |

### Occasionally useful

| flag | default | when to change |
|---|---|---|
| `--size` | 448 | `224` for ~4× speed at some quality cost. Must be a multiple of 16. |
| `--batch` | 0 (auto) | Auto picks 16 on GPU / 8 on CPU and **halves itself on out-of-memory**, so a 6 GB card self-throttles. Only set it manually to squeeze a big card (`32` gains ~10%). |
| `--device` | auto | `auto` tries CUDA, then CPU. `dml` exists but is opt-in only — the current model export crashes the DirectML provider. `cpu` forces the slow lane. |
| `--overlap` | 8 | Roughly how many photos show the same thing. Raise for very overlapping video frames, lower for sparse handheld sets. |
| `--model` | `%USERPROFILE%\.cache\vfold\model.onnx` | If you put the model elsewhere. |
| `--target-sat` | 0.70 | The auto-tuning target. `0.85` sometimes spreads slightly better. Anything below `0.5` degrades. |

### Leave these alone unless you know why

`--objective`, `--sat-mult`, `--min-neighbours`, `--gps-scale`, `--power`,
`--drop-pcs`, `--sigma-floor`, `--grid`.

Two warnings in particular:

- **`--objective proportional` is deliberately bad.** It exists only to
  reproduce a documented failure (it dumps the entire budget onto whichever area
  you photographed most). Never use it for real work.
- **`--sat-mult` overrides the auto-tuning.** The correct value depends on your
  photo set in a way that isn't guessable; leaving it at `0` lets the program
  work it out. Setting it by hand is how you get a bad result that still looks
  like it worked.

---

## 6. Troubleshooting

| symptom | cause and fix |
|---|---|
| `loading ...model.onnx: ...` at startup | Model missing or incomplete. Redo §1.2 and confirm **both** files exist, with `model.onnx_data` around 82 MB. |
| `no images in <dir>` | Wrong folder, or images are in a subfolder. `vfold` does not search subfolders — point it at the folder actually containing the `.jpg` files. |
| `-k 64 is not smaller than the 50 images available` | You asked for more photos than exist. Lower `-k`. |
| `gate left only N eligible frames for k=...` | Too many photos were rejected as unusable. Usually a genuinely bad photo set. As a last resort `--min-neighbours 1`, but inspect the rejects first. |
| `NO BETTER THAN RANDOM - appearance aliasing likely` | Your scene looks the same everywhere (crops, water, sand, uniform roof) so the program can't tell locations apart by looks. If photos have GPS this should already be handled; if not, GPS is the fix — or accept that a plain evenly-spaced selection is as good here. |
| Rejects lots of photos from a set that looks fine | Check the listed filenames. If they really are fine, your photos may have too little overlap for the default `--overlap 8`; try `--overlap 4`. |
| Very slow | Check the `device` line. On CUDA, 500 photos take ~30 s; on CPU ~3–4 minutes. If it fell back to CPU the parenthesis says why. `--size 224 --grid 2` is a further ~4× cheaper. |
| `device CPU (higher rungs unavailable...)` on a machine with an NVIDIA GPU | The CUDA 12 runtime DLLs aren't findable. Keep the DLLs that the build placed next to `vfold.exe` (`onnxruntime_providers_cuda.dll` etc.) with the exe if you move it, and make sure the NVIDIA driver is current. |
| Out of memory | Lower `--batch` to `4` or `2`. |
| Paths with spaces fail | Quote them: `vfold "D:\My Photos\Site" -k 64`. |
| Want to script it | Exit code `0` = success, `1` = error. `--help` also exits `0`. |

---

## 7. What it does not do

- It does **not** reconstruct anything. It only chooses photos. Feed the result
  to your actual 3D program.
- It does **not** look in subfolders.
- It does **not** modify, move, or delete any of your photos. It only reads them.
- It does **not** need internet after the one-time model download.
