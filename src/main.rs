//! vfold — squeeze a large image set down the thin funnel of a sparse model.
//!
//! Feedforward reconstruction models ingest a bounded number of views. DVLT is
//! trained on 2-18 and hits a VRAM wall near 80 frames at 504 px on 16 GB. A
//! 500-image capture does not fit. The usual answer is to chunk and stitch,
//! which makes seam alignment the accuracy ceiling.
//!
//! This tool takes the other path: pick the k views that best *represent* the
//! whole set and hand those over in one pass.
//!
//! It is deliberately detached. Output is a newline-delimited list of paths on
//! stdout, plus a JSON sidecar. Nothing links against it; delete the binary and
//! the existing pipeline is unchanged.

mod coverage;
mod decode;
mod descriptor;
mod embed;
mod gps;

use coverage::{Objective, Sim};
use std::path::{Path, PathBuf};
use std::time::Instant;

struct Args {
    dir: PathBuf,
    k: usize,
    model: PathBuf,
    size: usize,
    grid: usize,
    batch: usize,
    objective: String,
    /// Cap as a multiple of the median nearest-neighbour similarity. Anchoring
    /// to the data rather than fixing an absolute value keeps one setting valid
    /// across dense and sparse captures.
    /// Cap as a multiple of median nearest-neighbour similarity. 0 means
    /// auto-calibrate against `target_sat`, which is the default and the
    /// recommended setting — the right multiple depends on graph density.
    sat_mult: f32,
    /// Fraction of frames whose coverage should reach the cap. The quantity that
    /// empirically tracks selection quality; see `coverage::calibrate_cap`.
    target_sat: f32,
    /// Expected overlap of the capture — how many other views typically see the
    /// same ground. Sets the rectification floor. A capture property, not a
    /// budget property. See `descriptor::similarity`.
    overlap: usize,
    /// Genuine overlapping neighbours a frame needs to be selectable at all. A
    /// structural minimum (a frame with one neighbour is a dead end), not really
    /// a tuning knob. See `coverage::support_gate`.
    min_neighbours: usize,
    /// `auto` uses EXIF GPS when enough frames carry it; `off` forces pure
    /// appearance. See `gps` for when off is the better choice.
    gps: String,
    /// Widens or narrows the co-visibility radius relative to the capture's own
    /// local point spacing. 1.0 means "about as far as `overlap` neighbours".
    gps_scale: f32,
    power: f32,
    drop_pcs: usize,
    sigma_floor: f32,
    out: Option<PathBuf>,
    json: Option<PathBuf>,
    threads: usize,
}

const USAGE: &str = "\
vfold - fold a large image set down to the k views that best cover it

USAGE:
    vfold <IMAGE_DIR> [OPTIONS]

OPTIONS:
    -k, --keep <N>          views to select                        [default: 64]
        --model <PATH>      DINOv3 ONNX  [default: %USERPROFILE%\\.cache\\vfold\\model.onnx]
        --size <PX>         model input edge, multiple of 16       [default: 448]
        --grid <G>          pooling grid edge (GxG regions)        [default: 4]
        --batch <N>         frames per inference call              [default: 8]
        --objective <NAME>  saturated | facility | proportional    [default: saturated]
        --sat-mult <F>      cap = F x median nearest-neighbour sim;
                            0 auto-calibrates to --target-sat        [default: 0]
        --target-sat <F>    fraction of frames to saturate         [default: 0.70]
        --overlap <D>       expected views seeing the same ground;
                            sets the coverage-graph density        [default: 8]
        --min-neighbours <N>  neighbours a frame needs to be
                              selectable at all (structural)         [default: 2]
        --gps <MODE>        auto | off                            [default: auto]
        --gps-scale <F>     co-visibility radius, relative to the
                            capture's own local point spacing      [default: 1.0]
        --power <F>         similarity sharpening exponent         [default: 1.0]
        --drop-pcs <N>      leading principal components to remove [default: 1]
        --sigma-floor <F>   whitening floor as fraction of max     [default: 0.3]
    -o, --out <PATH>        write list here as well as stdout
        --json <PATH>       write diagnostics JSON
        --threads <N>       worker threads                    [default: cores-2]
    -h, --help

The objective defaults to saturated coverage with a UNIFORM cap. The textbook
per-item cap of Lin & Bilmes (--objective proportional) is available but is not
recommended: its threshold scales with how photographed a region already is, so
the densest region takes the whole budget. See src/coverage.rs.
";

fn parse() -> Result<Args, String> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    if argv.is_empty() || argv.iter().any(|a| a == "-h" || a == "--help") {
        return Err(USAGE.to_string());
    }
    let home = std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .unwrap_or_default();
    let mut a = Args {
        dir: PathBuf::new(),
        k: 64,
        model: PathBuf::from(&home).join(".cache").join("vfold").join("model.onnx"),
        size: 448,
        grid: 4,
        batch: 8,
        objective: "saturated".into(),
        sat_mult: 0.0,
        target_sat: 0.70,
        overlap: 8,
        min_neighbours: 2,
        gps: "auto".into(),
        gps_scale: 1.0,
        power: 1.0,
        drop_pcs: 1,
        sigma_floor: 0.3,
        out: None,
        json: None,
        threads: std::thread::available_parallelism().map(|n| n.get().saturating_sub(2).max(1)).unwrap_or(4),
    };

    let mut i = 0;
    let mut positional = false;
    while i < argv.len() {
        let next = |i: usize| -> Result<String, String> {
            argv.get(i + 1).cloned().ok_or_else(|| format!("{} needs a value", argv[i]))
        };
        let num = |s: String, what: &str| -> Result<f32, String> {
            s.parse::<f32>().map_err(|_| format!("{what}: '{s}' is not a number"))
        };
        match argv[i].as_str() {
            "-k" | "--keep" => { a.k = num(next(i)?, "-k")? as usize; i += 2 }
            "--model" => { a.model = PathBuf::from(next(i)?); i += 2 }
            "--size" => { a.size = num(next(i)?, "--size")? as usize; i += 2 }
            "--grid" => { a.grid = num(next(i)?, "--grid")? as usize; i += 2 }
            "--batch" => { a.batch = num(next(i)?, "--batch")? as usize; i += 2 }
            "--objective" => { a.objective = next(i)?; i += 2 }
            "--sat-mult" => { a.sat_mult = num(next(i)?, "--sat-mult")?; i += 2 }
            "--target-sat" => { a.target_sat = num(next(i)?, "--target-sat")?; i += 2 }
            "--overlap" => { a.overlap = num(next(i)?, "--overlap")? as usize; i += 2 }
            "--min-neighbours" | "--min-neighbors" => {
                a.min_neighbours = num(next(i)?, "--min-neighbours")? as usize;
                i += 2
            }
            "--gps" => { a.gps = next(i)?; i += 2 }
            "--gps-scale" => { a.gps_scale = num(next(i)?, "--gps-scale")?; i += 2 }
            "--power" => { a.power = num(next(i)?, "--power")?; i += 2 }
            "--drop-pcs" => { a.drop_pcs = num(next(i)?, "--drop-pcs")? as usize; i += 2 }
            "--sigma-floor" => { a.sigma_floor = num(next(i)?, "--sigma-floor")?; i += 2 }
            "-o" | "--out" => { a.out = Some(PathBuf::from(next(i)?)); i += 2 }
            "--json" => { a.json = Some(PathBuf::from(next(i)?)); i += 2 }
            "--threads" => { a.threads = num(next(i)?, "--threads")? as usize; i += 2 }
            s if s.starts_with('-') => return Err(format!("unknown flag {s}\n\n{USAGE}")),
            s => {
                if positional {
                    return Err(format!("unexpected second directory '{s}'"));
                }
                a.dir = PathBuf::from(s);
                positional = true;
                i += 1;
            }
        }
    }
    if !positional {
        return Err(format!("no image directory given\n\n{USAGE}"));
    }
    Ok(a)
}

fn main() {
    // Help is not an error: it belongs on stdout with exit 0, so `vfold --help`
    // pipes cleanly and a wrapper script does not read it as a failure.
    let argv: Vec<String> = std::env::args().skip(1).collect();
    if argv.is_empty() || argv.iter().any(|a| a == "-h" || a == "--help") {
        print!("{USAGE}");
        return;
    }
    if let Err(e) = run() {
        eprintln!("{e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let a = parse()?;
    rayon::ThreadPoolBuilder::new()
        .num_threads(a.threads)
        .build_global()
        .ok();

    let t0 = Instant::now();
    let paths = decode::list_images(&a.dir)?;
    if paths.is_empty() {
        return Err(format!("no images in {}", a.dir.display()));
    }
    eprintln!("[vfold] {} images in {}", paths.len(), a.dir.display());
    if a.k >= paths.len() {
        return Err(format!(
            "-k {} is not smaller than the {} images available; nothing to fold",
            a.k,
            paths.len()
        ));
    }

    // --- decode -------------------------------------------------------------
    let t = Instant::now();
    let (frames, failed) = decode::load_all(&paths, a.size as u32);
    for (p, e) in &failed {
        eprintln!("[vfold] skipped {}: {e}", p.display());
    }
    if frames.is_empty() {
        return Err("every image failed to decode".into());
    }
    eprintln!(
        "[vfold] decoded {} frames at {}px in {:.1}s ({} threads)",
        frames.len(),
        a.size,
        t.elapsed().as_secs_f32(),
        a.threads
    );

    // --- embed --------------------------------------------------------------
    let t = Instant::now();
    let mut embedder = embed::Embedder::new(&a.model, a.size, a.grid, a.threads)?;
    let mut descs: Vec<Vec<f32>> = Vec::with_capacity(frames.len());
    for (bi, chunk) in frames.chunks(a.batch).enumerate() {
        let batch: Vec<Vec<f32>> = chunk.iter().map(|f| f.tensor.clone()).collect();
        descs.extend(embedder.embed_batch(&batch)?);
        if bi % 8 == 0 {
            eprint!("\r[vfold] embedding {}/{}", descs.len(), frames.len());
        }
    }
    eprintln!(
        "\r[vfold] embedded {} frames -> {}-d ({}x{} grid x {}) in {:.1}s",
        descs.len(),
        embedder.width(),
        a.grid,
        a.grid,
        embed::DIM,
        t.elapsed().as_secs_f32()
    );

    // --- whiten and score ---------------------------------------------------
    let raw_spread = {
        let mut c = descs.clone();
        descriptor::l2_normalise(&mut c);
        descriptor::cosine_spread(&c)
    };
    descriptor::whiten(&mut descs, a.drop_pcs, a.sigma_floor);
    let wht_spread = descriptor::cosine_spread(&descs);

    // Spatial plausibility, if the capture carries positions. Appearance alone
    // cannot distinguish two frames 300 m apart over identical ground; position
    // can. Frames without GPS get neutral weight, so partial tagging degrades
    // smoothly rather than falling off a cliff.
    let kept_paths: Vec<PathBuf> = frames.iter().map(|f| f.path.clone()).collect();
    let spatial = if a.gps == "off" {
        None
    } else {
        let t = Instant::now();
        let pos = gps::positions(&kept_paths);
        let out = gps::spatial_weights(&pos, a.overlap, a.gps_scale);
        match &out {
            Some((_, info)) => eprintln!(
                "[vfold] gps         {}/{} tagged, site {:.0}x{:.0} m, co-visibility radius \
                 p05/median/p95 = {:.1}/{:.1}/{:.1} m ({:.1}x spread) in {:.1}s",
                info.tagged, info.total, info.extent.0, info.extent.1,
                info.scale_p05, info.median_scale, info.scale_p95,
                info.scale_p95 / info.scale_p05.max(1e-6),
                t.elapsed().as_secs_f32()
            ),
            None => eprintln!("[vfold] gps         no usable positions; appearance only"),
        }
        out
    };
    let spatial_w = spatial.as_ref().map(|(w, _)| w.as_slice());

    // Rectify so that 95% of frames retain at least `overlap` genuine
    // relationships — a guarantee about frames, so no region is starved of
    // coverage in preprocessing regardless of how oversampled its rivals are.
    let sim = descriptor::similarity(&descs, a.overlap, a.power, spatial_w);
    let sp = descriptor::spread(&sim);
    eprintln!(
        "[vfold] cosine      raw p05/p50/p95 = {:+.3}/{:+.3}/{:+.3}   whitened = {:+.3}/{:+.3}/{:+.3}",
        raw_spread.p05, raw_spread.p50, raw_spread.p95,
        wht_spread.p05, wht_spread.p50, wht_spread.p95
    );
    eprintln!(
        "[vfold] rectified   p50/p95 = {:.3}/{:.3}  mean nn1 = {:.3}  coverage-graph density = {:.1}%",
        sp.p50,
        sp.p95,
        sp.mean_nn1,
        100.0 * sp.nonzero_frac
    );

    // --- gate ---------------------------------------------------------------
    let gate = coverage::support_gate(&sim, a.min_neighbours);
    eprintln!(
        "[vfold] support gate need >={} neighbours (corpus median {}): kept {}, rejected {}",
        gate.min_neighbours,
        gate.median_neighbours,
        gate.eligible.len(),
        gate.rejected.len()
    );
    for &r in gate.rejected.iter().take(12) {
        eprintln!("[vfold]   rejected {}", frames[r].path.display());
    }
    if gate.rejected.len() > 12 {
        eprintln!("[vfold]   ... and {} more", gate.rejected.len() - 12);
    }
    if gate.eligible.len() <= a.k {
        return Err(format!(
            "gate left only {} eligible frames for k={}; lower --gate",
            gate.eligible.len(),
            a.k
        ));
    }

    // --- select -------------------------------------------------------------
    let nn1 = sim.nth_neighbour_sim(1);
    let mut med = nn1.clone();
    med.sort_unstable_by(|x, y| x.partial_cmp(y).unwrap());
    let median_nn1 = med[med.len() / 2];
    let t_cap = a.sat_mult * median_nn1;

    // Auto-calibrate the cap unless one was pinned. The right multiple of
    // median_nn1 moves with graph density, so a fixed one is a trap; the
    // saturation *fraction* is the quantity that actually tracks quality.
    let (t_cap, calibrated) = if a.sat_mult > 0.0 || a.objective != "saturated" {
        // The other objectives have no cap to calibrate; skip the probes rather
        // than logging a number that does not affect the result.
        (t_cap, None)
    } else {
        let (t, frac) =
            coverage::calibrate_cap(&sim, a.k, &gate.eligible, a.target_sat, median_nn1);
        eprintln!(
            "[vfold] calibrate   cap {:.3} ({:.2}x median nn1) -> {:.0}% saturated (target {:.0}%)",
            t,
            t / median_nn1.max(1e-6),
            frac * 100.0,
            a.target_sat * 100.0
        );
        (t, Some(frac))
    };
    let _ = calibrated;

    let objective = match a.objective.as_str() {
        "saturated" => Objective::Saturated { t: t_cap },
        "facility" => Objective::FacilityLocation,
        "proportional" => Objective::Proportional { alpha: 0.1 },
        other => return Err(format!("unknown objective '{other}'")),
    };
    eprintln!(
        "[vfold] objective {:?}  (median nn1 = {median_nn1:.3})",
        objective
    );

    let t = Instant::now();
    let sel = coverage::select(&sim, objective, a.k, &gate.eligible);
    eprintln!(
        "[vfold] selected {} of {} in {:.2}s  ({} gain evaluations vs {} naive, saturated {:.0}%)",
        sel.picked.len(),
        frames.len(),
        t.elapsed().as_secs_f32(),
        sel.evaluations,
        gate.eligible.len() * a.k,
        sel.saturated_frac * 100.0
    );

    // --- spatial audit ------------------------------------------------------
    // The honest test of a geometric claim is a geometric measurement. For every
    // frame, how far is the nearest selected frame? Appearance-only selection on
    // repetitive ground scores no better than random here even when its feature-
    // space coverage looks excellent, so this is the number that matters.
    // Measured whenever positions exist, independent of whether GPS drove the
    // selection — otherwise the two modes cannot be compared, which was the
    // whole point of measuring.
    let spatial_audit = gps::coverage_distances(&gps::positions(&kept_paths), &sel.picked)
        .map(|d| {
            eprintln!(
                "[vfold] spatial     nearest selected frame: median {:.1} m, p95 {:.1} m, \
                 worst {:.1} m   [random: {:.1} / {:.1}]  {}",
                d.median, d.p95, d.max, d.random_median, d.random_max,
                if d.max <= d.random_max * 0.85 {
                    "clearly beats chance"
                } else if d.max <= d.random_max {
                    "marginally beats chance"
                } else {
                    "NO BETTER THAN RANDOM - appearance aliasing likely"
                }
            );
            d
        });

    // --- emit ---------------------------------------------------------------
    let lines: Vec<String> =
        sel.picked.iter().map(|&i| frames[i].path.display().to_string()).collect();
    println!("{}", lines.join("\n"));
    if let Some(p) = &a.out {
        std::fs::write(p, format!("{}\n", lines.join("\n")))
            .map_err(|e| format!("{}: {e}", p.display()))?;
        eprintln!("[vfold] wrote {}", p.display());
    }
    if let Some(p) = &a.json {
        write_json(p, &a, &frames, &sim, &gate, &sel, &sp, spatial_audit.as_ref())?;
        eprintln!("[vfold] wrote {}", p.display());
    }
    eprintln!("[vfold] total {:.1}s", t0.elapsed().as_secs_f32());
    Ok(())
}

fn esc(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

fn write_json(
    path: &Path,
    a: &Args,
    frames: &[decode::Frame],
    sim: &Sim,
    gate: &coverage::Gate,
    sel: &coverage::Selection,
    sp: &descriptor::Spread,
    spatial: Option<&gps::CoverageDistances>,
) -> Result<(), String> {
    let mut s = String::new();
    s.push_str("{\n");
    s.push_str(&format!("  \"dir\": \"{}\",\n", esc(&a.dir.display().to_string())));
    s.push_str(&format!("  \"n\": {},\n  \"k\": {},\n", frames.len(), sel.picked.len()));
    s.push_str(&format!(
        "  \"params\": {{\"size\": {}, \"grid\": {}, \"objective\": \"{}\", \"sat_mult\": {}, \
         \"overlap\": {}, \"min_neighbours\": {}, \"gps\": \"{}\", \"gps_scale\": {}, \
         \"power\": {}, \"drop_pcs\": {}, \"sigma_floor\": {}}},\n",
        a.size, a.grid, a.objective, a.sat_mult, a.overlap, a.min_neighbours,
        a.gps, a.gps_scale, a.power, a.drop_pcs, a.sigma_floor
    ));
    s.push_str(&format!(
        "  \"similarity\": {{\"p05\": {:.4}, \"p50\": {:.4}, \"p95\": {:.4}, \"mean_nn1\": {:.4}}},\n",
        sp.p05, sp.p50, sp.p95, sp.mean_nn1
    ));
    s.push_str(&format!(
        "  \"gate\": {{\"min_neighbours\": {}, \"median_neighbours\": {}, \"kept\": {}, \"rejected\": {}}},\n",
        gate.min_neighbours,
        gate.median_neighbours,
        gate.eligible.len(),
        gate.rejected.len()
    ));
    s.push_str(&format!("  \"saturated_frac\": {:.4},\n", sel.saturated_frac));
    if let Some(d) = spatial {
        s.push_str(&format!(
            "  \"spatial_m\": {{\"median\": {:.2}, \"p95\": {:.2}, \"max\": {:.2}, \
             \"random_median\": {:.2}, \"random_max\": {:.2}}},\n",
            d.median, d.p95, d.max, d.random_median, d.random_max
        ));
    }

    s.push_str("  \"curve\": [");
    s.push_str(&sel.curve.iter().map(|v| format!("{v:.3}")).collect::<Vec<_>>().join(", "));
    s.push_str("],\n");

    s.push_str("  \"rejected\": [\n");
    let rej: Vec<String> = gate
        .rejected
        .iter()
        .map(|&i| format!("    \"{}\"", esc(&frames[i].path.display().to_string())))
        .collect();
    s.push_str(&rej.join(",\n"));
    s.push_str("\n  ],\n");

    s.push_str("  \"selected\": [\n");
    let nn1 = sim.nth_neighbour_sim(1);
    let rows: Vec<String> = sel
        .picked
        .iter()
        .enumerate()
        .map(|(rank, &i)| {
            format!(
                "    {{\"rank\": {rank}, \"index\": {i}, \"path\": \"{}\", \"focus\": {:.1}, \"nn1\": {:.4}}}",
                esc(&frames[i].path.display().to_string()),
                frames[i].focus,
                nn1[i]
            )
        })
        .collect();
    s.push_str(&rows.join(",\n"));
    s.push_str("\n  ]\n}\n");

    std::fs::write(path, s).map_err(|e| format!("{}: {e}", path.display()))
}
