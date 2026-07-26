//! Video input: a quality gate in front of the coverage funnel.
//!
//! A video is a firehose of near-duplicate frames, and the duplicates are not
//! equal: motion blur, focus hunting and codec smear vary frame to frame. The
//! coverage objective downstream is nearly blind to this — a smeared frame
//! embeds close to its sharp neighbour — so quality is decided here, before
//! embedding, where frames from the same half-second are genuine substitutes.
//!
//! The gate: decode every frame at scoring resolution through an ffmpeg pipe,
//! score sharpness as Laplacian variance (the same measure, in the same units,
//! as the `focus` diagnostic on photographs), and keep the sharpest frame per
//! fixed time window. The argmax is per-window rather than a global threshold
//! on purpose: absolute sharpness tracks scene texture as much as focus, so a
//! global cut would delete whole low-contrast regions. A window whose best
//! frame is still junk is not our problem — the support gate downstream
//! rejects frames with no genuine neighbours, whatever the reason.
//!
//! Winners are re-extracted from the source at full resolution into an images
//! directory the unchanged photo pipeline consumes, named
//! `<clip>_f<frame>_t<seconds>s.jpg` so any selected frame is traceable to its
//! clip and moment by eye; `frames_index.tsv` maps every candidate to clip,
//! frame number, timestamp and sharpness.
//!
//! ffmpeg does the decoding. Shipping a video stack inside vfold would triple
//! the binary for a worse decoder; the dependency is honest — if ffmpeg is
//! missing the error says so, and nothing else in vfold changes.

use rayon::prelude::*;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

const VIDEO_EXTS: [&str; 5] = ["mp4", "mov", "m4v", "avi", "mkv"];

fn is_video(p: &Path) -> bool {
    p.extension()
        .and_then(|e| e.to_str())
        .map(|e| VIDEO_EXTS.contains(&e.to_ascii_lowercase().as_str()))
        .unwrap_or(false)
}

/// Decide whether the input wants the video stage.
///
/// A single video file: yes. A directory of videos with no images: yes. A
/// directory with any images: no — the images are the capture, and silently
/// mixing extracted video frames into a photo set would change what `-k`
/// means, so the videos are noted and ignored.
pub fn detect(input: &Path) -> Result<Option<Vec<PathBuf>>, String> {
    if input.is_file() {
        return Ok(if is_video(input) { Some(vec![input.to_path_buf()]) } else { None });
    }
    let entries: Vec<PathBuf> = std::fs::read_dir(input)
        .map_err(|e| format!("{}: {e}", input.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_file())
        .collect();
    let mut videos: Vec<PathBuf> = entries.iter().filter(|p| is_video(p)).cloned().collect();
    if videos.is_empty() {
        return Ok(None);
    }
    let has_images = entries.iter().any(|p| {
        matches!(
            p.extension().and_then(|e| e.to_str()).map(|e| e.to_ascii_lowercase()).as_deref(),
            Some("jpg" | "jpeg" | "png" | "tif" | "tiff")
        )
    });
    if has_images {
        eprintln!(
            "[vfold] {} video file(s) ignored: directory also holds images, which win. \
             Point vfold at the videos alone to use them.",
            videos.len()
        );
        return Ok(None);
    }
    videos.sort();
    Ok(Some(videos))
}

/// `<dir>\vfold_frames`, or next to a single video file.
pub fn default_frames_dir(input: &Path) -> PathBuf {
    if input.is_file() {
        input.parent().unwrap_or(Path::new(".")).join("vfold_frames")
    } else {
        input.join("vfold_frames")
    }
}

struct Probe {
    w: u32,
    h: u32,
    fps: f64,
    /// r_frame_rate exactly as ffprobe printed it ("30000/1001") — passed back
    /// to ffmpeg's -r untouched, so the scoring timeline and the seek
    /// arithmetic share one definition of the rate with no float re-rounding.
    rate_raw: String,
    avg_fps: f64,
}

fn ffmpeg_missing(tool: &str, e: &std::io::Error) -> String {
    if e.kind() == std::io::ErrorKind::NotFound {
        format!(
            "{tool} not found on PATH — video input needs it. \
             Install: winget install Gyan.FFmpeg (or grab a static build) and reopen the shell."
        )
    } else {
        format!("{tool}: {e}")
    }
}

fn probe(video: &Path) -> Result<Probe, String> {
    let out = Command::new("ffprobe")
        .args([
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate,avg_frame_rate",
            "-of", "csv=p=0",
        ])
        .arg(video)
        .output()
        .map_err(|e| ffmpeg_missing("ffprobe", &e))?;
    if !out.status.success() {
        return Err(format!(
            "ffprobe failed on {}: {}",
            video.display(),
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let line = text.trim().lines().next().unwrap_or("");
    let f: Vec<&str> = line.trim_end_matches(',').split(',').collect();
    if f.len() < 4 {
        return Err(format!("ffprobe gave '{line}' for {}", video.display()));
    }
    let rate = |s: &str| -> Result<f64, String> {
        match s.split_once('/') {
            Some((n, d)) => {
                let n: f64 = n.parse().map_err(|_| format!("bad rate '{s}'"))?;
                let d: f64 = d.parse().map_err(|_| format!("bad rate '{s}'"))?;
                Ok(if d == 0.0 { 0.0 } else { n / d })
            }
            None => s.parse().map_err(|_| format!("bad rate '{s}'")),
        }
    };
    let w: u32 = f[0].parse().map_err(|_| format!("bad width '{}'", f[0]))?;
    let h: u32 = f[1].parse().map_err(|_| format!("bad height '{}'", f[1]))?;
    let fps = rate(f[2])?;
    let avg_fps = rate(f[3]).unwrap_or(0.0);
    if fps <= 0.0 || fps > 1000.0 {
        return Err(format!("implausible frame rate {fps} for {}", video.display()));
    }
    Ok(Probe { w, h, fps, rate_raw: f[2].to_string(), avg_fps })
}

/// Read exactly `buf.len()` bytes, or return Ok(false) on clean EOF at a frame
/// boundary. EOF mid-frame is an error: it means the pipe died, and a silently
/// short scoring pass would misalign every frame index after it.
fn read_frame(r: &mut impl Read, buf: &mut [u8]) -> Result<bool, String> {
    let mut got = 0;
    while got < buf.len() {
        match r.read(&mut buf[got..]) {
            Ok(0) if got == 0 => return Ok(false),
            Ok(0) => return Err(format!("truncated frame: {got} of {} bytes", buf.len())),
            Ok(n) => got += n,
            Err(e) => return Err(e.to_string()),
        }
    }
    Ok(true)
}

/// Laplacian-variance per frame, decoded at `score_w` px wide via ffmpeg.
fn score_frames(video: &Path, pr: &Probe, score_w: u32) -> Result<Vec<f32>, String> {
    let sw = score_w.min(pr.w).max(16);
    let sh = ((pr.h as u64 * sw as u64 / pr.w as u64) as u32 / 2 * 2).max(16);
    let mut child = Command::new("ffmpeg")
        .args(["-v", "error", "-i"])
        .arg(video)
        .args([
            "-map", "0:v:0",
            "-vf", &format!("scale={sw}:{sh}"),
            "-pix_fmt", "gray",
            // Pin the scoring timeline to CFR at r_frame_rate rather than rely
            // on the rawvideo muxer's default doing the same: frame n of this
            // pipe is the frame at wall time n/fps, which is exactly what the
            // extraction seek assumes. On VFR input ffmpeg fills the timeline
            // by dup/drop; probe() warns about the ±1-frame attribution slack.
            "-fps_mode", "cfr",
            "-r", &pr.rate_raw,
            "-f", "rawvideo",
            "-",
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .spawn()
        .map_err(|e| ffmpeg_missing("ffmpeg", &e))?;

    let mut stdout = child.stdout.take().ok_or("no ffmpeg stdout")?;
    let mut buf = vec![0u8; (sw * sh) as usize];
    let mut grey = vec![0f32; buf.len()];
    let mut scores = Vec::new();
    loop {
        match read_frame(&mut stdout, &mut buf) {
            Ok(false) => break,
            Ok(true) => {
                for (g, b) in grey.iter_mut().zip(&buf) {
                    *g = *b as f32;
                }
                scores.push(crate::decode::laplacian_variance(&grey, sw as usize));
            }
            Err(e) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(format!("{}: {e}", video.display()));
            }
        }
    }
    let status = child.wait().map_err(|e| e.to_string())?;
    if !status.success() {
        return Err(format!("ffmpeg failed decoding {}", video.display()));
    }
    Ok(scores)
}

struct Winner {
    video: PathBuf,
    frame: usize,
    fps: f64,
    time_s: f64,
    sharpness: f32,
    jpg: PathBuf,
}

/// Run the full gate: score, pick winners per window, extract, write manifest.
/// Returns the number of candidate frames present on disk afterwards.
pub fn extract(
    videos: &[PathBuf],
    frames_dir: &Path,
    score_w: u32,
    cands_per_sec: f32,
) -> Result<usize, String> {
    std::fs::create_dir_all(frames_dir)
        .map_err(|e| format!("{}: {e}", frames_dir.display()))?;

    // Extracted names are built from the clip stem, so two clips sharing a
    // stem (DJI_0001.MP4 beside DJI_0001.MOV; NTFS also collapses case) would
    // silently write to the same paths and misattribute frames. Refuse loudly.
    let mut stems: std::collections::HashMap<String, &PathBuf> = std::collections::HashMap::new();
    for v in videos {
        let stem =
            v.file_stem().and_then(|s| s.to_str()).unwrap_or("clip").to_ascii_lowercase();
        if let Some(prev) = stems.insert(stem.clone(), v) {
            return Err(format!(
                "{} and {} share the clip stem '{}': their extracted frames would \
                 overwrite each other. Rename one, or fold them separately.",
                prev.display(),
                v.display(),
                stem
            ));
        }
    }

    let t0 = std::time::Instant::now();
    let mut winners: Vec<Winner> = Vec::new();
    for video in videos {
        let pr = probe(video)?;
        if pr.avg_fps > 0.0 && (pr.fps - pr.avg_fps).abs() / pr.fps > 0.005 {
            eprintln!(
                "[vfold] video       WARNING {}: variable frame rate (r={:.2} vs avg={:.2} fps). \
                 The gate scores a CFR-resampled timeline; extracted winners can sit one frame \
                 off the scored frame near timing gaps.",
                video.display(),
                pr.fps,
                pr.avg_fps
            );
        }
        let t = std::time::Instant::now();
        let scores = score_frames(video, &pr, score_w)?;
        if scores.is_empty() {
            eprintln!("[vfold] video       {}: decoded 0 frames, skipping", video.display());
            continue;
        }
        let win = ((pr.fps / cands_per_sec.max(0.01) as f64).round() as usize).max(1);
        let stem = video
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("clip")
            .to_string();
        let before = winners.len();
        for (w_idx, chunk) in scores.chunks(win).enumerate() {
            let (best_off, best) = chunk
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap();
            let n = w_idx * win + best_off;
            let time_s = n as f64 / pr.fps;
            winners.push(Winner {
                video: video.clone(),
                frame: n,
                fps: pr.fps,
                time_s,
                sharpness: *best,
                jpg: frames_dir.join(format!("{stem}_f{n:06}_t{time_s:08.2}s.jpg")),
            });
        }
        eprintln!(
            "[vfold] video       {}: {} frames @ {:.2} fps {}x{} -> {} candidates \
             (sharpest per {} frames) in {:.1}s",
            stem,
            scores.len(),
            pr.fps,
            pr.w,
            pr.h,
            winners.len() - before,
            win,
            t.elapsed().as_secs_f32()
        );
    }
    if winners.is_empty() {
        return Err("no decodable video frames found".into());
    }

    // Full-resolution extraction, parallel across winners. Seeking to
    // (n - 0.5)/fps lands on frame n exactly under accurate seek: the first
    // packet with pts >= target is n, and the half-frame back-off absorbs
    // float rounding in either direction. One decode thread per ffmpeg — the
    // parallelism is across processes, and each only decodes forward from the
    // keyframe before its target.
    // A file only counts as extracted if it is non-empty AND was completed:
    // ffmpeg writes to a .part name and the rename below is the commit point,
    // so an interrupted or failed run can never leave a truncated jpg that a
    // later run would trust as cached. (.part leftovers are invisible to
    // list_images and get overwritten on retry.)
    let present = |p: &Path| std::fs::metadata(p).map(|m| m.len() > 0).unwrap_or(false);
    let t = std::time::Instant::now();
    let cached: usize = winners.iter().filter(|w| present(&w.jpg)).count();
    let results: Vec<Result<(), String>> = winners
        .par_iter()
        .map(|w| {
            if present(&w.jpg) {
                return Ok(());
            }
            let part = w.jpg.with_extension("part");
            let seek = ((w.frame as f64 - 0.5) / w.fps).max(0.0);
            let out = Command::new("ffmpeg")
                .args(["-v", "error", "-y", "-ss", &format!("{seek:.6}"), "-i"])
                .arg(&w.video)
                .args([
                    "-map", "0:v:0",
                    "-frames:v", "1",
                    "-q:v", "2",
                    "-threads", "1",
                    // .part hides the jpg extension, so name the muxer.
                    "-f", "image2",
                    "-update", "1",
                ])
                .arg(&part)
                .stdin(Stdio::null())
                .output()
                .map_err(|e| ffmpeg_missing("ffmpeg", &e))?;
            if !out.status.success() {
                let _ = std::fs::remove_file(&part);
                return Err(format!(
                    "extract {} frame {}: {}",
                    w.video.display(),
                    w.frame,
                    String::from_utf8_lossy(&out.stderr).trim()
                ));
            }
            if !present(&part) {
                let _ = std::fs::remove_file(&part);
                return Err(format!(
                    "extract {} frame {}: ffmpeg wrote nothing (seek past end?)",
                    w.video.display(),
                    w.frame
                ));
            }
            std::fs::rename(&part, &w.jpg)
                .map_err(|e| format!("{} -> {}: {e}", part.display(), w.jpg.display()))
        })
        .collect();

    let mut failed = 0;
    for r in &results {
        if let Err(e) = r {
            eprintln!("[vfold] video       {e}");
            failed += 1;
        }
    }
    let present = winners.len() - failed;
    if present == 0 {
        return Err("every frame extraction failed".into());
    }

    // Manifest: every candidate, whether or not selection later keeps it.
    let mut tsv = String::from("file\tvideo\tframe\ttime_s\tsharpness\n");
    for w in &winners {
        tsv.push_str(&format!(
            "{}\t{}\t{}\t{:.3}\t{:.1}\n",
            w.jpg.file_name().and_then(|s| s.to_str()).unwrap_or("?"),
            w.video.file_name().and_then(|s| s.to_str()).unwrap_or("?"),
            w.frame,
            w.time_s,
            w.sharpness
        ));
    }
    let manifest = frames_dir.join("frames_index.tsv");
    std::fs::write(&manifest, tsv).map_err(|e| format!("{}: {e}", manifest.display()))?;

    eprintln!(
        "[vfold] video       extracted {} full-res frames ({} cached) -> {} in {:.1}s; \
         manifest frames_index.tsv; video stage total {:.1}s",
        present.saturating_sub(cached),
        cached,
        frames_dir.display(),
        t.elapsed().as_secs_f32(),
        t0.elapsed().as_secs_f32()
    );
    Ok(present)
}
