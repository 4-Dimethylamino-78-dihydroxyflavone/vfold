//! Getting 4.5 GB of 6000x4000 JPEG into small normalised tensors, quickly.
//!
//! The whole cost of this stage is JPEG decode, and almost all of that is
//! avoidable. JPEG stores 8x8 blocks of DCT coefficients; reconstructing from
//! only the low-frequency corner of each block yields the image directly at 1/2,
//! 1/4 or 1/8 scale without ever materialising full resolution. For a 6000x4000
//! frame targeted at 448 px, 1/8 scale gives 750x500 — still well above target,
//! for roughly 1/64th of the decode work.
//!
//! The subsequent resample is then operating on a 750x500 image rather than a
//! 24-megapixel one, so a good filter costs nothing.

use image::imageops::FilterType;
use image::{DynamicImage, RgbImage};
use rayon::prelude::*;
use std::path::{Path, PathBuf};

/// ImageNet statistics, taken from the model's own `preprocessor_config.json`
/// rather than assumed. Using the wrong constants shifts every descriptor by a
/// fixed vector, which survives whitening but distorts angles.
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

pub struct Frame {
    pub path: PathBuf,
    /// CHW, f32, normalised. Length `3 * size * size`.
    pub tensor: Vec<f32>,
    /// Variance of the Laplacian on the greyscale image — a standard focus
    /// measure. Low means blurred or featureless. Reported as a diagnostic; the
    /// similarity-based support gate is the primary junk filter, because blur is
    /// only one of several ways a frame can be useless.
    pub focus: f32,
}

/// Decode one image to `size x size`, normalised CHW.
///
/// The model's preprocessor resizes to a square without cropping
/// (`default_to_square: true`, `do_center_crop: null`), so a 3:2 aerial frame is
/// squashed rather than cut. That is deliberate: cropping would discard the
/// left and right thirds of every nadir frame, which is exactly the content that
/// distinguishes overlapping strips from one another.
pub fn load(path: &Path, size: u32) -> Result<Frame, String> {
    let rgb = decode_downscaled(path, size)?;
    let resized = DynamicImage::ImageRgb8(rgb).resize_exact(size, size, FilterType::Lanczos3);
    let rgb = resized.to_rgb8();

    let n = (size * size) as usize;
    let mut tensor = vec![0.0f32; 3 * n];
    let mut grey = vec![0.0f32; n];
    for (i, px) in rgb.pixels().enumerate() {
        for c in 0..3 {
            tensor[c * n + i] = (px[c] as f32 / 255.0 - MEAN[c]) / STD[c];
        }
        grey[i] = 0.299 * px[0] as f32 + 0.587 * px[1] as f32 + 0.114 * px[2] as f32;
    }

    Ok(Frame { path: path.to_path_buf(), tensor, focus: laplacian_variance(&grey, size as usize) })
}

/// Decode at the smallest power-of-two reduction that still leaves at least
/// `target` pixels on the short edge.
fn decode_downscaled(path: &Path, target: u32) -> Result<RgbImage, String> {
    let is_jpeg = matches!(
        path.extension().and_then(|e| e.to_str()).map(|e| e.to_ascii_lowercase()).as_deref(),
        Some("jpg") | Some("jpeg")
    );

    if is_jpeg {
        if let Ok(img) = decode_jpeg_scaled(path, target) {
            return Ok(img);
        }
        // Fall through to the general decoder on any JPEG the fast path rejects
        // (progressive, unusual subsampling, truncated). Correctness beats speed.
    }

    image::open(path)
        .map_err(|e| format!("{}: {e}", path.display()))
        .map(|i| i.to_rgb8())
}

fn decode_jpeg_scaled(path: &Path, target: u32) -> Result<RgbImage, String> {
    use jpeg_decoder::{Decoder, PixelFormat};
    use std::fs::File;
    use std::io::BufReader;

    let file = File::open(path).map_err(|e| e.to_string())?;
    let mut dec = Decoder::new(BufReader::new(file));
    dec.read_info().map_err(|e| e.to_string())?;
    let info = dec.info().ok_or("no jpeg info")?;

    // Ask for the largest reduction whose short edge still covers the target.
    // scale() snaps to the nearest supported 1/1, 1/2, 1/4 or 1/8.
    let short = info.width.min(info.height) as u32;
    let mut denom = 1u32;
    while denom < 8 && short / (denom * 2) >= target {
        denom *= 2;
    }
    let (rw, rh) = (info.width as u32 / denom, info.height as u32 / denom);
    let (aw, ah) = dec.scale(rw as u16, rh as u16).map_err(|e| e.to_string())?;

    let pixels = dec.decode().map_err(|e| e.to_string())?;
    let (aw, ah) = (aw as u32, ah as u32);

    let rgb = match dec.info().ok_or("no jpeg info")?.pixel_format {
        PixelFormat::RGB24 => RgbImage::from_raw(aw, ah, pixels).ok_or("bad rgb buffer")?,
        PixelFormat::L8 => {
            let mut out = Vec::with_capacity(pixels.len() * 3);
            for g in &pixels {
                out.extend_from_slice(&[*g, *g, *g]);
            }
            RgbImage::from_raw(aw, ah, out).ok_or("bad grey buffer")?
        }
        other => return Err(format!("unsupported jpeg pixel format {other:?}")),
    };
    Ok(rgb)
}

/// Variance of the 4-neighbour Laplacian. Standard focus measure.
fn laplacian_variance(grey: &[f32], w: usize) -> f32 {
    let h = grey.len() / w;
    if w < 3 || h < 3 {
        return 0.0;
    }
    let mut vals = Vec::with_capacity((w - 2) * (h - 2));
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let i = y * w + x;
            vals.push(
                grey[i - 1] + grey[i + 1] + grey[i - w] + grey[i + w] - 4.0 * grey[i],
            );
        }
    }
    let mean = vals.iter().sum::<f32>() / vals.len() as f32;
    vals.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / vals.len() as f32
}

/// Decode a whole directory in parallel, preserving path order.
///
/// `par_iter().collect()` preserves input order regardless of completion order,
/// so the returned indices line up with `paths` and stay stable across runs —
/// which matters because the output is a reproducible frame list.
pub fn load_all(paths: &[PathBuf], size: u32) -> (Vec<Frame>, Vec<(PathBuf, String)>) {
    let results: Vec<Result<Frame, (PathBuf, String)>> = paths
        .par_iter()
        .map(|p| load(p, size).map_err(|e| (p.clone(), e)))
        .collect();

    let mut frames = Vec::with_capacity(results.len());
    let mut failed = Vec::new();
    for r in results {
        match r {
            Ok(f) => frames.push(f),
            Err(e) => failed.push(e),
        }
    }
    (frames, failed)
}

/// Every readable image in a directory, sorted for reproducibility.
pub fn list_images(dir: &Path) -> Result<Vec<PathBuf>, String> {
    let mut out: Vec<PathBuf> = std::fs::read_dir(dir)
        .map_err(|e| format!("{}: {e}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.is_file()
                && matches!(
                    p.extension().and_then(|e| e.to_str()).map(|e| e.to_ascii_lowercase()).as_deref(),
                    Some("jpg" | "jpeg" | "png" | "tif" | "tiff")
                )
        })
        .collect();
    out.sort();
    Ok(out)
}
