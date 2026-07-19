//! DINOv3 patch tokens, pooled into a spatial grid.
//!
//! The descriptor design is the whole point of this tool, so it is worth being
//! explicit about what is being avoided.
//!
//! The obvious image descriptor is the mean over all patch tokens. It is also
//! the one that fails. Measured on this project's own data (`benchmark_v15_hub.md`):
//! mean-pooled DVLT tokens gave a power-normalised cosine of 0.982 between
//! adjacent frames and 0.789 between distant ones, and every dedup threshold from
//! 0.90 to 0.99 selected exactly the min-keep guard rather than anything about the
//! images. A global mean answers "which scene is this?", and every frame in a
//! capture has the same answer.
//!
//! Pooling into a `G x G` grid instead keeps coarse spatial layout. Two nadir
//! frames with 80% forward overlap have near-identical global means but different
//! corners, and the corners are what distinguishes them. At 448 px the patch grid
//! is 28x28, so a 4x4 pooling averages 49 tokens per region — enough to be stable,
//! local enough to be informative.
//!
//! Token layout matters and is easy to get wrong. DINOv3 emits
//! `[CLS] [reg x num_register_tokens] [patch x P]`. This model's `config.json`
//! reports `num_register_tokens: 4`, so patch tokens begin at index **5**, not 1.
//! Slicing at 1 — the DINOv2 habit, since DINOv2 has no registers — silently mixes
//! four register tokens into the first spatial region and corrupts every
//! similarity downstream without any error.

use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::execution_providers::{CUDAExecutionProvider, DirectMLExecutionProvider};
use ort::value::Tensor;
use std::path::Path;

/// `[CLS]` plus the register tokens, from `config.json: num_register_tokens = 4`.
const PREFIX_TOKENS: usize = 5;
/// `config.json: hidden_size`.
pub const DIM: usize = 384;
/// `config.json: patch_size`.
const PATCH: usize = 16;

/// Where the forward pass actually runs. Not a request — the achieved outcome.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    Cuda,
    DirectMl,
    Cpu,
}

impl Device {
    /// Sensible frames-per-call when the user leaves `--batch` on auto.
    ///
    /// GPU wants larger batches to amortise launch overhead; 16 at 448 px keeps
    /// peak attention transients within a 6 GB card's budget, and the OOM
    /// halving in the embed loop absorbs anything tighter. CPU gains nothing
    /// past the point where GEMMs saturate the cores.
    pub fn default_batch(self) -> usize {
        match self {
            Device::Cuda | Device::DirectMl => 16,
            Device::Cpu => 8,
        }
    }
}

impl std::fmt::Display for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Device::Cuda => "CUDA",
            Device::DirectMl => "DirectML",
            Device::Cpu => "CPU",
        })
    }
}

pub struct Embedder {
    session: Session,
    /// Model input edge in pixels. Must be a multiple of `PATCH`.
    size: usize,
    /// Pooling grid edge, e.g. 4 for a 4x4 layout.
    grid: usize,
    pub device: Device,
}

/// One session-build attempt against a specific device.
///
/// `error_on_failure` is the load-bearing detail. ort's default execution-
/// provider registration *silently* falls back to CPU when a provider cannot
/// initialise (missing cuDNN, no device), which would leave us reporting
/// "CUDA" while quietly running 20x slower. Hard-failing each rung makes the
/// ladder honest: whatever session survives is genuinely running where we say
/// it is.
fn try_session(model: &Path, threads: usize, dev: Device) -> Result<Session, String> {
    // Inline closures on purpose: `ort::Error<T>` is generic over the builder
    // state it hands back, so a single shared closure cannot serve every call.
    let mut b = Session::builder()
        .map_err(|e| e.to_string())?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| e.to_string())?
        .with_intra_threads(threads)
        .map_err(|e| e.to_string())?;
    b = match dev {
        Device::Cuda => b
            .with_execution_providers([CUDAExecutionProvider::default()
                .build()
                .error_on_failure()])
            .map_err(|e| e.to_string())?,
        Device::DirectMl => b
            .with_execution_providers([DirectMLExecutionProvider::default()
                .build()
                .error_on_failure()])
            .map_err(|e| e.to_string())?,
        Device::Cpu => b,
    };
    b.commit_from_file(model)
        .map_err(|e| format!("loading {}: {e}", model.display()))
}

/// Rungs strictly below a device, for runtime demotion.
///
/// Session-init success does not prove a provider can run this graph, so the
/// fallback ladder must also exist at embed time — otherwise "capable of going
/// without" is only true for the failures that announce themselves early. This
/// only helps with failures that come back as errors; see `new` for why the
/// rung that can fail by access violation is excluded from auto entirely.
pub fn ladder_below(dev: Device) -> &'static [Device] {
    match dev {
        Device::Cuda | Device::DirectMl => &[Device::Cpu],
        Device::Cpu => &[],
    }
}

impl Embedder {
    /// Build a session on the fastest device that actually works.
    ///
    /// `device_pref`: `auto` walks CUDA -> DirectML -> CPU and takes the first
    /// rung that initialises; naming a device pins it and fails loudly rather
    /// than substituting. CUDA needs the CUDA 12 runtime plus cuDNN 9 on the
    /// machine; DirectML needs only a DX12-capable GPU (any vendor, ships with
    /// Windows); CPU always works.
    pub fn new(
        model: &Path,
        size: usize,
        grid: usize,
        threads: usize,
        device_pref: &str,
    ) -> Result<Self, String> {
        // DirectML is deliberately NOT in the auto ladder. Measured on this
        // model's export: the DML provider passes session init, then fails a
        // dynamic-shape Reshape on the first run() — and the failure is
        // nondeterministic between a returned error and a native access
        // violation. An AV cannot be caught from Rust, so no in-process
        // demotion can recover from it. A rung that sometimes kills the
        // process may be *chosen* (--device dml, e.g. for a future export
        // that works), but must never be *defaulted*.
        let ladder: &[Device] = match device_pref {
            "auto" => &[Device::Cuda, Device::Cpu],
            "cuda" => &[Device::Cuda],
            "dml" | "directml" => &[Device::DirectMl],
            "cpu" => &[Device::Cpu],
            other => return Err(format!("unknown --device '{other}' (auto|cuda|dml|cpu)")),
        };
        Self::new_on(model, size, grid, threads, ladder)
    }

    /// Build on the first rung of an explicit ladder that initialises. Used both
    /// for the initial build and for runtime demotion after a rung that
    /// initialised but cannot actually run the graph.
    pub fn new_on(
        model: &Path,
        size: usize,
        grid: usize,
        threads: usize,
        ladder: &[Device],
    ) -> Result<Self, String> {
        if !size.is_multiple_of(PATCH) {
            return Err(format!("input size {size} is not a multiple of patch size {PATCH}"));
        }
        let tokens = size / PATCH;
        if !tokens.is_multiple_of(grid) {
            return Err(format!(
                "patch grid {tokens}x{tokens} does not divide evenly into a {grid}x{grid} pooling grid"
            ));
        }

        let mut last_err = String::new();
        for &dev in ladder {
            match try_session(model, threads, dev) {
                Ok(session) => {
                    if dev != Device::Cuda && ladder.len() > 1 {
                        eprintln!("[vfold] device      {dev} (higher rungs unavailable: {last_err})");
                    }
                    return Ok(Self { session, size, grid, device: dev });
                }
                Err(e) => {
                    // Keep only the first line; ORT errors trail C++ context.
                    last_err = e.lines().next().unwrap_or("").to_string();
                }
            }
        }
        Err(format!("no execution provider initialised: {last_err}"))
    }

    /// Descriptor width: one `DIM`-vector per pooled region.
    pub fn width(&self) -> usize {
        self.grid * self.grid * DIM
    }

    /// Embed a batch of CHW tensors, returning one row per frame.
    ///
    /// Batching matters: per-call overhead dominates at batch 1, but the
    /// activation memory for a ViT scales with `batch * tokens^2` in attention,
    /// so at 448 px (784 tokens) the batch must stay modest.
    pub fn embed_batch(&mut self, frames: &[Vec<f32>]) -> Result<Vec<Vec<f32>>, String> {
        let b = frames.len();
        if b == 0 {
            return Ok(Vec::new());
        }
        let per = 3 * self.size * self.size;
        let mut flat = Vec::with_capacity(b * per);
        for f in frames {
            if f.len() != per {
                return Err(format!("frame tensor is {} floats, expected {per}", f.len()));
            }
            flat.extend_from_slice(f);
        }

        let input = Tensor::from_array((
            [b as i64, 3, self.size as i64, self.size as i64],
            flat,
        ))
        .map_err(|e| e.to_string())?;

        let outputs = self
            .session
            .run(ort::inputs!["pixel_values" => input])
            .map_err(|e| e.to_string())?;

        let (shape, data) = outputs["last_hidden_state"]
            .try_extract_tensor::<f32>()
            .map_err(|e| e.to_string())?;

        // [B, 1 + registers + P, DIM]
        let seq = shape[1] as usize;
        let dim = shape[2] as usize;
        if dim != DIM {
            return Err(format!("model hidden size is {dim}, expected {DIM}"));
        }
        let tokens = self.size / PATCH;
        let expected = PREFIX_TOKENS + tokens * tokens;
        if seq != expected {
            return Err(format!(
                "sequence length {seq} != {PREFIX_TOKENS} prefix + {tokens}x{tokens} patches = {expected}. \
                 Either the input resolution was not honoured, or num_register_tokens is not {}.",
                PREFIX_TOKENS - 1
            ));
        }

        // `outputs` is a view into ORT-owned arena memory and therefore holds a
        // borrow of the session for as long as it lives. Capturing `self` in the
        // closure below would be a second, conflicting borrow — so read the one
        // field needed into a local and pool through an associated function.
        // This keeps the extraction zero-copy; cloning `data` out first would
        // also compile, at the cost of a redundant B*seq*384 allocation.
        let grid = self.grid;
        Ok((0..b)
            .map(|i| Self::pool(grid, &data[i * seq * dim..(i + 1) * seq * dim], tokens))
            .collect())
    }

    /// Average patch tokens within each cell of a `grid x grid` layout, then L2
    /// normalise each cell independently.
    ///
    /// Per-cell normalisation rather than one global normalisation is deliberate.
    /// It stops a single high-energy region — a bright roof, a sun glint — from
    /// dominating the concatenated vector and reducing the descriptor back to a
    /// glorified brightness measure.
    fn pool(grid: usize, seq: &[f32], tokens: usize) -> Vec<f32> {
        let patches = &seq[PREFIX_TOKENS * DIM..];
        let cell = tokens / grid;
        let mut out = vec![0.0f32; grid * grid * DIM];

        for gy in 0..grid {
            for gx in 0..grid {
                let base = (gy * grid + gx) * DIM;
                for ty in gy * cell..(gy + 1) * cell {
                    for tx in gx * cell..(gx + 1) * cell {
                        let t = (ty * tokens + tx) * DIM;
                        for d in 0..DIM {
                            out[base + d] += patches[t + d];
                        }
                    }
                }
                let inv = 1.0 / (cell * cell) as f32;
                let mut norm = 0.0f32;
                for d in 0..DIM {
                    out[base + d] *= inv;
                    norm += out[base + d] * out[base + d];
                }
                let norm = norm.sqrt().max(1e-12);
                for d in 0..DIM {
                    out[base + d] /= norm;
                }
            }
        }
        out
    }
}
