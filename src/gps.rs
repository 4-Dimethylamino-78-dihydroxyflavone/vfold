//! Spatial plausibility from EXIF GPS.
//!
//! Visual similarity is a proxy for co-visibility, and on repetitive ground the
//! proxy breaks. Measured on a 482-frame UAV crosshatch over a 307x388 m site:
//! selection by appearance alone landed at 13.6 m median / 50.8 m worst distance
//! to the nearest selected frame, versus 14.2 / 48.9 for *random* — no better
//! than chance, spatially. Two frames 300 m apart showing identical crop rows are
//! near-neighbours in feature space, so coverage ends up covering *appearance
//! classes* rather than ground: one wheat-field frame "covers" every wheat field
//! on the site and the selector moves on.
//!
//! The fix is a false-edge filter, not a new selection algorithm. Multiply the
//! visual similarity by a spatial plausibility weight and every downstream
//! stage — rectification, support gate, saturated coverage, lazy greedy — runs
//! unchanged on a corrected matrix.
//!
//! # Why the scale is per-frame
//!
//! A single global radius assumes uniform capture density. Real captures are not
//! uniform: lawnmower grids, orbits, expanding spirals, sector searches, and
//! scattered swarm patterns all mix dense hovering with sparse transit. Even on
//! the near-uniform grid above, the distance to a frame's 8th-nearest neighbour
//! ranged 14.0-22.9 m between the 5th and 95th percentile, with nearest-neighbour
//! spacing from 0.2 m (hover) to 10 m.
//!
//! So each frame gets its own scale `r_i`, read off the local point density as
//! the distance to its own `degree`-th nearest neighbour:
//!
//! ```text
//!     u    = d_ij^2 / (sigma^2 * r_i * r_j)
//!     w_ij = exp( -u^2 )
//! ```
//!
//! Note the square on `u` — this is a super-Gaussian, and a plain Gaussian does
//! not work here. A Gaussian decays from the origin, so at `d = r` it returns
//! `exp(-1) = 0.37`, attenuating *genuine* overlapping pairs to a third of their
//! weight along with the false ones. Measured on the UAV set that dropped the
//! coverage graph from 5.1% to 2.7% density and left only 8% of frames reaching
//! the saturation cap; with almost nothing saturating, the objective degenerates
//! to a linear sum, loses its diminishing returns, and stops spreading — it left
//! a 77.5 m hole. The super-Gaussian holds ~0.94 at `d = r` and collapses beyond
//! `1.5 r`, which removes false edges while leaving real ones intact.
//!
//! This is the self-tuning kernel of Zelnik-Manor & Perona (2004). It is
//! symmetric because `r_i * r_j` is, it needs no assumption about flight
//! geometry, and it adapts to dense and sparse parts of the same capture. Nothing
//! here knows what a grid is.
//!
//! # When to turn it off
//!
//! For a tight orbit around a single object, cameras on opposite sides are far
//! apart yet do see the same object. Spatial gating would suppress that pair.
//! In practice such pairs are already visually dissimilar (different faces), so
//! the cost is small — but `--gps off` restores pure appearance behaviour.

use std::path::Path;

/// A frame's position in a local tangent plane, metres, origin at corpus centre.
#[derive(Debug, Clone, Copy)]
pub struct Enu {
    pub x: f32,
    pub y: f32,
}

/// Read WGS84 latitude/longitude from a file's EXIF GPS block.
///
/// Returns `None` for anything without usable coordinates — no EXIF, no GPS
/// IFD, or a null island reading. Missing GPS is normal (scanned imagery,
/// indoor capture, stripped metadata) and must never be an error.
fn read_latlon(path: &Path) -> Option<(f64, f64)> {
    let file = std::fs::File::open(path).ok()?;
    let mut buf = std::io::BufReader::new(file);
    let reader = exif::Reader::new();
    let meta = reader.read_from_container(&mut buf).ok()?;

    // GPS coordinates are stored as three rationals (degrees, minutes, seconds)
    // with a separate hemisphere reference tag. Sign lives in the ref, not the
    // number, so a southern or western capture silently mirrors without it.
    let dms = |tag: exif::Tag| -> Option<f64> {
        let f = meta.get_field(tag, exif::In::PRIMARY)?;
        match &f.value {
            exif::Value::Rational(v) if v.len() >= 3 => {
                Some(v[0].to_f64() + v[1].to_f64() / 60.0 + v[2].to_f64() / 3600.0)
            }
            _ => None,
        }
    };
    let refc = |tag: exif::Tag| -> Option<char> {
        let f = meta.get_field(tag, exif::In::PRIMARY)?;
        match &f.value {
            exif::Value::Ascii(v) if !v.is_empty() && !v[0].is_empty() => {
                Some(v[0][0] as char)
            }
            _ => None,
        }
    };

    let mut lat = dms(exif::Tag::GPSLatitude)?;
    let mut lon = dms(exif::Tag::GPSLongitude)?;
    if matches!(refc(exif::Tag::GPSLatitudeRef), Some('S') | Some('s')) {
        lat = -lat;
    }
    if matches!(refc(exif::Tag::GPSLongitudeRef), Some('W') | Some('w')) {
        lon = -lon;
    }

    // Exactly (0,0) is the classic "GPS wrote a zero" sentinel, not a capture in
    // the Gulf of Guinea.
    if lat == 0.0 && lon == 0.0 {
        return None;
    }
    if !(-90.0..=90.0).contains(&lat) || !(-180.0..=180.0).contains(&lon) {
        return None;
    }
    Some((lat, lon))
}

/// Read positions for a whole corpus and project to a local tangent plane.
///
/// Equirectangular projection about the corpus centroid. Exact enough by orders
/// of magnitude at site scale — the distortion over a few kilometres is far below
/// consumer GPS noise — and it avoids a projection dependency.
///
/// Horizontal only. Altitude is deliberately ignored: consumer GPS vertical error
/// is several times its horizontal error, and height differences that matter to
/// co-visibility (an orbit rising past a facade) show up in the visual term
/// anyway.
pub fn positions(paths: &[std::path::PathBuf]) -> Vec<Option<Enu>> {
    let ll: Vec<Option<(f64, f64)>> = paths.iter().map(|p| read_latlon(p)).collect();
    let present: Vec<(f64, f64)> = ll.iter().flatten().copied().collect();
    if present.is_empty() {
        return vec![None; paths.len()];
    }
    let lat0 = present.iter().map(|p| p.0).sum::<f64>() / present.len() as f64;
    let lon0 = present.iter().map(|p| p.1).sum::<f64>() / present.len() as f64;
    const M_PER_DEG_LAT: f64 = 111_132.0;
    let m_per_deg_lon = 111_320.0 * lat0.to_radians().cos();

    ll.into_iter()
        .map(|o| {
            o.map(|(lat, lon)| Enu {
                x: ((lon - lon0) * m_per_deg_lon) as f32,
                y: ((lat - lat0) * M_PER_DEG_LAT) as f32,
            })
        })
        .collect()
}

/// What the spatial kernel did, for the diagnostics line.
#[derive(Debug, Clone)]
pub struct SpatialInfo {
    pub tagged: usize,
    pub total: usize,
    /// Site bounding box, metres.
    pub extent: (f32, f32),
    /// Median per-frame scale `r_i`, metres — the effective co-visibility radius.
    pub median_scale: f32,
    /// Spread of `r_i` across frames. Well above 1 means the capture density is
    /// uneven and a single global radius would have been wrong.
    pub scale_p05: f32,
    pub scale_p95: f32,
}

/// Build the symmetric spatial plausibility matrix, row-major `n * n`.
///
/// Pairs where either frame lacks GPS get weight 1.0 — no position, no evidence
/// against the pair, so the visual term decides alone. That makes partial GPS
/// coverage degrade smoothly instead of falling off a cliff.
pub fn spatial_weights(
    pos: &[Option<Enu>],
    degree: usize,
    sigma: f32,
) -> Option<(Vec<f32>, SpatialInfo)> {
    let n = pos.len();
    let tagged = pos.iter().filter(|p| p.is_some()).count();
    if tagged < 3 {
        return None;
    }

    let d = |a: Enu, b: Enu| -> f32 { ((a.x - b.x).powi(2) + (a.y - b.y).powi(2)).sqrt() };

    // Per-frame scale: distance to this frame's own `degree`-th nearest tagged
    // neighbour. Reading the scale off local density is what makes the kernel
    // indifferent to flight topology.
    let mut scale = vec![f32::NAN; n];
    for i in 0..n {
        let Some(pi) = pos[i] else { continue };
        let mut ds: Vec<f32> = (0..n)
            .filter(|&j| j != i)
            .filter_map(|j| pos[j].map(|pj| d(pi, pj)))
            .collect();
        if ds.is_empty() {
            continue;
        }
        ds.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        // Guard against coincident positions (a hovering drone writes the same
        // fix repeatedly); a zero scale would divide by zero below.
        scale[i] = ds[degree.min(ds.len()) - 1].max(1e-3);
    }

    let mut finite: Vec<f32> = scale.iter().copied().filter(|s| s.is_finite()).collect();
    if finite.is_empty() {
        return None;
    }
    finite.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let median_scale = finite[finite.len() / 2];
    let scale_p05 = finite[((finite.len() - 1) as f32 * 0.05) as usize];
    let scale_p95 = finite[((finite.len() - 1) as f32 * 0.95) as usize];

    let xs: Vec<f32> = pos.iter().flatten().map(|p| p.x).collect();
    let ys: Vec<f32> = pos.iter().flatten().map(|p| p.y).collect();
    let extent = (
        xs.iter().copied().fold(f32::MIN, f32::max) - xs.iter().copied().fold(f32::MAX, f32::min),
        ys.iter().copied().fold(f32::MIN, f32::max) - ys.iter().copied().fold(f32::MAX, f32::min),
    );

    let s2 = (sigma * sigma).max(1e-6);
    let mut w = vec![1.0f32; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            let weight = match (pos[i], pos[j]) {
                (Some(pi), Some(pj)) if scale[i].is_finite() && scale[j].is_finite() => {
                    let dd = d(pi, pj);
                    // Super-Gaussian: flat inside the co-visibility radius,
                    // steep outside it. See the module docs for why a plain
                    // Gaussian starves the coverage graph.
                    let u = (dd * dd) / (s2 * scale[i] * scale[j]);
                    (-(u * u)).exp()
                }
                // Missing position on either side: no evidence, no penalty.
                _ => 1.0,
            };
            w[i * n + j] = weight;
            w[j * n + i] = weight;
        }
    }

    Some((
        w,
        SpatialInfo { tagged, total: n, extent, median_scale, scale_p05, scale_p95 },
    ))
}

/// How well a selection covers the ground, in metres.
#[derive(Debug, Clone)]
pub struct CoverageDistances {
    pub median: f32,
    pub p95: f32,
    pub max: f32,
    /// Median distance achieved by an average random selection of the same size.
    ///
    /// Reported because absolute metres are uninterpretable without it. A median
    /// of 13.6 m sounds excellent until random scores 14.2 m — which is exactly
    /// what appearance-only selection did on a UAV crosshatch, and the only
    /// reason the aliasing was noticed at all. Any spatial claim should be read
    /// against this column, never on its own.
    pub random_median: f32,
    pub random_max: f32,
}

/// For every tagged frame, the distance to the nearest *selected* frame.
///
/// This is the ground truth for a spatial claim, and it is worth measuring even
/// when the feature-space diagnostics look healthy: a selection can saturate
/// appearance coverage beautifully while leaving a whole corner of the site
/// unrepresented, because every frame in that corner resembles frames elsewhere.
pub fn coverage_distances(pos: &[Option<Enu>], picked: &[usize]) -> Option<CoverageDistances> {
    let all: Vec<Enu> = pos.iter().flatten().copied().collect();
    if all.is_empty() {
        return None;
    }
    let sel: Vec<Enu> = picked.iter().filter_map(|&i| pos.get(i).copied().flatten()).collect();
    if sel.is_empty() {
        return None;
    }

    let nearest = |sel: &[Enu]| -> Vec<f32> {
        let mut d: Vec<f32> = all
            .iter()
            .map(|p| {
                sel.iter()
                    .map(|s| ((p.x - s.x).powi(2) + (p.y - s.y).powi(2)).sqrt())
                    .fold(f32::INFINITY, f32::min)
            })
            .collect();
        d.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        d
    };

    let d = nearest(&sel);

    // Random baseline, averaged over several deterministic draws. Fixed seed so
    // the reported figure is reproducible across runs; a moving baseline would
    // make two runs incomparable for no benefit.
    let k = sel.len().min(all.len());
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut rnd = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    const DRAWS: usize = 24;
    let (mut med_sum, mut max_sum) = (0.0f64, 0.0f64);
    let mut idx: Vec<usize> = (0..all.len()).collect();
    for _ in 0..DRAWS {
        // Partial Fisher-Yates: shuffle only the first k slots.
        for i in 0..k {
            let j = i + (rnd() as usize) % (idx.len() - i);
            idx.swap(i, j);
        }
        let draw: Vec<Enu> = idx[..k].iter().map(|&i| all[i]).collect();
        let rd = nearest(&draw);
        med_sum += rd[rd.len() / 2] as f64;
        max_sum += rd[rd.len() - 1] as f64;
    }

    Some(CoverageDistances {
        median: d[d.len() / 2],
        p95: d[((d.len() - 1) as f32 * 0.95) as usize],
        max: d[d.len() - 1],
        random_median: (med_sum / DRAWS as f64) as f32,
        random_max: (max_sum / DRAWS as f64) as f32,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid(cols: usize, rows: usize, step: f32) -> Vec<Option<Enu>> {
        (0..rows)
            .flat_map(|r| {
                (0..cols).map(move |c| {
                    Some(Enu { x: c as f32 * step, y: r as f32 * step })
                })
            })
            .collect()
    }

    /// The core property: distant pairs must be suppressed, near pairs preserved.
    #[test]
    fn distant_pairs_are_suppressed_near_pairs_survive() {
        let pos = grid(10, 10, 5.0);
        let (w, info) = spatial_weights(&pos, 8, 1.0).unwrap();
        let n = pos.len();
        eprintln!("median scale {:.1} m, extent {:?}", info.median_scale, info.extent);

        let adjacent = w[0 * n + 1]; // 5 m apart
        let opposite = w[0 * n + (n - 1)]; // corner to corner, ~64 m
        eprintln!("adjacent w={adjacent:.4}  opposite w={opposite:.6}");
        assert!(adjacent > 0.5, "adjacent pair suppressed: {adjacent}");
        assert!(opposite < 0.01, "distant pair survived: {opposite}");
    }

    /// Topology independence: the kernel must adapt to a capture whose density
    /// varies wildly, which is what a search or swarm pattern looks like. A dense
    /// cluster and a sparse transit line should BOTH keep their local neighbours.
    #[test]
    fn adapts_to_uneven_density() {
        let mut pos: Vec<Option<Enu>> = Vec::new();
        // Dense hover cluster: 20 frames within a couple of metres.
        for i in 0..20 {
            pos.push(Some(Enu { x: (i % 5) as f32 * 0.5, y: (i / 5) as f32 * 0.5 }));
        }
        // Sparse transit: 20 frames strung out every 40 m.
        for i in 0..20 {
            pos.push(Some(Enu { x: 200.0 + i as f32 * 40.0, y: 0.0 }));
        }
        let n = pos.len();
        let (w, info) = spatial_weights(&pos, 4, 1.0).unwrap();
        eprintln!(
            "scales: p05={:.2} median={:.2} p95={:.2} (spread {:.1}x)",
            info.scale_p05,
            info.median_scale,
            info.scale_p95,
            info.scale_p95 / info.scale_p05
        );

        // Neighbours inside the dense cluster survive despite being 0.5 m apart.
        assert!(w[0 * n + 1] > 0.5, "dense-cluster neighbour suppressed");
        // Neighbours along the sparse line survive despite being 40 m apart —
        // a global radius tuned to the cluster would have killed these.
        assert!(w[20 * n + 21] > 0.3, "sparse-line neighbour suppressed: {}", w[20 * n + 21]);
        // The two groups are 200 m apart and must not be linked.
        assert!(w[0 * n + 20] < 0.01, "unrelated groups linked");
    }

    #[test]
    fn is_symmetric_with_unit_self_similarity() {
        let pos = grid(6, 6, 3.0);
        let n = pos.len();
        let (w, _) = spatial_weights(&pos, 4, 1.0).unwrap();
        for i in 0..n {
            assert!((w[i * n + i] - 1.0).abs() < 1e-6, "self weight != 1 at {i}");
            for j in 0..n {
                assert!(
                    (w[i * n + j] - w[j * n + i]).abs() < 1e-6,
                    "asymmetry at ({i},{j})"
                );
            }
        }
    }

    /// Frames without GPS must not be penalised — partial coverage degrades
    /// smoothly to pure appearance behaviour rather than falling off a cliff.
    #[test]
    fn untagged_frames_get_neutral_weight() {
        let mut pos = grid(5, 5, 5.0);
        pos[7] = None;
        let n = pos.len();
        let (w, info) = spatial_weights(&pos, 4, 1.0).unwrap();
        assert_eq!(info.tagged, n - 1);
        for j in 0..n {
            assert!(
                (w[7 * n + j] - 1.0).abs() < 1e-6,
                "untagged frame was penalised against {j}"
            );
        }
    }

    /// The audit must be able to tell a good spread from a bad one, and must
    /// report a random baseline that sits between them. Without a working
    /// baseline the metric is unreadable — 13.6 m looks like success until you
    /// learn random scores 14.2 m.
    #[test]
    fn spatial_audit_discriminates_spread_from_clumped() {
        let pos = grid(12, 12, 10.0); // 144 frames over 110x110 m
        let k = 16;

        // Evenly spread: every third row and column.
        let spread: Vec<usize> = (0..12)
            .step_by(3)
            .flat_map(|r| (0..12).step_by(3).map(move |c| r * 12 + c))
            .collect();
        // Clumped: one corner block.
        let clumped: Vec<usize> = (0..4).flat_map(|r| (0..4).map(move |c| r * 12 + c)).collect();
        assert_eq!(spread.len(), k);
        assert_eq!(clumped.len(), k);

        let s = coverage_distances(&pos, &spread).unwrap();
        let c = coverage_distances(&pos, &clumped).unwrap();
        eprintln!(
            "spread : median {:.1} worst {:.1}   clumped: median {:.1} worst {:.1}   \
             random: median {:.1} worst {:.1}",
            s.median, s.max, c.median, c.max, s.random_median, s.random_max
        );

        assert!(s.max < c.max, "spread should beat clumped on worst case");
        assert!(
            s.max < s.random_max,
            "an even spread should beat random ({:.1} vs {:.1})",
            s.max,
            s.random_max
        );
        assert!(
            c.max > c.random_max,
            "a corner clump should lose to random ({:.1} vs {:.1})",
            c.max,
            c.random_max
        );
        // The baseline is a property of the point set and k, not of the
        // selection, so both calls must report the same one.
        assert!((s.random_max - c.random_max).abs() < 1e-3, "baseline is not deterministic");
    }

    #[test]
    fn too_few_positions_disables_the_kernel() {
        let pos = vec![None, None, Some(Enu { x: 0.0, y: 0.0 })];
        assert!(spatial_weights(&pos, 4, 1.0).is_none());
    }
}
