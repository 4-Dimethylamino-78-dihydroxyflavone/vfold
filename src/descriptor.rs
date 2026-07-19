//! Turning raw descriptors into a similarity matrix worth selecting on.
//!
//! Self-supervised ViT features have a large, near-constant leading component.
//! It encodes something like "this is an aerial photograph of farmland in
//! afternoon light" — true of every frame in the capture, and therefore useless
//! for telling them apart. Left in, it dominates the cosine and compresses every
//! pairwise similarity into a narrow band near 1.0, which is precisely the
//! failure recorded for the mean-pooled DVLT descriptors: 0.982 adjacent versus
//! 0.789 distant, with thresholds landing on a cliff rather than a plateau.
//!
//! Projecting out the leading principal components restores dynamic range. The
//! per-dimension rescale that follows needs a floor: directions with almost no
//! variance are noise, and dividing by a tiny sigma amplifies that noise into
//! the signal band. The floor is expressed as a fraction of the largest sigma,
//! matching the convention already used elsewhere in this project.

use crate::coverage::Sim;

/// Centre, project out the leading `drop_pcs` principal components, rescale each
/// dimension by its standard deviation subject to a floor, then L2 normalise.
pub fn whiten(x: &mut [Vec<f32>], drop_pcs: usize, sigma_floor_frac: f32) {
    let n = x.len();
    if n < 2 {
        return;
    }
    let d = x[0].len();

    // Centre.
    let mut mean = vec![0.0f32; d];
    for row in x.iter() {
        for j in 0..d {
            mean[j] += row[j];
        }
    }
    for m in mean.iter_mut() {
        *m /= n as f32;
    }
    for row in x.iter_mut() {
        for j in 0..d {
            row[j] -= mean[j];
        }
    }

    // Leading principal components by power iteration with deflation. With
    // n << d the covariance is rank-deficient, so an explicit d x d
    // eigendecomposition would be both enormous and mostly meaningless; power
    // iteration touches only the directions the data actually spans.
    for _ in 0..drop_pcs {
        let Some(v) = leading_direction(x, 64) else { break };
        for row in x.iter_mut() {
            let dot: f32 = row.iter().zip(&v).map(|(a, b)| a * b).sum();
            for j in 0..d {
                row[j] -= dot * v[j];
            }
        }
    }

    // Per-dimension rescale with a floor relative to the largest sigma.
    let mut sigma = vec![0.0f32; d];
    for row in x.iter() {
        for j in 0..d {
            sigma[j] += row[j] * row[j];
        }
    }
    for s in sigma.iter_mut() {
        *s = (*s / n as f32).sqrt();
    }
    let smax = sigma.iter().copied().fold(0.0f32, f32::max);
    let floor = sigma_floor_frac * smax;
    for row in x.iter_mut() {
        for j in 0..d {
            row[j] /= sigma[j].max(floor).max(1e-12);
        }
    }

    l2_normalise(x);
}

fn leading_direction(x: &[Vec<f32>], iters: usize) -> Option<Vec<f32>> {
    let d = x.first()?.len();
    // Deterministic seed — a fixed pattern rather than a random one, so runs
    // reproduce exactly. Any vector not orthogonal to the leading eigenvector
    // converges; alternating signs avoids being orthogonal to a constant one.
    let mut v: Vec<f32> = (0..d).map(|j| if j % 2 == 0 { 1.0 } else { -1.0 }).collect();
    normalise(&mut v);

    for _ in 0..iters {
        // w = X^T (X v), i.e. apply the covariance without forming it.
        let mut w = vec![0.0f32; d];
        for row in x {
            let dot: f32 = row.iter().zip(&v).map(|(a, b)| a * b).sum();
            if dot != 0.0 {
                for (wj, &rj) in w.iter_mut().zip(row) {
                    *wj += dot * rj;
                }
            }
        }
        let norm: f32 = w.iter().map(|a| a * a).sum::<f32>().sqrt();
        if norm < 1e-20 {
            return None;
        }
        w.iter_mut().for_each(|wj| *wj /= norm);
        v = w;
    }
    Some(v)
}

fn normalise(v: &mut [f32]) {
    let n: f32 = v.iter().map(|a| a * a).sum::<f32>().sqrt().max(1e-12);
    v.iter_mut().for_each(|a| *a /= n);
}

pub fn l2_normalise(x: &mut [Vec<f32>]) {
    for row in x.iter_mut() {
        normalise(row);
    }
}

/// Cosine similarity, rectified at a data-derived point and optionally sharpened.
///
/// Coverage requires non-negative similarity: a negative value would let adding
/// a view *reduce* coverage, breaking the monotonicity the greedy guarantee
/// rests on. But clamping raw cosine at zero is wrong after whitening, and
/// silently so. Whitening centres the descriptors, which forces the mean
/// pairwise cosine to approximately zero — so a hard clamp at zero destroys the
/// entire lower half of the distribution. Measured on the building orbit that
/// produced p05 = p50 = 0.000: half of all pairs became indistinguishable.
///
/// So the rectification point is derived from the data instead:
///
/// ```text
///     s = max(0, (cos - c) / (1 - c))
/// ```
///
/// The subtlety is how to choose `c`. The obvious route — a quantile of the
/// all-pairs cosine distribution — fails silently on exactly the captures this
/// tool targets. When one region is photographed far more than the rest, its
/// internal pairs dominate the distribution by sheer count, so any high quantile
/// lands *inside* that region's similarity range. Every smaller region then falls
/// below the cut, is zeroed, and its frames end up with no neighbours at all.
/// Measured on a synthetic capture where one cluster held 78% of the frames, a
/// `q = 0.90` all-pairs rule erased three regions and left 18 of 252 junk-free
/// frames with zero neighbours. The regions the tool exists to protect were
/// destroyed in preprocessing, before selection ever ran.
///
/// So `c` is chosen to guarantee something about *frames* rather than pairs:
/// take each frame's `degree`-th strongest relationship and set `c` at a low
/// quantile of those. By construction, 95% of frames keep at least `degree`
/// genuine neighbours regardless of which region they belong to, while a frame
/// with no real overlap — blur, sky, obstruction — scores near-zero against
/// everything and is caught downstream by the support gate.
///
/// `degree` is the *expected overlap* of the capture: how many other views
/// typically see the same ground. It is a property of how the photos were taken
/// (forward and side overlap), not of the selection budget, and it must not be
/// confused with the gate's minimum-registrable count. Tying the two together —
/// rectifying at the 2-neighbour level to match a gate of 2 — sets the floor far
/// too high and collapses the coverage graph to a median of two neighbours per
/// frame, which is what an earlier version of this code did.
///
/// `power` further sharpens what survives. Coverage *sums* similarities, so
/// without it many weak relationships can outweigh a few strong ones.
/// `spatial` optionally supplies a symmetric `n * n` plausibility weight in
/// `(0, 1]` — see [`crate::gps`]. It is applied to the cosine *before* the
/// rectification point is chosen, not after, and the ordering matters. Visual
/// similarity is only a proxy for co-visibility, and on repetitive ground the
/// proxy produces false edges: two frames 300 m apart over identical crop rows
/// look like neighbours. If the floor were derived first, those false edges
/// would set it, and the genuine near neighbours they displaced would already
/// be gone. Weighting first means the "keep the degree-th strongest
/// relationship" guarantee is about relationships that could physically exist.
pub fn similarity(
    x: &[Vec<f32>],
    degree: usize,
    power: f32,
    spatial: Option<&[f32]>,
) -> Sim {
    let n = x.len();
    let mut cos = vec![0.0f32; n * n];
    for i in 0..n {
        for j in i..n {
            let dot: f32 = x[i].iter().zip(&x[j]).map(|(a, b)| a * b).sum();
            let dot = dot.clamp(-1.0, 1.0);
            cos[i * n + j] = dot;
            cos[j * n + i] = dot;
        }
    }

    // Suppress physically implausible pairs. Negatives are floored to zero here
    // rather than later: they are discarded by the rectification regardless, and
    // clearing them first keeps the weighting a plain product on [0, 1].
    if let Some(w) = spatial {
        assert_eq!(w.len(), n * n, "spatial weight matrix must be n*n");
        for idx in 0..n * n {
            cos[idx] = cos[idx].max(0.0) * w[idx];
        }
    }

    // Each frame's degree-th strongest relationship.
    let degree = degree.clamp(1, n.saturating_sub(1).max(1));
    let mut dth: Vec<f32> = (0..n)
        .map(|i| {
            let mut row: Vec<f32> = (0..n).filter(|&j| j != i).map(|j| cos[i * n + j]).collect();
            row.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap());
            row.get(degree - 1).copied().unwrap_or(-1.0)
        })
        .collect();
    dth.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    // p05 rather than the median: the median frame lives in the densest region,
    // and letting it set the bar reintroduces the starvation this avoids.
    let c = dth[((dth.len() - 1) as f32 * 0.05) as usize];
    let denom = (1.0 - c).max(1e-6);

    let mut data = vec![0.0f32; n * n];
    for i in 0..n {
        for j in i..n {
            let s = ((cos[i * n + j] - c) / denom).clamp(0.0, 1.0);
            let s = if (power - 1.0).abs() < 1e-6 { s } else { s.powf(power) };
            data[i * n + j] = s;
            data[j * n + i] = s;
        }
    }
    Sim::from_vec(n, data)
}

/// Spread of the raw cosine distribution, before rectification. Reported so the
/// rectification point can be judged rather than assumed.
pub fn cosine_spread(x: &[Vec<f32>]) -> Spread {
    let n = x.len();
    let mut off = Vec::with_capacity(n * (n - 1) / 2);
    let mut best = vec![f32::NEG_INFINITY; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let dot: f32 = x[i].iter().zip(&x[j]).map(|(a, b)| a * b).sum();
            off.push(dot);
            if dot > best[i] {
                best[i] = dot;
            }
            if dot > best[j] {
                best[j] = dot;
            }
        }
    }
    off.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let at = |q: f32| off[((off.len() - 1) as f32 * q) as usize];
    Spread {
        p05: at(0.05),
        p50: at(0.50),
        p95: at(0.95),
        mean_nn1: best.iter().sum::<f32>() / n as f32,
        nonzero_frac: off.iter().filter(|&&s| s > 0.0).count() as f32 / off.len() as f32,
    }
}

/// Summary of how much a descriptor set actually discriminates.
///
/// This is the yardstick the DVLT descriptors failed. If `p05` and `p95` sit
/// within a whisker of each other, the descriptor is reporting "same scene" and
/// no selection rule downstream can rescue it.
pub struct Spread {
    pub p05: f32,
    pub p50: f32,
    pub p95: f32,
    pub mean_nn1: f32,
    /// Fraction of off-diagonal pairs with non-zero similarity — the density of
    /// the coverage graph. Very low means frames barely relate to one another
    /// and the fold will struggle; very high means weak correlations are being
    /// counted as overlap.
    pub nonzero_frac: f32,
}

pub fn spread(sim: &Sim) -> Spread {
    let n = sim.n();
    let mut off = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            off.push(sim.col(i)[j]);
        }
    }
    off.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let at = |q: f32| off[((off.len() - 1) as f32 * q) as usize];
    let nn1 = sim.nth_neighbour_sim(1);
    let nonzero = off.iter().filter(|&&s| s > 0.0).count();
    Spread {
        p05: at(0.05),
        p50: at(0.50),
        p95: at(0.95),
        mean_nn1: nn1.iter().sum::<f32>() / n as f32,
        nonzero_frac: nonzero as f32 / off.len() as f32,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Whitening must widen the similarity distribution. Construct descriptors
    /// with a huge shared component plus a small distinguishing one — the
    /// pathology being corrected — and check the spread opens up.
    #[test]
    fn whitening_restores_dynamic_range() {
        let n = 40;
        let d = 128;
        let mut x: Vec<Vec<f32>> = (0..n)
            .map(|i| {
                (0..d)
                    .map(|j| {
                        // Dominant constant "this is the same scene" component,
                        // plus a weak per-item signal.
                        10.0 + 0.1 * ((i * 7 + j * 13) % 11) as f32
                    })
                    .collect()
            })
            .collect();

        let mut raw = x.clone();
        l2_normalise(&mut raw);
        let before = cosine_spread(&raw);

        whiten(&mut x, 1, 0.3);
        let after = cosine_spread(&x);

        eprintln!(
            "raw      p05={:.4} p50={:.4} p95={:.4}",
            before.p05, before.p50, before.p95
        );
        eprintln!(
            "whitened p05={:.4} p50={:.4} p95={:.4}",
            after.p05, after.p50, after.p95
        );

        let range_before = before.p95 - before.p05;
        let range_after = after.p95 - after.p05;
        assert!(
            range_after > range_before,
            "whitening narrowed the spread ({range_before:.4} -> {range_after:.4})"
        );
    }

    #[test]
    fn similarity_is_symmetric_and_non_negative() {
        let x: Vec<Vec<f32>> = (0..12)
            .map(|i| (0..16).map(|j| ((i * 5 + j * 3) % 7) as f32 - 3.0).collect())
            .collect();
        let mut x = x;
        l2_normalise(&mut x);
        let sim = similarity(&x, 2, 1.5, None);
        for i in 0..sim.n() {
            for j in 0..sim.n() {
                assert!(sim.col(i)[j] >= 0.0, "negative similarity at ({i},{j})");
                assert!(
                    (sim.col(i)[j] - sim.col(j)[i]).abs() < 1e-6,
                    "asymmetry at ({i},{j})"
                );
            }
        }
    }
}
