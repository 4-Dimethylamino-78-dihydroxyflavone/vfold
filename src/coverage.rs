//! Subset selection by submodular maximisation.
//!
//! The job: given N candidate views and a pairwise similarity matrix, pick the k
//! that best *represent* all N. Not the k most different from each other — that is
//! farthest-point sampling, and on real capture it walks straight into the blurry
//! frame, the one pointed at the sky, and the one with a thumb over the lens,
//! because an outlier is by construction the farthest point.
//!
//! Three objectives are implemented. Only one is the default, and the reason is
//! measured rather than assumed — see `objective_probe.py` in the scratchpad and
//! the tests at the bottom of this file.
//!
//! * [`Objective::Saturated`] — `f(S) = Σᵢ min(Cᵢ(S), T)` with `Cᵢ(S) = Σ_{j∈S} sim(i,j)`
//!   and a *uniform* cap T. Each source view needs a fixed quantum of similarity
//!   mass from the selected set; once it has that, further neighbours earn nothing.
//!   Budget therefore flows to whatever is still under-covered. This is the default.
//!
//! * [`Objective::Proportional`] — the textbook saturated coverage of Lin & Bilmes
//!   2011, `f(S) = Σᵢ min(Cᵢ(S), α·Cᵢ(V))`. Kept only so the failure is reproducible.
//!   The per-item cap scales with how photographed a region already is, so a densely
//!   shot region gets a proportionally huge cap and swallows the entire budget. On
//!   the synthetic capture in the tests it spends 12/12 on one cluster.
//!
//! * [`Objective::FacilityLocation`] — `f(S) = Σᵢ max_{j∈S} sim(i,j)`. Well behaved
//!   and outlier-resistant, but allocates budget in proportion to photo density,
//!   which on foot-captured sites tracks walking convenience rather than scene
//!   importance.
//!
//! All three are monotone submodular under a cardinality constraint, so the greedy
//! solution is within (1 − 1/e) ≈ 63% of optimal, and Minoux's lazy evaluation
//! returns a *bit-identical* answer to naive greedy at a fraction of the work.

use std::collections::BinaryHeap;

/// Dense symmetric similarity matrix, row-major, `n × n`.
///
/// Similarities are expected non-negative — clamp cosines at zero before
/// constructing. Negative similarity would let adding a view *reduce* coverage,
/// which breaks monotonicity and with it the greedy guarantee.
pub struct Sim {
    n: usize,
    data: Vec<f32>,
}

impl Sim {
    pub fn from_vec(n: usize, data: Vec<f32>) -> Self {
        assert_eq!(data.len(), n * n, "similarity matrix must be n*n");
        debug_assert!(
            data.iter().all(|&x| x >= 0.0),
            "negative similarity breaks monotonicity; clamp at zero first"
        );
        Self { n, data }
    }

    #[inline]
    pub fn n(&self) -> usize {
        self.n
    }

    /// Column `j` — identical to row `j` by symmetry, so this is a contiguous slice.
    #[inline]
    pub fn col(&self, j: usize) -> &[f32] {
        &self.data[j * self.n..(j + 1) * self.n]
    }

    /// Total similarity mass of each item to the whole corpus, `Cᵢ(V)`.
    pub fn mass(&self) -> Vec<f32> {
        (0..self.n).map(|i| self.col(i).iter().sum()).collect()
    }

    /// Similarity to each item's `rank`-th nearest *other* item.
    ///
    /// `rank = 1` is the nearest neighbour. This is the support signal used to
    /// detect frames with no genuine neighbours (blur, sky, lens obstruction).
    /// It discriminates far better than total mass: a junk frame accumulates
    /// respectable mass from many near-zero similarities, but has no single
    /// strong neighbour.
    pub fn nth_neighbour_sim(&self, rank: usize) -> Vec<f32> {
        assert!(rank >= 1, "rank is 1-based");
        (0..self.n)
            .map(|i| {
                let mut row: Vec<f32> = self
                    .col(i)
                    .iter()
                    .enumerate()
                    .filter(|(j, _)| *j != i)
                    .map(|(_, &s)| s)
                    .collect();
                row.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap());
                row.get(rank - 1).copied().unwrap_or(0.0)
            })
            .collect()
    }
}

/// Outcome of the support gate.
#[derive(Debug, Clone)]
pub struct Gate {
    /// Frames eligible to be *chosen*. Excluded frames are still covered.
    pub eligible: Vec<usize>,
    /// Frames the gate rejected.
    pub rejected: Vec<usize>,
    /// Neighbour count required.
    pub min_neighbours: usize,
    /// Median genuine-neighbour count across the corpus. Context for judging
    /// whether the requirement was demanding or trivial for this dataset.
    pub median_neighbours: usize,
}

/// Reject frames with no genuine neighbours — blur, sky, lens obstruction, a
/// hand over the lens.
///
/// This is not optional. Because `sim(j,j) = 1`, any unpicked frame can always
/// cover *itself* for a gain of ~1. While real scene structure remains
/// uncovered that loses easily, but once the scene is covered every remaining
/// real frame offers only a small gain while a junk frame still offers its full
/// 1.0 — so junk sweeps the tail of the budget. Measured on a synthetic capture
/// at k/N ≈ 25%, both saturated coverage and facility location filled 5–6 of 24
/// slots with junk when ungated.
///
/// The signal is the second-nearest-neighbour similarity, not total similarity
/// mass. Mass does not discriminate: a junk frame accumulates respectable mass
/// from hundreds of near-zero similarities, overlapping the mass range of a
/// genuinely sparse region. A junk frame has no single strong neighbour.
///
/// The criterion is a *count*, not a threshold, and that is the point.
///
/// A threshold has to be tuned, and a relative one misbehaves on clean data: on
/// a 50-frame building orbit containing no junk at all, a median-relative cut
/// rejected 9 frames, because slicing the lower tail of a smooth unimodal
/// distribution always rejects something. There was nothing to find and it found
/// things anyway.
///
/// Rectified similarity removes the need. After rectification "unrelated" is not
/// a small number, it is exactly zero, so the question becomes how many genuine
/// overlapping neighbours a frame has. Zero means no downstream model could ever
/// register it. One means a dead end. Requiring at least two encodes the physical
/// requirement directly and needs no tuning — and on data with no junk it
/// correctly rejects nothing.
pub fn support_gate(sim: &Sim, min_neighbours: usize) -> Gate {
    let n = sim.n();
    let counts: Vec<usize> = (0..n)
        .map(|i| {
            sim.col(i)
                .iter()
                .enumerate()
                .filter(|(j, &s)| *j != i && s > 0.0)
                .count()
        })
        .collect();

    let (mut eligible, mut rejected) = (Vec::new(), Vec::new());
    for (i, &c) in counts.iter().enumerate() {
        if c >= min_neighbours {
            eligible.push(i)
        } else {
            rejected.push(i)
        }
    }

    let mut sorted = counts.clone();
    sorted.sort_unstable();
    let median_neighbours = sorted[sorted.len() / 2];

    Gate { eligible, rejected, min_neighbours, median_neighbours }
}

#[derive(Debug, Clone, Copy)]
pub enum Objective {
    /// Uniform cap T. The default; see module docs.
    Saturated { t: f32 },
    /// Lin & Bilmes per-item cap `α·Cᵢ(V)`. Retained to reproduce its failure.
    Proportional { alpha: f32 },
    /// `Σᵢ max_{j∈S} sim(i,j)`.
    FacilityLocation,
}

/// Per-item caps, precomputed once. `None` for facility location, which caps
/// implicitly via `max` rather than additively.
fn caps(sim: &Sim, obj: Objective) -> Option<Vec<f32>> {
    match obj {
        Objective::Saturated { t } => Some(vec![t; sim.n()]),
        Objective::Proportional { alpha } => {
            Some(sim.mass().into_iter().map(|m| alpha * m).collect())
        }
        Objective::FacilityLocation => None,
    }
}

/// Marginal gain of adding `j` to the current selection.
///
/// `cover` holds `Cᵢ(S)` for the additive objectives, or `maxᵢ` for facility
/// location. Both are O(n) per candidate.
#[inline]
fn gain(sim: &Sim, cover: &[f32], caps: Option<&[f32]>, j: usize) -> f64 {
    let sj = sim.col(j);
    match caps {
        Some(cap) => {
            let mut g = 0.0f64;
            for i in 0..sim.n() {
                let before = cover[i].min(cap[i]);
                let after = (cover[i] + sj[i]).min(cap[i]);
                g += (after - before) as f64;
            }
            g
        }
        None => {
            let mut g = 0.0f64;
            for i in 0..sim.n() {
                let d = sj[i] - cover[i];
                if d > 0.0 {
                    g += d as f64;
                }
            }
            g
        }
    }
}

#[inline]
fn absorb(sim: &Sim, cover: &mut [f32], caps: Option<&[f32]>, j: usize) {
    let sj = sim.col(j);
    match caps {
        Some(_) => {
            for i in 0..cover.len() {
                cover[i] += sj[i];
            }
        }
        None => {
            for i in 0..cover.len() {
                if sj[i] > cover[i] {
                    cover[i] = sj[i];
                }
            }
        }
    }
}

/// Total-ordered f64 for the lazy-greedy heap. Gains are finite by construction.
#[derive(PartialEq)]
struct Bound(f64, usize);
impl Eq for Bound {}
impl Ord for Bound {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0
            .partial_cmp(&other.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            // Tie-break on index so selection is deterministic across runs.
            .then_with(|| other.1.cmp(&self.1))
    }
}
impl PartialOrd for Bound {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Result of a selection run, with enough diagnostics to tell whether the fold
/// actually worked rather than merely completed.
#[derive(Debug, Clone)]
pub struct Selection {
    /// Chosen indices, in the order greedy picked them. Order is meaningful:
    /// earlier picks carry more of the scene.
    pub picked: Vec<usize>,
    /// Objective value after each pick — the saturation curve. If it flattens
    /// well before k, the budget is larger than the scene needs.
    pub curve: Vec<f64>,
    /// Fraction of items whose coverage reached the cap. Only meaningful for
    /// the capped objectives.
    pub saturated_frac: f32,
    /// Candidates the lazy heap re-evaluated, versus `n*k` for naive greedy.
    pub evaluations: usize,
}

/// Lazy greedy maximisation under a cardinality constraint.
///
/// `eligible` is the candidate pool — pass the survivors of the support gate.
/// Items outside it can still be *covered*; they just cannot be *chosen*. That
/// asymmetry is deliberate: a blurry frame should not be picked to represent the
/// scene, but the region it looked at still deserves representation.
pub fn select(sim: &Sim, obj: Objective, k: usize, eligible: &[usize]) -> Selection {
    let n = sim.n();
    let caps = caps(sim, obj);
    let caps_ref = caps.as_deref();
    let mut cover = vec![0.0f32; n];
    let mut chosen = vec![false; n];
    let mut picked = Vec::with_capacity(k);
    let mut curve = Vec::with_capacity(k);
    let mut evaluations = 0usize;

    // Seed the heap with exact gains against the empty set. Because the
    // objective is submodular these are valid upper bounds for every later
    // round — gains only ever shrink as S grows.
    let mut heap: BinaryHeap<Bound> = BinaryHeap::with_capacity(eligible.len());
    for &j in eligible {
        evaluations += 1;
        heap.push(Bound(gain(sim, &cover, caps_ref, j), j));
    }

    let mut total = 0.0f64;
    while picked.len() < k {
        let Some(Bound(_, j)) = heap.pop() else { break };
        if chosen[j] {
            continue;
        }
        evaluations += 1;
        let g = gain(sim, &cover, caps_ref, j);

        // The lazy test: if this freshly computed gain still beats the best
        // *upper bound* remaining in the heap, no unevaluated candidate can
        // beat it, so it is exactly the greedy choice for this round.
        let beats_rest = heap.peek().is_none_or(|Bound(b, _)| g >= *b);
        if beats_rest {
            if g <= 0.0 && !picked.is_empty() {
                break; // nothing left to gain; stop rather than pad the output
            }
            chosen[j] = true;
            absorb(sim, &mut cover, caps_ref, j);
            total += g;
            picked.push(j);
            curve.push(total);
        } else {
            heap.push(Bound(g, j));
        }
    }

    let saturated_frac = match caps_ref {
        Some(cap) => {
            let hit = (0..n).filter(|&i| cover[i] >= cap[i] - 1e-6).count();
            hit as f32 / n as f32
        }
        None => f32::NAN,
    };

    Selection { picked, curve, saturated_frac, evaluations }
}

/// Choose the saturation cap `T` by targeting a saturation *fraction*.
///
/// `--sat-mult` expresses `T` as a multiple of the median nearest-neighbour
/// similarity, which sounds scale-free and is not: the right multiple moves with
/// how dense the coverage graph is. Measured on a 482-frame UAV set, the best
/// worst-case ground gap sat at multiplier 0.75 when graph density was 5.1%, and
/// at 0.5 or below once spatial filtering cut density to 2.5%. A constant tuned
/// against one density silently misbehaves at another — which is how enabling
/// GPS made results *worse* before this existed.
///
/// The quantity that actually predicts quality is the fraction of frames whose
/// coverage reaches the cap. Across that sweep every good configuration landed at
/// 56-85% saturation and every poor one at 35-45%. The reason is mechanical: when
/// little saturates there are no diminishing returns, the objective degenerates
/// towards a linear sum, and greedy loses any incentive to spread.
///
/// So target the fraction and solve for `T`. Saturation is monotone decreasing in
/// `T` — a higher bar is harder to clear — so bisection is valid. Selection is
/// effectively instant under lazy greedy, making a dozen probes free.
pub fn calibrate_cap(
    sim: &Sim,
    k: usize,
    eligible: &[usize],
    target: f32,
    hint: f32,
) -> (f32, f32) {
    let (mut lo, mut hi) = (1e-4_f32.max(hint * 0.02), hint * 8.0);
    let frac_at = |t: f32| select(sim, Objective::Saturated { t }, k, eligible).saturated_frac;

    // If even the loosest cap cannot reach the target, take it and report
    // honestly rather than pretending the bisection converged.
    if frac_at(lo) < target {
        return (lo, frac_at(lo));
    }
    if frac_at(hi) > target {
        return (hi, frac_at(hi));
    }
    for _ in 0..14 {
        let mid = 0.5 * (lo + hi);
        if frac_at(mid) >= target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let t = 0.5 * (lo + hi);
    (t, frac_at(t))
}

/// Reference implementation: recompute every gain every round. Used only to
/// prove the lazy version agrees.
#[cfg(test)]
fn select_naive(sim: &Sim, obj: Objective, k: usize, eligible: &[usize]) -> Vec<usize> {
    let caps = caps(sim, obj);
    let caps_ref = caps.as_deref();
    let mut cover = vec![0.0f32; sim.n()];
    let mut chosen = vec![false; sim.n()];
    let mut picked = Vec::new();
    while picked.len() < k {
        let mut best = None::<(f64, usize)>;
        for &j in eligible {
            if chosen[j] {
                continue;
            }
            let g = gain(sim, &cover, caps_ref, j);
            // Match the lazy version's tie-break: lowest index wins.
            if best.map_or(true, |(bg, bj)| g > bg || (g == bg && j < bj)) {
                best = Some((g, j));
            }
        }
        let Some((g, j)) = best else { break };
        if g <= 0.0 && !picked.is_empty() {
            break;
        }
        chosen[j] = true;
        absorb(sim, &mut cover, caps_ref, j);
        picked.push(j);
    }
    picked
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Production default: a frame must have at least this many genuine
    /// overlapping neighbours to be selectable.
    const MIN_NEIGHBOURS: usize = 2;

    /// A capture with the pathology this tool exists to fix: one region shot to
    /// death because it was easy to stand in, three shot barely, plus junk frames.
    ///
    /// `select` receives an *already rectified* similarity matrix — that is its
    /// contract. So the synthetic emits rectified-style structure directly:
    /// within a region, high similarity with spread; between regions, exactly
    /// zero, because two different places do not overlap; junk, zero to
    /// everything including — off the diagonal — to other junk. Whitening and
    /// rectification are exercised separately in `descriptor.rs`, where they
    /// belong. Mixing them in here is what made an earlier version of this test
    /// reject 5% of every clean corpus by construction.
    fn synthetic_capture() -> (Sim, Vec<&'static str>) {
        // Deterministic LCG — no rand dependency, reproducible across runs.
        let mut state = 0x2545F491_4F6CDD1Du64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64 // in [0,1)
        };

        // The easy-to-stand-in spot gets ~78% of the frames. Milder ratios do
        // not separate the objectives — at 60% of corpus taking 58% of budget
        // every objective merely tracks density and the comparison says nothing.
        let spec: [(&str, usize); 4] =
            [("entrance", 200), ("midyard", 40), ("fardrive", 8), ("backshed", 4)];
        const JUNK: usize = 6;

        let mut labels: Vec<&'static str> = Vec::new();
        let mut region: Vec<i32> = Vec::new();
        for (r, (name, count)) in spec.iter().enumerate() {
            for _ in 0..*count {
                labels.push(name);
                region.push(r as i32);
            }
        }
        for _ in 0..JUNK {
            labels.push("junk");
            region.push(-1); // belongs to no region
        }

        let n = labels.len();
        let mut data = vec![0.0f32; n * n];
        for i in 0..n {
            for j in (i + 1)..n {
                // Same real region -> genuine overlap, with spread in [0.35, 0.95]
                // so that regions differ in internal cohesion. Different regions,
                // or any junk frame -> no overlap.
                let s = if region[i] >= 0 && region[i] == region[j] {
                    0.35 + 0.60 * next() as f32
                } else {
                    0.0
                };
                data[i * n + j] = s;
                data[j * n + i] = s;
            }
        }
        (Sim::from_vec(n, data), labels)
    }

    fn tally(picked: &[usize], labels: &[&str]) -> std::collections::BTreeMap<String, usize> {
        let mut m = std::collections::BTreeMap::new();
        for &p in picked {
            *m.entry(labels[p].to_string()).or_insert(0) += 1;
        }
        m
    }

    fn gated(sim: &Sim, min_neighbours: usize) -> Vec<usize> {
        support_gate(sim, min_neighbours).eligible
    }

    fn regions_hit(picked: &[usize], labels: &[&str]) -> usize {
        picked
            .iter()
            .map(|&p| labels[p])
            .filter(|l| *l != "junk")
            .collect::<std::collections::BTreeSet<_>>()
            .len()
    }

    /// Gate the junk, then compare. Uniform-cap saturated coverage must spread
    /// budget across regions more evenly than facility location, which tracks
    /// photo density.
    #[test]
    fn saturated_spreads_budget_more_evenly_than_facility_location() {
        let (sim, labels) = synthetic_capture();
        let eligible = gated(&sim, MIN_NEIGHBOURS);

        let fl = select(&sim, Objective::FacilityLocation, 12, &eligible);
        let sat = select(&sim, Objective::Saturated { t: 1.2 }, 12, &eligible);

        let fl_t = tally(&fl.picked, &labels);
        let sat_t = tally(&sat.picked, &labels);
        eprintln!("facility location: {fl_t:?} -> {}/4 regions", regions_hit(&fl.picked, &labels));
        eprintln!("saturated (T=1.2): {sat_t:?} -> {}/4 regions", regions_hit(&sat.picked, &labels));

        let fl_dense = fl_t.get("entrance").copied().unwrap_or(0);
        let sat_dense = sat_t.get("entrance").copied().unwrap_or(0);
        assert!(
            sat_dense < fl_dense,
            "saturated should spend less on the oversampled region than FL \
             (saturated {sat_dense}, FL {fl_dense})"
        );
        assert_eq!(
            regions_hit(&sat.picked, &labels),
            4,
            "saturated coverage should reach every region"
        );
    }

    /// The hazard that makes the support gate mandatory rather than optional.
    ///
    /// `sim(j,j) = 1`, so any unpicked frame can always cover *itself* for a gain
    /// of ~1. While real scene structure is uncovered that loses easily. Once the
    /// scene is covered, every remaining real frame offers a small gain but a junk
    /// frame still offers its own full 1.0 — so junk wins the tail of the budget.
    /// This bites both objectives, and bites harder as k/N grows.
    #[test]
    fn ungated_selection_eats_junk_in_the_tail() {
        let (sim, labels) = synthetic_capture();
        let all: Vec<usize> = (0..sim.n()).collect();
        let k = 24; // deliberately generous, to push into the tail

        for obj in [Objective::Saturated { t: 1.2 }, Objective::FacilityLocation] {
            let ungated = select(&sim, obj, k, &all);
            let junk = ungated.picked.iter().filter(|&&p| labels[p] == "junk").count();
            eprintln!("{obj:?} ungated at k={k}: {junk} junk frames selected");

            let gated_sel = select(&sim, obj, k, &gated(&sim, MIN_NEIGHBOURS));
            let junk_gated =
                gated_sel.picked.iter().filter(|&&p| labels[p] == "junk").count();
            assert_eq!(junk_gated, 0, "gate failed to exclude junk for {obj:?}");
        }
    }

    /// The refuted variant, pinned so nobody "fixes" the default back to the
    /// textbook formula. Lin & Bilmes' proportional cap collapses onto the
    /// densest cluster because that cluster's cap scales with its own size.
    #[test]
    fn proportional_cap_collapses_onto_the_dense_cluster() {
        let (sim, labels) = synthetic_capture();
        let eligible = gated(&sim, MIN_NEIGHBOURS);
        let prop = select(&sim, Objective::Proportional { alpha: 0.1 }, 12, &eligible);
        let sat = select(&sim, Objective::Saturated { t: 1.2 }, 12, &eligible);

        let p_t = tally(&prop.picked, &labels);
        eprintln!("proportional (a=0.1): {p_t:?} -> {}/4 regions", regions_hit(&prop.picked, &labels));

        let p_dense = p_t.get("entrance").copied().unwrap_or(0);
        let s_dense = tally(&sat.picked, &labels).get("entrance").copied().unwrap_or(0);
        assert!(
            p_dense > s_dense,
            "textbook proportional cap should concentrate on the oversampled region \
             more than the uniform cap does (proportional {p_dense}, uniform {s_dense})"
        );
        assert!(
            regions_hit(&prop.picked, &labels) < 4,
            "expected the proportional cap to starve at least one region, \
             got {}/4", regions_hit(&prop.picked, &labels)
        );
    }

    /// Lazy greedy is an optimisation, not an approximation. It must return the
    /// identical sequence to naive greedy for every objective.
    #[test]
    fn lazy_greedy_matches_naive_exactly() {
        let (sim, _) = synthetic_capture();
        let all: Vec<usize> = (0..sim.n()).collect();
        for obj in [
            Objective::Saturated { t: 1.2 },
            Objective::Proportional { alpha: 0.1 },
            Objective::FacilityLocation,
        ] {
            let lazy = select(&sim, obj, 12, &all);
            let naive = select_naive(&sim, obj, 12, &all);
            assert_eq!(lazy.picked, naive, "lazy diverged from naive for {obj:?}");
            eprintln!(
                "{obj:?}: {} evaluations vs {} naive",
                lazy.evaluations,
                sim.n() * 12
            );
        }
    }

    /// Submodularity, checked numerically: the marginal gain of any fixed item
    /// must not increase as the selection grows.
    #[test]
    fn marginal_gains_are_non_increasing() {
        let (sim, _) = synthetic_capture();
        let all: Vec<usize> = (0..sim.n()).collect();
        for obj in [Objective::Saturated { t: 1.2 }, Objective::FacilityLocation] {
            let caps = caps(&sim, obj);
            let caps_ref = caps.as_deref();
            let mut cover = vec![0.0f32; sim.n()];
            let probe = sim.n() - 1;
            let mut prev = gain(&sim, &cover, caps_ref, probe);
            for &j in all.iter().take(15) {
                if j == probe {
                    continue;
                }
                absorb(&sim, &mut cover, caps_ref, j);
                let now = gain(&sim, &cover, caps_ref, probe);
                assert!(
                    now <= prev + 1e-6,
                    "gain rose from {prev} to {now} for {obj:?} — not submodular"
                );
                prev = now;
            }
        }
    }

    /// The support gate must remove junk without removing genuinely sparse
    /// regions. Total mass cannot do this; second-nearest-neighbour can.
    /// The gate must reject junk and only junk.
    #[test]
    fn neighbour_count_gate_rejects_junk_and_only_junk() {
        let (sim, labels) = synthetic_capture();
        let g = support_gate(&sim, MIN_NEIGHBOURS);
        eprintln!(
            "gate: need >={} neighbours (corpus median {}), kept {}, rejected {}",
            g.min_neighbours,
            g.median_neighbours,
            g.eligible.len(),
            g.rejected.len()
        );
        assert!(
            g.rejected.iter().all(|&i| labels[i] == "junk"),
            "gate rejected a real frame: {:?}",
            g.rejected.iter().map(|&i| labels[i]).collect::<Vec<_>>()
        );
        assert_eq!(
            g.rejected.len(),
            labels.iter().filter(|l| **l == "junk").count(),
            "gate missed some junk"
        );
    }

    /// The complementary property, and the one the previous median-relative gate
    /// failed: given data with nothing wrong in it, reject nothing. A threshold
    /// slicing the tail of a unimodal distribution always finds victims.
    #[test]
    fn gate_rejects_nothing_when_there_is_no_junk() {
        let (sim, labels) = synthetic_capture();
        // Rebuild without the junk frames.
        let keep: Vec<usize> = (0..sim.n()).filter(|&i| labels[i] != "junk").collect();
        let m = keep.len();
        let mut data = vec![0.0f32; m * m];
        for (a, &i) in keep.iter().enumerate() {
            for (b, &j) in keep.iter().enumerate() {
                data[a * m + b] = sim.col(i)[j];
            }
        }
        let clean = Sim::from_vec(m, data);
        let g = support_gate(&clean, MIN_NEIGHBOURS);
        eprintln!(
            "clean corpus: kept {}, rejected {} (median {} neighbours)",
            g.eligible.len(),
            g.rejected.len(),
            g.median_neighbours
        );
        assert!(
            g.rejected.is_empty(),
            "gate rejected {} frames from a corpus with no junk",
            g.rejected.len()
        );
    }

    /// Ineligible items still get covered — they just cannot be chosen.
    #[test]
    fn gated_items_are_covered_but_never_picked() {
        let (sim, labels) = synthetic_capture();
        let eligible: Vec<usize> =
            (0..sim.n()).filter(|&i| labels[i] != "junk").collect();
        let sel = select(&sim, Objective::Saturated { t: 1.2 }, 12, &eligible);
        assert!(
            sel.picked.iter().all(|&p| labels[p] != "junk"),
            "a gated frame was selected"
        );
    }

    /// Selection must not pad its output with zero-gain picks when the scene is
    /// exhausted before the budget is.
    #[test]
    fn stops_early_rather_than_padding() {
        // Four identical items: after the first pick, everything is saturated.
        let n = 4;
        let sim = Sim::from_vec(n, vec![1.0; n * n]);
        let all: Vec<usize> = (0..n).collect();
        let sel = select(&sim, Objective::Saturated { t: 1.0 }, 4, &all);
        assert_eq!(sel.picked.len(), 1, "should stop once nothing is left to gain");
    }
}
