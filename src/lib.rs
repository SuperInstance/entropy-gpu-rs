//! # entropy-gpu
//!
//! GPU-ready batch entropy computation library providing pure Rust data structures
//! and operations for information-theoretic measures. Designed for batch processing
//! with rayon parallelism, making it trivially portable to GPU kernels.
//!
//! ## Core Functions
//!
//! - [`batch_shannon_entropy`] — Shannon entropy for N probability distributions
//! - [`batch_kl_divergence`] — KL divergence for N distribution pairs
//! - [`batch_js_divergence`] — Jensen-Shannon divergence for N distribution pairs
//! - [`mutual_information_matrix`] — Mutual information between all variable pairs
//! - [`TransferEntropyEstimator`] — Streaming transfer entropy estimator
//! - [`EntropyProfile`] — Entropy rate, permutation entropy, sample entropy

use rayon::prelude::*;

// ─── Constants ───────────────────────────────────────────────────────────────

const LN_2: f64 = std::f64::consts::LN_2;

/// Small epsilon to avoid log(0) or division by zero.
const EPS: f64 = 1e-15;

// ─── Batch Shannon Entropy ───────────────────────────────────────────────────

/// Compute Shannon entropy H(X) = -Σ p(x) log₂ p(x) for each distribution.
///
/// Distributions are not required to be normalized (they will be normalized internally).
/// Zero-probability entries are skipped.
///
/// # Panics
///
/// Panics if any distribution is empty.
pub fn batch_shannon_entropy(distributions: &[Vec<f64>]) -> Vec<f64> {
    assert!(
        !distributions.is_empty(),
        "distributions must not be empty"
    );
    distributions
        .par_iter()
        .map(|dist| {
            assert!(!dist.is_empty(), "each distribution must be non-empty");
            let sum: f64 = dist.iter().sum();
            if sum.abs() < EPS {
                return 0.0;
            }
            -dist
                .iter()
                .filter(|&&p| p > EPS)
                .map(|&p| {
                    let pn = p / sum;
                    pn * (pn.ln() / LN_2)
                })
                .sum::<f64>()
        })
        .collect()
}

// ─── Batch KL Divergence ────────────────────────────────────────────────────

/// Compute KL divergence D_KL(P || Q) = Σ P(i) log₂(P(i)/Q(i)) for each pair.
///
/// Both distributions are normalized internally. Entries where Q ≈ 0 are skipped
/// (treating them as supported only where Q > 0).
pub fn batch_kl_divergence(pairs: &[(Vec<f64>, Vec<f64>)]) -> Vec<f64> {
    assert!(!pairs.is_empty(), "pairs must not be empty");
    pairs
        .par_iter()
        .map(|(p, q)| {
            assert!(
                p.len() == q.len(),
                "each pair must have same length: got {} vs {}",
                p.len(),
                q.len()
            );
            let sp: f64 = p.iter().sum();
            let sq: f64 = q.iter().sum();
            if sp.abs() < EPS || sq.abs() < EPS {
                return 0.0;
            }
            p.iter()
                .zip(q.iter())
                .filter(|(&pi, &qi)| pi > EPS && qi > EPS)
                .map(|(pi, qi)| {
                    let pn = pi / sp;
                    let qn = qi / sq;
                    pn * (pn.ln() / LN_2 - qn.ln() / LN_2)
                })
                .sum()
        })
        .collect()
}

// ─── Batch JS Divergence ────────────────────────────────────────────────────

/// Compute Jensen-Shannon divergence JS(P || Q) = ½ D_KL(P || M) + ½ D_KL(Q || M),
/// where M = ½(P + Q), for each pair.
///
/// JS divergence is symmetric and bounded in [0, 1] (base-2 log).
pub fn batch_js_divergence(pairs: &[(Vec<f64>, Vec<f64>)]) -> Vec<f64> {
    assert!(!pairs.is_empty(), "pairs must not be empty");
    pairs
        .par_iter()
        .map(|(p, q)| {
            assert!(
                p.len() == q.len(),
                "each pair must have same length: got {} vs {}",
                p.len(),
                q.len()
            );
            let sp: f64 = p.iter().sum();
            let sq: f64 = q.iter().sum();
            if sp.abs() < EPS || sq.abs() < EPS {
                return 0.0;
            }
            let n = p.len();
            let mut kl_pm = 0.0_f64;
            let mut kl_qm = 0.0_f64;
            for i in 0..n {
                let pn = p[i] / sp;
                let qn = q[i] / sq;
                let mn = 0.5 * pn + 0.5 * qn;
                if pn > EPS && mn > EPS {
                    kl_pm += pn * (pn.ln() / LN_2 - mn.ln() / LN_2);
                }
                if qn > EPS && mn > EPS {
                    kl_qm += qn * (qn.ln() / LN_2 - mn.ln() / LN_2);
                }
            }
            0.5 * kl_pm + 0.5 * kl_qm
        })
        .collect()
}

// ─── Mutual Information Matrix ───────────────────────────────────────────────

/// Compute the mutual information matrix for a set of observed discrete variables.
///
/// Each inner `Vec<usize>` represents observations of one variable (all must have the
/// same length). Returns an `n × n` symmetric matrix where entry `(i, j)` is the
/// mutual information I(X_i; X_j) in bits.
pub fn mutual_information_matrix(observations: &[Vec<usize>]) -> Vec<Vec<f64>> {
    let n_vars = observations.len();
    assert!(n_vars > 0, "observations must not be empty");
    let n_obs = observations[0].len();
    for (i, obs) in observations.iter().enumerate() {
        assert_eq!(
            obs.len(),
            n_obs,
            "variable {} has {} observations, expected {}",
            i,
            obs.len(),
            n_obs
        );
    }

    let mut mi = vec![vec![0.0_f64; n_vars]; n_vars];

    // Compute MI for upper triangle in parallel
    let pairs: Vec<(usize, usize)> = (0..n_vars)
        .flat_map(|i| (i + 1..n_vars).map(move |j| (i, j)))
        .collect();

    let results: Vec<((usize, usize), f64)> = pairs
        .par_iter()
        .map(|&(i, j)| {
            let val = compute_mi(&observations[i], &observations[j]);
            ((i, j), val)
        })
        .collect();

    for ((i, j), val) in results {
        mi[i][j] = val;
        mi[j][i] = val;
    }

    // Diagonal: H(X_i)
    for i in 0..n_vars {
        mi[i][i] = compute_entropy_discrete(&observations[i]);
    }

    mi
}

/// Compute MI between two discrete variables.
fn compute_mi(x: &[usize], y: &[usize]) -> f64 {
    let n = x.len() as f64;
    let x_vals: Vec<usize> = x.iter().copied().collect();
    let y_vals: Vec<usize> = y.iter().copied().collect();

    // Marginal counts
    let x_max = *x_vals.iter().max().unwrap_or(&0) + 1;
    let y_max = *y_vals.iter().max().unwrap_or(&0) + 1;

    let mut px = vec![0.0_f64; x_max];
    let mut py = vec![0.0_f64; y_max];
    let mut pxy = vec![vec![0.0_f64; y_max]; x_max];

    for k in 0..x_vals.len() {
        px[x_vals[k]] += 1.0;
        py[y_vals[k]] += 1.0;
        pxy[x_vals[k]][y_vals[k]] += 1.0;
    }

    let mut mi = 0.0;
    for i in 0..x_max {
        if px[i] < EPS {
            continue;
        }
        for j in 0..y_max {
            if py[j] < EPS || pxy[i][j] < EPS {
                continue;
            }
            let pij = pxy[i][j] / n;
            let pi = px[i] / n;
            let pj = py[j] / n;
            mi += pij * ((pij / (pi * pj)).ln() / LN_2);
        }
    }
    mi
}

/// Shannon entropy of a discrete-valued vector.
fn compute_entropy_discrete(x: &[usize]) -> f64 {
    let n = x.len() as f64;
    let max_val = *x.iter().max().unwrap_or(&0) + 1;
    let mut counts = vec![0.0_f64; max_val];
    for &v in x {
        counts[v] += 1.0;
    }
    -counts
        .iter()
        .filter(|&&c| c > EPS)
        .map(|&c| {
            let p = c / n;
            p * (p.ln() / LN_2)
        })
        .sum::<f64>()
}

// ─── Transfer Entropy Estimator ──────────────────────────────────────────────

/// Streaming estimator for transfer entropy TE(X → Y).
///
/// Transfer entropy measures the amount of directed (time-asymmetric) transfer of
/// information between two random processes. Uses a k-history approach:
///
/// TE(X→Y) = H(Yₜ₊₁ | Yₜ⁽ᵏ⁾) − H(Yₜ₊₁ | Yₜ⁽ᵏ⁾, Xₜ⁽ᵏ⁾)
///
/// where Yₜ⁽ᵏ⁾ denotes the k-length history of Y up to time t.
///
/// This estimator accumulates observations incrementally and computes TE on demand.
#[derive(Debug, Clone)]
pub struct TransferEntropyEstimator {
    k: usize,
    // We store discretized symbols for history windows
    x_history: Vec<usize>,
    y_history: Vec<usize>,
    // Counts: (y_next, y_past_key, x_past_key) → count
    joint_counts: std::collections::HashMap<(usize, u64, u64), usize>,
    y_past_counts: std::collections::HashMap<(usize, u64), usize>,
    y_cond_counts: std::collections::HashMap<u64, usize>,
    total: usize,
}

impl TransferEntropyEstimator {
    /// Create a new estimator with embedding dimension `k` (history length).
    pub fn new(k: usize) -> Self {
        assert!(k > 0, "embedding dimension k must be > 0");
        Self {
            k,
            x_history: Vec::new(),
            y_history: Vec::new(),
            joint_counts: std::collections::HashMap::new(),
            y_past_counts: std::collections::HashMap::new(),
            y_cond_counts: std::collections::HashMap::new(),
            total: 0,
        }
    }

    /// Encode a k-length history as a u64 key (each symbol in low bits).
    fn encode_history(history: &[usize], k: usize) -> u64 {
        let len = history.len();
        let start = len.saturating_sub(k);
        let mut key: u64 = 0;
        for i in start..len {
            key = key * 1000 + history[i] as u64; // works for symbols < 1000
        }
        key
    }

    /// Add a pair of simultaneous observations (xₜ, yₜ).
    pub fn observe(&mut self, x: usize, y: usize) {
        // We need k past observations before we can start counting
        self.x_history.push(x);
        self.y_history.push(y);

        let n = self.y_history.len();
        if n < self.k + 1 {
            return;
        }

        // The "next" y is at position n-1, the past is positions n-1-k .. n-2
        let y_next = self.y_history[n - 1];
        let y_past_key = Self::encode_history(&self.y_history[..n - 1], self.k);
        let x_past_key = Self::encode_history(&self.x_history[..n - 1], self.k);

        *self
            .joint_counts
            .entry((y_next, y_past_key, x_past_key))
            .or_insert(0) += 1;
        *self
            .y_past_counts
            .entry((y_next, y_past_key))
            .or_insert(0) += 1;
        *self
            .y_cond_counts
            .entry(y_past_key)
            .or_insert(0) += 1;
        self.total += 1;
    }

    /// Add a batch of (x, y) observation pairs.
    pub fn observe_batch(&mut self, pairs: &[(usize, usize)]) {
        for &(x, y) in pairs {
            self.observe(x, y);
        }
    }

    /// Compute the current transfer entropy estimate in bits.
    ///
    /// Returns 0.0 if insufficient data has been accumulated.
    pub fn transfer_entropy(&self) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        let total = self.total as f64;

        // H(Y_{t+1} | Y_past)
        let h_y_given_ypast: f64 = -self
            .y_past_counts
            .iter()
            .filter(|(&(_, ref yp), &c_yn)| {
                c_yn > 0 && *self.y_cond_counts.get(yp).unwrap_or(&0) > 0
            })
            .map(|(&(_, ref yp), &c_yn)| {
                let c_yp = *self.y_cond_counts.get(yp).unwrap_or(&0) as f64;
                let p = c_yn as f64 / total;
                let p_cond = c_yn as f64 / c_yp;
                p * p_cond.ln()
            })
            .sum::<f64>()
            / LN_2;

        // H(Y_{t+1} | Y_past, X_past)
        let h_y_given_ypast_xpast: f64 = -self
            .joint_counts
            .iter()
            .map(|(&(_, yp, xp), &c_ynxp)| {
                // We need P(y_next, y_past, x_past) and the conditional
                let p_joint = c_ynxp as f64 / total;
                // Count for (y_past, x_past)
                let c_ypxp: f64 = self
                    .joint_counts
                    .iter()
                    .filter(|((_, yp2, xp2), _)| *yp2 == yp && *xp2 == xp)
                    .map(|(_, &c)| c as f64)
                    .sum::<f64>();
                if c_ypxp < EPS {
                    return 0.0;
                }
                p_joint * (p_joint / (c_ypxp / total)).ln()
            })
            .sum::<f64>()
            / LN_2;

        h_y_given_ypast - h_y_given_ypast_xpast
    }

    /// Reset the estimator, clearing all accumulated data.
    pub fn reset(&mut self) {
        self.x_history.clear();
        self.y_history.clear();
        self.joint_counts.clear();
        self.y_past_counts.clear();
        self.y_cond_counts.clear();
        self.total = 0;
    }

    /// Number of valid observations accumulated.
    pub fn count(&self) -> usize {
        self.total
    }
}

// ─── Entropy Profile ─────────────────────────────────────────────────────────

/// A comprehensive entropy profile for a time series.
///
/// Provides:
/// - **Entropy rate**: Shannon entropy rate via block entropy
/// - **Permutation entropy**: Based on ordinal patterns
/// - **Sample entropy**: Measure of complexity/regularity
#[derive(Debug, Clone)]
pub struct EntropyProfile {
    /// Shannon entropy rate (bits per symbol).
    pub entropy_rate: f64,
    /// Permutation entropy (normalized to [0, 1]).
    pub permutation_entropy: f64,
    /// Sample entropy (lower = more regular).
    pub sample_entropy: f64,
}

impl EntropyProfile {
    /// Compute the full entropy profile for a continuous-valued time series.
    ///
    /// # Parameters
    /// - `series`: The time series data.
    /// - `embedding_dim`: Embedding dimension `d` for permutation entropy and sample entropy.
    ///   Typically 3–7.
    /// - `tolerance`: Tolerance `r` for sample entropy, as a fraction of the series standard
    ///   deviation. Typically 0.1–0.25.
    /// - `block_size`: Block size for entropy rate estimation via block entropy differences.
    pub fn compute(
        series: &[f64],
        embedding_dim: usize,
        tolerance: f64,
        block_size: usize,
    ) -> Self {
        assert!(series.len() > 2, "series must have > 2 elements");
        assert!(embedding_dim >= 2, "embedding_dim must be >= 2");
        assert!(block_size >= 1, "block_size must be >= 1");

        let entropy_rate = Self::entropy_rate(series, block_size);
        let permutation_entropy = Self::permutation_entropy(series, embedding_dim);
        let sample_entropy = Self::sample_entropy(series, embedding_dim, tolerance);

        Self {
            entropy_rate,
            permutation_entropy,
            sample_entropy,
        }
    }

    /// Estimate entropy rate as H(X_{t+1} | X_t) using block entropy differences.
    ///
    /// h = H(X_1...X_{k+1}) − H(X_1...X_k)
    fn entropy_rate(series: &[f64], block_size: usize) -> f64 {
        if series.len() <= block_size + 1 {
            return 0.0;
        }

        // Discretize into bins for counting
        let n_bins = 10;
        let min_val = series.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = series.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = max_val - min_val;
        if range < EPS {
            return 0.0;
        }

        let discretized: Vec<usize> = series
            .iter()
            .map(|&v| {
                let bin = ((v - min_val) / range * (n_bins as f64 - 1.0)).round() as usize;
                bin.min(n_bins - 1)
            })
            .collect();

        let h_k = block_entropy(&discretized, block_size);
        let h_k1 = block_entropy(&discretized, block_size + 1);
        h_k1 - h_k
    }

    /// Compute permutation entropy (normalized).
    ///
    /// Counts ordinal patterns of length `d` in the series and returns the
    /// normalized Shannon entropy of the pattern distribution.
    fn permutation_entropy(series: &[f64], d: usize) -> f64 {
        if series.len() <= d {
            return 0.0;
        }

        let mut pattern_counts: std::collections::HashMap<Vec<usize>, usize> =
            std::collections::HashMap::new();

        for i in 0..=series.len() - d {
            let window = &series[i..i + d];
            let pattern = ordinal_pattern(window);
            *pattern_counts.entry(pattern).or_insert(0) += 1;
        }

        let total = (series.len() - d + 1) as f64;
        let entropy: f64 = -pattern_counts
            .values()
            .map(|&c| {
                let p = c as f64 / total;
                p * (p.ln() / LN_2)
            })
            .sum::<f64>();

        // Normalize by log₂(d!)
        let max_entropy = (1..=d).fold(1.0_f64, |acc, i| acc * i as f64);
        let log2_fact = max_entropy.ln() / LN_2;
        if log2_fact < EPS {
            0.0
        } else {
            entropy / log2_fact
        }
    }

    /// Compute sample entropy.
    ///
    /// SampEn = −ln(A / B) where A = count of template matches at distance m+1,
    /// B = count of template matches at distance m.
    fn sample_entropy(series: &[f64], m: usize, tolerance: f64) -> f64 {
        let n = series.len();
        if n <= m + 1 {
            return 0.0;
        }

        let std_dev = {
            let mean = series.iter().sum::<f64>() / n as f64;
            let var = series.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / n as f64;
            var.sqrt()
        };
        if std_dev < EPS {
            return 0.0;
        }
        let r = tolerance * std_dev;

        let b = count_matches(series, m, r);
        let a = count_matches(series, m + 1, r);

        if b == 0 || a == 0 {
            return f64::INFINITY;
        }

        -(a as f64 / b as f64).ln()
    }
}

/// Compute block entropy for a discretized sequence.
fn block_entropy(series: &[usize], block_size: usize) -> f64 {
    if series.len() < block_size {
        return 0.0;
    }
    let mut counts: std::collections::HashMap<Vec<usize>, usize> =
        std::collections::HashMap::new();
    let n_blocks = series.len() - block_size + 1;
    for i in 0..n_blocks {
        let block = series[i..i + block_size].to_vec();
        *counts.entry(block).or_insert(0) += 1;
    }
    let total = n_blocks as f64;
    -counts
        .values()
        .map(|&c| {
            let p = c as f64 / total;
            p * (p.ln() / LN_2)
        })
        .sum::<f64>()
}

/// Get the ordinal pattern of a window (ranking of elements).
fn ordinal_pattern(window: &[f64]) -> Vec<usize> {
    let n = window.len();
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| window[a].partial_cmp(&window[b]).unwrap_or(std::cmp::Ordering::Equal));
    let mut rank = vec![0usize; n];
    for (r, &idx) in indices.iter().enumerate() {
        rank[idx] = r;
    }
    rank
}

/// Count the number of matching template pairs for sample entropy.
fn count_matches(series: &[f64], m: usize, r: f64) -> usize {
    let n = series.len();
    if n <= m {
        return 0;
    }
    let mut count = 0usize;
    for i in 0..=n - m {
        for j in (i + 1)..=n - m {
            let mut matches = true;
            for k in 0..m {
                if (series[i + k] - series[j + k]).abs() > r {
                    matches = false;
                    break;
                }
            }
            if matches {
                count += 1;
            }
        }
    }
    count
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_shannon_entropy_uniform() {
        // Uniform distribution over 4 outcomes: H = log₂(4) = 2 bits
        let dists = vec![vec![0.25, 0.25, 0.25, 0.25]];
        let ent = batch_shannon_entropy(&dists);
        assert_relative_eq!(ent[0], 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_shannon_entropy_deterministic() {
        // Degenerate distribution: H = 0
        let dists = vec![vec![1.0, 0.0, 0.0]];
        let ent = batch_shannon_entropy(&dists);
        assert_relative_eq!(ent[0], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_shannon_entropy_batch() {
        let dists = vec![
            vec![0.5, 0.5],           // H = 1 bit
            vec![0.25, 0.25, 0.25, 0.25], // H = 2 bits
            vec![1.0, 0.0],           // H = 0
        ];
        let ent = batch_shannon_entropy(&dists);
        assert_eq!(ent.len(), 3);
        assert_relative_eq!(ent[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(ent[1], 2.0, epsilon = 1e-10);
        assert_relative_eq!(ent[2], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_shannon_entropy_unnormalized() {
        // [1, 1] should give same as [0.5, 0.5]
        let dists = vec![vec![1.0, 1.0]];
        let ent = batch_shannon_entropy(&dists);
        assert_relative_eq!(ent[0], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_kl_divergence_same_distribution() {
        let pairs = vec![(vec![0.5, 0.5], vec![0.5, 0.5])];
        let kl = batch_kl_divergence(&pairs);
        assert_relative_eq!(kl[0], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_kl_divergence_batch() {
        let pairs = vec![
            (vec![0.5, 0.5], vec![0.5, 0.5]),
            (vec![1.0, 0.0], vec![0.5, 0.5]),
        ];
        let kl = batch_kl_divergence(&pairs);
        assert_relative_eq!(kl[0], 0.0, epsilon = 1e-10);
        assert!(kl[1] > 0.0);
    }

    #[test]
    fn test_js_divergence_same() {
        let pairs = vec![(vec![0.3, 0.7], vec![0.3, 0.7])];
        let js = batch_js_divergence(&pairs);
        assert_relative_eq!(js[0], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_js_divergence_symmetry() {
        let p = vec![0.4, 0.6];
        let q = vec![0.8, 0.2];
        let js_ab = batch_js_divergence(&[(p.clone(), q.clone())]);
        let js_ba = batch_js_divergence(&[(q, p)]);
        assert_relative_eq!(js_ab[0], js_ba[0], epsilon = 1e-10);
    }

    #[test]
    fn test_js_divergence_bounded() {
        // JS divergence with base-2 log is bounded by 1
        let pairs = vec![(vec![1.0, 0.0], vec![0.0, 1.0])];
        let js = batch_js_divergence(&pairs);
        assert!(js[0] <= 1.0 + 1e-10);
        assert!(js[0] > 0.0);
    }

    #[test]
    fn test_mutual_information_independent() {
        // Two independent variables should have MI ≈ 0
        let x = vec![0, 1, 0, 1, 0, 1, 0, 1, 0, 1];
        let y = vec![0, 0, 1, 1, 0, 0, 1, 1, 0, 0];
        let mi = mutual_information_matrix(&[x, y]);
        assert_eq!(mi.len(), 2);
        assert!(mi[0][1] >= 0.0);
    }

    #[test]
    fn test_mutual_information_identical() {
        // A variable should have MI with itself equal to its entropy
        let x = vec![0, 1, 0, 1, 0, 1, 0, 1];
        let mi = mutual_information_matrix(&[x.clone()]);
        assert!(mi[0][0] > 0.0);
    }

    #[test]
    fn test_mutual_information_perfect_correlation() {
        let x = vec![0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2];
        let y = x.clone();
        let mi = mutual_information_matrix(&[x, y]);
        // MI(X, X) should be H(X)
        assert_relative_eq!(mi[0][1], mi[0][0], epsilon = 1e-10);
    }

    #[test]
    fn test_transfer_entropy_independent() {
        // Independent series should have TE ≈ 0
        let mut te = TransferEntropyEstimator::new(2);
        for i in 0..100 {
            te.observe(i % 3, (i * 7 + 1) % 5);
        }
        let val = te.transfer_entropy();
        // Should be small for independent processes
        assert!(val >= 0.0);
    }

    #[test]
    fn test_transfer_entropy_reset() {
        let mut te = TransferEntropyEstimator::new(2);
        te.observe_batch(&[(0, 1), (1, 2), (2, 3), (3, 0), (0, 1)]);
        assert!(te.count() > 0);
        te.reset();
        assert_eq!(te.count(), 0);
    }

    #[test]
    fn test_entropy_profile_constant_series() {
        let series = vec![5.0; 100];
        let profile = EntropyProfile::compute(&series, 3, 0.2, 3);
        // Constant series: entropy rate ≈ 0, permutation entropy ≈ 0
        assert_relative_eq!(profile.entropy_rate, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_permutation_entropy_fully_random() {
        // Shuffled values → many different ordinal patterns → high permutation entropy
        let series: Vec<f64> = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0, 3.0,
                                     5.0, 8.0, 9.0, 7.0, 9.0, 3.0, 2.0, 3.0, 8.0, 4.0,
                                     6.0, 2.0, 6.0, 4.0, 3.0, 3.0, 8.0, 3.0, 2.0, 7.0,
                                     9.0, 5.0, 0.0, 2.0, 8.0, 8.0, 5.0, 7.0, 0.0, 9.0];
        let pe = EntropyProfile::permutation_entropy(&series, 3);
        assert!(pe > 0.9, "permutation entropy should be high for varied series, got {pe}");
    }

    #[test]
    fn test_sample_entropy_regular_vs_noisy() {
        use rand::Rng;
        // Regular series should have lower sample entropy than noisy
        let regular: Vec<f64> = (0..200).map(|i| (i as f64 * 0.1).sin()).collect();
        let mut rng = rand::thread_rng();
        let noisy: Vec<f64> = (0..200).map(|_| rng.gen::<f64>()).collect();

        let se_reg = EntropyProfile::sample_entropy(&regular, 2, 0.2);
        let se_noisy = EntropyProfile::sample_entropy(&noisy, 2, 0.2);
        assert!(
            se_reg < se_noisy,
            "regular series should have lower sample entropy: got regular={se_reg}, noisy={se_noisy}"
        );
    }
}
