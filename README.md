# entropy-gpu

**Batch information-theoretic computation library** providing Shannon entropy, KL divergence, Jensen-Shannon divergence, mutual information matrices, transfer entropy, and entropy profiles (permutation entropy, sample entropy) — designed for parallel batch processing with `rayon` and trivial GPU portability.

## Why It Matters

Information-theoretic measures are the backbone of modern data analysis: feature selection (mutual information), causal inference (transfer entropy), anomaly detection (entropy profile shifts), and model comparison (KL divergence between predicted and true distributions). However, computing these measures for large numbers of distributions or long time series is computationally expensive.

This library addresses three practical needs:

1. **Batch computation**: Process N distributions or N distribution pairs in parallel, not one at a time.
2. **GPU portability**: All operations are expressed as element-wise or reduction operations on flat `Vec<f64>` buffers — directly mappable to GPU kernels (CUDA, ROCm, Metal).
3. **Comprehensive profile**: Beyond simple Shannon entropy, the `EntropyProfile` struct computes permutation entropy (ordinal pattern complexity) and sample entropy (regularity measure) for time-series analysis.

## How It Works

### Batch Shannon Entropy

$$H(X) = -\sum_{i} p_i \log_2 p_i$$

Distributions are normalized internally (Σ p_i need not equal 1). Zero-probability entries are skipped using the limit `lim_{p→0} p × log(p) = 0`. The batch version processes N distributions in parallel via `rayon::par_iter`.

**Complexity**: O(n × k) where n = number of distributions, k = average distribution length. Parallel speedup: up to min(n, num_cores)×.

### Batch KL Divergence

$$D_{KL}(P \| Q) = \sum_{i} p_i \log_2 \frac{p_i}{q_i}$$

Entries where Q ≈ 0 are skipped (convention: treating the distribution as having support only where Q > 0). Both P and Q are normalized internally.

### Jensen-Shannon Divergence

$$JS(P \| Q) = \frac{1}{2} D_{KL}(P \| M) + \frac{1}{2} D_{KL}(Q \| M), \quad M = \frac{P + Q}{2}$$

JS divergence is symmetric (unlike KL), always finite, and bounded in [0, 1] for base-2 logarithm. It is the square of a metric (JS distance).

### Mutual Information Matrix

$$I(X_i; X_j) = \sum_{x,y} p(x,y) \log_2 \frac{p(x,y)}{p(x) \cdot p(y)}$$

Computes the full n × n symmetric MI matrix for n discrete variables. Only the upper triangle is computed (in parallel), then mirrored. The diagonal is the Shannon entropy H(X_i).

**Complexity**: O(n² × m) where n = number of variables, m = number of observations. The n² factor comes from computing all pairs — this is the main bottleneck for large variable counts.

### Transfer Entropy

$$TE_{X \to Y} = H(Y_{t+1} | Y_t^{(k)}) - H(Y_{t+1} | Y_t^{(k)}, X_t^{(k)})$$

Measures directed information flow from process X to process Y, using a k-length history embedding. The estimator uses hash maps over (y_next, y_past_key, x_past_key) tuples, where history windows are encoded as `u64` keys.

**Properties**: TE ≥ 0 (by the data processing inequality). TE = 0 for independent processes. TE is asymmetric: TE(X→Y) ≠ TE(Y→X) in general.

### Entropy Profile

Three measures for time-series complexity:

1. **Entropy rate** (block entropy difference): `h = H(X₁...X_{k+1}) − H(X₁...X_k)` — measures predictability of the next symbol given k-length history.

2. **Permutation entropy** (Bandt-Pompe method): Ordinal patterns of length d are counted, and the normalized Shannon entropy of the pattern distribution is computed:

   $$PE = \frac{H(\text{pattern distribution})}{\log_2(d!)}$$

   PE ∈ [0, 1]. PE ≈ 0 for periodic series, PE ≈ 1 for white noise.

3. **Sample entropy** (Richman-Moorman method):

   $$SampEn = -\ln\left(\frac{A}{B}\right)$$

   Where A = matching template pairs at distance m+1, B = matching pairs at distance m. Lower values indicate more regular/predictable series.

**Complexity of sample entropy**: O(n²) — all template pairs are compared. This is the most expensive operation and the prime candidate for GPU acceleration.

### GPU Portability

All functions operate on flat `Vec<f64>` / `&[f64]` slices and perform only element-wise arithmetic and reductions. Porting to CUDA/OpenCL requires:
1. Replace `rayon::par_iter` with kernel launches
2. Replace `HashMap` (in TE estimator) with hash table on device
3. Shared-memory tiling for MI matrix computation

## Quick Start

```rust
use entropy_gpu::{batch_shannon_entropy, batch_kl_divergence, mutual_information_matrix};

// Shannon entropy for 3 distributions
let dists = vec![
    vec![0.5, 0.5],                     // H = 1 bit
    vec![0.25, 0.25, 0.25, 0.25],       // H = 2 bits
    vec![1.0, 0.0],                     // H = 0
];
let entropies = batch_shannon_entropy(&dists);

// Mutual information matrix
let x = vec![0, 1, 0, 1, 0, 1, 0, 1];
let y = vec![0, 0, 1, 1, 0, 0, 1, 1];
let mi = mutual_information_matrix(&[x, y]);
```

## API

### Batch Functions
- `batch_shannon_entropy(&[Vec<f64>]) -> Vec<f64>` — H(X) for N distributions
- `batch_kl_divergence(&[(Vec<f64>, Vec<f64>)]) -> Vec<f64>` — D_KL for N pairs
- `batch_js_divergence(&[(Vec<f64>, Vec<f64>)]) -> Vec<f64>` — JS divergence, symmetric, bounded [0,1]
- `mutual_information_matrix(&[Vec<usize>]) -> Vec<Vec<f64>>` — n×n MI matrix

### Transfer Entropy
- `TransferEntropyEstimator::new(k)` — Create with embedding dimension k
- `.observe(x: usize, y: usize)` — Add observation pair
- `.transfer_entropy() -> f64` — Current TE estimate in bits

### Entropy Profile
- `EntropyProfile::compute(series, embedding_dim, tolerance, block_size) -> EntropyProfile`
- Fields: `entropy_rate`, `permutation_entropy`, `sample_entropy`

## Architecture Notes

This crate provides the information-theoretic measurement layer for the SuperInstance stack. The γ + η = C conservation link connects here:

- **γ** (gamma) = measured entropy of the system's active state distribution
- **η** (eta) = KL divergence between expected and observed distributions (the "leakage")
- **C** (constant) = maximum entropy H_max = log₂(n) for n possible states

If γ + η > C, the system has more uncertainty than physically possible — a sign of measurement error or state corruption.

See the full architecture: [ARCHITECTURE.md](https://github.com/SuperInstance/SuperInstance/blob/main/ARCHITECTURE.md)

## References

1. Cover, T.M. & Thomas, J.A. (2006). *Elements of Information Theory,* 2nd ed. Wiley.
2. Schreiber, T. (2000). "Measuring Information Transfer." *Physical Review Letters, 85(2).* — Original transfer entropy paper.
3. Bandt, C. & Pompe, B. (2002). "Permutation Entropy: A Natural Complexity Measure for Time Series." *Physical Review Letters, 88(17).*
4. Richman, J.S. & Moorman, J.R. (2000). "Physiological Time-Series Analysis Using Approximate Entropy and Sample Entropy." *Am. J. Physiology, 278(6).*
5. Lin, J. (1991). "Divergence Measures Based on the Shannon Entropy." *IEEE Trans. Information Theory, 37(1).* — JS divergence properties.

## License

MIT
