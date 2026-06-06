# entropy-gpu

GPU-ready batch entropy computation in pure Rust.

This crate provides **batch-vectorized** information-theoretic measures designed with a data layout that maps trivially to GPU kernels. All batch functions use **rayon** for CPU parallelism today and can be ported to CUDA/OpenCL with minimal changes (flat arrays of `f64`, no heap allocation in hot paths).

## Features

| Function | Description |
|---|---|
| `batch_shannon_entropy` | Shannon entropy H(X) for N distributions |
| `batch_kl_divergence` | KL divergence D_KL(P‖Q) for N pairs |
| `batch_js_divergence` | Jensen-Shannon divergence for N pairs (symmetric, bounded) |
| `mutual_information_matrix` | Full MI matrix for M discrete variables |
| `TransferEntropyEstimator` | Streaming transfer entropy TE(X→Y) with configurable embedding dimension |
| `EntropyProfile` | Entropy rate + permutation entropy + sample entropy for time series |

## Usage

```toml
[dependencies]
entropy-gpu = "0.1"
```

```rust
use entropy_gpu::*;

// Batch Shannon entropy
let distributions = vec![
    vec![0.25, 0.25, 0.25, 0.25],  // uniform → 2 bits
    vec![1.0, 0.0],                 // deterministic → 0 bits
];
let entropies = batch_shannon_entropy(&distributions);

// Batch KL divergence
let pairs = vec![
    (vec![0.5, 0.5], vec![0.5, 0.5]),  // same → 0
    (vec![1.0, 0.0], vec![0.5, 0.5]),  // divergent → > 0
];
let kl = batch_kl_divergence(&pairs);

// Mutual information matrix
let observations = vec![
    vec![0, 1, 0, 1, 0, 1],  // variable X
    vec![0, 0, 1, 1, 0, 0],  // variable Y
];
let mi = mutual_information_matrix(&observations);

// Transfer entropy (streaming)
let mut te = TransferEntropyEstimator::new(2);
te.observe_batch(&[(0, 1), (1, 2), (2, 0), (0, 1), (1, 2)]);
println!("TE = {} bits", te.transfer_entropy());

// Entropy profile for a time series
let series: Vec<f64> = (0..1000).map(|i| (i as f64 * 0.01).sin()).collect();
let profile = EntropyProfile::compute(&series, 3, 0.2, 5);
println!("Entropy rate: {} bits", profile.entropy_rate);
println!("Permutation entropy: {}", profile.permutation_entropy);
println!("Sample entropy: {}", profile.sample_entropy);
```

## Design Principles

1. **Batch-first**: All functions process arrays of inputs, enabling SIMD/GPU vectorization.
2. **Flat data, no GPU dependency**: Pure Rust data structures that can be copied to GPU memory without transformation.
3. **Rayon parallelism**: CPU parallelism out of the box via `par_iter()`.
4. **Numerically stable**: Avoids log(0) with epsilon guards, normalizes internally.

## Testing

```bash
cargo test    # 17 tests
```

## License

MIT
