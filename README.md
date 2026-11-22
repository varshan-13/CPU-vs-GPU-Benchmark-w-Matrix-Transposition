# CPU vs GPU Matrix Transposition Benchmark

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/varshan-13/cpu-gpu-matrix-transpose/blob/main/Matrix_Transposition_CPU_vs_GPU_Benchmark_final.ipynb)

A performance study of matrix transposition on CPU and GPU, written in Python with Numba and CUDA and run on Google Colab (Tesla T4).

The project builds a CPU version, a naive GPU version and an optimized shared-memory GPU version. It then measures how memory access patterns, block configuration, matrix size and data transfer affect performance. The results are compared with NVIDIA's published transpose benchmark.

> Computer Architecture and Organization (CAO) course project.

## Why transposition?

Transposing a matrix (`B[j][i] = A[i][j]`) does no arithmetic. Every element is read once and written once, so its speed depends entirely on the memory system. That makes it a simple, clean way to compare how CPUs and GPUs handle memory: caches, coalescing, shared memory and bandwidth.

## What's inside

| # | Section | What it does |
|---|---|---|
| 1 | Introduction and objectives | Project goals |
| 2 | Environment and hardware | Detects CPU, RAM, GPU, VRAM and shared memory |
| 3 | Background | CPU vs GPU architecture and memory hierarchy |
| 4 | CPU implementation | Transpose with Numba `@njit` |
| 5 | Naive GPU implementation | One CUDA thread per element, checked against NumPy |
| 6 | Initial benchmark | CPU vs naive GPU at 4096 × 4096 |
| 7 | Optimized GPU | Shared-memory tiled transpose, with and without padding |
| 8 | Block and grid configuration | Tests 11 thread-block shapes |
| 9 | Matrix size benchmark | 256 × 256 up to 8192 × 8192 |
| 10 | Comparison with published results | Against NVIDIA's transpose benchmark and a CPU article |
| 11 | Full-capacity test | Largest matrix the session can hold, plus memory use and RAM ↔ VRAM transfer |
| 12 | Conclusions | Summary of results, future work and CSV export |

## Implementations

| Implementation | Description |
|---|---|
| **CPU (1 core)** | Simple nested loop compiled with Numba `@njit` |
| **CPU (all cores)** | Parallel version using `@njit(parallel=True)` and `prange` |
| **GPU naive** | Each thread copies one element. Reads are scattered (uncoalesced), writes are coalesced |
| **GPU tiled (bank conflicts)** | Each block loads a 32 × 32 tile into shared memory, so all VRAM reads and writes are coalesced |
| **GPU tiled (padded)** | Same, with a 32 × 33 tile to remove shared-memory bank conflicts. This is the final optimized kernel |
| **GPU copy** | Plain copy with perfect access. Used as the practical speed limit for a transpose |

## Measurements

- **Execution time.** GPU kernels are timed as the average of repeated launches after a warm-up.
- **Effective bandwidth.** `2 × matrix bytes / time`, the measure NVIDIA uses for transpose.
- **% of copy speed.** Lets results from different GPUs be compared fairly.
- **Speedup.** Both kernel-only and end-to-end (including RAM ↔ VRAM transfer).
- **Memory use and transfer speed.** Measured at the largest matrix size.

All results are checked for correctness against NumPy's `A.T`.

## Results

Run the notebook to produce the full results. Section 12.1 prints a summary of the key numbers, and every table is saved to `results/` as CSV.

**Reference: NVIDIA published results** (1024 × 1024, % of copy bandwidth)

| Kernel | Tesla M2050 | Tesla K20c |
|---|---|---|
| Naive | 18% | 41% |
| Coalesced (shared memory) | 49% | 72% |
| No bank conflicts (padded) | 95% | 106% |

**This project** (Tesla T4)

| Kernel | Time at 4096 × 4096 | Effective bandwidth | % of copy |
|---|---|---|---|
| Naive | TBD | TBD | TBD |
| Tiled (bank conflicts) | TBD | TBD | TBD |
| Tiled (padded) | TBD | TBD | TBD |

## How to run

1. Click **Open in Colab** above, or upload the notebook to [Google Colab](https://colab.research.google.com).
2. Go to **Runtime → Change runtime type** and select **T4 GPU**.
3. Click **Runtime → Run all**.
4. Download `transpose_results.zip` from Colab's **Files** panel for the CSV results.

The full run takes a few minutes. Most of the time goes to the single-core CPU tests in Sections 9 and 11.

> The size of the full-capacity test in Section 11 is set automatically from the free RAM and VRAM in your Colab session (about 19,200 × 19,200 in testing). RAM, not the GPU, is usually the limit.

## Requirements

Everything is pre-installed on Google Colab:

- Python 3
- NumPy, Pandas, Matplotlib, psutil
- Numba with CUDA support
- An NVIDIA GPU (tested on Tesla T4)

## Repository structure

```
├── Matrix_Transposition_CPU_vs_GPU_Benchmark_final.ipynb   # main notebook
├── results/                                                # CSV output from a full run
└── README.md
```

## Key findings

- Transposition is **memory-bound**, so effective bandwidth describes its performance better than raw time.
- **Memory access pattern matters more than the number of threads.** Staging tiles in shared memory makes every VRAM access coalesced.
- **Padding the shared tile** (32 × 33) removes bank conflicts.
- **Data transfer can outweigh the computation.** Moving the matrix over PCIe takes much longer than transposing it inside VRAM.
- **A speedup depends on the baseline.** The CPU baseline here is a naive loop, and transfers change the GPU speedup a lot.

## Future work

- Cache-blocked, multi-core CPU baseline for a fairer comparison
- Pinned memory and CUDA streams to speed up transfers
- Profiling with NVIDIA Nsight Compute
- Rectangular matrices and other data types (`float16`, `float64`)
- Newer GPUs (L4, A100) and library comparison (CuPy, PyTorch)

## References

- Mark Harris, [An Efficient Matrix Transpose in CUDA C/C++](https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/), NVIDIA Developer Blog
- [Why a Mathematically Free Matrix Transpose Is 10× Slower on CPUs](https://www.besthub.dev/articles/why-a-mathematically-free-matrix-transpose-is-10-slower-on-cpus-73363c8650d8)
- NVIDIA Tesla T4 product brief
- [Numba CUDA documentation](https://numba.readthedocs.io/en/stable/cuda/index.html)

## Author

**Logavarshan Sakthivel** · [GitHub](https://github.com/varshan-13)
