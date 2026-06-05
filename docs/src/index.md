# SparseColumnPivotedQR.jl: rank-revealing sparse Householder QR

SparseColumnPivotedQR.jl is a component of the
[SciML](https://sciml.ai/) ecosystem providing a pure-Julia,
rank-revealing, column-pivoted Householder QR factorization that operates
directly on
[`SparseMatrixCSC`](https://docs.julialang.org/en/v1/stdlib/SparseArrays/)
sparse matrices.

The package targets the same "small-to-medium sparse" niche as KLU does for
LU — low symbolic-phase overhead, no BLAS-3 / multifrontal machinery — while
preserving the rank-revealing guarantees of LAPACK's column-pivoted QR. It
follows KLU's `analyze` / `factor` / `refactor!` split so the symbolic phase
can be reused across calls with the same sparsity pattern.

## Installation

```julia
using Pkg
Pkg.add("SparseColumnPivotedQR")
```

## Quick start

```julia
using SparseArrays, SparseColumnPivotedQR
using AMD  # enables the recommended AMD column ordering

# A 5×5 sparse matrix and a right-hand side.
A = sparse([1.0  0   2   0   0;
            0    3   0   0   1;
            4    0   5   0   0;
            0    0   0   6   0;
            0    7   0   0   8])
b = [1.0, 2.0, 3.0, 4.0, 5.0]

# One-shot factor + solve.
F = csr_qr(A)
x = F \ b

# Rank and dimensions.
rank(F), size(F)
```

## Column ordering

`csr_qr` / `csr_analyze` accept an `ordering` keyword. Available choices:

| ordering    | meaning                                                              |
|-------------|----------------------------------------------------------------------|
| `:default`  | **(default)** `:amd` when the AMD.jl extension is loaded, else `:natural` |
| `:natural`  | identity column ordering (opt-in; usually ~2× slower than `:amd` on dense-fill matrices) |
| `:amd`      | AMD on `AᵀA` via the `AMD.jl` weak dep (`using AMD` to enable)        |
| `:colamd`   | currently an alias for `:amd`                                        |
| `:adaptive` | build both `:amd` and `:natural` symbolics, keep the shallower-etree one (~30 µs overhead) |

`:default` is the recommended choice: it gives CXSparse-class
performance out of the box whenever `using AMD` has been executed in the
session (directly or via a transitive dep), and falls back to `:natural`
without warning otherwise. Use `ordering = :natural` to opt out
explicitly for matrices that are already well-ordered (block diagonal,
banded, etc.).

```julia
F1 = csr_qr(A)                          # = :default (= :amd if AMD loaded)
F2 = csr_qr(A; ordering = :natural)     # opt-in to the chain etree
F3 = csr_qr(A; ordering = :amd)         # explicit :amd; errors if AMD.jl not loaded
F4 = csr_qr(A; ordering = :adaptive)    # build both, keep the shallower etree
```

## Approximate factorization

`csr_qr` accepts a `drop_tol::Real` keyword (default `0`). When
`drop_tol > 0`, entries of each Householder vector `V[:, k]` with
`|v_i| <= drop_tol * ‖v‖` are discarded after the reflector is built;
`β_k` is rescaled for the truncated vector so `H̃ = I - β̃ ṽ ṽᵀ` remains
a proper Householder. The result is an approximate QR — the residual
`‖A x - b‖` grows with `drop_tol` — but the apply step walks fewer
entries on every subsequent call. Useful when you can tolerate a larger
residual to shrink the factorized form.

```julia
F_exact = csr_qr(A)                    # drop_tol = 0
F_approx = csr_qr(A; drop_tol = 1e-8)  # smaller V, larger ‖A x - b‖
```

## Rank-deficient inputs

The factorization is rank-revealing: on rank-deficient inputs `F` reports
a `rank(F)` smaller than `min(size(F)...)`, and `F \ b` returns a finite
least-squares solution whose residual matches the SVD pseudoinverse minimum
to floating-point precision.

```julia
A_singular = sparse([1.0 2.0; 0.5 1.0; 2.0 4.0])  # rank 1
b = [1.0, 1.0, 1.0]
F = csr_qr(A_singular)
rank(F)              # → 1
norm(A_singular * (F \ b) - b)   # matches the SVD minimum residual
```
