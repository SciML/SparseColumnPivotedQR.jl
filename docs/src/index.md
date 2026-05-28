# SparseColumnPivotedQR.jl: rank-revealing sparse Householder QR

SparseColumnPivotedQR.jl is a component of the
[SciML](https://sciml.ai/) ecosystem providing a pure-Julia,
rank-revealing, column-pivoted Householder QR factorization that operates
directly on
[`SparseMatrixCSR`](https://github.com/JuliaSmoothOptimizers/SparseMatricesCSR.jl)
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
using SparseArrays, SparseMatricesCSR, SparseColumnPivotedQR

# A 5×5 sparse matrix and a right-hand side.
A_csc = sparse([1.0  0   2   0   0;
                0    3   0   0   1;
                4    0   5   0   0;
                0    0   0   6   0;
                0    7   0   0   8])
A = SparseMatrixCSR(A_csc)
b = [1.0, 2.0, 3.0, 4.0, 5.0]

# One-shot factor + solve.
F = csr_qr(A)
x = F \ b

# Rank and dimensions.
rank(F), size(F)
```

## Rank-deficient inputs

The factorization is rank-revealing: on rank-deficient inputs `F` reports
a `rank(F)` smaller than `min(size(F)...)`, and `F \ b` returns a finite
least-squares solution whose residual matches the SVD pseudoinverse minimum
to floating-point precision.

```julia
A_singular = SparseMatrixCSR(sparse([1.0 2.0; 0.5 1.0; 2.0 4.0]))  # rank 1
b = [1.0, 1.0, 1.0]
F = csr_qr(A_singular)
rank(F)              # → 1
norm(A_singular * (F \ b) - b)   # matches the SVD minimum residual
```
