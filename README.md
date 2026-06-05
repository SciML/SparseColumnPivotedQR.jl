# SparseColumnPivotedQR.jl

[![Join the chat at https://julialang.zulipchat.com #sciml-bridged](https://img.shields.io/static/v1?label=Zulip&message=chat&color=9558b2&labelColor=389826)](https://julialang.zulipchat.com/#narrow/stream/279055-sciml-bridged)
[![Global Docs](https://img.shields.io/badge/docs-SciML-blue.svg)](https://docs.sciml.ai/SparseColumnPivotedQR/stable/)

[![codecov](https://codecov.io/gh/SciML/SparseColumnPivotedQR.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/SciML/SparseColumnPivotedQR.jl)
[![Build Status](https://github.com/SciML/SparseColumnPivotedQR.jl/workflows/Tests/badge.svg)](https://github.com/SciML/SparseColumnPivotedQR.jl/actions?query=workflow%3ATests)

[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor%27s%20Guide-blueviolet)](https://github.com/SciML/ColPrac)
[![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle)

A pure-Julia, rank-revealing, column-pivoted Householder QR factorization that
operates directly on
[`SparseMatrixCSC`](https://docs.julialang.org/en/v1/stdlib/SparseArrays/)
sparse matrices. Targets the same "small-to-medium sparse" niche as KLU does for LU
— low symbolic-phase overhead, no BLAS-3 / multifrontal machinery — while preserving
the rank-revealing guarantees of LAPACK's column-pivoted QR.

## Tutorials and Documentation

For information on using the package,
[see the stable documentation](https://docs.sciml.ai/SparseColumnPivotedQR/stable/). Use the
[in-development documentation](https://docs.sciml.ai/SparseColumnPivotedQR/dev/) for the
version of the documentation, which contains the unreleased features.

## Usage

```julia
using Pkg
Pkg.add("SparseColumnPivotedQR")

using SparseArrays, SparseColumnPivotedQR

A = sparse([1.0  0   2   0   0;
            0    3   0   0   1;
            4    0   5   0   0;
            0    0   0   6   0;
            0    7   0   0   8])
b = [1.0, 2.0, 3.0, 4.0, 5.0]

F = csr_qr(A)
x = F \ b
```
