module SparseColumnPivotedQRSparseMatricesCSRExt

using SparseColumnPivotedQR
using LinearAlgebra
using SparseArrays
using SparseMatricesCSR
using PrecompileTools

import SparseColumnPivotedQR: csr_qr, csr_analyze, csr_factor, csr_refactor!,
    CSRQRSymbolic, CSRQRFactorization

# Convert a `SparseMatrixCSR` to the `SparseMatrixCSC` the core operates on.
# This is the same CSR -> CSC conversion the kernel performed internally before
# the CSC-native refactor; it now lives here so the core never depends on
# `SparseMatricesCSR`.
@inline _to_csc(A::SparseMatrixCSR) = SparseMatrixCSC(A)

function csr_analyze(A::SparseMatrixCSR; ordering::Symbol = :default)
    return csr_analyze(_to_csc(A); ordering = ordering)
end

function csr_factor(A::SparseMatrixCSR, sym::CSRQRSymbolic; kwargs...)
    return csr_factor(_to_csc(A), sym; kwargs...)
end

function csr_qr(A::SparseMatrixCSR; kwargs...)
    return csr_qr(_to_csc(A); kwargs...)
end

function csr_refactor!(F::CSRQRFactorization, A::SparseMatrixCSR; kwargs...)
    return csr_refactor!(F, _to_csc(A); kwargs...)
end

# Keep the CSR entry points specialized in the package image. Mirrors the
# core workload, but on `SparseMatrixCSR` inputs through the conversion path.
@setup_workload begin
    @compile_workload begin
        for T in (Float64, Float32, ComplexF64, ComplexF32)
            for Ti in (Int32, Int64)
                rows = Ti[1, 2, 3, 4, 5, 6, 1, 2, 3, 4]
                cols = Ti[1, 2, 3, 4, 5, 6, 2, 3, 4, 5]
                vals = T[4, 4, 4, 4, 4, 4, 1, 1, 1, 1]
                A = sparsecsr(rows, cols, vals, 6, 6)
                b = ones(T, 6)

                F = csr_qr(A; ordering = :natural)
                F \ b
                rank(F)

                sym = csr_analyze(A; ordering = :natural)
                G = csr_factor(A, sym)
                csr_refactor!(G, A)
                G \ b

                drows = Ti[1, 2, 3, 4, 5, 1, 2, 3, 4, 6]
                dcols = Ti[1, 2, 3, 4, 5, 2, 3, 4, 5, 1]
                dvals = T[4, 4, 4, 4, 4, 1, 1, 1, 1, 4]
                Ad = sparsecsr(drows, dcols, dvals, 6, 6)
                Fd = csr_qr(Ad; ordering = :natural)
                Fd \ b
                rank(Fd)
            end
        end
    end
end

end # module
