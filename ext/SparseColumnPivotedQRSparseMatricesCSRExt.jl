module SparseColumnPivotedQRSparseMatricesCSRExt

using SparseColumnPivotedQR: SparseColumnPivotedQR
using SparseArrays: SparseMatrixCSC
using SparseMatricesCSR: SparseMatrixCSR, sparsecsr
using LinearAlgebra: rank
using PrecompileTools: PrecompileTools, @setup_workload, @compile_workload

import SparseColumnPivotedQR: scpqr, scpqr_analyze, scpqr_factor, scpqr_refactor!,
    SparseColumnPivotedQRSymbolic, SparseColumnPivotedQRFactorization

# Convert a `SparseMatrixCSR` to the `SparseMatrixCSC` the core operates on, so
# the core never depends on `SparseMatricesCSR`.
@inline _to_csc(A::SparseMatrixCSR) = SparseMatrixCSC(A)

function scpqr_analyze(A::SparseMatrixCSR; ordering::Symbol = :default)
    return scpqr_analyze(_to_csc(A); ordering = ordering)
end

function scpqr_factor(
        A::SparseMatrixCSR, sym::SparseColumnPivotedQRSymbolic; kwargs...
    )
    return scpqr_factor(_to_csc(A), sym; kwargs...)
end

function scpqr(A::SparseMatrixCSR; kwargs...)
    return scpqr(_to_csc(A); kwargs...)
end

function scpqr_refactor!(
        F::SparseColumnPivotedQRFactorization, A::SparseMatrixCSR; kwargs...
    )
    return scpqr_refactor!(F, _to_csc(A); kwargs...)
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

                F = scpqr(A; ordering = :natural)
                F \ b
                rank(F)

                sym = scpqr_analyze(A; ordering = :natural)
                G = scpqr_factor(A, sym)
                scpqr_refactor!(G, A)
                G \ b

                drows = Ti[1, 2, 3, 4, 5, 1, 2, 3, 4, 6]
                dcols = Ti[1, 2, 3, 4, 5, 2, 3, 4, 5, 1]
                dvals = T[4, 4, 4, 4, 4, 1, 1, 1, 1, 4]
                Ad = sparsecsr(drows, dcols, dvals, 6, 6)
                Fd = scpqr(Ad; ordering = :natural)
                Fd \ b
                rank(Fd)
            end
        end
    end
end

end # module
