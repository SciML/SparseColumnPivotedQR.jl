module SparseColumnPivotedQRAMDExt

using SparseColumnPivotedQR
using SparseArrays
using AMD

# Flag the host module so `:default` ordering resolves to `:amd` and so the
# `:amd` opt-in doesn't error. Set on extension load, never cleared.
function __init__()
    SparseColumnPivotedQR._AMD_EXT_LOADED[] = true
    return nothing
end

# Override the AMD column ordering hook in SparseColumnPivotedQR.
# Builds a CSC SparseMatrixCSC from the (rowptr, colval) CSR pattern and asks
# AMD.colamd for an unsymmetric column ordering. Falls back to natural if AMD
# fails for any reason (zero rows, ill-formed pattern, etc).
function SparseColumnPivotedQR._amd_colperm(rowptr::Vector{Int}, colval::Vector{Int},
                                            m::Int, n::Int)
    # Build a SparseMatrixCSC pattern with dummy values; AMD only needs the pattern.
    # We have the CSR pattern, so convert to CSC.
    nnz_total = length(colval)
    colcounts = zeros(Int, n)
    @inbounds for p in 1:nnz_total
        colcounts[colval[p]] += 1
    end
    colptr = Vector{Int}(undef, n + 1)
    colptr[1] = 1
    @inbounds for j in 1:n
        colptr[j + 1] = colptr[j] + colcounts[j]
    end
    rowidx = Vector{Int}(undef, nnz_total)
    nzval = ones(Float64, nnz_total)
    work = copy(colptr)
    @inbounds for i in 1:m
        r1 = rowptr[i]; r2 = rowptr[i + 1] - 1
        for p in r1:r2
            j = colval[p]
            rowidx[work[j]] = i
            work[j] += 1
        end
    end
    A = SparseMatrixCSC(m, n, colptr, rowidx, nzval)
    try
        p = AMD.colamd(A)
        # AMD.colamd returns a Vector of Int (or Int32); normalize to Int.
        return Int.(p)
    catch
        return collect(1:n)
    end
end

end # module
