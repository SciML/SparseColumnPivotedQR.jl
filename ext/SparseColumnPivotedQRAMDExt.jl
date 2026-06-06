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
#
# `colamd_l` needs only the structural pattern (no numeric values), in 0-based
# CSC layout: `p` = column pointers (length n+1) and a workspace array whose
# leading `nnz` entries are the row indices in column-major order, padded out
# to `colamd_l_recommended(nnz, m, n)` (colamd uses the slack in place).
#
# We have the CSR pattern (rowptr, colval). Rather than materialize a full
# `SparseMatrixCSC` with a dummy `nzval` vector (which `colamd` never reads),
# we scatter the CSR pattern directly into colamd's workspace in CSC order.
# This avoids the `ones(Float64, nnz)` value buffer, the `Int.(p)` output copy,
# and AMD.jl's internal `p .- 1` / `workspace .- 1` temporaries.
#
# Falls back to natural ordering if AMD reports failure for any reason.
function SparseColumnPivotedQR._amd_colperm(
        rowptr::Vector{Int}, colval::Vector{Int},
        m::Int, n::Int
    )
    SS = AMD.SS_Int
    nnz_total = length(colval)

    # 0-based CSC column pointers in `p` (length n+1).
    p = Vector{SS}(undef, n + 1)
    @inbounds for j in 1:(n + 1)
        p[j] = zero(SS)
    end
    @inbounds for q in 1:nnz_total
        p[colval[q] + 1] += one(SS)   # count per column (shifted by one)
    end
    @inbounds for j in 1:n
        p[j + 1] += p[j]              # prefix sum -> 0-based colptr
    end

    # Workspace: leading `nnz` entries are 0-based row indices in CSC order,
    # padded to the recommended length. colamd overwrites/uses the tail.
    len = AMD.colamd_l_recommended(SS(nnz_total), SS(m), SS(n))
    workspace = Vector{SS}(undef, len)
    # Running write cursor per column (CSC starts), kept in a small scratch.
    cursor = Vector{Int}(undef, n)
    @inbounds for j in 1:n
        cursor[j] = p[j]             # 0-based start offset of column j
    end
    @inbounds for i in 1:m
        r1 = rowptr[i]; r2 = rowptr[i + 1] - 1
        for q in r1:r2
            j = colval[q]
            workspace[cursor[j] + 1] = SS(i - 1)   # 0-based row index
            cursor[j] += 1
        end
    end
    @inbounds for q in (nnz_total + 1):len
        workspace[q] = zero(SS)
    end

    meta = AMD.Colamd{SS}()
    AMD.colamd_set_defaults(meta.knobs)
    valid = AMD.colamd_l(SS(m), SS(n), SS(len), workspace, p, meta.knobs, meta.stats)
    if !Bool(valid)
        return collect(1:n)
    end
    # colamd writes the column permutation (0-based) into p[1:n]; convert to
    # a 1-based `Vector{Int}` in place (SS_Int == Int on 64-bit, but keep the
    # explicit conversion so this stays correct on a hypothetical 32-bit SS).
    perm = Vector{Int}(undef, n)
    @inbounds for k in 1:n
        perm[k] = Int(p[k]) + 1
    end
    return perm
end

end # module
