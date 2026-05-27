module SparseColumnPivotedQR

using LinearAlgebra
using SparseArrays
using SparseMatricesCSR

import LinearAlgebra: ldiv!, rank
import Base: \, size, eltype

export csr_qr, CSRQRFactorization

# SparseMatricesCSR offset: rowptr stores 1-based if Bi == 1, 0-based if Bi == 0.
# For Bi == 1 (Julia default), no offset needed. For Bi == 0, add 1 to indices.
@inline function getoffset(::SparseMatrixCSR{Bi}) where {Bi}
    return Bi == 1 ? 0 : 1
end

# Factorization object.
#
# Storage layout:
# - R is stored as a list of row-sparse vectors (R_cols[i], R_vals[i]), each kept
#   sorted by column index.
# - The k-th Householder vector v_k is stored "step-wise" as (Vstep_idx[k],
#   Vstep_val[k]): the row indices where v_k is nonzero (sorted), and the
#   corresponding values. This makes `applyQ` / `applyQH` O(sum nnz(v_k))
#   rather than O(m * nstep).
struct CSRQRFactorization{T, RT}
    m::Int
    n::Int
    R_cols::Vector{Vector{Int}}
    R_vals::Vector{Vector{T}}
    Vstep_idx::Vector{Vector{Int}}
    Vstep_val::Vector{Vector{T}}
    tau::Vector{T}
    perm::Vector{Int}
    rnk::Int
    tol::RT
end

LinearAlgebra.rank(F::CSRQRFactorization) = F.rnk
Base.size(F::CSRQRFactorization) = (F.m, F.n)
Base.size(F::CSRQRFactorization, d::Integer) = d == 1 ? F.m : (d == 2 ? F.n : 1)
Base.eltype(::CSRQRFactorization{T}) where {T} = T

# ---- Small helpers on sorted-(cols,vals) sparse rows ----

@inline function row_get(cols::Vector{Int}, vals::Vector{T}, c::Int) where {T}
    idx = searchsortedfirst(cols, c)
    if idx <= length(cols) && cols[idx] == c
        return vals[idx]
    end
    return zero(T)
end

@inline function row_remove!(cols::Vector{Int}, vals::Vector{T}, c::Int) where {T}
    idx = searchsortedfirst(cols, c)
    if idx <= length(cols) && cols[idx] == c
        deleteat!(cols, idx)
        deleteat!(vals, idx)
        return true
    end
    return false
end

@inline function row_set!(cols::Vector{Int}, vals::Vector{T}, c::Int, v::T) where {T}
    idx = searchsortedfirst(cols, c)
    if idx <= length(cols) && cols[idx] == c
        vals[idx] = v
    else
        insert!(cols, idx, c)
        insert!(vals, idx, v)
    end
    return nothing
end

# Convert a SparseMatrixCSR to a list of (cols, vals) row arrays (sorted by col).
function _csr_to_rows(A::SparseMatrixCSR{Bi, T}) where {Bi, T}
    m, n = size(A)
    cols = Vector{Vector{Int}}(undef, m)
    vals = Vector{Vector{T}}(undef, m)
    rowptr = A.rowptr
    colval = A.colval
    nzval = A.nzval
    off = getoffset(A)
    for i in 1:m
        r1 = rowptr[i] + off
        r2 = rowptr[i + 1] + off - 1
        nz = r2 - r1 + 1
        ci = Vector{Int}(undef, nz)
        vi = Vector{T}(undef, nz)
        k = 1
        for p in r1:r2
            ci[k] = Int(colval[p]) + off
            vi[k] = nzval[p]
            k += 1
        end
        if !issorted(ci)
            pp = sortperm(ci)
            ci = ci[pp]
            vi = vi[pp]
        end
        cols[i] = ci
        vals[i] = vi
    end
    return cols, vals
end

"""
    csr_qr(A::SparseMatrixCSR; tol=nothing) -> CSRQRFactorization

Column-pivoted Householder QR factorization of `A` operating directly on CSR storage.

The factorization satisfies `A[:, perm] ≈ Q * R` where `Q = H_1 * H_2 * ... * H_rnk`
is a product of Householder reflectors `H_k = I - tau_k * v_k * v_k^H` and `R` is
upper-triangular in the first `rnk` columns; the remaining `n - rnk` columns of `R`
and the bottom `m - rnk` rows of `R` are truncated to zero by the rank-revealing stop.

If `tol === nothing`, a default of `eps(real(T)) * max(m, n) * ||A||_F` is used,
matching LAPACK's `xgeqp3`-style rank-revealing default.

The solve `F \\ b` returns the back-substituted least-squares solution with the trailing
`n - rnk` coordinates of the rotated solution set to zero (a "basic" solution; not the
minimum-norm pseudoinverse solution, but it always satisfies `A x ≈ projection(b, range(A))`).
"""
function csr_qr(A::SparseMatrixCSR{Bi, T}; tol::Union{Nothing, Real}=nothing) where {Bi, T}
    m, n = size(A)
    R_cols, R_vals = _csr_to_rows(A)

    Vstep_idx = Vector{Vector{Int}}()
    Vstep_val = Vector{Vector{T}}()
    tau       = T[]

    perm = collect(1:n)

    RT = real(T)
    col_nrm2 = zeros(RT, n)
    @inbounds for i in 1:m
        ci = R_cols[i]; vi = R_vals[i]
        for q in eachindex(ci)
            col_nrm2[ci[q]] += abs2(vi[q])
        end
    end
    # Save initial column norms² so we can detect catastrophic cancellation
    # during downdate (Drmac-Bujanovic style) and trigger recompute.
    col_nrm2_init = copy(col_nrm2)

    fro = sqrt(sum(col_nrm2))
    tol_use = tol === nothing ? RT(eps(RT) * max(m, n)) * fro : RT(max(tol, 0))
    tol2 = tol_use * tol_use

    kmax = min(m, n)
    rnk = kmax

    # Reusable workspaces
    w = zeros(T, n)
    w_touched = falses(n)   # bool flag — robust to w[j] passing through 0 by cancellation
    mark_cols = Int[]
    # Scratch buffers for the per-row merge in the Householder application step.
    new_cols_buf = Int[]
    new_vals_buf = T[]
    sizehint!(new_cols_buf, 2n)
    sizehint!(new_vals_buf, 2n)
    # Householder x-vector scratch
    x_idx = Int[]
    x_val = T[]
    v_vals = T[]

    for k in 1:kmax
        # --- Pivot column selection: argmax col_nrm2[k..n] ---
        p = k
        best = col_nrm2[k]
        @inbounds for j in (k + 1):n
            if col_nrm2[j] > best
                best = col_nrm2[j]
                p = j
            end
        end

        # Rank-deficiency stop. Before declaring full rank, do a final recompute of
        # the candidate pivot column's norm if it's suspiciously small relative to
        # initial (avoid keeping a column whose downdated norm is just rounding noise).
        if col_nrm2_init[p] > 0 && best <= sqrt(eps(RT)) * col_nrm2_init[p]
            s = zero(RT)
            @inbounds for ii in k:m
                ci2 = R_cols[ii]; vi2 = R_vals[ii]
                idx2 = searchsortedfirst(ci2, p)
                if idx2 <= length(ci2) && ci2[idx2] == p
                    s += abs2(vi2[idx2])
                end
            end
            best = s
            col_nrm2[p] = s
            col_nrm2_init[p] = s
        end
        if best <= tol2
            rnk = k - 1
            break
        end

        # --- Swap columns k and p in R, perm, and norms ---
        # Must swap entries in ALL rows (including rows above k) because column p's
        # values in rows < k are also "R" entries that participate in back-substitution
        # and must be in the new column position.
        if p != k
            col_nrm2[k], col_nrm2[p] = col_nrm2[p], col_nrm2[k]
            col_nrm2_init[k], col_nrm2_init[p] = col_nrm2_init[p], col_nrm2_init[k]
            perm[k], perm[p] = perm[p], perm[k]
            @inbounds for i in 1:m
                ci = R_cols[i]; vi = R_vals[i]
                vk = row_get(ci, vi, k)
                vp = row_get(ci, vi, p)
                if vk == 0 && vp == 0
                    continue
                end
                vk != 0 && row_remove!(ci, vi, k)
                vp != 0 && row_remove!(ci, vi, p)
                vp != 0 && row_set!(ci, vi, k, vp)
                vk != 0 && row_set!(ci, vi, p, vk)
            end
        end

        # --- Gather column-k entries from rows k..m to build Householder x ---
        empty!(x_idx); empty!(x_val)
        @inbounds for i in k:m
            ci = R_cols[i]; vi = R_vals[i]
            idx = searchsortedfirst(ci, k)
            if idx <= length(ci) && ci[idx] == k
                v = vi[idx]
                if v != 0
                    push!(x_idx, i)
                    push!(x_val, v)
                end
            end
        end

        if isempty(x_idx)
            rnk = k - 1
            break
        end

        # Ensure row k is first (so v[1] corresponds to row k -> diagonal slot)
        if x_idx[1] != k
            pushfirst!(x_idx, k)
            pushfirst!(x_val, zero(T))
        end

        # --- Compute Householder reflector: alpha, v, tau ---
        normx2 = zero(RT)
        @inbounds for q in eachindex(x_val)
            normx2 += abs2(x_val[q])
        end
        normx = sqrt(normx2)

        x1 = x_val[1]
        # Choose sign to avoid cancellation
        if T <: Real
            sgn = x1 >= 0 ? one(T) : -one(T)
        else
            sgn = x1 == 0 ? one(T) : x1 / abs(x1)
        end
        alpha = -sgn * normx

        resize!(v_vals, length(x_val))
        copyto!(v_vals, x_val)
        v_vals[1] = x1 - alpha

        # tau = 2 / (v^H v), works for real and complex with H = I - tau v v^H
        vnorm2 = zero(RT)
        @inbounds for q in eachindex(v_vals)
            vnorm2 += abs2(v_vals[q])
        end

        if vnorm2 == 0
            rnk = k - 1
            break
        end

        tau_k = T(2) / T(vnorm2)

        # --- Apply H = I - tau v v^H to columns j > k of rows in x_idx ---
        # Step 1: w[j] = v^H * R[:, j] = sum_i conj(v[i]) * R[i, j]   for j > k
        # We use w_touched[j] (Bool) to detect first-time visits so the same j is
        # pushed onto mark_cols only once, even if w[j] happens to pass through 0
        # due to cancellation.
        empty!(mark_cols)
        nrows_v = length(x_idx)
        @inbounds for q in 1:nrows_v
            i = x_idx[q]
            vi_q = v_vals[q]
            if vi_q == 0
                continue
            end
            cvi_q = conj(vi_q)
            ci = R_cols[i]; vi = R_vals[i]
            start = searchsortedfirst(ci, k + 1)
            for p2 in start:length(ci)
                j = ci[p2]
                if !w_touched[j]
                    w_touched[j] = true
                    push!(mark_cols, j)
                end
                w[j] += cvi_q * vi[p2]
            end
        end

        # Sort marked cols once.
        sort!(mark_cols)

        # Step 2: for each row i with v[i] != 0, update R[i, j>k] -= tau * v[i] * w[j]
        # using a merge of existing-row-tail with marked cols. Uses scratch buffers
        # `new_cols_buf` and `new_vals_buf` to avoid per-row allocations.
        @inbounds for q in 1:nrows_v
            i = x_idx[q]
            vi_q = v_vals[q]
            if vi_q == 0
                continue
            end
            factor = tau_k * vi_q
            ci = R_cols[i]; vi = R_vals[i]
            start = searchsortedfirst(ci, k + 1)
            la = length(ci) - start + 1
            lb = length(mark_cols)

            empty!(new_cols_buf); empty!(new_vals_buf)

            a = 1; b = 1
            while a <= la && b <= lb
                ca = ci[start + a - 1]
                cb = mark_cols[b]
                if ca == cb
                    nv = vi[start + a - 1] - factor * w[cb]
                    if nv != 0
                        push!(new_cols_buf, ca); push!(new_vals_buf, nv)
                    end
                    a += 1; b += 1
                elseif ca < cb
                    push!(new_cols_buf, ca); push!(new_vals_buf, vi[start + a - 1])
                    a += 1
                else
                    nv = -factor * w[cb]
                    if nv != 0
                        push!(new_cols_buf, cb); push!(new_vals_buf, nv)
                    end
                    b += 1
                end
            end
            while a <= la
                push!(new_cols_buf, ci[start + a - 1]); push!(new_vals_buf, vi[start + a - 1])
                a += 1
            end
            while b <= lb
                cb = mark_cols[b]
                nv = -factor * w[cb]
                if nv != 0
                    push!(new_cols_buf, cb); push!(new_vals_buf, nv)
                end
                b += 1
            end

            # Replace ci[start:end] with new_cols_buf (and vi correspondingly).
            new_tail_len = length(new_cols_buf)
            new_total_len = start - 1 + new_tail_len
            old_total_len = length(ci)
            if new_total_len > old_total_len
                resize!(ci, new_total_len)
                resize!(vi, new_total_len)
            elseif new_total_len < old_total_len
                resize!(ci, new_total_len)
                resize!(vi, new_total_len)
            end
            for t in 1:new_tail_len
                ci[start + t - 1] = new_cols_buf[t]
                vi[start + t - 1] = new_vals_buf[t]
            end
        end

        # Reset w and w_touched
        @inbounds for j in mark_cols
            w[j] = zero(T)
            w_touched[j] = false
        end

        # --- Set R[k, k] = alpha; drop R[i, k] for i > k (they go to v storage) ---
        @inbounds begin
            ck = R_cols[k]; vk = R_vals[k]
            idx = searchsortedfirst(ck, k)
            if idx <= length(ck) && ck[idx] == k
                vk[idx] = T(alpha)
            else
                insert!(ck, idx, k)
                insert!(vk, idx, T(alpha))
            end
        end
        @inbounds for q in 1:nrows_v
            i = x_idx[q]
            i == k && continue
            row_remove!(R_cols[i], R_vals[i], k)
        end

        # --- Store Householder vector step-wise (efficient applyQ/applyQH) ---
        push!(tau, tau_k)
        # Collect nonzero (i, v_i) pairs from x_idx/v_vals; x_idx is already sorted in
        # increasing row order, so the step storage is sorted too.
        vidx = Int[];   sizehint!(vidx, nrows_v)
        vval = T[];     sizehint!(vval, nrows_v)
        @inbounds for q in 1:nrows_v
            vv = v_vals[q]
            vv == 0 && continue
            push!(vidx, x_idx[q])
            push!(vval, vv)
        end
        push!(Vstep_idx, vidx)
        push!(Vstep_val, vval)

        # --- Downdate column norms for j > k using the new R[k, j] ---
        # Drmac-Bujanovic-style: subtract |R[k, j]|². If the relative size of the
        # remaining norm to the initial norm becomes too small (loss of significant
        # digits), recompute from scratch to maintain accuracy.
        @inbounds begin
            ck = R_cols[k]; vk_ = R_vals[k]
            startc = searchsortedfirst(ck, k + 1)
            # If remaining/initial < sqrt(eps), recompute exactly (lost half the digits).
            recompute_thresh = RT(sqrt(eps(RT)))
            for p2 in startc:length(ck)
                j = ck[p2]
                old = col_nrm2[j]
                col_nrm2[j] = old - abs2(vk_[p2])
                if col_nrm2[j] < 0
                    col_nrm2[j] = zero(RT)
                end
                # If the downdated norm is tiny compared to its initial value, recompute
                # exactly by scanning rows k+1..m at column j.
                if col_nrm2[j] <= recompute_thresh * col_nrm2_init[j]
                    s = zero(RT)
                    for ii in (k + 1):m
                        ci2 = R_cols[ii]; vi2 = R_vals[ii]
                        idx2 = searchsortedfirst(ci2, j)
                        if idx2 <= length(ci2) && ci2[idx2] == j
                            s += abs2(vi2[idx2])
                        end
                    end
                    col_nrm2[j] = s
                    col_nrm2_init[j] = s  # reset reference too
                end
            end
        end
        col_nrm2[k] = zero(RT)
    end

    return CSRQRFactorization{T, RT}(m, n, R_cols, R_vals, Vstep_idx, Vstep_val, tau, perm, rnk, tol_use)
end

# ----- Apply Q^H to a vector y in-place: y <- Q^H y -----
# Q = H_1 ... H_rnk, so Q^H y = H_rnk^H ... H_1^H y. Apply in order k = 1..rnk.
# H_k = I - tau_k v_k v_k^H, so H_k^H = I - conj(tau_k) v_k v_k^H.
function applyQH!(y::AbstractVector{T}, F::CSRQRFactorization{T}) where {T}
    nstep = length(F.tau)
    @inbounds for k in 1:nstep
        vidx = F.Vstep_idx[k]; vval = F.Vstep_val[k]
        nz = length(vidx)
        nz == 0 && continue
        beta = zero(T)
        for q in 1:nz
            beta += conj(vval[q]) * y[vidx[q]]
        end
        if beta == 0
            continue
        end
        tk_conj = conj(F.tau[k])
        for q in 1:nz
            y[vidx[q]] -= tk_conj * vval[q] * beta
        end
    end
    return y
end

# Apply Q to a vector: y <- Q y = H_1 ... H_rnk y. Loop k = rnk..1.
function applyQ!(y::AbstractVector{T}, F::CSRQRFactorization{T}) where {T}
    nstep = length(F.tau)
    @inbounds for k in nstep:-1:1
        vidx = F.Vstep_idx[k]; vval = F.Vstep_val[k]
        nz = length(vidx)
        nz == 0 && continue
        beta = zero(T)
        for q in 1:nz
            beta += conj(vval[q]) * y[vidx[q]]
        end
        if beta == 0
            continue
        end
        tk = F.tau[k]
        for q in 1:nz
            y[vidx[q]] -= tk * vval[q] * beta
        end
    end
    return y
end

# Back-substitute R[1:rnk, 1:rnk] z = c[1:rnk]; trailing entries of z set to 0.
function backsub_R!(z::AbstractVector{T}, F::CSRQRFactorization{T}, c::AbstractVector{T}) where {T}
    rnk = F.rnk
    @inbounds for i in 1:rnk
        z[i] = c[i]
    end
    @inbounds for i in rnk:-1:1
        ci = F.R_cols[i]; vi = F.R_vals[i]
        diag_idx = searchsortedfirst(ci, i)
        if diag_idx > length(ci) || ci[diag_idx] != i
            error("Missing R diagonal at row $i (factorization is corrupt or rank stopped early)")
        end
        s = z[i]
        for p in (diag_idx + 1):length(ci)
            j = ci[p]
            j > rnk && break
            s -= vi[p] * z[j]
        end
        z[i] = s / vi[diag_idx]
    end
    @inbounds for i in (rnk + 1):length(z)
        z[i] = zero(T)
    end
    return z
end

function LinearAlgebra.ldiv!(x::AbstractVector{T}, F::CSRQRFactorization{T}, b::AbstractVector{T}) where {T}
    length(b) == F.m || throw(DimensionMismatch("b length $(length(b)) != m=$(F.m)"))
    length(x) == F.n || throw(DimensionMismatch("x length $(length(x)) != n=$(F.n)"))
    y = copy(b)
    applyQH!(y, F)
    z = Vector{T}(undef, F.n)
    backsub_R!(z, F, y)
    @inbounds for k in 1:F.n
        x[F.perm[k]] = z[k]
    end
    return x
end

function Base.:\(F::CSRQRFactorization{T}, b::AbstractVector{T}) where {T}
    x = zeros(T, F.n)
    ldiv!(x, F, b)
    return x
end

function Base.:\(F::CSRQRFactorization{T}, b::AbstractVector) where {T}
    bb = convert(Vector{T}, b)
    x = zeros(T, F.n)
    ldiv!(x, F, bb)
    return x
end

end # module
